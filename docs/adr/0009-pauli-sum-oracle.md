# ADR 0009 — Pauli-sum oracle: one circuit per measurement group, in one planner call per window

- **Status:** Proposed (2026-11)
- **Contracts:** [C-3 · Measurement counts format](../CONTRACTS.md#c-3--measurement-counts-format), [C-4 · Measurement placement](../CONTRACTS.md#c-4--measurement-placement-terminal-measurements), [C-5 · Optimizer ↔ oracle](../CONTRACTS.md#c-5--optimizer--oracle-contract-polypus-optimizers), [C-7 · Seeding & run manifest](../CONTRACTS.md#c-7--seeding--run-manifest-python-entry-points); C-12 (call trace) is unchanged
- **Supersedes:** —
- **Issue / PRs:** #531 (epic #528, step 3 of #424); builds on #529 (multi-qubit `PauliSum`, exact H2) and #530 (qubit-wise grouping and parity reducer)
- **Amended by:** #667 (2026-11): decision 6 runs the candidates in windows, one call per window; every group's counts must total the shots; zero-coefficient terms are in no group
- **Amended by:** #536 (2026-11): decision 1 gains an exact parameter-shift gradient for templates whose parameters are single-frequency coordinates (decision 8)

## Context

Only costs diagonal in the computational basis can be trained today. A
`CostObservable` reduces **one** counts map per candidate, and `VqcOracle`
builds **one** circuit per candidate. A Hamiltonian such as H2's is a sum of
multi-qubit Pauli strings (`Z0 Z1`, `X0 X1 Y2 Y3`, …): hardware measures every
qubit in the `Z` basis, so a term with `X` or `Y` factors is measured by
rotating those qubits first.

#530 added the pure pieces, in `polypus-observable`: `group_qubit_wise`
partitions a sum into qubit-wise-commuting groups, each a `MeasurementGroup`
with its basis rotations (`BasisRotation::H`, `BasisRotation::SdgThenH`) and a
`ParityObservable` that turns that group's counts into its contribution, while
identity terms go to `Grouping::constant`. The energy of a state is

```text
E = constant + Σ_g ⟨P_g⟩,   ⟨P_g⟩ = the group's ParityObservable on the counts
                                     of the template followed by its rotations
```

What is missing is an oracle that, for each candidate, binds the template,
measures it once per group, and combines the groups into one value. The
4-qubit Jordan–Wigner H2 of #529 at its equilibrium bond length (0.7414 Å)
has 15 terms, which `group_qubit_wise` splits into 5 groups plus the constant
(`crates/polypus-observable/tests/pauli_sum.rs`). Grouping is greedy and
depends on the term order, and a term can vanish below
`H2_ROUNDING_TOLERANCE` at other bond lengths, so G is not fixed in general.

## Decision

1. **A new, additive oracle, `PauliSumOracle`, and its factory,
   `PauliSumOracleFactory`, in `polypus-evaluation`.** The factory implements
   the current `OracleFactory` contract and returns `Oracles` with an
   evaluation oracle, no minibatch oracle, and, since #536, the exact
   parameter-shift gradient of decision 8 when the template allows it
   *(amended by #536: this was an evaluation oracle only)*. Without that
   gradient Adam falls back to finite differences, as with any oracle
   without one.
   `EvaluationOracle`, the `Planner` and `VqcOracle` do not change, and
   neither does C-5's rule (one finite value per candidate). C-5's text does
   change: its `final_shots` paragraph, which lists the oracles that put
   `shots` in every task (`VqcOracle`, `QmlOracle`), gains `PauliSumOracle`
   and its enforcing test in #531, and its sentence "A candidate is one
   circuit in `train`" gains "and G, one per measurement group, for a
   Pauli-sum cost" when #532 lets `train` use this oracle.

2. **The factory takes the template and the Hamiltonian ungrouped** —
   `PauliSumOracleFactory::new(template, terms)`, `template` a native
   `ParameterizedCircuit` (Qiskit and OpenQASM templates are #533) and
   `terms` `(coefficient, PauliMask)` pairs — and groups it itself with
   `group_qubit_wise`. Grouping is how the sum is measured, not what the
   caller trains, and the factory takes the ungrouped terms, so the exact path
   of #534 can keep them. `OracleFactory::build` cannot fail, so **every check runs in `new`**,
   which returns a `Result`:
   - a template that already measures (a `Measure` or `MeasureAll` anywhere)
     is rejected with a clear error: the rotations are appended after the
     template, and contract C-4 requires measurements to be terminal;
   - the Hamiltonian is checked by `group_qubit_wise` over the template's
     width: at most 64 qubits (`MAX_QUBITS`), finite coefficients, no term
     acting outside the template, and `Σ|c|` within `MAX_WEIGHT_BOUND`;
   - a Hamiltonian with no non-identity term (empty, or identity terms only;
     since #667 also when its other terms all have a zero coefficient, which
     `group_qubit_wise` leaves out of every group) is rejected: it has no
     group to measure, its value does not depend on the candidate, and
     nothing would be trained.

   The G measurement templates (template, then group `g`'s rotations in
   application order, then `measure_all`) are built once, in `new`; each
   candidate only binds them.

3. **The oracle returns `−E(θ)`, the energy with its sign flipped.** The
   optimizers maximise, and a VQE minimises the energy. The sign is fixed, not
   a `scale` argument as on `QuboObservable`: a forgotten `scale = -1` would
   make the optimizer look for the highest energy with no error to say so. A
   run's `best_fitness` and `final_fitness` are therefore `−E`; a caller that
   reports the energy (the Python bindings of #532) flips them back.

4. **Shots are per group.** Each of the G circuits of a candidate is measured
   with the run's `shots`, so a candidate costs `G · shots` shots. This is
   what `shots` already means for `VqcOracle` and the QML oracle (shots per
   circuit), and every task of a batch has the same shots, which is what
   `SequentialPlanner` supports: it runs a whole wave with its first task's
   shots, without checking the rest. With independent groups (decision 6),
   the statistical error of a candidate's value is

   ```text
   σ² = Σ_g Var_g / shots
   ```

   where `Var_g` is the variance of the **per-shot value of group `g`'s
   `ParityObservable`**, `f_g(b) = Σ_{k∈g} c_k (−1)^{popcount(b ∧ m_k)}`. The
   terms of one group are measured on the same shots, so `Var_g` includes
   their covariances; it is **not** `Σ_k c_k² Var(P_k)`. Groups are measured
   on separate circuits, so there is no covariance term between groups.
   Allocating shots across groups is #534 (see Consequences).

5. **`final_shots` measures every group with `final_shots` shots.** The oracle
   overrides `evaluate_batch_with_shots`; otherwise the trait's default would
   score the final batch with the training's shots, silently. Its noise is
   independent of the training's because every backend call, and on the
   native backend every circuit, has its own sampling stream (C-7).

6. **One `Planner::execute` call per window of whole candidates, with
   `W · G` tasks, and the oracle reduces.** *(Amended by #667: this was one
   call per batch, with `N · G` tasks.)* `W` is
   `candidate_window_size(backend, G)`, as the QML oracle sizes its windows,
   so at most `W · G` bound circuits are alive at once; a failing window
   stops the batch before the next is bound. The tasks are candidate-major:
   task `i · G + g` is the window's candidate `i`'s circuit for group `g`,
   groups in `Grouping::groups()` order. `Planner::execute` returns "merged
   counts, one per task, in input order", however many waves it splits the
   window into. The windows change no expected value, since the tasks keep
   their order across windows. The sampled values are the same bit for bit
   on the native backend, which seeds each circuit by its position in the
   run (`circuit_seed`, C-7), not by its call; a backend seeded per call
   (Local/Aer, CUNQA) gives each window's call its own stream, so there the
   noise depends on `W`, and so on the host, as it already does for the QML
   oracle. An empty batch makes no call. The oracle:
   - returns finite sentinels without running anything once the run's
     `OracleErrorSlot` holds an error, as `VqcOracle` does (C-12's trace
     relies on this short-circuit);
   - checks it got `W · G` counts maps (`EvaluationError::WrongLength`
     otherwise), and that each totals the shots asked for (C-3; otherwise
     `EvaluationError::Observable`, naming the group and the candidate, #667:
     the built-in planners already check it, a caller's own may not);
   - reduces map `i · G + g` with group `g`'s `ParityObservable` and returns
     `−(constant + value_{i,0} + … + value_{i,G−1})` for candidate `i`, summed
     in that order (the constant, then the groups in order) so that a seeded
     run is reproducible bit for bit;
   - checks C-5 as `VqcOracle` does (one finite value per candidate) and
     records any failure in the error slot, with finite sentinels.

   It does not call `Planner::evaluate`, whose reducer sees each task's counts
   on their own. Each group's noise is independent of the other groups' where
   the backend gives each circuit of a call its own sampling stream: on the
   native backend (`circuit_seed`, C-7), the only one #531 supports, and on
   CUNQA (per QPU) and Aer (Aer's own per-experiment seeding, which no Polypus
   contract or test pins). C-7 guarantees a third-party backend only one
   stream per call, and `docs/backends.md` allows it to send a whole call
   with one seed; there, two groups of a candidate measured in the same call
   may share random numbers. The bindings of #532/#533 must take this into
   account before this oracle runs on such backends.

7. **The 4-qubit H2 ansatz of the issue lives in the tests** of
   `polypus-evaluation`: the one-parameter double excitation from the
   Hartree–Fock state `|0101⟩` within span{`|0101⟩`, `|1010⟩`}, built from
   `polypus-circuit` gates. States are written with bit `i` for qubit `i`
   (bit 0 rightmost), in the spin-orbital order of `polypus-physics`:
   qubits 0..3 are `σ↑, σ*↑, σ↓, σ*↓`
   (`crates/polypus-physics/src/chemistry/h2.rs`), so `|0101⟩` has `σ↑` and
   `σ↓` occupied and `|1010⟩` has `σ*↑` and `σ*↓` occupied, the closed-shell
   pair of the singlet ground state. It is exact for neutral H2/STO-3G only;
   the oracle is generic, and the general chemistry ansatz generator is #538.
   `polypus-physics` (for H2's Hamiltonian and its exact energy) becomes a
   dev-dependency of `polypus-evaluation`, as it is of `polypus-observable`.

8. **An exact parameter-shift gradient, by shifting the parameter, when every
   parameter is a single-frequency coordinate** *(added by #536)*. `new`
   runs `ParameterizedCircuit::check_single_frequency` on the template once,
   the check Rotosolve's landscape already relies on: every parameter is the
   angle `±θ_k + c` of exactly one `rx`, `ry`, `rz`, `rxx`, `rzz`, `p`, `u1`,
   `u`, `u2` or `u3`. Then the energy, as a function of `θ_k` with the others
   fixed, is one sinusoid `a + b·cos θ_k + c·sin θ_k` for every group, so

   ```text
   ∂E/∂θ_k = [E(θ + π/2·e_k) − E(θ − π/2·e_k)] / 2
   ```

   exactly, with no circuit surgery: the shifted points are ordinary
   candidates, measured by the oracle's own evaluation path (decision 6:
   windows, the `W · G` length check, the shot totals), each group with the
   run's `shots`. The oracle implements `GradientOracle` for the fitness
   `−E` (decision 3), uphill as C-5 requires, and `build` returns **one**
   `Arc<PauliSumOracle>` as both the run's evaluation oracle and its
   gradient, as `QmlOracleFactory` does: C-5 requires `value_and_gradient`'s
   value to come from the optimizer's own evaluation oracle. The oracle:
   - `value_and_gradient(θ, P)` scores `[θ, θ+π/2·e_0, θ−π/2·e_0, …]`, the
     layout of `FiniteDifferenceGradient`, in one batch of `2P + 1`
     candidates (`(2P + 1) · G` circuits), and returns `θ`'s value with the
     gradient, so an Adam iteration is one batch; `scores_theta` is `true`;
   - `gradient_batch` scores the `2P` shifted candidates alone, and
     `gradient(θ, k)` only the two of component `k`;
   - a failure is recorded in the error slot and returns finite zeros, as
     `evaluate_batch` does (decision 6).

   The shifted candidates may fall outside an optimizer's `bounds`; that is
   harmless, since the oracle does not enforce bounds and the angles are
   periodic. Each component carries the noise of two independent values,
   `σ(g_k)² = (σ₊² + σ₋²) / 4`, but no bias, unlike a finite difference,
   whose step trades bias against noise.

   When the check fails (a parameter in two angles, or in a controlled
   rotation, a `u0`, a declared gate, an expression such as `2·θ`, or in no
   angle at all), the factory provides **no** gradient and Adam takes the
   finite difference, as before #536. That is not silent:
   `PauliSumOracleFactory::gradient_unavailable()` returns the
   `SingleFrequencyError` naming the parameter and the gate, and `new` logs
   it once with `log::warn!`. It is not an error either, because the factory
   does not know the optimizer, and DE, PSO and the others need no gradient.

## Deviation from issue #536

The issue asks to "reuse the QML implementation where possible" and to test
the gradient "on the exact path". Neither is possible as written:

- **The QML parameter shift covers fewer gates and is bound to its
  models.** `polypus-qml` shifts each occurrence of a parameter separately
  (`untie` and `classify`, private to its `problem` module), which also
  handles a parameter shared by several gates; the shift and its
  combination are `QmlProblem` methods (`gradient_circuits`). It shifts only
  the Pauli rotations `rx`, `ry`, `rz`, `rxx`, `rzz` with plain `Param`
  angles: there `p`, `u1`, `u`, `u2`, `u3` are differentiated by the
  adjoint only, and expression angles not at all. Reusing it would lose
  gates and `±θ + c` angles that decision 8 accepts, and extracting it would
  refactor `polypus-qml` for shared parameters, a case no Pauli-sum template
  has yet. Decision 8 reuses instead what the Rotosolve path shares,
  `check_single_frequency` in `polypus-circuit`, and shifts the parameter
  itself. A parameter shared by several gates gets no exact gradient for
  now; untying it can extend decision 8 later without changing its API.
- **There is no exact path for Pauli sums yet.** The exact `⟨ψ|H|ψ⟩` on the
  native backend is #534, blocked on #423 at the time of writing. Integer
  counts cannot reach 1e-10, so no fake backend can meet the criterion
  either. The oracle's gradient therefore builds its candidates and combines
  their values in one private function that takes the evaluator as an
  argument: the oracle passes its windowed, shot-based evaluation, and the
  test passes an exact evaluator built from the native backend's exact
  probabilities (`run_circuits_exact`) and each group's `ParityObservable`,
  the same pieces #534's exact path will use. The test cannot pass without
  running the code the oracle ships.

## Deviation from issue #531

### The planner call

The issue offers two ways to call the `Planner` and says to start with the
first:

- *One call per group* (start here): needs no new guarantee and, since every
  call has its own seed (C-7), groups get independent noise; groups run one
  after another.
- *One call* with N×G tasks and a **composite reducer**: faster, but only safe
  if the `Planner` contract guarantees the whole batch reaches the reducer in
  order, never split into waves.

This ADR takes **one call** (per window of whole candidates since #667),
without a composite reducer (decision 6), because the guarantee the issue
asks for already exists, on `execute`:

- The concern is about `Planner::evaluate`, which runs the tasks and hands the
  counts to a `CostObservable`. A reducer that needs all G groups of a
  candidate at once would depend on how `evaluate` passes the counts to it.
- `Planner::execute` does not reduce. Its documented contract
  (`crates/polypus-backend/src/planner.rs`) is: *"Run every task and return
  merged counts, one per task, in input order. Owns the wave loop, the
  concurrency cap, the between-wave cancellation / interrupt check and, if
  shots are split, the C-3 merge."*
- A training run uses its backend's `default_planner()`, which is
  `SequentialPlanner` for every built-in backend (wrapped in `TracingPlanner`
  when the run is traced, which forwards `execute` unchanged).
  `SequentialPlanner` runs the tasks in waves sized by the backend
  (`capabilities_for`) and concatenates their counts in order; every wave is
  checked by `validate_run_results` (`crates/polypus-backend/src/lib.rs`),
  which rejects a wave with the wrong number of counts maps or shots, so a
  misbehaving backend is an error, never a silent mispairing.
  `ShotDistributingPlanner` takes exactly one task and rejects more with a
  typed error (`InvalidCircuitCount`), and the `polypus` edge selects it only
  for a single-circuit run with `n_qpus > 1`, never for training. A
  third-party backend's or a Rust caller's own planner is bound by the same
  documented `execute` contract.
- With the counts of every task in order, the oracle pairs map `i · G + g`
  with candidate `i` and group `g` itself. No composite reducer exists, and
  nothing in the `Planner` contract changes.
- This is how the supervised QML oracle already works
  (`crates/polypus-evaluation/src/qml_oracle.rs`): for each window of
  candidates it calls `execute` once, with `candidates × rows` tasks, and
  pairs each counts map with its candidate and label itself.

One call gives full waves: 10 candidates of 5 groups on a backend that runs 8
circuits at a time take 7 waves in one call, and 10 (8 + 2, five times) in
five. With windows (#667), a window holding all 10 candidates is still one
call; smaller windows lose at most one partial wave each. A test pins the
guarantee decision 6 relies on, through a backend that runs 3 circuits at a
time and encodes each circuit's identity in its counts, with `N · G` not a
multiple of 3, so that a candidate's groups straddle two waves: every
candidate must still get its own value.

What one call (per window) gives up, compared with the issue's starting point:

- **Noise independence on every backend.** With one call per group, two
  groups of a candidate are always in different calls, which C-7 gives
  different streams on every backend. With one call they usually share a
  call, and their noise is independent only where the backend seeds each
  circuit of a call separately (decision 6): the native backend, the only one
  #531 supports, does.
- **Different shots per group.** Every task of one call has the same shots
  (decision 4); see the consequence for #534 below.

### The H2 states

The issue writes the ansatz's states as `|1100⟩` and `|0011⟩`, in an
interleaved, big-endian notation. In `polypus-physics`'s block spin-orbital
order with bit `i` for qubit `i` they are `|0101⟩` and `|1010⟩` (decision 7);
taken literally in that order, `|1100⟩` and `|0011⟩` would put both electrons
in the same spin, outside the singlet sector, and the ansatz could not reach
the FCI energy.

## Consequences

- VQE-style costs can be trained from Rust on native templates. Python
  (#532), Qiskit/OpenQASM templates (#533), shot allocation and the exact
  native expectation (#534) build on this oracle.
- A candidate costs `G` circuits and `G · shots` shots; the docs say so.
- **Shot allocation (#534) cannot reuse one call as it is.** #534 gives
  groups different shots, also in the `final_shots` batch. `SequentialPlanner`
  runs a whole wave with its first task's shots without checking the rest, so
  tasks with different shots in one `execute` call would be measured with the
  wrong shots, silently. #534 must either make one `execute` call per
  distinct shot count (with per-group allocation, close to one call per
  group, which needs no engine change), or change the `Planner` so that it
  splits waves where the shots change (an engine-contract change, with its own
  ADR and the matching C-12 trace change).
- All `N · G` bound circuits of a batch were held at once, fine for H2
  (G = 5) but not once G grows (FCIDUMP molecules, #535, with the deeper
  ansatzes of #538). #667 windows the candidates as the QML oracle does, with
  `candidate_window_size(backend, G)`; windows are whole candidates, so the
  pairing of decision 6 is unchanged. A trace (C-12) then shows one `batch`
  line per window, as it does for the QML oracle.
- C-5's text gains this oracle (decision 1), and since #536 its exact
  gradient (decision 8): Adam on a single-frequency ansatz, such as the H2
  ansatz of decision 7, steps on an unbiased gradient at `2P + 1` candidates
  per iteration, the cost of the finite difference it replaces.

## Validation

The acceptance criteria of #531, and the decisions above, map to tests:

- **Oracle at the analytic optimum** of the H2 ansatz, fixed seed: `−value`
  within 3σ of exact diagonalization (`polypus-physics`), σ from decision 4's
  formula with each `Var_g` computed from that evaluation's sampled counts.
  This also pins the sign of decision 3.
- **End to end:** unreduced 4-qubit H2 with DE, the native backend and
  `final_shots`. `final_fitness` is `−E` of `final_params`, still slightly
  optimistic with DE (the best of up to 5 candidates, C-5), so, following
  the epic's statistical rules, the test then evaluates `final_params` once
  more with `evaluate_batch_with_shots(&[final_params], final_shots)`, a new
  backend call and so a new stream (C-7), and compares `−value` with FCI
  within 3σ, σ from that evaluation's group variances. Seeds are fixed
  beforehand.
- **`final_shots`** uses shots independent of training: every task of the
  final batch carries `final_shots`, and its values do not repeat the
  optimizer's.
- **`new`'s checks:** a template that measures; a term acting outside the
  template; a non-finite coefficient; more than 64 qubits; `Σ|c|` above the
  weight bound; a Hamiltonian with no non-identity term. Each is rejected
  with a clear error.
- **The basis rotations** are pinned by a term with an odd number of `Y` on a
  state with complex amplitudes (`⟨Y₀⟩ = −sin θ` after `Rx(θ)`), which the H2
  test alone cannot do: every H2 term has an even number of `Y`, so `S` for
  `S†` would cancel there.
- **The pairing across waves:** the test described in the deviation above.
- **Failures:** a wrong number of counts maps is `WrongLength`, and a counts
  map that does not total the shots (empty, short, over, or overflowing
  `u64`) is an `Observable` error naming the group and the candidate in the
  batch (#667); both are recorded in the error slot with finite sentinels,
  and a recorded failure short-circuits later batches without calling the
  backend.
- **Windows (#667):** on the native backend, five noisy candidates give the
  same values bit for bit in windows of 1, 2, 3 or all five, each call
  carrying at most `W · G` tasks; a failing window stops the batch before
  the next call; an empty batch makes no call.
- **Zero terms (#667):** a zero-coefficient term opens no group and widens
  none, and a repeated string joins its twin's group.
- **Gradient (#536), exact:** the oracle's shifted candidates and their
  combination, with each value from the exact evaluator of the deviation
  above, give the value `−E(θ)` and the gradient `−∂E/∂θ` of the H2
  ansatz's analytic `E(θ)` within 1e-10 at several `θ`, the optimum
  included; a two-parameter template with a closed-form energy pins that
  component `k` is `−∂E/∂θ_k`.
- **Gradient (#536), with shots:** one batch of `2P + 1` candidates per
  `value_and_gradient`; H2's gradient within 3σ of the analytic derivative,
  σ from the group variances, fixed seed; a template that fails
  `check_single_frequency` gets no gradient, names the reason and logs it;
  a failure is recorded with finite zeros; and Adam, in a training run,
  takes this gradient instead of the finite difference.
- The determinism gate stays green and existing flows are unchanged.

## Alternatives considered

- **A `scale` argument** instead of the fixed sign (decision 3): consistent
  with `QuboObservable`, but a forgotten `scale = -1` maximises the energy
  silently.
- **`shots` as the total per candidate, split across groups** (decision 4):
  closer in cost to a diagonal cost, but uneven splits (`1000 / 3`) and
  `shots < G` need rules, groups would no longer share one shot count, and
  allocating shots across groups is #534's subject.
- **One planner call per group** (decision 6): the issue's starting point; see
  the deviation above for what it would keep and what one call gains.
- **A `Grouping` as input** (decision 2): would let a caller choose another
  grouping, but every caller would have to group first, and the oracle would
  lose the ungrouped terms #534 needs. A second constructor can add it later
  without breaking anything.
- **The QML occurrence shift, extracted** (decision 8): handles shared
  parameters, but shifts only `rx`, `ry`, `rz`, `rxx`, `rzz` with plain
  angles, and refactors `polypus-qml` for a case no Pauli-sum template has
  yet; a copy of it in `polypus-evaluation` would duplicate it instead.
- **The exact path now** (decision 8): the oracle's own exact `⟨ψ|H|ψ⟩`
  is #534's subject, blocked on #423 at the time of writing.
- **Rejecting a template without a gradient** (decision 8): `new` does not
  know the optimizer, so it would forbid DE or PSO on a template they can
  train.
- **A public H2 ansatz** (decision 7): reusable at once, but public API that
  the general generator of #538 would replace, and in `polypus-physics` a new
  dependency on `polypus-circuit`.
