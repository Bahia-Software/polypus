//! PyO3 bindings for the native Rust circuit layer (`polypus-circuit`).
//!
//! Exposes `polypus.Circuit` and `polypus.Param` so Python users can build
//! GIL-free circuits and pass them to `polypus.run_quantum_circuit` /
//! `polypus.train` exactly like a Qiskit `QuantumCircuit`.

use numpy::{IntoPyArray, PyArray1};
use polypus_circuit::{
    CircuitError, ConcreteCircuit, GateInstruction, GateParam, ParameterizedCircuit,
};
use polypus_sim::{SimError, Simulator, Statevector, StatevectorSimulator};
use pyo3::exceptions::{PyKeyboardInterrupt, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::infrastructure::{attach_or, check_statevector_fits};

/// Map a native [`CircuitError`] onto a Python `ValueError`.
fn to_py_err(e: CircuitError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// Reference to the trainable parameter at a given index.
///
/// ```python
/// qc = polypus.Circuit(2).rx(0, polypus.Param(0)).rzz(0, 1, polypus.Param(1))
/// ```
#[pyclass(module = "polypus", frozen, skip_from_py_object)]
#[derive(Clone, Copy)]
pub struct Param {
    /// Index into the parameter vector bound at execution time.
    #[pyo3(get)]
    pub index: usize,
}

#[pymethods]
impl Param {
    #[new]
    fn new(index: usize) -> Self {
        Param { index }
    }

    fn __repr__(&self) -> String {
        format!("Param({})", self.index)
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.index == other.index
    }
}

/// Accepts either a plain number (fixed angle) or a [`Param`] reference.
pub(crate) enum AngleArg {
    Fixed(f64),
    Param(usize),
}

impl From<AngleArg> for GateParam {
    fn from(value: AngleArg) -> Self {
        match value {
            AngleArg::Fixed(v) => GateParam::Fixed(v),
            AngleArg::Param(i) => GateParam::Param(i),
        }
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for AngleArg {
    type Error = PyErr;

    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        if let Ok(p) = ob.extract::<PyRef<'_, Param>>() {
            return Ok(AngleArg::Param(p.index));
        }
        if let Ok(v) = ob.extract::<f64>() {
            return Ok(AngleArg::Fixed(v));
        }
        Err(PyTypeError::new_err(
            "angle must be a number or polypus.Param",
        ))
    }
}

/// Native Rust quantum circuit with OpenQASM 2.0 and OpenQASM 3 import and
/// export.
///
/// Construction and parameter binding run in pure Rust (no GIL), which makes
/// per-candidate binding during training parallelisable in ways a Qiskit
/// circuit can never be. Methods return `self`, so calls can be chained:
///
/// ```python
/// qc = (polypus.Circuit(3)
///       .h(0).h(1).h(2)
///       .rzz(0, 1, polypus.Param(0))
///       .rx(0, polypus.Param(1))
///       .measure_all())
/// polypus.train(qc, polypus.DE(...), dimensions=2, ...)
/// ```
#[pyclass(module = "polypus")]
pub struct Circuit {
    pub(crate) inner: ParameterizedCircuit,
}

impl Circuit {
    /// Access the wrapped native circuit (used by the entry points to build
    /// `CircuitSource::Native` / `BoundCircuit::Qasm2`).
    pub(crate) fn native(&self) -> &ParameterizedCircuit {
        &self.inner
    }
}

/// Push `gate` onto the builder, translating errors to Python exceptions and
/// returning `self` for chaining.
fn push(mut slf: PyRefMut<'_, Circuit>, gate: GateInstruction) -> PyResult<PyRefMut<'_, Circuit>> {
    slf.inner.try_push(gate).map_err(to_py_err)?;
    Ok(slf)
}

/// Compute the full statevector of a native `polypus.Circuit` with the pure-Rust
/// `polypus-sim` backend (no Qiskit, no OpenQASM round-trip).
///
/// `params` supplies one value per free parameter; omit it for a circuit that
/// has none. Returns the `2^n` complex amplitudes as a **`numpy.ndarray` of
/// `dtype=complex128`**, in Qiskit little-endian order (qubit 0 is the
/// least-significant bit), so it can be compared directly with
/// `qiskit.quantum_info.Statevector` (whose `.data` is the same dtype) and fed
/// straight into NumPy without a conversion step.
///
/// The simulation runs **GIL-free** — only the cheap parameter binding and the
/// conversion of the result hold the GIL — so other Python threads keep making
/// progress while it runs (see `docs/ENGINEERING.md` §3).
///
/// **Interruptible.** A `KeyboardInterrupt` takes effect *mid-simulation*
/// (issue #110): the gate loop calls back into this function at checkpoints at
/// least ~25ms apart, each of which reacquires the GIL just long enough to
/// check for a pending signal — and, if one is pending, abandons the run and
/// raises that original exception verbatim. The throttling lives inside
/// `polypus-sim` and is what keeps this cheap: a circuit that finishes inside
/// one interval never calls back at all, the frequency does not depend on how
/// expensive an individual gate is, and the interval stretches to keep the
/// callback under ~5% of the run (which matters when another Python thread
/// holds the GIL and reacquiring it costs milliseconds). What is *not*
/// interruptible is the `2^n` allocation below, which is a single `vec![]`.
///
/// **Qubit ceiling.** A dense statevector needs `2^n` complex amplitudes
/// (`16 · 2^n` bytes), so the backend refuses circuits with more than
/// [`polypus_sim::MAX_QUBITS`] qubits (30 ≈ 16 GiB) and raises a `ValueError`
/// naming the requested and the supported count **before** attempting any
/// allocation. `polypus.Circuit` itself is deliberately unbounded (see
/// `Circuit::new`): the ceiling belongs to this backend, not to the IR. Note
/// that what a call near the ceiling spends its time on is *allocating* that
/// buffer, not handing it to Python: even a gateless 30-qubit circuit costs ~5s
/// (measured), essentially all of it `Statevector::new`'s 16 GiB `vec![]`. The
/// ceiling bounds memory, not wall-clock time — cost scales with gates × `2^n`
/// and nothing bounds the gate count, which is why the run has to stay
/// interruptible (see above) rather than relying on being short.
///
/// **Memory budget.** Below the ceiling, the `16 · 2^n`-byte statevector is also
/// checked against the same memory budget the native backend applies (issue
/// #215): `POLYPUS_MEM_BUDGET` when set, else the detected RAM/cgroup limit minus
/// a safety reserve. One that does not fit raises
/// `polypus.InsufficientMemoryError` (a `polypus.PolypusError`) before anything
/// is allocated, instead of the process being killed by the out-of-memory killer.
/// With no limit detected (macOS, Windows) nothing is refused. The ceiling's
/// `ValueError` is checked first and is unchanged.
///
/// **`fusion`** (default `true`) controls [`polypus_sim::StatevectorSimulator::fusion`]:
/// whether consecutive gates may be fused into fewer buffer passes before
/// applying them. Fusion only changes performance, never the result (up to
/// floating-point rounding order); pass `fusion=False` for a strictly
/// gate-by-gate simulation of exactly the circuit as written.
///
/// ```python
/// import polypus
/// qc = polypus.Circuit(2).h(0).cx(0, 1)
/// amps = polypus.statevector(qc)          # array([0.707…+0j, 0j, 0j, 0.707…+0j])
/// ```
#[pyfunction(signature = (qc, params = None, fusion = true))]
pub fn statevector<'py>(
    qc: PyRef<'py, Circuit>,
    params: Option<Vec<f64>>,
    fusion: bool,
) -> PyResult<Bound<'py, PyArray1<polypus_sim::C64>>> {
    let params = params.unwrap_or_default();
    // This entry point is always the native statevector path, so surface the
    // one-time, default-visible warning if the gate-parallel threshold fell back
    // to the static default (uncalibrated machine, or stale calibration).
    super::calibration::warn_if_using_default_threshold(qc.py())?;
    // Binding stays on this side of the release: it is O(gates), allocates
    // nothing of size `2^n`, and reads the circuit through the `PyRef`.
    let concrete = qc.native().assign_parameters(&params).map_err(to_py_err)?;
    // Refuse a statevector the memory budget cannot hold before allocating it
    // (issue #215). Above the ceiling the simulator's `ValueError` stays the error.
    if concrete.num_qubits <= polypus_sim::MAX_QUBITS {
        check_statevector_fits(concrete.num_qubits)
            .map_err(|e| crate::exceptions::insufficient_memory_to_pyerr(&e))?;
    }
    // The simulation is the expensive, pure-Rust part: release the GIL for it
    // so it cannot stall every other Python thread (docs/ENGINEERING.md §3).
    //
    // Cancellation hook: `polypus-sim`'s gate loop calls this closure
    // periodically (throttled on its side to ≥25ms apart *and* to a small
    // fraction of the run, so the frequency is decoupled from gate cost and a
    // fast circuit never pays for it). Reacquiring the GIL from inside
    // `detach` is a *fresh* attach, not a reuse: `detach` resets the thread's
    // attachment, so `Python::attach` here could panic, or attach to a
    // finalizing interpreter, if the interpreter were shutting down (e.g. a
    // daemon thread at exit, on any Python version). The hook therefore
    // attaches with `attach_or` (§9, see `poll_for_cancellation`) and, when the
    // interpreter cannot be reached, reports a cancellation: the run's result
    // could never be handed back to Python anyway, so it stops at once.
    // `check_signals()` is what turns a pending SIGINT into a
    // `KeyboardInterrupt` *mid-run*, since releasing the GIL does not by itself
    // process signals (§3).
    //
    // The `PyErr` cannot travel back through `SimError` (`polypus-sim` is
    // Python-free by design, §2), so it is stashed here and recovered below —
    // the same problem `OracleErrorSlot` solves for the optimizer oracles, and
    // the same reason: a generic trait boundary must not downgrade the real
    // exception. No `Arc<Mutex<...>>` is needed, though: `statevector` is
    // single-shot, and the hook runs on this very thread (the simulation is
    // inline in `detach`, not on a worker), so a `&mut` local is enough.
    // That is also why `check_signals()` is effective at all: it is a no-op off
    // the main thread, and this closure runs on whichever thread called us.
    let mut pending: Option<PyErr> = None;
    let outcome = qc
        .py()
        .detach(|| simulate_cancellable(&concrete, fusion, &mut pending));
    let sv = match outcome {
        Ok(sv) => sv,
        // Propagate the *real* pending exception verbatim. Mapping this through
        // `PyValueError::new_err(e.to_string())` like the variants below would
        // silently downgrade a `KeyboardInterrupt` into a bogus `ValueError`
        // (and `check_signals()` cannot be re-run to recover it: raising it
        // cleared the pending flag).
        Err(SimError::Cancelled) => return Err(cancellation_error(pending)),
        Err(e) => return Err(PyValueError::new_err(e.to_string())),
    };
    // A signal that arrived after the last checkpoint (or during the `2^n`
    // allocation, which has no checkpoint) is still honored here, the moment
    // the GIL is reacquired and before the amplitudes are handed to NumPy.
    qc.py().check_signals()?;
    // `into_amplitudes` (not `amplitudes().to_vec()`) so the `2^n`-element
    // buffer moves into the array instead of being copied: `sv` is owned here
    // and dropped immediately after.
    Ok(sv.into_amplitudes().into_pyarray(qc.py()))
}

/// The part of `statevector` that runs detached: simulate `circuit`, polling
/// [`poll_for_cancellation`] at the simulator's checkpoints. A pending signal
/// is stashed in `pending`. Pure Rust apart from that hook, so it can be tested
/// without an interpreter.
pub(super) fn simulate_cancellable(
    circuit: &ConcreteCircuit,
    fusion: bool,
    pending: &mut Option<PyErr>,
) -> Result<Statevector, SimError> {
    let mut cancelled = || poll_for_cancellation(pending);
    StatevectorSimulator {
        fusion,
        ..StatevectorSimulator::new()
    }
    .run_cancellable(circuit, Some(&mut cancelled))
}

/// The exception a cancelled `statevector` raises: the pending signal's own
/// (a Ctrl+C stays a `KeyboardInterrupt`), or, when the hook cancelled because
/// the interpreter became unreachable (shutdown), a `KeyboardInterrupt` too.
/// That matches what the planner's guard raises for the same event
/// (`BackendError::Aborted`): it is an abort, not an invalid input.
pub(super) fn cancellation_error(pending: Option<PyErr>) -> PyErr {
    pending.unwrap_or_else(|| {
        PyKeyboardInterrupt::new_err(
            "the simulation was aborted: the Python interpreter is no longer available",
        )
    })
}

/// Body of the `statevector` cancellation hook, run while detached: `true`
/// cancels the simulation. A pending signal is stashed in `pending` so the real
/// exception can be re-raised; an interpreter that cannot be attached to
/// (shutting down) cancels too, without a pending exception.
pub(super) fn poll_for_cancellation(pending: &mut Option<PyErr>) -> bool {
    attach_or(
        || true,
        |py| match py.check_signals() {
            Ok(()) => false,
            Err(err) => {
                *pending = Some(err);
                true
            }
        },
    )
}

#[pymethods]
impl Circuit {
    /// Build an empty circuit on `num_qubits` qubits.
    ///
    /// **Intentionally unbounded.** A `Circuit` is backend-agnostic IR: the same
    /// object is routed by `polypus.run_quantum_circuit` to the native
    /// statevector simulator, CUNQA, QMIO or Aer, and those capacities differ.
    /// [`polypus_sim::MAX_QUBITS`] is the dense-statevector limit and is
    /// enforced where it applies — when a circuit is *simulated* (see
    /// [`statevector`]) — so checking it here would reject circuits that are
    /// perfectly valid for the other backends.
    #[new]
    fn new(num_qubits: usize) -> Self {
        Circuit {
            inner: ParameterizedCircuit::new(num_qubits),
        }
    }

    /// Import an OpenQASM 2.0 program (inverse of [`to_qasm2`](Circuit::to_qasm2)).
    ///
    /// Accepts the QASM this class exports plus Qiskit's `qasm2.dumps` output,
    /// `gate` declarations included (every instruction — `p`, `u1`, `u2`, `u`,
    /// `id` and calls of declared gates included — is kept one-to-one under its
    /// own spelling; only the builtins `U`/`CX` become `u`/`cx`; multiple
    /// registers are flattened in declaration order). The
    /// result is fully concrete (`num_params == 0`); builder methods can keep
    /// extending it.
    ///
    /// ```python
    /// qc = polypus.Circuit.from_qasm2(qiskit.qasm2.dumps(qiskit_circuit))
    /// ```
    #[staticmethod]
    fn from_qasm2(source: &str) -> PyResult<Self> {
        Ok(Circuit {
            inner: ParameterizedCircuit::from_qasm2(source).map_err(to_py_err)?,
        })
    }

    /// Import a program in the **OpenQASM 3 profile with Qiskit phase
    /// conventions** (inverse of [`to_qasm3`](Circuit::to_qasm3)).
    ///
    /// The profile is the straight-line part of OpenQASM 3 that carries a
    /// parameterised, terminal-measurement circuit: `include "stdgates.inc";`
    /// (provided internally; no file is ever read), `qubit` and `bit`
    /// registers, `input float[64]` parameters, calls of `U`, of the
    /// `stdgates.inc` gates and of gates declared with `gate` blocks,
    /// `barrier`, and measurements assigned to bits. Angles are expressions of
    /// the inputs and the constants `pi`, `tau` and `euler`, evaluated in
    /// binary64 exactly as written. Anything else — control flow, `reset`,
    /// classical computation, subroutines, gate modifiers, `gphase`, timing,
    /// arrays, physical qubits — raises `ValueError` naming the construct and
    /// its line, as does a division of two integers (`1/2`, integer division
    /// in OpenQASM 3: write `1.0/2`).
    ///
    /// Each `input` is a free parameter, in declaration order (unused ones
    /// included) and under its name ([`param_names`](Circuit::param_names)):
    /// bind values in that order. `U` becomes `u`, `CX` becomes `cx`, `phase`
    /// becomes `p` and `cphase` becomes `cp`; every other gate keeps its name,
    /// and a declared gate stays a declared gate whatever its name.
    ///
    /// **Phase convention.** `U`, `u2` and `u3` are read with Qiskit's
    /// matrices, which differ from the OpenQASM 3 specification's by the
    /// global phases e^{-iθ/2} (`U`) and e^{i(φ+λ)/2} (`u2`, `u3`).
    /// Statevector amplitudes may therefore differ from those of a reader that
    /// follows the specification by these factors; probabilities, counts and
    /// expectation values do not. Every other `stdgates.inc` gate follows the
    /// specification. That Polypus reads Qiskit's `qasm3.dumps` output as
    /// Qiskit does is tested for the Qiskit versions the test suite pins, not
    /// guaranteed in general.
    ///
    /// ```python
    /// qc = polypus.Circuit.from_qasm3(qiskit.qasm3.dumps(qiskit_circuit))
    /// qc.param_names          # Qiskit's (mangled) input names, in order
    /// ```
    #[staticmethod]
    fn from_qasm3(py: Python<'_>, source: &str) -> PyResult<Self> {
        // Pure Rust, bounded by the importer's budgets but not instant on a
        // large program: do not hold the GIL (docs/ENGINEERING.md §3).
        let inner = py
            .detach(|| ParameterizedCircuit::from_qasm3(source))
            .map_err(to_py_err)?;
        Ok(Circuit { inner })
    }

    /// The name of each free parameter, in index order (`num_params` names).
    ///
    /// Parameters imported from OpenQASM 3 keep their `input` names; every
    /// other one is `theta_<index>` (or, if that is taken, the first free
    /// `theta_<index>_<k>`), including indices no gate uses. The names belong
    /// to the circuit: they survive copies and the OpenQASM 3 round trip, and
    /// `to_qasm3` writes them as the program's inputs.
    #[getter]
    fn param_names(&self) -> Vec<String> {
        self.inner.param_names()
    }

    /// Number of qubits in the quantum register.
    #[getter]
    fn num_qubits(&self) -> usize {
        self.inner.num_qubits
    }

    /// Number of free (trainable) parameters.
    #[getter]
    fn num_params(&self) -> usize {
        self.inner.num_params
    }

    /// Size of the implicit classical register.
    #[getter]
    fn num_clbits(&self) -> usize {
        self.inner.num_clbits()
    }

    // ── Single-qubit gates ───────────────────────────────────────────────

    fn h(slf: PyRefMut<'_, Self>, qubit: usize) -> PyResult<PyRefMut<'_, Self>> {
        push(slf, GateInstruction::H(qubit))
    }

    fn x(slf: PyRefMut<'_, Self>, qubit: usize) -> PyResult<PyRefMut<'_, Self>> {
        push(slf, GateInstruction::X(qubit))
    }

    fn y(slf: PyRefMut<'_, Self>, qubit: usize) -> PyResult<PyRefMut<'_, Self>> {
        push(slf, GateInstruction::Y(qubit))
    }

    fn z(slf: PyRefMut<'_, Self>, qubit: usize) -> PyResult<PyRefMut<'_, Self>> {
        push(slf, GateInstruction::Z(qubit))
    }

    fn s(slf: PyRefMut<'_, Self>, qubit: usize) -> PyResult<PyRefMut<'_, Self>> {
        push(slf, GateInstruction::S(qubit))
    }

    fn t(slf: PyRefMut<'_, Self>, qubit: usize) -> PyResult<PyRefMut<'_, Self>> {
        push(slf, GateInstruction::T(qubit))
    }

    fn sdg(slf: PyRefMut<'_, Self>, qubit: usize) -> PyResult<PyRefMut<'_, Self>> {
        push(slf, GateInstruction::Sdg(qubit))
    }

    fn tdg(slf: PyRefMut<'_, Self>, qubit: usize) -> PyResult<PyRefMut<'_, Self>> {
        push(slf, GateInstruction::Tdg(qubit))
    }

    /// Identity gate `id`: no effect on the state, but kept as an instruction
    /// (it counts towards gate count and depth, as in Qiskit).
    fn id(slf: PyRefMut<'_, Self>, qubit: usize) -> PyResult<PyRefMut<'_, Self>> {
        push(slf, GateInstruction::Id(qubit))
    }

    fn rx(slf: PyRefMut<'_, Self>, qubit: usize, theta: AngleArg) -> PyResult<PyRefMut<'_, Self>> {
        push(
            slf,
            GateInstruction::Rx {
                qubit,
                theta: theta.into(),
            },
        )
    }

    fn ry(slf: PyRefMut<'_, Self>, qubit: usize, theta: AngleArg) -> PyResult<PyRefMut<'_, Self>> {
        push(
            slf,
            GateInstruction::Ry {
                qubit,
                theta: theta.into(),
            },
        )
    }

    fn rz(slf: PyRefMut<'_, Self>, qubit: usize, theta: AngleArg) -> PyResult<PyRefMut<'_, Self>> {
        push(
            slf,
            GateInstruction::Rz {
                qubit,
                theta: theta.into(),
            },
        )
    }

    /// Generic single-qubit gate `u3(theta, phi, lam)`.
    fn u(
        slf: PyRefMut<'_, Self>,
        qubit: usize,
        theta: AngleArg,
        phi: AngleArg,
        lam: AngleArg,
    ) -> PyResult<PyRefMut<'_, Self>> {
        push(
            slf,
            GateInstruction::U {
                qubit,
                theta: theta.into(),
                phi: phi.into(),
                lam: lam.into(),
            },
        )
    }

    // ── Two-qubit gates ──────────────────────────────────────────────────

    fn cx(slf: PyRefMut<'_, Self>, control: usize, target: usize) -> PyResult<PyRefMut<'_, Self>> {
        push(slf, GateInstruction::Cx(control, target))
    }

    fn cz(slf: PyRefMut<'_, Self>, control: usize, target: usize) -> PyResult<PyRefMut<'_, Self>> {
        push(slf, GateInstruction::Cz(control, target))
    }

    /// SWAP: exchange the states of qubits `q0` and `q1`.
    fn swap(slf: PyRefMut<'_, Self>, q0: usize, q1: usize) -> PyResult<PyRefMut<'_, Self>> {
        push(slf, GateInstruction::Swap(q0, q1))
    }

    fn rzz(
        slf: PyRefMut<'_, Self>,
        q0: usize,
        q1: usize,
        theta: AngleArg,
    ) -> PyResult<PyRefMut<'_, Self>> {
        push(
            slf,
            GateInstruction::Rzz {
                q0,
                q1,
                theta: theta.into(),
            },
        )
    }

    fn rxx(
        slf: PyRefMut<'_, Self>,
        q0: usize,
        q1: usize,
        theta: AngleArg,
    ) -> PyResult<PyRefMut<'_, Self>> {
        push(
            slf,
            GateInstruction::Rxx {
                q0,
                q1,
                theta: theta.into(),
            },
        )
    }

    /// Controlled phase gate `cp(theta)` on `(q0, q1)`.
    fn cp(
        slf: PyRefMut<'_, Self>,
        q0: usize,
        q1: usize,
        theta: AngleArg,
    ) -> PyResult<PyRefMut<'_, Self>> {
        push(
            slf,
            GateInstruction::Cp {
                q0,
                q1,
                theta: theta.into(),
            },
        )
    }

    // ── The rest of qelib1.inc ───────────────────────────────────────────

    /// √X gate `sx`.
    fn sx(slf: PyRefMut<'_, Self>, qubit: usize) -> PyResult<PyRefMut<'_, Self>> {
        push(slf, GateInstruction::Sx(qubit))
    }

    /// √X† gate `sxdg`.
    fn sxdg(slf: PyRefMut<'_, Self>, qubit: usize) -> PyResult<PyRefMut<'_, Self>> {
        push(slf, GateInstruction::Sxdg(qubit))
    }

    /// Controlled-Y `cy(control, target)`.
    fn cy(slf: PyRefMut<'_, Self>, control: usize, target: usize) -> PyResult<PyRefMut<'_, Self>> {
        push(slf, GateInstruction::Cy(control, target))
    }

    /// Controlled-Hadamard `ch(control, target)`.
    fn ch(slf: PyRefMut<'_, Self>, control: usize, target: usize) -> PyResult<PyRefMut<'_, Self>> {
        push(slf, GateInstruction::Ch(control, target))
    }

    /// Controlled-√X `csx(control, target)`.
    fn csx(slf: PyRefMut<'_, Self>, control: usize, target: usize) -> PyResult<PyRefMut<'_, Self>> {
        push(slf, GateInstruction::Csx(control, target))
    }

    /// Toffoli `ccx(control0, control1, target)`.
    fn ccx(
        slf: PyRefMut<'_, Self>,
        control0: usize,
        control1: usize,
        target: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        push(slf, GateInstruction::Ccx(control0, control1, target))
    }

    /// Fredkin `cswap(control, target0, target1)`.
    fn cswap(
        slf: PyRefMut<'_, Self>,
        control: usize,
        target0: usize,
        target1: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        push(slf, GateInstruction::Cswap(control, target0, target1))
    }

    /// Controlled X-rotation `crx(control, target, theta)`.
    fn crx(
        slf: PyRefMut<'_, Self>,
        control: usize,
        target: usize,
        theta: AngleArg,
    ) -> PyResult<PyRefMut<'_, Self>> {
        push(
            slf,
            GateInstruction::Crx {
                control,
                target,
                theta: theta.into(),
            },
        )
    }

    /// Controlled Y-rotation `cry(control, target, theta)`.
    fn cry(
        slf: PyRefMut<'_, Self>,
        control: usize,
        target: usize,
        theta: AngleArg,
    ) -> PyResult<PyRefMut<'_, Self>> {
        push(
            slf,
            GateInstruction::Cry {
                control,
                target,
                theta: theta.into(),
            },
        )
    }

    /// Controlled Z-rotation `crz(control, target, theta)`.
    fn crz(
        slf: PyRefMut<'_, Self>,
        control: usize,
        target: usize,
        theta: AngleArg,
    ) -> PyResult<PyRefMut<'_, Self>> {
        push(
            slf,
            GateInstruction::Crz {
                control,
                target,
                theta: theta.into(),
            },
        )
    }

    /// Controlled phase in its `cu1` spelling, `cu1(q0, q1, theta)`: the same
    /// operator as `cp`, exported as `cu1`.
    fn cu1(
        slf: PyRefMut<'_, Self>,
        q0: usize,
        q1: usize,
        theta: AngleArg,
    ) -> PyResult<PyRefMut<'_, Self>> {
        push(
            slf,
            GateInstruction::Cu1 {
                q0,
                q1,
                theta: theta.into(),
            },
        )
    }

    /// Controlled `u3`: `cu3(control, target, theta, phi, lam)`.
    fn cu3(
        slf: PyRefMut<'_, Self>,
        control: usize,
        target: usize,
        theta: AngleArg,
        phi: AngleArg,
        lam: AngleArg,
    ) -> PyResult<PyRefMut<'_, Self>> {
        push(
            slf,
            GateInstruction::Cu3 {
                control,
                target,
                theta: theta.into(),
                phi: phi.into(),
                lam: lam.into(),
            },
        )
    }

    /// Controlled `u` with phase `gamma` on the controlled branch (Qiskit's
    /// `cu`): `cu(control, target, theta, phi, lam, gamma)`.
    fn cu(
        slf: PyRefMut<'_, Self>,
        control: usize,
        target: usize,
        theta: AngleArg,
        phi: AngleArg,
        lam: AngleArg,
        gamma: AngleArg,
    ) -> PyResult<PyRefMut<'_, Self>> {
        push(
            slf,
            GateInstruction::Cu {
                control,
                target,
                theta: theta.into(),
                phi: phi.into(),
                lam: lam.into(),
                gamma: gamma.into(),
            },
        )
    }

    /// Phase gate `p(qubit, lam)`.
    fn p(slf: PyRefMut<'_, Self>, qubit: usize, lam: AngleArg) -> PyResult<PyRefMut<'_, Self>> {
        push(
            slf,
            GateInstruction::P {
                qubit,
                lam: lam.into(),
            },
        )
    }

    /// `u1(qubit, lam)`: the phase gate in its `u1` spelling.
    fn u1(slf: PyRefMut<'_, Self>, qubit: usize, lam: AngleArg) -> PyResult<PyRefMut<'_, Self>> {
        push(
            slf,
            GateInstruction::U1 {
                qubit,
                lam: lam.into(),
            },
        )
    }

    /// `u2(qubit, phi, lam)` (= `u3(π/2, phi, lam)`).
    fn u2(
        slf: PyRefMut<'_, Self>,
        qubit: usize,
        phi: AngleArg,
        lam: AngleArg,
    ) -> PyResult<PyRefMut<'_, Self>> {
        push(
            slf,
            GateInstruction::U2 {
                qubit,
                phi: phi.into(),
                lam: lam.into(),
            },
        )
    }

    /// `u0(gamma)`: the identity ("idle for `gamma` units"), kept as an
    /// instruction like `id`.
    fn u0(slf: PyRefMut<'_, Self>, qubit: usize, gamma: AngleArg) -> PyResult<PyRefMut<'_, Self>> {
        push(
            slf,
            GateInstruction::U0 {
                qubit,
                gamma: gamma.into(),
            },
        )
    }

    /// Simplified Toffoli `rccx(control0, control1, target)`, up to relative
    /// phases.
    fn rccx(
        slf: PyRefMut<'_, Self>,
        control0: usize,
        control1: usize,
        target: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        push(slf, GateInstruction::Rccx(control0, control1, target))
    }

    /// Simplified 3-controlled Toffoli `rc3x(c0, c1, c2, target)`, up to
    /// relative phases.
    fn rc3x(
        slf: PyRefMut<'_, Self>,
        c0: usize,
        c1: usize,
        c2: usize,
        target: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        push(slf, GateInstruction::Rc3x(c0, c1, c2, target))
    }

    /// 3-controlled X `c3x(c0, c1, c2, target)`.
    fn c3x(
        slf: PyRefMut<'_, Self>,
        c0: usize,
        c1: usize,
        c2: usize,
        target: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        push(slf, GateInstruction::C3x(c0, c1, c2, target))
    }

    /// 3-controlled √X `c3sqrtx(c0, c1, c2, target)`.
    fn c3sqrtx(
        slf: PyRefMut<'_, Self>,
        c0: usize,
        c1: usize,
        c2: usize,
        target: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        push(slf, GateInstruction::C3sqrtx(c0, c1, c2, target))
    }

    /// 4-controlled X `c4x(c0, c1, c2, c3, target)`.
    fn c4x(
        slf: PyRefMut<'_, Self>,
        c0: usize,
        c1: usize,
        c2: usize,
        c3: usize,
        target: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        push(slf, GateInstruction::C4x(c0, c1, c2, c3, target))
    }

    // ── Non-unitary instructions ─────────────────────────────────────────

    /// Barrier on all qubits, or on `qubits` when given.
    #[pyo3(signature = (qubits=None))]
    fn barrier(
        slf: PyRefMut<'_, Self>,
        qubits: Option<Vec<usize>>,
    ) -> PyResult<PyRefMut<'_, Self>> {
        push(slf, GateInstruction::Barrier(qubits.unwrap_or_default()))
    }

    /// Measure `qubit` into classical bit `cbit`.
    fn measure(slf: PyRefMut<'_, Self>, qubit: usize, cbit: usize) -> PyResult<PyRefMut<'_, Self>> {
        push(slf, GateInstruction::Measure { qubit, cbit })
    }

    /// Measure every qubit `i` into classical bit `i`.
    fn measure_all(slf: PyRefMut<'_, Self>) -> PyResult<PyRefMut<'_, Self>> {
        push(slf, GateInstruction::MeasureAll)
    }

    // ── Export ───────────────────────────────────────────────────────────

    /// Serialize to OpenQASM 2.0.
    ///
    /// For a parameterised circuit, pass `params` (one value per free
    /// parameter). For a fully fixed circuit, call with no arguments. A gate
    /// declared in OpenQASM 3 is written from its definition, renamed where
    /// OpenQASM 2.0 cannot take its name; one whose body uses `arcsin`,
    /// `arccos` or `arctan`, which OpenQASM 2.0 lacks, raises `ValueError`.
    #[pyo3(signature = (params=None))]
    fn to_qasm2(&self, params: Option<Vec<f64>>) -> PyResult<String> {
        self.inner
            .to_qasm2_with_params(&params.unwrap_or_default())
            .map_err(to_py_err)
    }

    /// Serialize to the **OpenQASM 3 profile with Qiskit phase conventions**
    /// (see [`from_qasm3`](Circuit::from_qasm3)).
    ///
    /// Without `params`, each free parameter is an `input float[64]` under its
    /// name ([`param_names`](Circuit::param_names)) and angles are written as
    /// the expressions they are. With `params` (one value per free parameter),
    /// the circuit is bound first and the program has no inputs.
    ///
    /// The output is canonical: importing it and exporting again gives the
    /// same text. `u` is written as the builtin `U` (with Qiskit's matrix, see
    /// the phase convention of `from_qasm3`); instructions `stdgates.inc`
    /// lacks (`rzz`, `rxx`, `sxdg`, `csx`, `cu1`, `cu3`, `u0`, `rccx`,
    /// `rc3x`, `c3x`, `c3sqrtx`, `c4x`) are calls of gates the output defines
    /// with the same matrices. Raises `ValueError` for a declared gate
    /// OpenQASM 3 cannot express (an OpenQASM 2.0 body with a barrier) or a
    /// program beyond what `from_qasm3` reads back (its size limits).
    ///
    /// ```python
    /// text = polypus.Circuit(1).rx(0, polypus.Param(0)).to_qasm3()
    /// qiskit.qasm3.loads(text)   # needs qiskit-qasm3-import
    /// ```
    #[pyo3(signature = (params=None))]
    fn to_qasm3(&self, py: Python<'_>, params: Option<Vec<f64>>) -> PyResult<String> {
        let circuit = &self.inner;
        py.detach(|| match &params {
            None => circuit.to_qasm3(),
            Some(values) => circuit.to_qasm3_with_params(values),
        })
        .map_err(to_py_err)
    }

    /// Serialize to a QIR Base Profile LLVM IR module (text `.ll`).
    ///
    /// For a parameterised circuit, pass `params` (one value per free
    /// parameter); for a fully fixed circuit, call with no arguments. Most
    /// gates map to a standard QIS intrinsic; `rzz`/`rxx`/`u3` are decomposed
    /// to the standard set and `barrier` is dropped. The output targets QIR
    /// Alliance consumers (e.g. Azure Quantum, Quantinuum).
    ///
    /// ```python
    /// qc = polypus.Circuit(2).h(0).cx(0, 1).measure_all()
    /// qir = qc.to_qir()                       # complete LLVM IR module
    /// ```
    #[pyo3(signature = (params=None))]
    fn to_qir(&self, params: Option<Vec<f64>>) -> PyResult<String> {
        self.inner
            .to_qir_with_params(&params.unwrap_or_default())
            .map_err(to_py_err)
    }

    /// Serialize to QIR LLVM bitcode (`.bc`) as Python `bytes`.
    ///
    /// Requires `llvm-as` to be available on `PATH`.
    #[pyo3(signature = (params=None))]
    fn to_qir_bitcode<'py>(
        &self,
        py: Python<'py>,
        params: Option<Vec<f64>>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let bitcode = self
            .inner
            .to_qir_bitcode_with_params(&params.unwrap_or_default())
            .map_err(to_py_err)?;
        Ok(PyBytes::new(py, &bitcode))
    }

    fn __len__(&self) -> usize {
        self.inner.gates.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "Circuit(num_qubits={}, num_params={}, gates={})",
            self.inner.num_qubits,
            self.inner.num_params,
            self.inner.gates.len()
        )
    }
}

/// Quantum Fourier Transform on `num_qubits` qubits, as a `polypus.Circuit`.
///
/// Follows the Qiskit `QFT` convention (big-endian, trailing qubit-reversal
/// swaps). The returned circuit has no free parameters and no measurements, so
/// it composes like any hand-built circuit — keep chaining gates, add
/// `measure_all()`, or run it directly.
///
/// * `inverse` — build the inverse transform (QFT†), the exact adjoint of the
///   forward circuit with the same arguments.
/// * `swaps` — include the qubit-reversal swaps (default `True`); set
///   `False` when the surrounding circuit already handles bit order.
///
/// ```python
/// import polypus
/// qft = polypus.circuits.templates.qft(4)
/// counts = polypus.run_quantum_circuit(qft.measure_all(), shots=1024)
/// ```
#[pyfunction]
#[pyo3(signature = (num_qubits, inverse = false, swaps = true))]
pub fn qft(num_qubits: usize, inverse: bool, swaps: bool) -> Circuit {
    Circuit {
        inner: polypus_circuit::templates::qft_with_options(num_qubits, inverse, swaps),
    }
}
