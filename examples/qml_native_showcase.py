"""Native ``polypus.qml`` showcase: eleven scenarios, each one verified.

``examples/basic_qml.py`` shows the *Qiskit* form of QML (a ``QuantumCircuit``
feature map and ansatz scored by your own ``expectation_function``). This script
is its counterpart for **native models**, ``polypus.qml.Model``: all four
encoders (angle, amplitude, IQP and computational basis), the QCNN layers, every
optimizer, exact and sampled runs, minibatches, inference and saving, the
``Dataset`` utilities, weighted observables, and the convergence curve.

Every scenario *checks learning*, not just that nothing raised: the toy datasets
are two or three well-separated clusters in ``[0, π]`` (or, for the
``basis_encoder`` scenario, 3-bit parity, which no hyperplane separates), and
each scenario asserts a fitness or an accuracy only a fitted model reaches.
Every source of randomness (splits, optimizer seeds, shot sampling) is seeded,
so the whole script is deterministic.

Each scenario builds its ``Model`` once and reuses it: a model holds no trained
state, so the same object trains, binds to its parameters and trains again.
Every scenario also checks two things that hold for any run:
``model.num_params`` (known before training) is the length of the
``best_params`` the optimizer returns, and ``result.fitness_history`` has one
non-decreasing entry per iteration. Scenario 7 shows the one case where that
curve need not end at ``best_fitness`` (a gradient-free optimizer under
minibatches, contract C-5), and the case where it still does (Adam, which tracks
its best on the whole training set).

Run it directly::

    python examples/qml_native_showcase.py

It prints a ✅/❌ table and exits non-zero if any scenario failed.
"""

import math
import os
import sys
import tempfile
import time

import polypus
from polypus import qml

PI = math.pi

# Every run uses the native simulator, the default on infrastructure="local":
# seeded end to end by Polypus (contract C-7), no Aer needed, and the only
# backend with `exact=True`.
Z, X = qml.Z, qml.X


# ─────────────────────────────────────────────────────────────────────────────
# Datasets — fixed literals (no unseeded randomness anywhere)
# ─────────────────────────────────────────────────────────────────────────────


def two_clusters_2f():
    """Eight samples, two features, two clusters far apart inside ``[0, π]``
    (around 0.5 and around 2.6). Labels ``∓1``, the ``hinge`` domain."""
    x = [
        [0.50, 0.55],
        [0.60, 0.45],
        [0.45, 0.60],
        [0.55, 0.50],
        [2.55, 2.60],
        [2.65, 2.50],
        [2.50, 2.65],
        [2.60, 2.55],
    ]
    y = [-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0]
    return x, y


def two_clusters_4f():
    """Six samples over **four** features — the width the QCNN and the IQP
    encoder read — same two well-separated clusters in ``[0, π]``."""
    x = [
        [0.50, 0.55, 0.45, 0.60],
        [0.60, 0.45, 0.55, 0.50],
        [0.45, 0.60, 0.50, 0.55],
        [2.55, 2.60, 2.50, 2.65],
        [2.65, 2.50, 2.60, 2.55],
        [2.50, 2.65, 2.55, 2.60],
    ]
    y = [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]
    return x, y


def three_clusters_3f():
    """Nine samples over three features in three well-separated clusters
    (around 0.4, 1.55 and 2.7). Labels are **class indices** ``{0, 1, 2}``, the
    domain ``categorical_cross_entropy`` requires."""
    x = [
        [0.35, 0.40, 0.45],
        [0.45, 0.35, 0.40],
        [0.40, 0.45, 0.35],
        [1.50, 1.55, 1.60],
        [1.60, 1.50, 1.55],
        [1.55, 1.60, 1.50],
        [2.70, 2.75, 2.65],
        [2.65, 2.70, 2.75],
        [2.75, 2.65, 2.70],
    ]
    y = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0]
    return x, y


def direction_clusters_4f():
    """Six samples whose classes differ in **direction**, not magnitude: class
    ``−1`` puts its mass on features 0–1, class ``+1`` on features 2–3.

    Amplitude encoding normalizes the sample before loading it into the
    amplitudes, so two positive multiples of one vector prepare the *same*
    state — a dataset separated only by overall scale would be unlearnable
    there. These clusters are separated by direction, which survives the
    normalization.
    """
    x = [
        [0.90, 0.80, 0.10, 0.05],
        [0.85, 0.75, 0.05, 0.10],
        [0.95, 0.70, 0.10, 0.10],
        [0.10, 0.05, 0.90, 0.80],
        [0.05, 0.10, 0.85, 0.75],
        [0.10, 0.10, 0.95, 0.70],
    ]
    y = [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]
    return x, y


def raw_scale_2f():
    """Twelve samples on two features with **unscaled**, wildly different raw
    ranges (roughly ``[10, 46]`` and ``[−2, 7]``) — the input the ``Dataset``
    scaling utilities exist for. Still two well-separated clusters, so the data
    stays learnable once mapped onto ``[0, π]``."""
    x = [
        [10.0, -2.0],
        [11.5, -1.6],
        [12.0, -1.2],
        [13.0, -1.8],
        [11.0, -1.4],
        [14.0, -1.0],
        [40.0, 5.0],
        [41.5, 5.4],
        [43.0, 6.0],
        [44.0, 5.6],
        [42.0, 6.4],
        [46.0, 7.0],
    ]
    y = [-1.0] * 6 + [1.0] * 6
    return x, y


def parity_3bits():
    """All eight 3-bit combinations, labelled by **parity** (XOR of the three
    bits). Not linearly separable — no hyperplane in the raw features
    separates the two classes — so a model that learns it must be using real
    entanglement, not just fitting clusters an angle encoder already spreads
    apart. Exhaustive: covers the entire input domain, not a sample of it."""
    x = [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0],
    ]
    y = [1.0 if sum(row) % 2 == 0 else -1.0 for row in x]
    return x, y


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def verdict(checks, summary):
    """Fold ``(name, passed)`` checks and a summary into ``(ok, message)``. A
    failure names the checks that did not hold, so the table diagnoses itself."""
    failed = [name for name, passed in checks if not passed]
    if failed:
        return False, f"{summary} | FAILED: {', '.join(failed)}"
    return True, summary


def accuracy(trained, rows, labels):
    """The fraction of ``rows`` labelled correctly, from exact (shot-free,
    deterministic) inference. Returns ``(accuracy, predictions)``."""
    predictions = trained.predict(rows, exact=True)
    hits = sum(1 for p, y in zip(predictions, labels) if p == y)
    return hits / len(labels), predictions


def history_checks(result, ends_at_best=True):
    """The ``fitness_history`` guarantees (contract C-5): one entry per
    iteration run, never decreasing, and, except for a gradient-free optimizer
    under minibatches, ending at ``best_fitness``."""
    history = result.fitness_history
    checks = [
        ("history length == iterations_run", len(history) == result.iterations_run),
        (
            "history is non-decreasing",
            all(b >= a for a, b in zip(history, history[1:])),
        ),
    ]
    if ends_at_best:
        checks.append(
            ("history ends at best_fitness", history[-1] == result.best_fitness)
        )
    return checks


def params_check(model, result):
    return (
        "num_params matches best_params",
        model.num_params == len(result.best_params),
    )


def sign_model_2q(reps=2):
    """The workhorse binary model: 2 qubits, ``ry`` angle encoding, a
    hardware-efficient ansatz, and ``sign(⟨Z₀⟩)`` as the decision."""
    return qml.Model(
        qml.Encoder(2).angle_encoder("ry"),
        qml.Ansatz(2).hardware_efficient(reps),
        qml.Readout([Z(0)], decision="sign"),
    )


def qng_variance(theta, param_index):
    """QNG's ``variance_function``, called as ``fn(theta, param_index)``: here
    the flat ``0.25`` of a single-Pauli generator, which makes QNG a
    gradient step preconditioned by a fixed metric."""
    return 0.25


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 1: the simplest native path end to end
# ─────────────────────────────────────────────────────────────────────────────


def scenario_1_angle_de_hinge_exact():
    """Angle encoding + hardware-efficient ansatz + ``DE`` + ``hinge``, exact.

    No shot noise, a gradient-free optimizer, and clusters the model must
    separate to reach a near-zero hinge loss. ``patience=generations`` keeps the
    fitness-stagnation stop (contract C-5) out of the budget, here and below.
    With 40 generations the final fitness varies across seeds around the −0.1
    bar (−0.08 to −0.13), so this scenario runs 80: still milliseconds.
    """
    x, y = two_clusters_2f()
    model = sign_model_2q()
    print(f"  {model!r}")
    result = model.fit(
        x,
        y,
        method=polypus.DE(
            generations=80, population_size=20, tolerance=1e-9, patience=80
        ),
        loss="hinge",
        exact=True,
        seed=7,
        id="showcase_1",
    )
    acc, _ = accuracy(model.bind(result.best_params), x, y)
    # 12 parameters: 2 axes (ry, rz) × 2 qubits × (reps + final rotation layer).
    checks = [
        ("12 parameters", len(result.best_params) == 12),
        params_check(model, result),
        # Fitness is −mean hinge loss: above −0.1, every sample is essentially
        # outside the margin.
        ("hinge fitness > -0.1", result.best_fitness > -0.1),
        ("train accuracy == 1.0", acc == 1.0),
    ] + history_checks(result)
    return verdict(
        checks,
        f"DE/hinge/exact: fitness={result.best_fitness:+.4f} acc={acc:.0%} "
        f"iters={result.iterations_run}",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 2: amplitude encoding + PSO
# ─────────────────────────────────────────────────────────────────────────────


def scenario_2_amplitude_pso():
    """Amplitude encoding (4 features on 2 qubits) + ``PSO``.

    The amplitude encoder normalises each sample, so the classes are separated
    by direction (see ``direction_clusters_4f``), and the scenario checks that
    defining property directly: doubling a sample leaves its prediction
    unchanged.
    """
    x, y = direction_clusters_4f()
    model = qml.Model(
        qml.Encoder(2).amplitude_encoder(),
        qml.Ansatz(2).hardware_efficient(2),
        qml.Readout([Z(0)], decision="sign"),
    )
    result = model.fit(
        x,
        y,
        method=polypus.PSO(generations=40, population_size=20, tolerance=1e-9),
        loss="hinge",
        exact=True,
        seed=11,
        id="showcase_2",
    )
    trained = model.bind(result.best_params)
    acc, predictions = accuracy(trained, x, y)
    doubled = trained.predict([[2.0 * v for v in x[0]]], exact=True)
    checks = [
        ("12 parameters", len(result.best_params) == 12),
        params_check(model, result),
        ("hinge fitness > -0.1", result.best_fitness > -0.1),
        ("train accuracy == 1.0", acc == 1.0),
        ("scale invariant", doubled[0] == predictions[0]),
    ] + history_checks(result)
    return verdict(
        checks,
        f"PSO/amplitude(4f→2q): fitness={result.best_fitness:+.4f} acc={acc:.0%}",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 3: IQP encoding + Adam, with real shot noise
# ─────────────────────────────────────────────────────────────────────────────


def scenario_3_iqp_adam_shots():
    """IQP encoding + ``Adam`` on **sampled** results.

    The one scenario without ``exact=True``: every fitness and every
    parameter-shift gradient Adam sees comes from 4096 sampled shots, the
    hardware-like path. The seed still fixes the sampling, so the run repeats
    exactly (C-7). Learning is judged on the learned parameters, scored without
    noise afterwards, since the training fitness is itself an estimate.
    """
    x, y = two_clusters_2f()
    model = qml.Model(
        qml.Encoder(2).iqp_encoder("full"),
        qml.Ansatz(2).hardware_efficient(1),
        qml.Readout([Z(0)], decision="sign"),
    )
    result = model.fit(
        x,
        y,
        method=polypus.Adam(max_iters=40, learning_rate=0.2),
        loss="hinge",
        shots=4096,
        seed=5,
        id="showcase_3",
    )
    acc, _ = accuracy(model.bind(result.best_params), x, y)
    checks = [
        ("8 parameters", len(result.best_params) == 8),
        params_check(model, result),
        # A random θ scores around −1; a fitted model, even estimated from
        # shots, is far above −0.35.
        ("shot fitness > -0.35", result.best_fitness > -0.35),
        ("train accuracy == 1.0", acc == 1.0),
    ] + history_checks(result)
    return verdict(
        checks,
        f"Adam/IQP/shots=4096: fitness={result.best_fitness:+.4f} acc={acc:.0%} "
        f"iters={result.iterations_run}",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 4: QCNN (conv → pool → conv) + QNG
# ─────────────────────────────────────────────────────────────────────────────


def scenario_4_qcnn_qng():
    """A QCNN, ``conv`` → ``pool`` → ``conv``, trained with ``QNG``.

    Each layer shares one block across its qubit pairs, so the parameter count
    depends on the blocks alone: 4 (basic conv) + 3 (pool) + 4 (basic conv) = 11.
    Pooling halves the active qubits, so the second convolution acts on the two
    survivors.
    """
    x, y = two_clusters_4f()
    model = qml.Model(
        qml.Encoder(4).angle_encoder("ry"),
        qml.Ansatz(4)
        .conv("basic")
        .pool("basic", keep="even_positions")
        .conv("basic", pairing="even_pairs"),
        qml.Readout([Z(0)], decision="sign"),
    )
    result = model.fit(
        x,
        y,
        method=polypus.QNG(
            variance_function=qng_variance, max_iters=60, learning_rate=0.3
        ),
        loss="hinge",
        exact=True,
        seed=3,
        id="showcase_4",
    )
    acc, _ = accuracy(model.bind(result.best_params), x, y)
    checks = [
        ("11 shared parameters", len(result.best_params) == 11),
        params_check(model, result),
        ("hinge fitness > -0.15", result.best_fitness > -0.15),
        ("train accuracy == 1.0", acc == 1.0),
    ] + history_checks(result)
    return verdict(
        checks,
        f"QNG/QCNN(conv→pool→conv): fitness={result.best_fitness:+.4f} acc={acc:.0%} "
        f"iters={result.iterations_run}",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 5: three classes, argmax + categorical cross-entropy
# ─────────────────────────────────────────────────────────────────────────────


def scenario_5_multiclass_argmax():
    """One observable per class, ``decision="argmax"`` and
    ``loss="categorical_cross_entropy"``, which only go together, on class
    indices ``0, 1, 2``. The bar is clearly beating the 1/3 of chance."""
    x, y = three_clusters_3f()
    model = qml.Model(
        qml.Encoder(3).angle_encoder("ry"),
        qml.Ansatz(3).hardware_efficient(1),
        qml.Readout([Z(0), Z(1), Z(2)], decision="argmax"),
    )
    result = model.fit(
        x,
        y,
        method=polypus.DE(
            generations=60, population_size=24, tolerance=1e-9, patience=60
        ),
        loss="categorical_cross_entropy",
        exact=True,
        seed=17,
        id="showcase_5",
    )
    acc, predictions = accuracy(model.bind(result.best_params), x, y)
    checks = [
        ("12 parameters", len(result.best_params) == 12),
        params_check(model, result),
        # Three equal expectations score −ln 3 ≈ −1.1; a fitted model beats it.
        ("fitness > -1.0", result.best_fitness > -1.0),
        ("accuracy >= 2/3 (chance 1/3)", acc >= 2 / 3),
        ("predictions are class indices", set(predictions) <= {0, 1, 2}),
    ] + history_checks(result)
    return verdict(
        checks,
        f"DE/argmax/3 classes: fitness={result.best_fitness:+.4f} acc={acc:.0%} "
        f"predictions={predictions}",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 6: a readout in the X basis
# ─────────────────────────────────────────────────────────────────────────────


def scenario_6_x_basis_readout():
    """``sign(⟨X₀⟩)`` instead of ``sign(⟨Z₀⟩)``: the model adds the basis change
    (an ``H`` before the measurement) itself. The model must separate both
    clusters, including two samples it never saw."""
    x, y = two_clusters_2f()
    model = qml.Model(
        qml.Encoder(2).angle_encoder("ry"),
        qml.Ansatz(2).hardware_efficient(2),
        qml.Readout([X(0)], decision="sign"),
    )
    result = model.fit(
        x,
        y,
        # The X expectation of this circuit is a harder surface for DE than
        # scenario 1's, so a larger budget; still milliseconds.
        method=polypus.DE(
            generations=120, population_size=30, tolerance=1e-9, patience=120
        ),
        loss="hinge",
        exact=True,
        seed=23,
        id="showcase_6",
    )
    trained = model.bind(result.best_params)
    acc, _ = accuracy(trained, x, y)
    held_acc, held = accuracy(trained, [[0.52, 0.58], [2.58, 2.52]], [-1, 1])
    checks = [
        params_check(model, result),
        ("hinge fitness > -0.1", result.best_fitness > -0.1),
        ("train accuracy == 1.0", acc == 1.0),
        ("held-out accuracy == 1.0", held_acc == 1.0),
    ] + history_checks(result)
    return verdict(
        checks,
        f"DE/⟨X₀⟩ readout: fitness={result.best_fitness:+.4f} acc={acc:.0%} held-out={held}",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 7: minibatches
# ─────────────────────────────────────────────────────────────────────────────


def scenario_7_minibatches():
    """``batch_size=4`` of 12 samples, against a full-batch reference.

    Each evaluation scores a fresh 4-sample subset, but the reported
    ``best_fitness`` is always on the whole training set (contract C-5): Adam
    uses the subsets only for its gradient and tracks its best on the whole set,
    so its curve still ends at ``best_fitness``. A gradient-free optimizer
    (``DE``) steers by the subset estimates and has its ``best_fitness``
    re-scored on the whole set at the end, so its curve keeps the estimates.
    """
    x, y = raw_scale_2f()
    ranges = [(10.0, 46.0), (-2.0, 7.0)]
    scaled = [
        [(v - lo) / (hi - lo) * PI for v, (lo, hi) in zip(row, ranges)] for row in x
    ]
    model = sign_model_2q(reps=1)

    def run(method, batch_size):
        return model.fit(
            scaled,
            y,
            method=method,
            loss="hinge",
            exact=True,
            seed=31,
            batch_size=batch_size,
            id="showcase_7",
        )

    full = run(polypus.Adam(max_iters=60, learning_rate=0.2), None)
    adam = run(polypus.Adam(max_iters=60, learning_rate=0.2), 4)
    de = run(
        polypus.DE(generations=30, population_size=12, tolerance=1e-9, patience=30), 4
    )
    acc, _ = accuracy(model.bind(adam.best_params), scaled, y)
    checks = (
        [
            params_check(model, adam),
            ("Adam minibatch fitness > -0.2", adam.best_fitness > -0.2),
            (
                "agrees with full batch (< 0.15)",
                abs(adam.best_fitness - full.best_fitness) < 0.15,
            ),
            ("train accuracy == 1.0", acc == 1.0),
        ]
        + history_checks(adam)
        + history_checks(de, ends_at_best=False)
    )
    return verdict(
        checks,
        f"batch_size=4 of 12: Adam fitness={adam.best_fitness:+.4f} "
        f"(full batch {full.best_fitness:+.4f}), DE fitness={de.best_fitness:+.4f} "
        f"acc={acc:.0%}",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 8: save, load, and predictions against their results
# ─────────────────────────────────────────────────────────────────────────────


def scenario_8_save_load_predict():
    """``TrainedModel.save`` / ``qml.load``, then ``predict`` against ``run``.

    The parameters read back bit for bit. On two samples the model never saw,
    its decisions are checked three ways: ``predict`` itself, the sign of
    ``⟨Z₀⟩`` computed by hand from ``run(exact=True)``'s probabilities, and the
    same from ``run``'s sampled counts. (C-3 keys put qubit 0 last.)
    """
    x, y = two_clusters_2f()
    model = sign_model_2q()
    result = model.fit(
        x,
        y,
        method=polypus.DE(
            generations=40, population_size=20, tolerance=1e-9, patience=40
        ),
        loss="hinge",
        exact=True,
        seed=7,
        id="showcase_8",
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "showcase_model.json")
        model.bind(result.best_params).save(path)
        loaded = qml.load(path)

    def z0(distribution):
        total = sum(distribution.values())
        return (
            sum((1 if k[-1] == "0" else -1) * w for k, w in distribution.items())
            / total
        )

    new_x, expected = [[0.52, 0.58], [2.58, 2.52]], [-1, 1]
    via_predict = loaded.predict(new_x, exact=True)
    expectations = [z0(p) for p in loaded.run(new_x, exact=True).probabilities]
    via_probabilities = [1 if e >= 0 else -1 for e in expectations]
    via_counts = [
        1 if z0(c) >= 0 else -1 for c in loaded.run(new_x, shots=1000, seed=3).counts
    ]
    checks = [
        params_check(model, result),
        ("parameters round-trip exactly", loaded.params == result.best_params),
        ("predict matches the expected class", via_predict == expected),
        ("probabilities agree", via_probabilities == via_predict),
        ("counts agree", via_counts == via_predict),
        # Decisive expectations, or the agreement could be a near-zero value
        # rounding the same way three times.
        ("|⟨Z₀⟩| > 0.5 on both samples", all(abs(e) > 0.5 for e in expectations)),
    ] + history_checks(result)
    return verdict(
        checks,
        f"save/load + 3 readings: ⟨Z₀⟩={[round(e, 3) for e in expectations]} "
        f"predictions={via_predict}",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 9: Dataset utilities: split, scale, frozen scaler
# ─────────────────────────────────────────────────────────────────────────────


def scenario_9_dataset_utilities():
    """``train_test_split`` + ``scale_features_to`` + ``scale_features_with``.

    Split, scale the **train** part onto ``[0, π]``, freeze its
    ``feature_ranges()`` and replay them on the test part. The replay is checked
    arithmetically, and a test value beyond the train range is expected to land
    outside ``[0, π]``. The model trains on the scaled train part and is scored
    on all twelve rows through the same frozen map: nine it learned from, three
    it never saw.
    """
    x, y = raw_scale_2f()
    train, test = qml.Dataset(x, y).train_test_split(0.25, seed=19)
    raw_test_ranges = test.feature_ranges()
    frozen = train.feature_ranges()
    train.scale_features_to(0.0, PI)
    test.scale_features_with(frozen, 0.0, PI)

    def apply_frozen(value, feature):
        lo, hi = frozen[feature]
        return (value - lo) / (hi - lo) * PI

    expected = [
        (apply_frozen(lo, j), apply_frozen(hi, j))
        for j, (lo, hi) in enumerate(raw_test_ranges)
    ]
    actual = test.feature_ranges()
    replayed = all(
        math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
        for pair_a, pair_b in zip(actual, expected)
        for a, b in zip(pair_a, pair_b)
    )
    model = sign_model_2q()
    result = model.fit(
        train,
        method=polypus.DE(
            generations=40, population_size=20, tolerance=1e-9, patience=40
        ),
        loss="hinge",
        exact=True,
        seed=29,
        id="showcase_9",
    )
    all_scaled = [[apply_frozen(v, j) for j, v in enumerate(row)] for row in x]
    acc, _ = accuracy(model.bind(result.best_params), all_scaled, y)
    checks = [
        ("split sizes == (9, 3)", (train.num_samples, test.num_samples) == (9, 3)),
        (
            "train spans exactly [0, π]",
            train.feature_ranges() == [(0.0, PI), (0.0, PI)],
        ),
        ("frozen scaler replayed exactly", replayed),
        params_check(model, result),
        ("fitness > -0.1", result.best_fitness > -0.1),
        ("accuracy on all 12 rows == 1.0", acc == 1.0),
    ] + history_checks(result)
    return verdict(
        checks,
        f"split(0.25)+scale[0,π]: test ranges={[(round(lo, 3), round(hi, 3)) for lo, hi in actual]} "
        f"fitness={result.best_fitness:+.4f} acc={acc:.0%}",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 10: a weighted observable
# ─────────────────────────────────────────────────────────────────────────────


def scenario_10_weighted_observable():
    """``qml.Observable``, the weighted sum ``0.5·Z₀ + 0.5·Z₀Z₁``: a readout a
    single Pauli product cannot express, trained like any other."""
    x, y = two_clusters_2f()
    model = qml.Model(
        qml.Encoder(2).angle_encoder("ry"),
        qml.Ansatz(2).hardware_efficient(2),
        qml.Readout(
            [qml.Observable([(0.5, Z(0)), (0.5, Z(0) @ Z(1))])], decision="sign"
        ),
    )
    result = model.fit(
        x,
        y,
        method=polypus.DE(
            generations=40, population_size=20, tolerance=1e-9, patience=40
        ),
        loss="hinge",
        exact=True,
        seed=13,
        id="showcase_10",
    )
    acc, _ = accuracy(model.bind(result.best_params), x, y)
    checks = [
        params_check(model, result),
        # A harder margin to flatten than a single ⟨Z₀⟩, but a random θ sits
        # near −2 here, so −0.25 still only admits a fitted model.
        ("hinge fitness > -0.25", result.best_fitness > -0.25),
        ("train accuracy == 1.0", acc == 1.0),
    ] + history_checks(result)
    return verdict(
        checks,
        f"DE/0.5·Z₀+0.5·Z₀Z₁: fitness={result.best_fitness:+.4f} acc={acc:.0%}",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 11: basis encoding on 3-bit parity
# ─────────────────────────────────────────────────────────────────────────────


def scenario_11_basis_encoder_parity():
    """Basis encoding (one ``X`` per set bit) and an entangling ansatz learn
    3-bit **parity**, which no hyperplane over the bits separates. The encoder
    alone (θ = 0) scores chance, so 100 % on all eight inputs means the
    entangling layers do the work."""
    x, y = parity_3bits()
    model = qml.Model(
        qml.Encoder(3).basis_encoder(),
        qml.Ansatz(3).hardware_efficient(2, entangler="cx", entanglement="full"),
        qml.Readout([Z(0)], decision="sign"),
    )
    result = model.fit(
        x,
        y,
        method=polypus.DE(
            generations=150, population_size=40, tolerance=1e-9, patience=150
        ),
        loss="hinge",
        exact=True,
        seed=11,
        id="showcase_11",
    )
    acc, _ = accuracy(model.bind(result.best_params), x, y)
    checks = [
        params_check(model, result),
        # θ = 0 scores exactly −1 (every sample on the margin).
        ("hinge fitness > -0.5", result.best_fitness > -0.5),
        ("accuracy == 1.0 on all 8 inputs (not linearly separable)", acc == 1.0),
    ] + history_checks(result)
    return verdict(
        checks,
        f"DE/basis/3-bit parity: fitness={result.best_fitness:+.4f} acc={acc:.0%} "
        f"params={model.num_params}",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

SCENARIOS = [
    ("1. Angle encoding + DE + hinge (exact)", scenario_1_angle_de_hinge_exact),
    ("2. Amplitude encoding + PSO", scenario_2_amplitude_pso),
    ("3. IQP encoding + Adam (shots)", scenario_3_iqp_adam_shots),
    ("4. QCNN conv/pool/conv + QNG", scenario_4_qcnn_qng),
    ("5. Three classes, argmax + cross-entropy", scenario_5_multiclass_argmax),
    ("6. X-basis readout", scenario_6_x_basis_readout),
    ("7. Minibatches (Adam and DE)", scenario_7_minibatches),
    ("8. Save, load and predictions", scenario_8_save_load_predict),
    ("9. Dataset split + scaling", scenario_9_dataset_utilities),
    ("10. Weighted observable", scenario_10_weighted_observable),
    ("11. Basis encoding + parity", scenario_11_basis_encoder_parity),
]


def main():
    print(f"Native QML showcase: {len(SCENARIOS)} scenarios\n")
    results = []
    for name, scenario in SCENARIOS:
        print(f"▶ {name} ... ", end="", flush=True)
        started = time.perf_counter()
        try:
            ok, message = scenario()
        except Exception as exc:  # noqa: BLE001 - report, never abort the table
            ok, message = False, f"unexpected {type(exc).__name__}: {exc}"
        elapsed = time.perf_counter() - started
        print(f"{'✅' if ok else '❌'} ({elapsed:.1f}s)")
        results.append((name, ok, message, elapsed))

    width = max(len(name) for name, _, _, _ in results)
    print("\n" + "═" * 100)
    print("SUMMARY")
    print("═" * 100)
    for name, ok, message, _ in results:
        print(f"{'✅' if ok else '❌'} {name:<{width}}  {message}")
    total = sum(elapsed for _, _, _, elapsed in results)
    passed = sum(1 for _, ok, _, _ in results if ok)
    print("═" * 100)
    print(f"{passed}/{len(results)} scenarios passed in {total:.1f}s")
    if passed != len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
