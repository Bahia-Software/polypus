"""Native ``polypus.qml`` pipeline showcase — ten scenarios, all verified.

``examples/basic_qml.py`` demonstrates the *Qiskit* QML path (a
``QuantumCircuit`` feature map + ansatz + a user ``expectation_function``). This
script is its counterpart for the **native** path built in ``polypus-qml``: the
``polypus.qml.Model`` builder, ``polypus.qml.Dataset``, the four optimizers, the
exact and shot-sampled backends, minibatching, ``TrainedModel`` inference, the
dataset split/scaling utilities, and the weighted-sum ``Observable`` readout.

Every scenario *checks learning*, not merely the absence of an exception: the
toy datasets are two or three well-separated clusters in ``[0, π]``, and each
scenario asserts a fitness and/or a training-set accuracy that a model can only
reach by actually fitting the data. Every source of randomness (dataset order is
fixed, splits, optimizer seeds, shot sampling) is seeded explicitly, so the
whole script is deterministic run to run.

A ``Model`` is reusable without limit — ``Model.train()``/``TrainedModel`` both
clone it internally before compiling, so the same object can be trained,
wrapped in a ``TrainedModel``, and even extended with further builder calls
afterwards. Every scenario below builds its model **once** and reuses that one
object; an earlier version of this script built a fresh model per use, which
was defensive code against a limitation that does not exist.

Training reads as a fluent step on the model itself — ``model.train(dataset,
...)`` — rather than the free ``polypus.qml.train(model, dataset, ...)``, and a
simple readout term is ``Z(0)`` / ``Z(0) @ Z(1)`` rather than the bare
``[("z", 0)]`` tuple form; both older spellings still work everywhere (every
change in this crate's Python ergonomics is additive), this script just uses
the more idiomatic one throughout.

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

PI = math.pi

# Every run stays on the native, local, in-process statevector backend: it is
# seeded end to end by Polypus (contract C-7), needs no Aer, and supports the
# shot-free `exact=True` path.
BACKEND = dict(infrastructure="local", backend="polypus")

# `Z(0)`, `X(0) @ Y(1)`, … — the idiomatic spelling of a simple Pauli readout
# term, in place of the bare `[("z", 0)]` tuple form both still accept.
Z, X, Y = polypus.qml.Z, polypus.qml.X, polypus.qml.Y


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


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def verdict(checks, summary):
    """Fold a list of ``(name, passed)`` checks plus a metrics summary into the
    uniform ``(ok, message)`` every scenario returns. A failure names the exact
    checks that did not hold, so the summary table is diagnostic on its own."""
    failed = [name for name, passed in checks if not passed]
    if failed:
        return False, f"{summary} | FAILED: {', '.join(failed)}"
    return True, summary


def accuracy(trained, rows, labels, run_id):
    """Fraction of ``rows`` the trained model labels correctly, via the exact
    (shot-free, deterministic) inference path. Returns ``(accuracy, preds)``."""
    preds = trained.predict(rows, id=run_id, exact=True, **BACKEND)
    hits = sum(1 for p, y in zip(preds, labels) if p == y)
    return hits / len(labels), preds


def sign_model_2q(reps=2):
    """The workhorse binary model: 2 qubits, ``Ry`` angle encoding, a
    hardware-efficient ansatz, and ``sign(⟨Z₀⟩)`` as the decision.

    Build one instance and reuse it: a ``Model`` is not consumed by ``train``
    or ``TrainedModel`` (both clone it internally before compiling), so the
    same object can serve both.
    """
    return (
        polypus.qml.Model(2)
        .angle_encoder(axis="ry")
        .hardware_efficient(reps=reps)
        .readout(observables=[Z(0)], decision="sign")
    )


def raw_model_2q(reps=2):
    """``sign_model_2q``'s twin with ``decision="raw"``: the identical circuit
    and θ layout, but the readout returns ⟨Z₀⟩ unchanged instead of its sign.
    Used in scenario 8 to obtain the exact expectation of a sample."""
    return (
        polypus.qml.Model(2)
        .angle_encoder(axis="ry")
        .hardware_efficient(reps=reps)
        .readout(observables=[Z(0)], decision="raw")
    )


def qng_variance(theta, param_index):
    """The ``variance_function`` QNG requires: the diagonal of the metric
    approximation, called as ``fn(theta: list[float], param_index: int)``.

    A constant ``0.25`` is the standard flat approximation for a single-Pauli
    generator (``Var[P] = 1 − ⟨P⟩² ≤ 1``, scaled by the ½ of the generator), and
    it keeps the example free of a second oracle; QNG then behaves as a
    metric-preconditioned gradient step with a fixed metric.
    """
    return 0.25


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 1 — the simplest native path end to end
# ─────────────────────────────────────────────────────────────────────────────


def scenario_1_angle_de_hinge_exact():
    """``AngleEncoder`` + ``hardware_efficient`` + ``DE`` + ``hinge``, exact.

    The reference native pipeline: no shot noise (``exact=True``), a
    gradient-free optimizer, and a dataset the model must separate to score a
    near-zero hinge loss. Verified by both the reported fitness and a 100 %
    training-set accuracy read back through ``TrainedModel``.
    """
    x, y = two_clusters_2f()
    model = sign_model_2q()
    dataset = polypus.qml.Dataset(x, y)
    result = model.train(
        dataset,
        method=polypus.DE(generations=40, population_size=20, tolerance=1e-9),
        loss="hinge",
        id="showcase_1_basic",
        seed=7,
        exact=True,
        **BACKEND,
    )
    # `model` is reused as-is: training does not consume it.
    trained = polypus.qml.TrainedModel(model, dataset, result.best_params)
    acc, _ = accuracy(trained, x, y, "showcase_1_predict")

    # 12 θ: 2 axes (ry, rz) × 2 qubits × (reps + final rotation layer) blocks.
    checks = [
        ("theta count == 12", len(result.best_params) == 12),
        ("fitness finite", math.isfinite(result.best_fitness)),
        # Hinge fitness is −mean hinge loss; > −0.1 means every sample sits
        # essentially outside the margin, i.e. the clusters were separated.
        ("hinge fitness > -0.1", result.best_fitness > -0.1),
        ("train accuracy == 1.0", acc == 1.0),
    ]
    return verdict(
        checks,
        f"DE/hinge/exact: fitness={result.best_fitness:+.4f} acc={acc:.0%} "
        f"iters={result.iterations_run} theta={len(result.best_params)}",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 2 — amplitude encoding + PSO
# ─────────────────────────────────────────────────────────────────────────────


def scenario_2_amplitude_pso():
    """``AmplitudeEncoder`` (first layer, 4 features on 2 qubits) + ``PSO``.

    Amplitude encoding must be the model's first layer and needs
    ``num_features <= 2^num_qubits`` — here exactly ``4 == 2²``, the dense case.
    Because the encoder normalizes each sample, the dataset is separated by
    direction (see ``direction_clusters_4f``). Beyond fitness and accuracy, the
    scenario checks the encoder's defining property directly: doubling a sample
    leaves the prediction bit-identical.
    """
    x, y = direction_clusters_4f()
    model = (
        polypus.qml.Model(2)
        .amplitude_encoder()
        .hardware_efficient(reps=2)
        .readout(observables=[Z(0)], decision="sign")
    )
    dataset = polypus.qml.Dataset(x, y)
    result = model.train(
        dataset,
        method=polypus.PSO(generations=40, population_size=20, tolerance=1e-9),
        loss="hinge",
        id="showcase_2_amplitude",
        seed=11,
        exact=True,
        **BACKEND,
    )
    trained = polypus.qml.TrainedModel(model, dataset, result.best_params)
    acc, preds = accuracy(trained, x, y, "showcase_2_predict")
    # Scale invariance: 2× a sample is the same normalized state, so the same
    # prediction — an angle encoder would answer differently.
    doubled = trained.predict(
        [[2.0 * v for v in x[0]]], id="showcase_2_scaled", exact=True, **BACKEND
    )

    checks = [
        ("theta count == 12", len(result.best_params) == 12),
        ("hinge fitness > -0.1", result.best_fitness > -0.1),
        ("train accuracy == 1.0", acc == 1.0),
        ("scale invariant", doubled[0] == preds[0]),
    ]
    return verdict(
        checks,
        f"PSO/amplitude(4f→2q): fitness={result.best_fitness:+.4f} acc={acc:.0%} "
        f"scale-invariant={doubled[0] == preds[0]}",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 3 — IQP encoder + Adam, with real shot noise
# ─────────────────────────────────────────────────────────────────────────────


def scenario_3_iqp_adam_shots():
    """``IqpEncoder`` + ``Adam`` (default ``patience``) in **shot mode**.

    The only scenario that leaves ``exact=False``: every fitness and every
    parameter-shift gradient Adam sees is estimated from 4096 sampled shots, so
    this is the noisy, hardware-like path. The seed still pins the sampling, so
    the run reproduces exactly (C-7).

    Learning is judged on the *learned θ*, scored noiselessly afterwards through
    the exact inference path — the training fitness itself is a shot estimate,
    so the accuracy is the honest signal.
    """
    x, y = two_clusters_2f()
    model = (
        polypus.qml.Model(2)
        .iqp_encoder()
        .hardware_efficient(reps=1)
        .readout(observables=[Z(0)], decision="sign")
    )
    dataset = polypus.qml.Dataset(x, y)
    result = model.train(
        dataset,
        method=polypus.Adam(max_iters=40, learning_rate=0.2, tolerance=1e-4),
        loss="hinge",
        shots=4096,
        id="showcase_3_iqp_shots",
        seed=5,
        **BACKEND,
    )
    trained = polypus.qml.TrainedModel(model, dataset, result.best_params)
    acc, _ = accuracy(trained, x, y, "showcase_3_predict")

    checks = [
        ("theta count == 8", len(result.best_params) == 8),
        ("fitness finite", math.isfinite(result.best_fitness)),
        # A shot-estimated hinge fitness on separated clusters: −0.35 is well
        # inside what a fitted model reaches and far from an unfitted one (a
        # random θ scores around −1).
        ("shot fitness > -0.35", result.best_fitness > -0.35),
        ("train accuracy == 1.0", acc == 1.0),
    ]
    return verdict(
        checks,
        f"Adam/IQP/shots=4096: fitness={result.best_fitness:+.4f} acc={acc:.0%} "
        f"iters={result.iterations_run} converged={result.converged}",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 4 — QCNN (conv → pool → conv) + QNG
# ─────────────────────────────────────────────────────────────────────────────


def scenario_4_qcnn_qng():
    """A QCNN stack — ``conv`` → ``pool`` → ``conv`` — trained with ``QNG``.

    Parameters are *shared* inside each layer, so the θ count depends on the
    blocks alone: 4 (basic conv) + 3 (basic pool) + 4 (basic conv) = 11,
    independent of the 4 qubits. Pooling halves the active set, which is why the
    second convolution reads only the two surviving qubits.
    """
    x, y = two_clusters_4f()
    model = (
        polypus.qml.Model(4)
        .angle_encoder(axis="ry")
        .conv(block="basic")
        .pool(block="basic")
        .conv(block="basic", pairing="even_pairs")
        .readout(observables=[Z(0)], decision="sign")
    )
    dataset = polypus.qml.Dataset(x, y)
    result = model.train(
        dataset,
        method=polypus.QNG(
            variance_function=qng_variance,
            max_iters=60,
            learning_rate=0.3,
            tolerance=1e-4,
        ),
        loss="hinge",
        id="showcase_4_qcnn",
        seed=3,
        exact=True,
        **BACKEND,
    )
    trained = polypus.qml.TrainedModel(model, dataset, result.best_params)
    acc, _ = accuracy(trained, x, y, "showcase_4_predict")

    checks = [
        ("theta count == 11 (shared)", len(result.best_params) == 11),
        ("hinge fitness > -0.15", result.best_fitness > -0.15),
        ("train accuracy == 1.0", acc == 1.0),
    ]
    return verdict(
        checks,
        f"QNG/QCNN(conv→pool→conv): fitness={result.best_fitness:+.4f} acc={acc:.0%} "
        f"iters={result.iterations_run} theta={len(result.best_params)}",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 5 — multiclass: Argmax + categorical cross-entropy
# ─────────────────────────────────────────────────────────────────────────────


def scenario_5_multiclass_argmax():
    """Three classes: one observable per class, ``decision="argmax"`` and
    ``loss="categorical_cross_entropy"``.

    The two must appear together — ``QmlProblem`` rejects ``Argmax`` under a
    scalar loss and a categorical loss under a scalar decision — and the labels
    are class indices ``{0, 1, 2}``. All three observables are ``Z``, so they
    share one measurement basis and one circuit per sample.

    Perfect accuracy is not the bar here; clearly beating the 1/3 of chance is.
    """
    x, y = three_clusters_3f()
    model = (
        polypus.qml.Model(3)
        .angle_encoder(axis="ry")
        .hardware_efficient(reps=1)
        .readout(
            observables=[Z(0), Z(1), Z(2)],
            decision="argmax",
        )
    )
    dataset = polypus.qml.Dataset(x, y)
    result = model.train(
        dataset,
        method=polypus.DE(generations=60, population_size=24, tolerance=1e-9),
        loss="categorical_cross_entropy",
        id="showcase_5_multiclass",
        seed=17,
        exact=True,
        **BACKEND,
    )
    trained = polypus.qml.TrainedModel(model, dataset, result.best_params)
    acc, preds = accuracy(trained, x, y, "showcase_5_predict")

    checks = [
        ("theta count == 12", len(result.best_params) == 12),
        ("fitness finite", math.isfinite(result.best_fitness)),
        # Softmax cross-entropy over three near-equal expectations starts around
        # −ln 3 ≈ −1.1 (as fitness); a fitted model beats that clearly.
        ("fitness > -1.0", result.best_fitness > -1.0),
        # Chance is 1/3 on three balanced classes.
        ("accuracy >= 2/3 (chance 1/3)", acc >= 2 / 3),
        ("predictions are class indices", set(preds) <= {0.0, 1.0, 2.0}),
    ]
    return verdict(
        checks,
        f"DE/argmax/3-class: fitness={result.best_fitness:+.4f} acc={acc:.0%} "
        f"(chance 33%) preds={[int(p) for p in preds]}",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 6 — readout in the X basis
# ─────────────────────────────────────────────────────────────────────────────


def scenario_6_x_basis_readout():
    """``sign(⟨X₀⟩)`` instead of ``sign(⟨Z₀⟩)``.

    ``compile`` inserts the basis change (an ``H`` before the terminal
    measurement) for the non-``Z`` readout; from Python nothing else changes.
    The scenario trains a real model against the X-basis expectation and
    requires it to separate the two clusters perfectly, including on two samples
    it never saw.
    """
    x, y = two_clusters_2f()
    model = (
        polypus.qml.Model(2)
        .angle_encoder(axis="ry")
        .hardware_efficient(reps=2)
        .readout(observables=[X(0)], decision="sign")
    )
    dataset = polypus.qml.Dataset(x, y)
    result = model.train(
        dataset,
        # A slightly larger budget than scenario 1's: the X-basis expectation of
        # this circuit is a harder surface for DE to flatten to the same hinge
        # loss, and the whole run still takes milliseconds.
        method=polypus.DE(generations=120, population_size=30, tolerance=1e-9),
        loss="hinge",
        id="showcase_6_xbasis",
        seed=23,
        exact=True,
        **BACKEND,
    )
    trained = polypus.qml.TrainedModel(model, dataset, result.best_params)
    acc, _ = accuracy(trained, x, y, "showcase_6_predict")
    # Two fresh samples, one near each cluster, never used in training.
    held_out = [[0.52, 0.58], [2.58, 2.52]]
    held_acc, held_preds = accuracy(
        trained, held_out, [-1.0, 1.0], "showcase_6_holdout"
    )

    checks = [
        ("hinge fitness > -0.1", result.best_fitness > -0.1),
        ("train accuracy == 1.0", acc == 1.0),
        ("held-out accuracy == 1.0", held_acc == 1.0),
    ]
    return verdict(
        checks,
        f"DE/⟨X₀⟩ readout: fitness={result.best_fitness:+.4f} acc={acc:.0%} "
        f"held-out={[int(p) for p in held_preds]}",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 7 — minibatching with Adam
# ─────────────────────────────────────────────────────────────────────────────


def scenario_7_minibatch_adam():
    """``batch_size`` active with ``Adam``, against a full-batch reference.

    Each optimizer evaluation scores a deterministic 4-sample subset of the 12,
    so the run is cheaper per step — but the reported ``best_fitness`` is *not*
    a minibatch estimate: it is recomputed once against the whole training set
    when the run ends (design doc §17 / contract C-5). This scenario pins that
    down by running the identical configuration with and without ``batch_size``
    and requiring the two reported fitnesses to agree, plus the usual accuracy
    check on the minibatched model's θ.
    """
    x, y = raw_scale_2f()
    # Reuse the 12-sample geometry, already inside [0, π] after scaling — do it
    # here in Python so the dataset is the same for both runs.
    ranges = [(10.0, 46.0), (-2.0, 7.0)]
    scaled = [
        [(v - lo) / (hi - lo) * PI for v, (lo, hi) in zip(row, ranges)] for row in x
    ]
    # One model, one dataset, reused across both `train` calls below and the
    # final `TrainedModel` — none of the three consumes them.
    model = sign_model_2q(reps=1)
    dataset = polypus.qml.Dataset(scaled, y)

    def run(batch_size):
        return model.train(
            dataset,
            method=polypus.Adam(max_iters=60, learning_rate=0.2, tolerance=1e-4),
            loss="hinge",
            id="showcase_7_minibatch",
            seed=31,
            exact=True,
            batch_size=batch_size,
            **BACKEND,
        )

    full = run(None)
    mini = run(4)
    trained = polypus.qml.TrainedModel(model, dataset, mini.best_params)
    acc, _ = accuracy(trained, scaled, y, "showcase_7_predict")

    checks = [
        ("minibatch fitness finite", math.isfinite(mini.best_fitness)),
        # A full-dataset value on separated data, never a rosy ≈ 0 minibatch
        # estimate nor a wild one.
        ("minibatch fitness > -0.2", mini.best_fitness > -0.2),
        (
            "agrees with full-batch (<0.15)",
            abs(mini.best_fitness - full.best_fitness) < 0.15,
        ),
        ("train accuracy == 1.0", acc == 1.0),
    ]
    return verdict(
        checks,
        f"Adam/batch_size=4 of 12: fitness={mini.best_fitness:+.4f} "
        f"(full-batch {full.best_fitness:+.4f}) acc={acc:.0%}",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 8 — save / load and the three inference entry points
# ─────────────────────────────────────────────────────────────────────────────


def scenario_8_save_load_predict():
    """``TrainedModel.save`` / ``load``, then all three inference entry points.

    After the round trip (θ survives byte for byte — the JSON is written with
    float round-tripping) the reloaded model is exercised on two samples it
    never trained on, through:

    * ``predict(exact=True)`` — the end-to-end path: bind, run, decide;
    * ``predict_from_probabilities`` — the readout fed an exact distribution;
    * ``predict_from_counts`` — the readout fed integer counts.

    The distribution handed to the last two is *derived from the state*, not
    from the expected label: a ``decision="raw"`` twin of the model (same
    circuit, same θ) reports the exact ⟨Z₀⟩ of each sample, and ⟨Z₀⟩ fixes the
    two-outcome distribution ``p("00") = (1 + ⟨Z₀⟩)/2`` over qubit 0 — which is
    all the readout consumes. So the three entry points are compared on the same
    physical state and must agree with each other and with the expected class.
    """
    x, y = two_clusters_2f()
    model = sign_model_2q()
    dataset = polypus.qml.Dataset(x, y)
    result = model.train(
        dataset,
        method=polypus.DE(generations=40, population_size=20, tolerance=1e-9),
        loss="hinge",
        id="showcase_8_train",
        seed=7,
        exact=True,
        **BACKEND,
    )
    trained = polypus.qml.TrainedModel(model, dataset, result.best_params)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "showcase_model.json")
        trained.save(path)
        loaded = polypus.qml.TrainedModel.load(path)
        theta_survived = loaded.theta == result.best_params

        # Samples the model never trained on, one near each cluster.
        new_x = [[0.52, 0.58], [2.58, 2.52]]
        expected = [-1.0, 1.0]

        # The raw twin is a genuinely different model (a different readout
        # decision), so it gets its own build — identical circuit and θ, but
        # returning ⟨Z₀⟩ itself instead of its sign.
        raw_twin = polypus.qml.TrainedModel(raw_model_2q(), dataset, loaded.theta)
        expectations = raw_twin.predict(
            new_x, id="showcase_8_raw", exact=True, **BACKEND
        )

        shots = 1000
        via_predict = loaded.predict(
            new_x, id="showcase_8_predict", exact=True, **BACKEND
        )
        via_probs, via_counts = [], []
        for e in expectations:
            p0 = (1.0 + e) / 2.0  # P(qubit 0 = |0⟩); C-3 keys are little-endian.
            via_probs.append(
                loaded.predict_from_probabilities({"00": p0, "01": 1.0 - p0})
            )
            zeros = round(shots * p0)
            via_counts.append(
                loaded.predict_from_counts({"00": zeros, "01": shots - zeros})
            )

    checks = [
        ("theta round-trips exactly", theta_survived),
        ("predict matches expected class", via_predict == expected),
        ("predict_from_probabilities agrees", via_probs == via_predict),
        ("predict_from_counts agrees", via_counts == via_predict),
        # The expectations must be decisive, or the agreement above would be a
        # coincidence of a near-zero value rounding the same way three times.
        ("|⟨Z₀⟩| > 0.5 on both samples", all(abs(e) > 0.5 for e in expectations)),
    ]
    return verdict(
        checks,
        "save/load + 3 inference paths: "
        f"⟨Z₀⟩={[round(e, 3) for e in expectations]} "
        f"preds={[int(p) for p in via_predict]} (all three paths agree)",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 9 — Dataset utilities: split, scale, frozen scaler
# ─────────────────────────────────────────────────────────────────────────────


def scenario_9_dataset_utilities():
    """``train_test_split`` + ``scale_features_to`` + ``scale_features_with``.

    The canonical preprocessing workflow: split, scale the **train** partition
    onto ``[0, π]`` (the range angle encoding wants), freeze that partition's
    ``feature_ranges()`` and replay them on the test partition. The frozen
    scaler is checked arithmetically — the test partition's post-scaling ranges
    must equal the hand-computed image of its pre-scaling ranges under the
    train map — and a test value beyond the train range is *expected* to land
    outside ``[0, π]``, not treated as an error.

    Then the scaled train partition is trained on, and the model is scored on
    all twelve rows scaled through the same frozen map: nine it learned from and
    three it never saw.
    """
    x, y = raw_scale_2f()
    dataset = polypus.qml.Dataset(x, y)
    train, test = dataset.train_test_split(0.25, seed=19)

    raw_test_ranges = test.feature_ranges()  # before any scaling
    frozen = train.feature_ranges()  # the scaler to replay
    train.scale_features_to(0.0, PI)
    test.scale_features_with(frozen, 0.0, PI)

    def apply_frozen(value, feature):
        lo, hi = frozen[feature]
        return (value - lo) / (hi - lo) * PI

    expected_test_ranges = [
        (apply_frozen(lo, j), apply_frozen(hi, j))
        for j, (lo, hi) in enumerate(raw_test_ranges)
    ]
    actual_test_ranges = test.feature_ranges()
    scaler_replayed = all(
        math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
        for (a_lo, a_hi), (b_lo, b_hi) in zip(actual_test_ranges, expected_test_ranges)
        for a, b in ((a_lo, b_lo), (a_hi, b_hi))
    )
    # Documented behaviour, reported rather than asserted: whether the test
    # partition happens to exceed the train range on some feature.
    outside = any(lo < 0.0 or hi > PI for lo, hi in actual_test_ranges)

    model = sign_model_2q()
    result = model.train(
        train,
        method=polypus.DE(generations=40, population_size=20, tolerance=1e-9),
        loss="hinge",
        id="showcase_9_scaled_train",
        seed=29,
        exact=True,
        **BACKEND,
    )
    # Score every original row through the same frozen map — the three the split
    # held back are genuinely unseen.
    all_scaled = [[apply_frozen(v, j) for j, v in enumerate(row)] for row in x]
    trained = polypus.qml.TrainedModel(model, train, result.best_params)
    acc, _ = accuracy(trained, all_scaled, y, "showcase_9_predict")

    checks = [
        ("split sizes == (9, 3)", (train.num_samples, test.num_samples) == (9, 3)),
        ("both keep 2 features", train.num_features == test.num_features == 2),
        (
            "train spans exactly [0, π]",
            train.feature_ranges() == [(0.0, PI), (0.0, PI)],
        ),
        ("frozen scaler replayed exactly", scaler_replayed),
        ("trains on scaled data (fitness > -0.1)", result.best_fitness > -0.1),
        ("accuracy on all 12 rows == 1.0", acc == 1.0),
    ]
    return verdict(
        checks,
        f"split(0.25)+scale[0,π]: sizes=(9,3) test-ranges="
        f"{[(round(lo, 3), round(hi, 3)) for lo, hi in actual_test_ranges]} "
        f"outside[0,π]={outside} fitness={result.best_fitness:+.4f} acc={acc:.0%}",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 10 — the weighted multi-term Observable
# ─────────────────────────────────────────────────────────────────────────────


def scenario_10_weighted_observable():
    """``polypus.qml.Observable`` — the weighted sum ``Σ cᵢ·Pᵢ`` (design doc §17).

    The bare ``[("z", 0)]`` form ``readout`` has always accepted only ever
    builds a single Pauli string at coefficient ``1.0``.
    ``Observable([(c, term), …])`` is the additive way to reach a genuine sum
    — here ``0.5·Z₀ + 0.5·Z₀Z₁``, a readout the bare form cannot express at
    all — and it trains like any other native model.
    """
    x, y = two_clusters_2f()
    observable = polypus.qml.Observable(
        [(0.5, [("z", 0)]), (0.5, [("z", 0), ("z", 1)])]
    )
    model = (
        polypus.qml.Model(2)
        .angle_encoder(axis="ry")
        .hardware_efficient(reps=2)
        .readout(observables=[observable], decision="sign")
    )
    dataset = polypus.qml.Dataset(x, y)
    result = model.train(
        dataset,
        method=polypus.DE(generations=40, population_size=20, tolerance=1e-9),
        loss="hinge",
        id="showcase_10_observable",
        seed=13,
        exact=True,
        **BACKEND,
    )
    trained = polypus.qml.TrainedModel(model, dataset, result.best_params)
    acc, _ = accuracy(trained, x, y, "showcase_10_predict")

    checks = [
        # This weighted sum's margin is a harder surface to flatten than a
        # bare ⟨Z₀⟩ (measured around −0.12 to −0.17 across several seeds), so
        # the bar sits lower than scenario 1's — but a random θ predicts every
        # sample's *wrong* class here (fitness ≈ −2), so −0.25 still only
        # admits a genuinely fitted model.
        ("hinge fitness > -0.25", result.best_fitness > -0.25),
        ("train accuracy == 1.0", acc == 1.0),
    ]
    return verdict(
        checks,
        f"DE/weighted 0.5·Z₀+0.5·Z₀Z₁: fitness={result.best_fitness:+.4f} "
        f"acc={acc:.0%}",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

SCENARIOS = [
    ("1. AngleEncoder + DE + hinge (exact)", scenario_1_angle_de_hinge_exact),
    ("2. AmplitudeEncoder + PSO", scenario_2_amplitude_pso),
    ("3. IqpEncoder + Adam (shots)", scenario_3_iqp_adam_shots),
    ("4. QCNN conv/pool/conv + QNG", scenario_4_qcnn_qng),
    ("5. Multiclass Argmax + cat. CE", scenario_5_multiclass_argmax),
    ("6. X-basis readout + sign", scenario_6_x_basis_readout),
    ("7. Minibatching + Adam", scenario_7_minibatch_adam),
    ("8. TrainedModel save/load/predict", scenario_8_save_load_predict),
    ("9. Dataset split + scaling", scenario_9_dataset_utilities),
    ("10. Weighted Observable", scenario_10_weighted_observable),
]


def main():
    print(f"polypus-qml native pipeline showcase — {len(SCENARIOS)} scenarios\n")
    results = []
    for name, scenario in SCENARIOS:
        print(f"▶ {name} ... ", end="", flush=True)
        started = time.perf_counter()
        try:
            ok, message = scenario()
        except Exception as exc:  # noqa: BLE001 — report, never abort the table
            ok, message = False, f"unexpected {type(exc).__name__}: {exc}"
        elapsed = time.perf_counter() - started
        print(f"{'✅' if ok else '❌'} ({elapsed:.1f}s)")
        results.append((name, ok, message, elapsed))

    width = max(len(name) for name, _, _, _ in results)
    print("\n" + "═" * 100)
    print("SUMMARY")
    print("═" * 100)
    for name, ok, message, elapsed in results:
        print(f"{'✅' if ok else '❌'} {name:<{width}}  {message}")
    total = sum(elapsed for _, _, _, elapsed in results)
    passed = sum(1 for _, ok, _, _ in results if ok)
    print("═" * 100)
    print(f"{passed}/{len(results)} scenarios passed in {total:.1f}s")

    if passed != len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
