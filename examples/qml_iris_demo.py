"""A real, small end-to-end run on the classic Iris dataset.

Not a synthetic toy cluster: 150 real flower measurements (Fisher, 1936; the
`sklearn`-distributed CSV), 4 continuous features, 3 species. Embedded here as
a literal so the demo runs offline, deterministically, forever.

The whole native `polypus.qml` workflow, start to finish: load real data, a
leakage-free train/test split (the test set's scaling comes from the train
set's ranges only), a native `Model` (angle encoding + a hardware-efficient
ansatz + a 3-observable `Argmax` readout), training with `DE` against
`categorical_cross_entropy`, and evaluation on held-out data the model never
saw. `seed=42` throughout means this prints the exact same numbers every run
— rerun it as many times as you like (contract C-7).

Run it directly::

    python examples/qml_iris_demo.py
"""

import math
import random
import time

import polypus

PI = math.pi

_IRIS_CSV = """\
5.1,3.5,1.4,0.2,0
4.9,3.0,1.4,0.2,0
4.7,3.2,1.3,0.2,0
4.6,3.1,1.5,0.2,0
5.0,3.6,1.4,0.2,0
5.4,3.9,1.7,0.4,0
4.6,3.4,1.4,0.3,0
5.0,3.4,1.5,0.2,0
4.4,2.9,1.4,0.2,0
4.9,3.1,1.5,0.1,0
5.4,3.7,1.5,0.2,0
4.8,3.4,1.6,0.2,0
4.8,3.0,1.4,0.1,0
4.3,3.0,1.1,0.1,0
5.8,4.0,1.2,0.2,0
5.7,4.4,1.5,0.4,0
5.4,3.9,1.3,0.4,0
5.1,3.5,1.4,0.3,0
5.7,3.8,1.7,0.3,0
5.1,3.8,1.5,0.3,0
5.4,3.4,1.7,0.2,0
5.1,3.7,1.5,0.4,0
4.6,3.6,1.0,0.2,0
5.1,3.3,1.7,0.5,0
4.8,3.4,1.9,0.2,0
5.0,3.0,1.6,0.2,0
5.0,3.4,1.6,0.4,0
5.2,3.5,1.5,0.2,0
5.2,3.4,1.4,0.2,0
4.7,3.2,1.6,0.2,0
4.8,3.1,1.6,0.2,0
5.4,3.4,1.5,0.4,0
5.2,4.1,1.5,0.1,0
5.5,4.2,1.4,0.2,0
4.9,3.1,1.5,0.2,0
5.0,3.2,1.2,0.2,0
5.5,3.5,1.3,0.2,0
4.9,3.6,1.4,0.1,0
4.4,3.0,1.3,0.2,0
5.1,3.4,1.5,0.2,0
5.0,3.5,1.3,0.3,0
4.5,2.3,1.3,0.3,0
4.4,3.2,1.3,0.2,0
5.0,3.5,1.6,0.6,0
5.1,3.8,1.9,0.4,0
4.8,3.0,1.4,0.3,0
5.1,3.8,1.6,0.2,0
4.6,3.2,1.4,0.2,0
5.3,3.7,1.5,0.2,0
5.0,3.3,1.4,0.2,0
7.0,3.2,4.7,1.4,1
6.4,3.2,4.5,1.5,1
6.9,3.1,4.9,1.5,1
5.5,2.3,4.0,1.3,1
6.5,2.8,4.6,1.5,1
5.7,2.8,4.5,1.3,1
6.3,3.3,4.7,1.6,1
4.9,2.4,3.3,1.0,1
6.6,2.9,4.6,1.3,1
5.2,2.7,3.9,1.4,1
5.0,2.0,3.5,1.0,1
5.9,3.0,4.2,1.5,1
6.0,2.2,4.0,1.0,1
6.1,2.9,4.7,1.4,1
5.6,2.9,3.6,1.3,1
6.7,3.1,4.4,1.4,1
5.6,3.0,4.5,1.5,1
5.8,2.7,4.1,1.0,1
6.2,2.2,4.5,1.5,1
5.6,2.5,3.9,1.1,1
5.9,3.2,4.8,1.8,1
6.1,2.8,4.0,1.3,1
6.3,2.5,4.9,1.5,1
6.1,2.8,4.7,1.2,1
6.4,2.9,4.3,1.3,1
6.6,3.0,4.4,1.4,1
6.8,2.8,4.8,1.4,1
6.7,3.0,5.0,1.7,1
6.0,2.9,4.5,1.5,1
5.7,2.6,3.5,1.0,1
5.5,2.4,3.8,1.1,1
5.5,2.4,3.7,1.0,1
5.8,2.7,3.9,1.2,1
6.0,2.7,5.1,1.6,1
5.4,3.0,4.5,1.5,1
6.0,3.4,4.5,1.6,1
6.7,3.1,4.7,1.5,1
6.3,2.3,4.4,1.3,1
5.6,3.0,4.1,1.3,1
5.5,2.5,4.0,1.3,1
5.5,2.6,4.4,1.2,1
6.1,3.0,4.6,1.4,1
5.8,2.6,4.0,1.2,1
5.0,2.3,3.3,1.0,1
5.6,2.7,4.2,1.3,1
5.7,3.0,4.2,1.2,1
5.7,2.9,4.2,1.3,1
6.2,2.9,4.3,1.3,1
5.1,2.5,3.0,1.1,1
5.7,2.8,4.1,1.3,1
6.3,3.3,6.0,2.5,2
5.8,2.7,5.1,1.9,2
7.1,3.0,5.9,2.1,2
6.3,2.9,5.6,1.8,2
6.5,3.0,5.8,2.2,2
7.6,3.0,6.6,2.1,2
4.9,2.5,4.5,1.7,2
7.3,2.9,6.3,1.8,2
6.7,2.5,5.8,1.8,2
7.2,3.6,6.1,2.5,2
6.5,3.2,5.1,2.0,2
6.4,2.7,5.3,1.9,2
6.8,3.0,5.5,2.1,2
5.7,2.5,5.0,2.0,2
5.8,2.8,5.1,2.4,2
6.4,3.2,5.3,2.3,2
6.5,3.0,5.5,1.8,2
7.7,3.8,6.7,2.2,2
7.7,2.6,6.9,2.3,2
6.0,2.2,5.0,1.5,2
6.9,3.2,5.7,2.3,2
5.6,2.8,4.9,2.0,2
7.7,2.8,6.7,2.0,2
6.3,2.7,4.9,1.8,2
6.7,3.3,5.7,2.1,2
7.2,3.2,6.0,1.8,2
6.2,2.8,4.8,1.8,2
6.1,3.0,4.9,1.8,2
6.4,2.8,5.6,2.1,2
7.2,3.0,5.8,1.6,2
7.4,2.8,6.1,1.9,2
7.9,3.8,6.4,2.0,2
6.4,2.8,5.6,2.2,2
6.3,2.8,5.1,1.5,2
6.1,2.6,5.6,1.4,2
7.7,3.0,6.1,2.3,2
6.3,3.4,5.6,2.4,2
6.4,3.1,5.5,1.8,2
6.0,3.0,4.8,1.8,2
6.9,3.1,5.4,2.1,2
6.7,3.1,5.6,2.4,2
6.9,3.1,5.1,2.3,2
5.8,2.7,5.1,1.9,2
6.8,3.2,5.9,2.3,2
6.7,3.3,5.7,2.5,2
6.7,3.0,5.2,2.3,2
6.3,2.5,5.0,1.9,2
6.5,3.0,5.2,2.0,2
6.2,3.4,5.4,2.3,2
5.9,3.0,5.1,1.8,2
"""

SPECIES = ["setosa", "versicolor", "virginica"]


def load_iris():
    """Parse the embedded CSV into (features, labels)."""
    x, y = [], []
    for line in _IRIS_CSV.strip().splitlines():
        *features, label = line.split(",")
        x.append([float(v) for v in features])
        y.append(float(label))
    return x, y


def split(x, y, test_fraction, seed):
    """A stratified-free 80/20 shuffle-split, seeded for reproducibility."""
    indices = list(range(len(x)))
    random.Random(seed).shuffle(indices)
    n_test = int(len(indices) * test_fraction)
    test_idx, train_idx = indices[:n_test], indices[n_test:]
    return (
        [x[i] for i in train_idx],
        [y[i] for i in train_idx],
        [x[i] for i in test_idx],
        [y[i] for i in test_idx],
    )


def scale_to_pi(rows, ranges):
    """Map every feature onto [0, π] using the (frozen) per-feature ranges."""
    return [
        [(v - lo) / (hi - lo) * PI for v, (lo, hi) in zip(row, ranges)] for row in rows
    ]


def feature_ranges(rows):
    n = len(rows[0])
    return [(min(r[j] for r in rows), max(r[j] for r in rows)) for j in range(n)]


def accuracy(predictions, labels):
    hits = sum(1 for p, y in zip(predictions, labels) if p == y)
    return hits / len(labels)


def main():
    print("polypus-qml — a real end-to-end run: Iris flower classification\n")

    x, y = load_iris()
    x_train, y_train, x_test, y_test = split(x, y, test_fraction=0.2, seed=42)
    print(f"Iris dataset: {len(x)} real flowers, 4 features, 3 species")
    print(f"Split: {len(x_train)} train / {len(x_test)} test (seed=42)\n")

    # Scale by the TRAIN set's own ranges only — the test set never leaks into
    # this, exactly as a real evaluation requires.
    ranges = feature_ranges(x_train)
    x_train_s = scale_to_pi(x_train, ranges)
    x_test_s = scale_to_pi(x_test, ranges)

    dataset = polypus.qml.Dataset(x_train_s, y_train)

    model = (
        polypus.qml.Model(4)
        .angle_encoder(axis="ry")
        .hardware_efficient(reps=2)
        .readout(
            observables=[polypus.qml.Z(0), polypus.qml.Z(1), polypus.qml.Z(2)],
            decision="argmax",
        )
    )
    print(f"Model: {model!r}")
    print(f"Trainable parameters: {model.num_params()}\n")

    print("Training (DE, exact statevector simulation, no shot noise)...")
    started = time.perf_counter()
    result = model.train(
        dataset,
        method=polypus.DE(generations=150, population_size=60, tolerance=1e-9),
        loss="categorical_cross_entropy",
        infrastructure="local",
        backend="polypus",
        id="iris_demo",
        seed=42,
        exact=True,
    )
    elapsed = time.perf_counter() - started

    print(
        f"Done in {elapsed:.1f}s — {result.iterations_run} generations "
        f"(converged={result.converged})"
    )
    print(
        f"Fitness: {result.fitness_history[0]:+.4f} -> "
        f"{result.fitness_history[-1]:+.4f} (higher is better)\n"
    )

    train_preds = result.trained_model.predict(
        x_train_s,
        infrastructure="local",
        backend="polypus",
        id="iris_train_eval",
        exact=True,
    )
    test_preds = result.trained_model.predict(
        x_test_s,
        infrastructure="local",
        backend="polypus",
        id="iris_test_eval",
        exact=True,
    )
    train_acc = accuracy(train_preds, y_train)
    test_acc = accuracy(test_preds, y_test)

    print(f"Train accuracy: {train_acc:.1%} ({len(x_train)} samples)")
    print(
        f"Test accuracy:  {test_acc:.1%} ({len(x_test)} samples, never seen in training)\n"
    )

    print("Per-sample test predictions:")
    for row, pred, true in zip(x_test, test_preds, y_test):
        mark = "✅" if pred == true else "❌"
        print(
            f"  {mark} {row} -> predicted {SPECIES[int(pred)]:<10} (actual {SPECIES[int(true)]})"
        )


if __name__ == "__main__":
    main()
