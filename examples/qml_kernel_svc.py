"""A quantum-kernel support vector machine with ``polypus.qml.kernel_matrix``.

A fidelity kernel ``K[i, j] = |<phi(x_i)|phi(x_j)>|^2`` compares two samples by
how much the quantum states an encoder prepares for them overlap. Polypus
computes the Gram matrix, and scikit-learn's ``SVC(kernel="precomputed")``
trains on it:

- train on ``K(train, train)``: ``kernel_matrix(encoder, x_train)``, symmetric,
  only its upper triangle computed;
- predict from ``K(test, train)``: ``kernel_matrix(encoder, x_test, x_train)``,
  one row per test sample and one column per training sample.

The data is the XOR of two quadrants (the label says whether the two features
have the same sign), which no straight line separates: a linear SVM stays
near chance. The IQP encoder's ``Rzz(x_0 * x_1)`` reads exactly that product,
so its kernel separates the classes. The features are scaled to ``[-1, 1]``
with the training set's ranges; IQP's pair angles are products, so a wider
range would grow them quadratically.

The kernel is computed twice: exactly (``exact=True``, inner products of the
native simulator's statevectors) and with 1024 shots per entry, an estimate
whose matrix need not be positive semi-definite (scikit-learn takes it as it
is). Both are seeded, so the run is deterministic.

Run it directly (``pip install -r requirements-examples.txt`` brings
scikit-learn)::

    python examples/qml_kernel_svc.py

It prints the accuracies and fails if the quantum kernel's test accuracy is
below 90 %: far above the 50 % of chance (and of the linear baseline) on this
data, and below the 100 % both kernels reach, which leaves room for shot noise.
"""

import time

import numpy as np
import numpy.typing as npt
from polypus import qml
from sklearn.svm import SVC

TRAIN, TEST = 40, 30
SHOTS = 1024
THRESHOLD = 0.9


def xor_quadrants(
    n: int, seed: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """`n` points of `[-1, 1]^2` labelled 1 when both features have the same
    sign, away from the axes (|x_0 * x_1| > 0.08) so the classes have a margin."""
    rng = np.random.default_rng(seed)
    points: list[npt.NDArray[np.float64]] = []
    while len(points) < n:
        point = rng.uniform(-1.0, 1.0, 2)
        if abs(point[0] * point[1]) > 0.08:
            points.append(point)
    x = np.array(points)
    return x, (x[:, 0] * x[:, 1] > 0).astype(np.int64)


def main() -> None:
    x_train, y_train = xor_quadrants(TRAIN, seed=1)
    x_test, y_test = xor_quadrants(TEST, seed=2)
    # Scale both sets with the training set's ranges.
    train = qml.Dataset(x_train.tolist())
    test = qml.Dataset(x_test.tolist())
    ranges = train.feature_ranges()
    train.scale_features_to(-1.0, 1.0)
    test.scale_features_with(ranges, -1.0, 1.0)

    encoder = qml.Encoder(2).iqp_encoder()
    linear = SVC(kernel="linear").fit(x_train, y_train).score(x_test, y_test)
    print(f"linear SVM test accuracy: {linear:.2f}")

    for exact in (True, False):
        name = "exact" if exact else f"{SHOTS} shots"
        start = time.perf_counter()
        # `shots` and `seed` only matter with exact=False.
        k_train = qml.kernel_matrix(encoder, train, exact=exact, shots=SHOTS, seed=7)
        k_test = qml.kernel_matrix(
            encoder, test, train, exact=exact, shots=SHOTS, seed=8
        )
        elapsed = time.perf_counter() - start
        svm = SVC(kernel="precomputed").fit(k_train, y_train)
        accuracy = svm.score(k_test, y_test)
        print(
            f"quantum kernel ({name}): train {svm.score(k_train, y_train):.2f}, "
            f"test {accuracy:.2f}, kernels {k_train.shape} and {k_test.shape} "
            f"in {elapsed:.2f} s"
        )
        assert accuracy >= THRESHOLD, (
            f"the {name} quantum kernel scored {accuracy:.2f} on the test set, "
            f"below {THRESHOLD}"
        )


if __name__ == "__main__":
    main()
