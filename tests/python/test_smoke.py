"""
Smoke tests — verify that polypus and polypus_python are correctly installed
and that all expected public symbols are present and callable/instantiable.

These tests do NOT execute any quantum circuit; they only check the installation.
"""

import importlib
import inspect
import pytest


# ---------------------------------------------------------------------------
# polypus (Rust/PyO3 extension)
# ---------------------------------------------------------------------------

class TestPolypusImport:
    def test_import_polypus(self):
        import polypus  # noqa: F401

    def test_run_quantum_circuit_exists(self):
        import polypus
        assert hasattr(polypus, "run_quantum_circuit")
        assert callable(polypus.run_quantum_circuit)

    def test_train_exists(self):
        import polypus
        assert hasattr(polypus, "train")
        assert callable(polypus.train)

    def test_DE_class_exists(self):
        import polypus
        assert hasattr(polypus, "DE")

    def test_PSO_class_exists(self):
        import polypus
        assert hasattr(polypus, "PSO")

    def test_QNG_class_exists(self):
        import polypus
        assert hasattr(polypus, "QNG")


class TestPolypusInstantiation:
    def test_DE_default_instantiation(self):
        import polypus
        de = polypus.DE()
        assert de.generations == 100
        assert de.population_size == 50
        assert de.tolerance == pytest.approx(0.01)

    def test_DE_custom_instantiation(self):
        import polypus
        de = polypus.DE(generations=10, population_size=5, tolerance=0.05)
        assert de.generations == 10
        assert de.population_size == 5
        assert de.tolerance == pytest.approx(0.05)

    def test_PSO_default_instantiation(self):
        import polypus
        pso = polypus.PSO()
        assert pso.generations == 100
        assert pso.population_size == 50

    def test_PSO_custom_instantiation(self):
        import polypus
        pso = polypus.PSO(generations=20, population_size=10, bounds=(-3.14, 3.14))
        assert pso.generations == 20
        assert pso.population_size == 10

    def test_QNG_instantiation(self):
        import polypus

        def dummy_variance(theta, a):
            return 0.5

        qng = polypus.QNG(variance_function=dummy_variance)
        assert qng.max_iters == 100
        assert qng.learning_rate == pytest.approx(0.1)

    def test_QNG_custom_instantiation(self):
        import polypus

        def dummy_variance(theta, a):
            return 0.5

        qng = polypus.QNG(
            variance_function=dummy_variance,
            max_iters=50,
            learning_rate=0.01,
        )
        assert qng.max_iters == 50
        assert qng.learning_rate == pytest.approx(0.01)


# ---------------------------------------------------------------------------
# polypus_python (pure Python wrapper)
# ---------------------------------------------------------------------------

class TestPolypusPythonImport:
    def test_import_polypus_python(self):
        import polypus_python  # noqa: F401

    def test_run_qc_in_qpu_exists(self):
        import polypus_python
        assert hasattr(polypus_python, "run_qc_in_qpu")
        assert callable(polypus_python.run_qc_in_qpu)

    def test_run_qcs_in_qpu_exists(self):
        import polypus_python
        assert hasattr(polypus_python, "run_qcs_in_qpu")
        assert callable(polypus_python.run_qcs_in_qpu)

    def test_serialize_quantum_circuit_exists(self):
        import polypus_python
        assert hasattr(polypus_python, "serialize_quantum_circuit")
        assert callable(polypus_python.serialize_quantum_circuit)

    def test_build_qaoa_circuit_exists(self):
        import polypus_python
        assert hasattr(polypus_python, "build_qaoa_circuit")
        assert callable(polypus_python.build_qaoa_circuit)

    def test_expectation_value_exists(self):
        import polypus_python
        assert hasattr(polypus_python, "expectation_value")
        assert callable(polypus_python.expectation_value)

    def test_expectation_values_exists(self):
        import polypus_python
        assert hasattr(polypus_python, "expectation_values")
        assert callable(polypus_python.expectation_values)

    def test_connect_to_infrastructure_exists(self):
        import polypus_python
        assert hasattr(polypus_python, "connect_to_infrastructure")
        assert callable(polypus_python.connect_to_infrastructure)

    def test_disconnect_from_infrastructure_exists(self):
        import polypus_python
        assert hasattr(polypus_python, "disconnect_from_infrastructure")
        assert callable(polypus_python.disconnect_from_infrastructure)
