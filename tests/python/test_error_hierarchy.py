"""
Custom Python exception hierarchy tests.

Failures that previously *panicked* on paths reachable from ``#[pyfunction]``
now cross the FFI boundary as typed, catchable exceptions rooted at
``polypus.PolypusError`` — never a ``pyo3_runtime.PanicException`` or an
interpreter abort. These tests pin the hierarchy and one representative
Rust-originated failure (mirrors the panic-to-typed-error pattern documented in
``test_vqc.py`` and ``test_backend_selection.py``).
"""

import pytest


class TestExceptionHierarchy:
    def test_classes_are_exposed(self):
        import polypus

        for name in (
            "PolypusError",
            "BackendError",
            "CunqaError",
            "QmioError",
            "NativeCircuitError",
            "InsufficientMemoryError",
            "EvaluationError",
        ):
            assert hasattr(polypus, name), f"polypus.{name} is not exported"
            assert issubclass(getattr(polypus, name), Exception)

    def test_domain_subclassing(self):
        import polypus

        assert issubclass(polypus.PolypusError, Exception)
        assert issubclass(polypus.BackendError, polypus.PolypusError)
        assert issubclass(polypus.CunqaError, polypus.BackendError)
        assert issubclass(polypus.QmioError, polypus.BackendError)
        assert issubclass(polypus.NativeCircuitError, polypus.BackendError)
        assert issubclass(polypus.InsufficientMemoryError, polypus.BackendError)
        assert issubclass(polypus.EvaluationError, polypus.PolypusError)


class TestRustOriginatedFailuresAreTyped:
    def test_invalid_native_qasm_raises_native_circuit_error(self):
        """An unparseable OpenQASM 2.0 string handed to the native backend used
        to panic in ``simulate_one``; it now raises ``NativeCircuitError``."""
        import polypus

        with pytest.raises(polypus.NativeCircuitError):
            polypus.run_quantum_circuit(
                "this is definitely not valid openqasm",
                shots=64,
                infrastructure="local",
                backend="polypus",
            )

    def test_native_circuit_error_is_catchable_as_base_classes(self):
        """The typed error is catchable at every level of the hierarchy, so
        callers can be as specific or as broad as they like."""
        import polypus

        bad_qasm = "still not valid openqasm"
        kwargs = dict(shots=32, infrastructure="local", backend="polypus")

        for exc_type in (
            polypus.NativeCircuitError,
            polypus.BackendError,
            polypus.PolypusError,
            Exception,
        ):
            with pytest.raises(exc_type):
                polypus.run_quantum_circuit(bad_qasm, **kwargs)

    def test_unknown_infrastructure_is_value_error_not_panic(self):
        """Contract C-1: an unknown infrastructure is a ``ValueError`` (never a
        ``PanicException``)."""
        import polypus

        qc = polypus.Circuit(1).h(0).measure_all()
        with pytest.raises(ValueError):
            polypus.run_quantum_circuit(qc, shots=10, infrastructure="does-not-exist")

    def test_wide_classical_register_raises_native_circuit_error(self):
        """Issue #346: a ``creg`` of 65536 bits used to panic formatting the
        counts (a ``PanicException``); the native backend now refuses any
        register wider than its 64-bit counts key with a typed error, both as
        an OpenQASM string and as an imported ``Circuit``."""
        import polypus

        qasm = (
            'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\ncreg c[65536];\n'
            "x q[0];\nmeasure q[0] -> c[0];\n"
        )
        kwargs = dict(shots=16, infrastructure="local", backend="polypus")
        for circuit in (qasm, polypus.Circuit.from_qasm2(qasm)):
            with pytest.raises(polypus.NativeCircuitError, match="65536 bits wide"):
                polypus.run_quantum_circuit(circuit, **kwargs)

    def test_measure_beyond_the_counts_key_raises_native_circuit_error(self):
        """Issue #355: ``Circuit(1).measure(0, 70)`` on the native backend used
        to panic shifting the bit into the counts key (a ``PanicException``,
        not catchable as ``Exception``); it now raises a ``polypus.*Error``."""
        import polypus

        qc = polypus.Circuit(1).x(0).measure(0, 70)
        with pytest.raises(polypus.PolypusError) as caught:
            polypus.run_quantum_circuit(
                qc, shots=16, infrastructure="local", backend="polypus"
            )
        assert isinstance(caught.value, polypus.NativeCircuitError)
        assert "71 bits wide" in str(caught.value)

    def test_widest_supported_register_still_runs_on_native(self):
        """The bound is the 64-bit key, not lower: ``measure(0, 63)`` runs and
        reports the bit in its own place."""
        import polypus

        qc = polypus.Circuit(1).x(0).measure(0, 63)
        result = polypus.run_quantum_circuit(
            qc, shots=16, infrastructure="local", backend="polypus"
        )
        assert result.counts[0] == {"1" + "0" * 63: 16}
