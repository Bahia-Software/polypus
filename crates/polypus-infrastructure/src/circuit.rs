//! The Qiskit half of the circuit boundary: [`QiskitCircuit`], the concrete
//! [`ForeignCircuit`] that carries a Qiskit `QuantumCircuit` through the pyo3-free
//! [`BoundCircuit`] enum, and [`to_py_object`], the GIL-holding conversion the
//! Python-backed backends (Aer/CUNQA) use to hand a [`BoundCircuit`] to the
//! `polypus_python` seam.
//!
//! This is where PyO3 re-enters after `polypus-backend`'s clean, provider-agnostic
//! contract: `polypus-backend` knows only `Native`/`Qasm2`/`Foreign(Box<dyn
//! ForeignCircuit>)`, and *this* crate is the one that teaches it what a Qiskit
//! object is. A GIL-free backend (native, QMIO) never touches this module; it
//! rejects any `Foreign` variant up front.

use std::any::Any;

use polypus_backend::{BackendError, BoundCircuit, ForeignCircuit};
use pyo3::prelude::*;

/// Box a Python exception from the `polypus_python` seam into the pyo3-free
/// [`BackendError::External`] variant, so it crosses the backend contract
/// type-erased and the FFI edge (`polypus::exceptions`) can downcast it back and
/// re-raise the original exception verbatim (preserving its
/// `ValueError`/`TypeError`/`KeyboardInterrupt` class — contract C-1).
///
/// This replaces the pre-extraction `BackendError::Seam(PyErr)` variant: the same
/// verbatim re-raise, but the boxing keeps PyO3 out of `polypus-backend`.
pub fn seam_error(err: PyErr) -> BackendError {
    BackendError::External(Box::new(err))
}

/// A bound Qiskit `QuantumCircuit`, wrapped so it can travel inside
/// [`BoundCircuit::Foreign`] without `polypus-backend` ever depending on PyO3.
///
/// [`clone_boxed`](ForeignCircuit::clone_boxed) bumps the Python object's
/// reference count under the GIL (a `Py::clone_ref`), which is what lets a
/// [`BoundCircuit`] carrying one be [`duplicate`](BoundCircuit::duplicate)-d by the
/// planner between waves; [`as_any`](ForeignCircuit::as_any) lets the Aer/CUNQA
/// backends recover the `Py<PyAny>` at submission time.
#[derive(Debug)]
pub struct QiskitCircuit(Py<PyAny>);

impl QiskitCircuit {
    /// Wrap a bound Qiskit `QuantumCircuit` Python object.
    pub fn new(circuit: Py<PyAny>) -> Self {
        QiskitCircuit(circuit)
    }

    /// Borrow the wrapped Python object.
    pub fn object(&self) -> &Py<PyAny> {
        &self.0
    }

    /// Wrap a Qiskit circuit directly as a ready [`BoundCircuit::Foreign`].
    pub fn into_bound(circuit: Py<PyAny>) -> BoundCircuit {
        BoundCircuit::Foreign(Box::new(QiskitCircuit(circuit)))
    }
}

impl ForeignCircuit for QiskitCircuit {
    fn clone_boxed(&self) -> Box<dyn ForeignCircuit> {
        // Reference-count bump under the GIL, plus one small heap allocation for the
        // box — the only per-circuit cost the `Foreign` escape hatch adds over the
        // old `BoundCircuit::Qiskit(Py<PyAny>)` duplicate (a bare `clone_ref`). Its
        // hot-path relevance is the planner's per-wave `duplicate()`; measured at
        // ~75 ns/call vs ~23 ns for the bare `clone_ref`, i.e. a ~50 ns delta —
        // negligible next to one Aer/CUNQA submission (a GIL crossing plus circuit
        // execution, tens of microseconds and up). See the `#[ignore]`d
        // `bench_foreign_duplicate_overhead` in this file. Left as-is deliberately.
        Box::new(QiskitCircuit(Python::with_gil(|py| self.0.clone_ref(py))))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Recover the wrapped Qiskit object from a [`BoundCircuit::Foreign`], or `None`
/// if the foreign payload is some other provider type this crate does not own.
pub fn as_qiskit(circuit: &BoundCircuit) -> Option<&Py<PyAny>> {
    match circuit {
        BoundCircuit::Foreign(foreign) => foreign
            .as_any()
            .downcast_ref::<QiskitCircuit>()
            .map(QiskitCircuit::object),
        _ => None,
    }
}

/// Convert a [`BoundCircuit`] to the Python object expected by
/// `polypus_python.run_qcs`: the Qiskit circuit as-is, or the QASM program as a
/// `str` (the Python layer parses/forwards it per infrastructure).
///
/// A [`Foreign`](BoundCircuit::Foreign) payload that is **not** a
/// [`QiskitCircuit`] is a circuit representation the Python backends cannot submit,
/// so it is rejected as [`BackendError::UnsupportedCircuit`] rather than silently
/// mishandled.
pub fn to_py_object(circuit: &BoundCircuit, py: Python<'_>) -> Result<Py<PyAny>, BackendError> {
    let conv = |e: PyErr| BackendError::Conversion(e.to_string());
    match circuit {
        BoundCircuit::Foreign(_) => {
            let qc = as_qiskit(circuit).ok_or_else(|| {
                BackendError::UnsupportedCircuit(
                    "this backend received a foreign circuit it does not recognise; the Aer/CUNQA \
                     backends can only submit a Qiskit QuantumCircuit, a polypus.Circuit or an \
                     OpenQASM 2.0 string"
                        .to_string(),
                )
            })?;
            Ok(qc.clone_ref(py))
        }
        BoundCircuit::Qasm2(qasm) => Ok(qasm
            .into_pyobject(py)
            .map_err(|e| conv(e.into()))?
            .into_any()
            .unbind()),
        // Native circuits reach a Python backend (Aer/CUNQA) as OpenQASM 2.0,
        // exactly like the `Qasm2` variant; the conversion is pure Rust.
        BoundCircuit::Native(circuit) => Ok(circuit
            .to_qasm2()
            .into_pyobject(py)
            .map_err(|e| conv(e.into()))?
            .into_any()
            .unbind()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use polypus_circuit::ParameterizedCircuit;

    #[test]
    fn qiskit_wrapper_round_trips_through_foreign() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let bound = QiskitCircuit::into_bound(py.None());
            assert!(bound.is_foreign());
            // Recoverable as a Qiskit object via the downcast helper.
            assert!(as_qiskit(&bound).is_some());
            // `duplicate` preserves the Qiskit payload (ref-count bump under GIL).
            assert!(as_qiskit(&bound.duplicate()).is_some());
        });
    }

    #[test]
    fn to_py_object_serialises_native_and_qasm_without_the_gil_payload() {
        pyo3::prepare_freethreaded_python();
        let native = BoundCircuit::Native(
            ParameterizedCircuit::new(1)
                .x(0)
                .measure_all()
                .assign_parameters(&[])
                .unwrap(),
        );
        let qasm = BoundCircuit::Qasm2("OPENQASM 2.0;\nqreg q[1];\n".to_string());
        Python::with_gil(|py| {
            // Native → a QASM `str`.
            let obj = to_py_object(&native, py).unwrap();
            assert!(obj
                .bind(py)
                .extract::<String>()
                .unwrap()
                .contains("OPENQASM"));
            // Qasm2 → the same string, verbatim.
            let obj = to_py_object(&qasm, py).unwrap();
            assert_eq!(
                obj.bind(py).extract::<String>().unwrap(),
                "OPENQASM 2.0;\nqreg q[1];\n"
            );
            // A Qiskit foreign object is returned as-is (here `None` stands in).
            let obj = to_py_object(&QiskitCircuit::into_bound(py.None()), py).unwrap();
            assert!(obj.bind(py).is_none());
        });
    }

    /// Perf evidence (issue #191 review, point 7): the `Foreign` variant's
    /// per-circuit cost versus the pre-extraction `BoundCircuit::Qiskit(Py<PyAny>)`.
    ///
    /// The only new cost the escape hatch adds on the hot Aer/CUNQA path is that a
    /// circuit `duplicate()` (which the planner does once per circuit per wave) goes
    /// from a bare `Py::clone_ref` (a ref-count bump, no allocation) to
    /// `Box::new(QiskitCircuit(clone_ref))` — one small heap allocation plus the
    /// `dyn` box. This isolates exactly that delta, side by side in one binary. It is
    /// `#[ignore]`d and assertion-free (it prints timings), like native.rs's
    /// `bench_identity_path_avoids_clone_and_qasm_roundtrip`.
    ///
    /// Run with:
    /// ```text
    /// LD_LIBRARY_PATH=~/miniconda3/lib cargo test -p polypus-infrastructure \
    ///   circuit::tests::bench_foreign_duplicate_overhead -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "perf micro-benchmark: prints timings, run explicitly with --ignored --nocapture"]
    fn bench_foreign_duplicate_overhead() {
        use std::hint::black_box;
        use std::time::Instant;

        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let obj: Py<PyAny> = py.None();
            const N: usize = 200_000;

            // (a) old cost: a bare `Py::clone_ref` per circuit `duplicate`.
            let t0 = Instant::now();
            for _ in 0..N {
                black_box(obj.clone_ref(py));
            }
            let old_ns = t0.elapsed().as_secs_f64() * 1e9 / N as f64;

            // (b) new cost: the `Foreign` variant's `duplicate` == `clone_boxed`,
            // i.e. a boxed `QiskitCircuit(clone_ref)`.
            let foreign = QiskitCircuit::into_bound(obj.clone_ref(py));
            let t0 = Instant::now();
            for _ in 0..N {
                black_box(foreign.duplicate());
            }
            let new_ns = t0.elapsed().as_secs_f64() * 1e9 / N as f64;

            println!(
                "\nissue #191 Foreign duplicate overhead (mean over {N} calls):\n  \
                 clone_ref only (old Qiskit variant): {old_ns:8.1} ns/call\n  \
                 Foreign duplicate (clone_boxed)    : {new_ns:8.1} ns/call\n  \
                 delta (extra Box alloc + dyn)      : {:8.1} ns/call",
                new_ns - old_ns
            );
            println!(
                "  For scale: one Aer submission crosses the GIL and runs a circuit \
                 (>= tens of microseconds); the delta above is a few tens of ns."
            );
        });
    }
}
