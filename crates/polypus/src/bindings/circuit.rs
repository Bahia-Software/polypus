//! PyO3 bindings for the native Rust circuit layer (`polypus-circuit`).
//!
//! Exposes `polypus.Circuit` and `polypus.Param` so Python users can build
//! GIL-free circuits and pass them to `polypus.run_quantum_circuit` /
//! `polypus.train` exactly like a Qiskit `QuantumCircuit`.

use numpy::{IntoPyArray, PyArray1};
use polypus_circuit::{CircuitError, GateInstruction, GateParam, ParameterizedCircuit};
use polypus_sim::{SimError, Simulator, StatevectorSimulator};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

/// Map a native [`CircuitError`] onto a Python `ValueError`.
fn to_py_err(e: CircuitError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// Reference to the trainable parameter at a given index.
///
/// ```python
/// qc = polypus.Circuit(2).rx(0, polypus.Param(0)).rzz(0, 1, polypus.Param(1))
/// ```
#[pyclass(module = "polypus", frozen)]
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

impl<'py> FromPyObject<'py> for AngleArg {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
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

/// Native Rust quantum circuit with OpenQASM 2.0 export.
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
/// ```python
/// import polypus
/// qc = polypus.Circuit(2).h(0).cx(0, 1)
/// amps = polypus.statevector(qc)          # array([0.707…+0j, 0j, 0j, 0.707…+0j])
/// ```
#[pyfunction(signature = (qc, params = None))]
pub fn statevector<'py>(
    qc: PyRef<'py, Circuit>,
    params: Option<Vec<f64>>,
) -> PyResult<Bound<'py, PyArray1<polypus_sim::C64>>> {
    let params = params.unwrap_or_default();
    // Binding stays on this side of the release: it is O(gates), allocates
    // nothing of size `2^n`, and reads the circuit through the `PyRef`.
    let concrete = qc.native().assign_parameters(&params).map_err(to_py_err)?;
    // The simulation is the expensive, pure-Rust part: release the GIL for it
    // so it cannot stall every other Python thread (docs/ENGINEERING.md §3).
    //
    // Cancellation hook: `polypus-sim`'s gate loop calls this closure
    // periodically (throttled on its side to ≥25ms apart *and* to a small
    // fraction of the run, so the frequency is decoupled from gate cost and a
    // fast circuit never pays for it). Reacquiring the GIL from inside
    // `allow_threads` is safe and re-entrant — the same `Python::with_gil`
    // guarantee `cunqa.rs` relies on to acquire the GIL from a `Drop` (§9) —
    // and `check_signals()` here is what turns a pending SIGINT into a
    // `KeyboardInterrupt` *mid-run*, since releasing the GIL does not by itself
    // process signals (§3).
    //
    // The `PyErr` cannot travel back through `SimError` (`polypus-sim` is
    // Python-free by design, §2), so it is stashed here and recovered below —
    // the same problem `OracleErrorSlot` solves for the optimizer oracles, and
    // the same reason: a generic trait boundary must not downgrade the real
    // exception. No `Arc<Mutex<...>>` is needed, though: `statevector` is
    // single-shot, and the hook runs on this very thread (the simulation is
    // inline in `allow_threads`, not on a worker), so a `&mut` local is enough.
    // That is also why `check_signals()` is effective at all: it is a no-op off
    // the main thread, and this closure runs on whichever thread called us.
    let mut pending: Option<PyErr> = None;
    let outcome = qc.py().allow_threads(|| {
        let mut cancelled = || {
            Python::with_gil(|py| match py.check_signals() {
                Ok(()) => false,
                Err(err) => {
                    pending = Some(err);
                    true
                }
            })
        };
        StatevectorSimulator::new().run_cancellable(&concrete, Some(&mut cancelled))
    });
    let sv = match outcome {
        Ok(sv) => sv,
        // Propagate the *real* pending exception verbatim. Mapping this through
        // `PyValueError::new_err(e.to_string())` like the variants below would
        // silently downgrade a `KeyboardInterrupt` into a bogus `ValueError`
        // (and `check_signals()` cannot be re-run to recover it: raising it
        // cleared the pending flag).
        Err(SimError::Cancelled) => {
            return match pending {
                Some(err) => Err(err),
                // Unreachable today — the only hook installed above records a
                // `PyErr` before it returns `true`. Handled rather than
                // `unwrap()`ed (§9) so a future cancellation source still
                // surfaces a typed error instead of a `PanicException`.
                None => Err(PyValueError::new_err(SimError::Cancelled.to_string())),
            };
        }
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
    /// Accepts the QASM this class exports plus Qiskit's `qasm2.dumps` output
    /// (`u`/`p`/`u1`/`u2` are canonicalised to `u3`, `swap` to 3×`cx`, `id` is
    /// dropped; multiple registers are flattened in declaration order). The
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
    /// parameter). For a fully fixed circuit, call with no arguments.
    #[pyo3(signature = (params=None))]
    fn to_qasm2(&self, params: Option<Vec<f64>>) -> PyResult<String> {
        self.inner
            .to_qasm2_with_params(&params.unwrap_or_default())
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
