//! The circuits the battery feeds a backend, and the throw-away
//! [`ForeignCircuit`] it uses to probe foreign-circuit rejection.
//!
//! Every circuit here is fully **bound** (parameter-free) and carries a terminal
//! measurement, so a backend's counts have a well-defined bitstring width. They are
//! built in the `Native` representation by default; [`as_qasm2`] re-emits one as an
//! OpenQASM 2.0 string for the `Qasm2` code path.

use std::any::Any;

use polypus_backend::{BoundCircuit, ForeignCircuit};
use polypus_circuit::ParameterizedCircuit;

/// An `n`-qubit circuit with **no gates** and a terminal measurement: every shot
/// reads the all-zeros bitstring. Deterministic (no shot noise), so the battery can
/// assert exact key widths and shot conservation without statistical flakiness.
pub fn zeros(n: usize) -> BoundCircuit {
    BoundCircuit::Native(
        ParameterizedCircuit::new(n)
            .measure_all()
            .assign_parameters(&[])
            .expect("a gate-free bound circuit always assigns"),
    )
}

/// An `n`-qubit circuit with an `X` on every qubit and a terminal measurement:
/// every shot reads the all-ones bitstring. Deterministic, and its outcome differs
/// from [`zeros`] — used where the battery needs two circuits a *simulating* backend
/// distinguishes (the ordering check tolerates a non-simulating backend by keying on
/// width, not value).
pub fn ones(n: usize) -> BoundCircuit {
    let mut pc = ParameterizedCircuit::new(n);
    for q in 0..n {
        pc = pc.x(q);
    }
    BoundCircuit::Native(
        pc.measure_all()
            .assign_parameters(&[])
            .expect("a bound X-layer circuit always assigns"),
    )
}

/// A 2-qubit Bell pair with a terminal measurement: outcomes are the correlated
/// `00`/`11`, spread by shot noise. Used where a non-trivial distribution is wanted.
pub fn bell() -> BoundCircuit {
    BoundCircuit::Native(
        ParameterizedCircuit::new(2)
            .h(0)
            .cx(0, 1)
            .measure_all()
            .assign_parameters(&[])
            .expect("a bound Bell circuit always assigns"),
    )
}

/// Re-emit a [`BoundCircuit`] as the `Qasm2` representation, so the battery can
/// exercise a backend's OpenQASM 2.0 code path. A `Native` circuit is serialised; a
/// circuit that is already `Qasm2`/`Foreign` is returned unchanged (duplicated).
pub fn as_qasm2(circuit: &BoundCircuit) -> BoundCircuit {
    match circuit {
        BoundCircuit::Native(cc) => BoundCircuit::Qasm2(cc.to_qasm2()),
        other => other.duplicate(),
    }
}

/// A provider-native circuit object that belongs to **no** real provider — the
/// battery submits it to probe that a backend rejects an unrecognised
/// [`Foreign`](BoundCircuit::Foreign) payload with
/// [`UnsupportedCircuit`](polypus_backend::BackendError::UnsupportedCircuit) rather
/// than mishandling it. Deliberately not a Qiskit circuit, so even the Aer/CUNQA
/// backends (which *do* accept a Qiskit `Foreign`) reject it.
#[derive(Debug)]
struct AlienCircuit;

impl ForeignCircuit for AlienCircuit {
    fn clone_boxed(&self) -> Box<dyn ForeignCircuit> {
        Box::new(AlienCircuit)
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// A [`BoundCircuit::Foreign`] wrapping a provider type no backend recognises.
pub fn alien_foreign() -> BoundCircuit {
    BoundCircuit::Foreign(Box::new(AlienCircuit))
}
