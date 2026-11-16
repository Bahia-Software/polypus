//! `ConcreteCircuit::inverse` (`polypus-circuit`) against the reference
//! semantics, the simulator: for every invertible instruction kind, `U` then
//! `U†`, and `U†` then `U`, leave a generic state where it was — exactly the
//! operator, global phase included. The instruction-level shape of each
//! inverse is `polypus-circuit`'s `tests/inverse.rs`; this file is the C-2
//! half that needs the simulator.

use polypus_circuit::{ConcreteCircuit, GateInstruction, GateParam, ParameterizedCircuit};
use polypus_sim::{Simulator, Statevector, StatevectorSimulator, C64};

const N: usize = 5;

/// A preparation that leaves every amplitude of 5 qubits distinct and non-zero,
/// with relative phases, so a gate that is not undone exactly (a phase, a
/// swapped pair of operands) cannot hide in a zero or a symmetry.
fn generic_state() -> Statevector {
    let mut sv = Statevector::new(N).unwrap();
    let prep = ParameterizedCircuit::new(N)
        .ry(0, 0.31)
        .ry(1, 1.17)
        .ry(2, 0.73)
        .ry(3, 2.05)
        .ry(4, 0.52)
        .cx(0, 1)
        .cx(2, 3)
        .rz(1, 0.9)
        .rx(3, 0.4)
        .cx(1, 4)
        .u(2, 0.6, 1.3, -0.8)
        .cp(0, 4, 1.9)
        .ry(4, 0.27)
        .rz(0, -1.4)
        .assign_parameters(&[])
        .unwrap();
    for gate in prep.gates() {
        sv.apply(gate).unwrap();
    }
    let amplitudes = sv.amplitudes();
    for (b, a) in amplitudes.iter().enumerate() {
        assert!(a.norm() > 1e-3, "amplitude {b} is too small to test with");
    }
    sv
}

/// One instruction of every invertible kind, on 5 qubits with fixed angles:
/// every kind but the measurements and the calls of declared gates (which
/// `inverse` refuses), so `polypus-circuit`'s `GateInstruction::KINDS - 3`.
fn every_invertible_kind() -> Vec<GateInstruction> {
    ParameterizedCircuit::new(N)
        .h(0)
        .x(1)
        .y(2)
        .z(0)
        .s(1)
        .t(2)
        .sdg(0)
        .tdg(1)
        .id(2)
        .rx(0, 0.25)
        .ry(1, 0.5)
        .rz(2, -1.5)
        .u(0, 0.1, 0.2, 0.3)
        .cx(0, 1)
        .cz(1, 2)
        .swap(0, 2)
        .rzz(0, 2, 0.4)
        .rxx(1, 2, 2.0)
        .cp(0, 1, 0.75)
        .sx(1)
        .sxdg(2)
        .cy(2, 0)
        .ch(1, 2)
        .csx(0, 2)
        .ccx(2, 0, 1)
        .cswap(1, 2, 0)
        .crx(1, 0, 0.6)
        .cry(2, 1, -0.6)
        .crz(0, 2, 1.1)
        .cu1(2, 1, 0.8)
        .cu3(1, 0, 0.2, -0.4, 0.9)
        .cu(0, 2, 0.3, 0.1, -0.1, 0.9)
        .u0(3, 0.7)
        .rccx(4, 0, 2)
        .rc3x(3, 1, 4, 0)
        .c3x(2, 4, 0, 3)
        .c3sqrtx(4, 3, 1, 2)
        .c4x(1, 4, 0, 3, 2)
        .p(4, 0.2)
        .u1(3, 0.45)
        .u2(2, -0.2, 0.6)
        .push(GateInstruction::UGate {
            qubit: 1,
            theta: GateParam::Fixed(0.1),
            phi: GateParam::Fixed(0.2),
            lam: GateParam::Fixed(-0.7),
        })
        .barrier()
        .barrier_on(&[0, 3])
        .assign_parameters(&[])
        .unwrap()
        .gates()
        .to_vec()
}

fn apply_all(sv: &mut Statevector, gates: &[GateInstruction]) {
    for gate in gates {
        sv.apply(gate).unwrap();
    }
}

fn assert_same_state(got: &Statevector, want: &Statevector, what: &str) {
    for (b, (a, s)) in got.amplitudes().iter().zip(want.amplitudes()).enumerate() {
        assert!((a - s).norm() < 1e-12, "{what}: amplitude {b} {a} != {s}");
    }
}

#[test]
fn every_invertible_instruction_kind_is_undone_by_its_inverse() {
    let gates = every_invertible_kind();
    let mut seen = [false; GateInstruction::KINDS];
    for gate in &gates {
        seen[gate.kind_index()] = true;
        let circuit = ConcreteCircuit::new(N, vec![gate.clone()]);
        let inverse = circuit.inverse().expect("an invertible instruction");
        let start = generic_state();

        let mut sv = start.clone();
        apply_all(&mut sv, circuit.gates());
        apply_all(&mut sv, inverse.gates());
        assert_same_state(&sv, &start, &format!("U·U† for {gate:?}"));

        let mut sv = start.clone();
        apply_all(&mut sv, inverse.gates());
        apply_all(&mut sv, circuit.gates());
        assert_same_state(&sv, &start, &format!("U†·U for {gate:?}"));
    }
    // Everything but `measure`, `measure_all` and a call of a declared gate.
    let missing = (0..GateInstruction::KINDS).filter(|&k| !seen[k]).count();
    assert_eq!(
        missing, 3,
        "every invertible kind must be exercised; only the three refused ones may be missing"
    );
}

#[test]
fn a_whole_circuit_is_undone_by_its_inverse() {
    // The order matters for non-commuting gates: the inverse of the whole
    // vocabulary in one circuit undoes it on the generic state.
    let circuit = ConcreteCircuit::new(N, every_invertible_kind());
    let inverse = circuit.inverse().unwrap();
    let start = generic_state();
    let mut sv = start.clone();
    apply_all(&mut sv, circuit.gates());
    apply_all(&mut sv, inverse.gates());
    assert_same_state(&sv, &start, "the whole vocabulary");
}

#[test]
fn compute_uncompute_of_a_state_preparation_returns_to_zero() {
    // The kernel's use: `U(x)` then `U(x)†` from `|0…0⟩` is `|0…0⟩` again, run
    // through the simulator as one circuit (fusion included).
    let u = ParameterizedCircuit::new(3)
        .h(0)
        .h(1)
        .rz(0, 0.8)
        .rz(1, -0.3)
        .rzz(0, 1, -0.24)
        .ry(2, 1.1)
        .cx(2, 0)
        .assign_parameters(&[])
        .unwrap();
    let mut gates = u.gates().to_vec();
    gates.extend(u.inverse().unwrap().into_gates());
    let sv = StatevectorSimulator::new()
        .run(&ConcreteCircuit::new(3, gates))
        .unwrap();
    assert!((sv.amplitudes()[0] - C64::new(1.0, 0.0)).norm() < 1e-12);
}
