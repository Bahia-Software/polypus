"""
Native circuit drawings: `Circuit.draw`, `print(circuit)` and the SVG that
Jupyter displays. No Qiskit involved.
"""

import math

import polypus
import pytest


def bell() -> "polypus.Circuit":
    return polypus.Circuit(2).h(0).cx(0, 1).measure_all()


def test_text_drawing_shows_wires_gates_and_measurements():
    text = bell().draw()
    lines = text.splitlines()
    assert lines[1].startswith("q0: ")
    assert lines[4].startswith("q1: ")
    assert "┤ H ├" in text and "┤ X ├" in text and "■" in text
    assert any(line.startswith("c2: ═") for line in lines)
    assert text.count("╩") == 2


def test_str_is_the_text_drawing_and_repr_is_unchanged():
    qc = bell()
    assert str(qc) == qc.draw()
    assert repr(qc).startswith("Circuit(")


def test_angles_show_parameter_names_and_multiples_of_pi():
    p = polypus.Param(0)
    qc = polypus.Circuit(1).rx(0, p).rz(0, math.pi / 2).ry(0, 2 * p)
    text = qc.draw(fold=None)
    assert "Rx(theta_0)" in text
    assert "Rz(π/2)" in text
    assert "Ry(2.0*theta_0)" in text


def test_ascii_drawing_is_pure_ascii():
    text = polypus.Circuit(1).rz(0, math.pi).measure_all().draw(ascii=True)
    assert text.isascii()
    assert "Rz(pi)" in text


def test_fold_wraps_wide_circuits():
    qc = polypus.Circuit(1)
    for _ in range(30):
        qc = qc.h(0)
    folded = qc.draw(fold=40)
    assert all(len(line) <= 40 for line in folded.splitlines())
    assert "»" in folded
    assert "»" not in qc.draw(fold=None)


def test_svg_drawing_and_jupyter_hook():
    qc = bell()
    svg = qc.draw("svg")
    assert svg.startswith("<svg") and svg.endswith("</svg>")
    assert ">H<" in svg
    assert qc._repr_svg_() == svg


def test_unknown_output_is_a_value_error():
    with pytest.raises(ValueError, match="output"):
        bell().draw("png")  # type: ignore[arg-type]
