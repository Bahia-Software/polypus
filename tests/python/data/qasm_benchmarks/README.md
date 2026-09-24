# Real benchmark files (test fixtures)

A handful of small, unmodified circuits from public benchmark suites, used by
`tests/python/test_qasm_benchmark_files.py` to check that Polypus imports a
real file, re-emits it as the same program (the same instruction counts per
name, size and depth for Qiskit), and that the re-emitted text runs on Aer.
They are inputs for that test only — no benchmark results live in this repo.

| file | source | notes |
|---|---|---|
| `qasmbench/adder_n10.qasm` | QASMBench, commit `357b942` (2025-01-19), `small/adder_n10/adder_n10.qasm` | `gate` blocks with `ccx`, four registers, broadcast `x b;`, comments, CRLF line endings |
| `qasmbench/pea_n5.qasm` | QASMBench, same commit, `small/pea_n5/pea_n5.qasm` | nested `gate` blocks (`ctu` calls `cu1fixed`), `u1` in a body, top-level `cu1` |
| `mqtbench/grover_n3.qasm` | MQT Bench 2.3.0, `get_benchmark("grover", BenchmarkLevel.INDEP, 3)`, exported with Qiskit 2.5.2 `qasm2.dumps` | declared `mcphase` called from a declared `gate_Q`; `crz`, `p` |
| `mqtbench/cdkm_ripple_carry_adder_n4.qasm` | MQT Bench 2.3.0, `cdkm_ripple_carry_adder`, indep, n=4 | declared `gate_MAJ`/`gate_UMA` with `ccx` |
| `mqtbench/qaoa_n4_nativegates.qasm` | MQT Bench 2.3.0, `qaoa`, nativegates (`ibm_falcon`: id, x, sx, rz, cx), n=4 | typical variational circuit at the native-gate level |

Licenses: QASMBench is distributed under the Battelle Memorial Institute
license in `qasmbench/LICENSE`; MQT Bench under the MIT license in
`mqtbench/LICENSE`. Both notices are kept with the files, as they require.
