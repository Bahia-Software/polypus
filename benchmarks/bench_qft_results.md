## QFT benchmark — 20 qubits (210 gates)
_2026-07-28 09:59:54_ · best of 5 runs

max amplitude error (qiskit vs native): 1.73e-18

| engine | time |
|---|---|
| qiskit (Terra Statevector) | 2082.695 ms |
| qiskit (Aer statevector) | 600.336 ms |
| polypus (native Rust) | 77.356 ms |

speedup vs Terra Statevector: 26.92x  
speedup vs Aer statevector: 7.76x

## QFT benchmark — 20 qubits (210 gates)
_2026-07-28 10:04:31_ · best of 5 runs

max amplitude error (qiskit vs native): 1.73e-18

| engine | time |
|---|---|
| qiskit (Terra Statevector) | 2054.103 ms |
| qiskit (Aer statevector) | 598.993 ms |
| polypus (native Rust) | 75.813 ms |

speedup vs Terra Statevector: 27.09x  
speedup vs Aer statevector: 7.90x

## QFT benchmark — 20 qubits (210 gates)
_2026-07-28 10:10:03_ · best of 8 runs

max amplitude error (qiskit vs native): 1.73e-18

| engine | time |
|---|---|
| qiskit (Terra Statevector) | 2134.849 ms |
| qiskit (Aer statevector) | 600.988 ms |
| polypus (native Rust) | 77.824 ms |

speedup vs Terra Statevector: 27.43x  
speedup vs Aer statevector: 7.72x

## QFT benchmark — 20 qubits (210 gates)
_2026-07-28 10:14:07_ · 15 interleaved rounds · load average: 18.86 10.02 5.72 (1/5/15 min)

max amplitude error (qiskit vs native): 1.73e-18

Machine is shared (multi-tenant, CPU governor `powersave`); min/median/max are reported instead of a single number because wall-clock time for the multi-threaded engines has been observed to vary by up to ~20x under load.

| engine | min (ms) | median (ms) | max (ms) |
|---|---|---|---|
| qiskit (Terra Statevector) | 2102.955 | 2232.023 | 2508.470 |
| qiskit (Aer statevector) | 666.138 | 677.509 | 700.213 |
| polypus (native Rust) | 87.222 | 89.449 | 94.388 |

speedup of polypus vs median — Terra: 24.95x, Aer: 7.57x

## QFT benchmark — 20 qubits (210 gates)
_2026-07-28 10:17:23_ · 10 rounds per engine, each in its own subprocess · load average: 15.95 9.36 6.07 (1/5/15 min)

max amplitude error (qiskit vs native): 1.73e-18

Each engine runs in an isolated subprocess: a single call to Terra's `Statevector.from_instruction` earlier in a process leaves later Aer runs ~20x slower in that same process (an OpenBLAS/OpenMP thread-affinity interaction, not machine noise), so in-process interleaving understated Aer.

| engine | min (ms) | median (ms) | max (ms) |
|---|---|---|---|
| qiskit (Terra Statevector) | 2084.683 | 2150.147 | 2259.001 |
| qiskit (Aer statevector) | 29.722 | 606.481 | 644.033 |
| polypus (native Rust) | 77.926 | 80.484 | 95.759 |

speedup of polypus vs median — Terra: 26.72x, Aer: 7.54x

## QFT benchmark — 26 qubits (351 gates)
_2026-07-28 10:33:40_ · 10 rounds per engine, each in its own subprocess · load average: 28.55 17.87 9.90 (1/5/15 min)

max amplitude error (aer vs native): 2.98e-19

| engine | min (ms) | median (ms) | max (ms) |
|---|---|---|---|
| qiskit (Aer statevector) | 3846.909 | 3943.831 | 4078.306 |
| polypus (native Rust) | 17284.405 | 17344.376 | 17447.709 |

speedup vs Aer statevector (median): 0.23x

## QFT via Aer backend — 20 qubits, shots=4096
_2026-07-28 10:44:10_ · 10 rounds per engine, each in its own subprocess · load average: 2.66 2.74 5.20 (1/5/15 min)

Isolates polypus's wrapper overhead: all three paths run the identical qiskit-aer engine underneath.

| path | min (ms) | median (ms) | max (ms) |
|---|---|---|---|
| qiskit -> Aer directly (no polypus) | 42.627 | 45.481 | 48.629 |
| qiskit circuit -> polypus.run_quantum_circuit(backend=aer) | 43.703 | 46.384 | 49.338 |
| native circuit -> polypus.run_quantum_circuit(backend=aer) | 42.678 | 349.243 | 644.294 |

overhead vs raw qiskit+Aer (median) — qiskit circuit via polypus: 1.02x, native circuit via polypus: 7.68x

## QFT via Aer backend — 26 qubits, shots=4096
_2026-07-28 10:47:13_ · 10 rounds per engine, each in its own subprocess · load average: 29.03 13.23 8.63 (1/5/15 min)

Isolates polypus's wrapper overhead: all three paths run the identical qiskit-aer engine underneath.

| path | min (ms) | median (ms) | max (ms) |
|---|---|---|---|
| qiskit -> Aer directly (no polypus) | 4062.443 | 4161.414 | 4360.462 |
| qiskit circuit -> polypus.run_quantum_circuit(backend=aer) | 3901.215 | 4085.555 | 4235.314 |
| native circuit -> polypus.run_quantum_circuit(backend=aer) | 3960.591 | 4072.967 | 4243.547 |

overhead vs raw qiskit+Aer (median) — qiskit circuit via polypus: 0.98x, native circuit via polypus: 0.98x

## QFT benchmark — 4 qubits (10 gates)
_2026-07-28 12:56:21_ · 1 rounds per engine, each in its own subprocess · load average: 0.60 0.81 0.52 (1/5/15 min)

max amplitude error (aer vs native): 1.39e-16

| engine | min (ms) | median (ms) | max (ms) |
|---|---|---|---|
| qiskit (Aer statevector) | 0.612 | 0.612 | 0.612 |
| polypus (native Rust) | 0.002 | 0.002 | 0.002 |

speedup vs Aer statevector (median): 359.82x

