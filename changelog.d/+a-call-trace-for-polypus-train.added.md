A call trace for `polypus.train` and `polypus.run_quantum_circuit`
(contract C-12): the keyword-only `trace=` (a path to a new file) or
`POLYPUS_CALL_TRACE=<directory>` writes one JSON Lines line per planner
batch, backend wave, optimizer iteration and QNG metric call, with sizes and
times only, capped by `POLYPUS_CALL_TRACE_MAX_MB` (512 MiB by default). It
never changes a result.
