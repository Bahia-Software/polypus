The call trace in Rust: the `trace` module of `polypus-orchestration`, with
`CallTracer`, which writes, caps and closes a trace, the types of its lines
(`TraceHeader`, `TraceEvent`, `RunStatus`, …), `DEFAULT_MAX_BYTES`, and four
decorators, `TracingPlanner`, `TracingBackend`, `TracingObserver` and
`TracingVariance`, which record the calls they forward; and
`Method::name()`, the optimizer's name that a trace's header records.
