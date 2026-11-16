**BREAKING:** for Rust users, `TrainFlow` and `RunCircuitFlow`
(`polypus-orchestration`) have a new public field,
`trace: Option<Arc<CallTracer>>`, so a struct literal must set it;
`trace: None` keeps today's behaviour.
