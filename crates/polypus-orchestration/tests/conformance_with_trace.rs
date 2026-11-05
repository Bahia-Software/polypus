//! Contract C-12: the conformance battery passes against the native backend
//! wrapped in a `TracingBackend`, so the wrapper forwards every call a backend
//! is certified on.
use std::sync::Arc;

use polypus_backend_conformance::{Conformance, Status};
use polypus_infrastructure::{NativeStatevectorBackend, QuantumBackend};
use polypus_orchestration::trace::{
    CallTracer, RunStatus, TraceFlow, TraceHeader, TracingBackend, DEFAULT_MAX_BYTES,
};

#[test]
fn conformance_with_trace() {
    let dir =
        std::env::temp_dir().join(format!("polypus-conformance-trace-{}", std::process::id()));
    if dir.exists() {
        std::fs::remove_dir_all(&dir).unwrap();
    }
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("conformance.jsonl");
    let header = TraceHeader {
        flow: TraceFlow::RunCircuit,
        id: "conformance".into(),
        polypus: "test".into(),
        method: None,
        dims: None,
        backend: "polypus".into(),
        infrastructure: "local".into(),
        n_qpus: 1,
        shots: 1000,
        seed: Some(0xABCD),
    };
    let tracer = CallTracer::create(&path, header, DEFAULT_MAX_BYTES).unwrap();
    let for_battery = Arc::clone(&tracer);
    let report = Conformance::new("native+trace", move || {
        let native: Arc<dyn QuantumBackend> = Arc::new(NativeStatevectorBackend::new(0xABCD));
        Ok(
            Arc::new(TracingBackend::new(native, Arc::clone(&for_battery), None))
                as Arc<dyn QuantumBackend>,
        )
    })
    .shots(1000)
    .run();
    println!("\n{report}");
    report.assert_conformant();
    for name in [
        "unresponsive_backend_is_classified",
        "interrupted_call_is_aborted",
        "clean_provider_error_is_classified",
    ] {
        let check = report.checks.iter().find(|c| c.name == name).unwrap();
        assert!(
            matches!(check.status, Status::Skipped(_)),
            "{name}: {:?}",
            check.status
        );
    }
    tracer.finish(RunStatus::Ok);
    let text = std::fs::read_to_string(&path).unwrap();
    assert!(text
        .lines()
        .any(|l| l.starts_with(r#"{"ev":"wave","k":null,"#)));
    assert!(text.lines().last().unwrap().contains(r#""status":"ok""#));
    std::fs::remove_dir_all(&dir).unwrap();
}
