use std::sync::OnceLock;

static TOKIO_RT: OnceLock<tokio::runtime::Runtime> = OnceLock::new();

/// Returns a reference to the global multi-threaded Tokio runtime used by Polypus.
/// The runtime is created lazily on first access and lives for the process lifetime.
pub fn tokio_runtime() -> &'static tokio::runtime::Runtime {
    TOKIO_RT.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            .thread_name("polypus-worker")
            .enable_all()
            .build()
            .expect("Failed to create Tokio runtime")
    })
}
