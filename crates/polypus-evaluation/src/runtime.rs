use std::sync::OnceLock;

static TOKIO_RT: OnceLock<tokio::runtime::Runtime> = OnceLock::new();

/// Returns a reference to the global multi-threaded Tokio runtime used by Polypus.
///
/// The runtime is created lazily on first access and lives for the process
/// lifetime. Building it can only fail on OS resource exhaustion; that failure
/// is returned (never a panic) so callers on the FFI path can surface it as a
/// typed error. The runtime is built *outside* the `OnceLock` initializer so a
/// failure is returned to the caller and simply retried on the next call,
/// rather than poisoning a lazy `get_or_init`.
pub fn tokio_runtime() -> std::io::Result<&'static tokio::runtime::Runtime> {
    // Fast path: already built.
    if let Some(rt) = TOKIO_RT.get() {
        return Ok(rt);
    }
    let rt = tokio::runtime::Builder::new_multi_thread()
        .thread_name("polypus-worker")
        .enable_all()
        .build()?;
    // Store it; if another thread raced us, keep whichever won (both are valid).
    let _ = TOKIO_RT.set(rt);
    match TOKIO_RT.get() {
        Some(rt) => Ok(rt),
        // Unreachable in practice (a `set` just succeeded, ours or the racer's),
        // but returned as an error rather than unwrapped so this stays panic-free.
        None => Err(std::io::Error::other(
            "the global Tokio runtime is unexpectedly unavailable",
        )),
    }
}
