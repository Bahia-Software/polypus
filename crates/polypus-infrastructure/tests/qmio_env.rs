//! `QmioBackend::new` reads the `QMIO_*` tuning variables from the real process
//! environment. A single test in its own binary, so mutating the environment cannot
//! race any other test.
#![cfg(feature = "qmio")]

use polypus_infrastructure::qmio::{QmioBackend, QmioError, QmioProgramFormat};

const VARS: [&str; 4] = [
    "QMIO_RECV_TIMEOUT_MS",
    "QMIO_MAX_RETRIES",
    "QMIO_RETRY_BACKOFF_MS",
    "QMIO_GLOBAL_TIMEOUT_MS",
];

fn build() -> Result<QmioBackend, QmioError> {
    QmioBackend::new(
        "tcp://127.0.0.1:59999".to_string(),
        QmioProgramFormat::OpenQasm,
        0,
        None,
        "binary_count".to_string(),
    )
}

fn clear() {
    for name in VARS {
        std::env::remove_var(name);
    }
}

#[test]
fn invalid_tuning_variables_fail_construction() {
    clear();
    assert!(build().is_ok(), "unset variables must use the defaults");

    for name in VARS {
        for bad in ["3000O", "0"] {
            // Zero is only invalid for the two timeouts.
            let zero_ok =
                bad == "0" && (name == "QMIO_MAX_RETRIES" || name == "QMIO_RETRY_BACKOFF_MS");
            std::env::set_var(name, bad);
            let result = build();
            clear();
            if zero_ok {
                assert!(result.is_ok(), "{name}=0 must be accepted");
                continue;
            }
            match result {
                Err(QmioError::Config(msg)) => assert!(msg.contains(name), "{msg}"),
                Err(other) => panic!("{name}={bad}: expected Config, got {other}"),
                Ok(_) => panic!("{name}={bad}: expected Config, got Ok"),
            }
        }
    }

    std::env::set_var("QMIO_RECV_TIMEOUT_MS", "1500");
    let result = build();
    clear();
    assert!(result.is_ok(), "a valid override must be accepted");
}
