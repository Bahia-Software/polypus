//! The `command` option accepts a JSON argv array, so a worker script (or the
//! interpreter) living under a path with spaces can be launched end to end. The plain
//! string form is split on whitespace and cannot express such a path.

mod common;

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Command;

use polypus_backend::{BackendBuildContext, OptLevel, RunParams};
use polypus_subprocess_backend::SubprocessBackend;

/// A scratch directory with a space in its name, removed on drop (also on panic).
struct ScratchDir(PathBuf);

impl ScratchDir {
    /// `tag` keeps concurrently running tests (same pid) out of each other's directory.
    fn new(tag: &str) -> ScratchDir {
        let dir = std::env::temp_dir().join(format!(
            "polypus dir with spaces {tag} {}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).expect("create the scratch directory");
        ScratchDir(dir)
    }
}

impl Drop for ScratchDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn context(command: String) -> BackendBuildContext {
    BackendBuildContext {
        id: "bridge-argv".to_string(),
        shots: 128,
        n_qpus: 1,
        seed: Some(5),
        opt_level: OptLevel::default(),
        options: HashMap::from([("command".to_string(), command)]),
    }
}

/// Build the backend from `command` through the registry path, run one circuit and
/// check the counts: the end-to-end proof that the argv reached `exec` intact.
fn assert_worker_answers(command: String) {
    assert!(
        command.starts_with('['),
        "the JSON form is what is under test"
    );
    let backend = SubprocessBackend::from_context(&context(command))
        .expect("a JSON argv with a spaced path spawns and handshakes");
    let params = RunParams {
        id: "bridge-argv".to_string(),
        shots: 128,
        seed: Some(5),
        opt_level: OptLevel::default(),
    };
    let out = backend
        .run_circuits(&[common::native(2)], &params)
        .expect("the spaced-path worker answers a run");
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].values().sum::<u64>(), 128);

    backend.close();
}

#[test]
fn json_argv_launches_a_worker_whose_script_path_has_spaces() {
    let python = match common::resolve_python() {
        Some(p) => p,
        None => {
            eprintln!("skipping: no Python interpreter found (set POLYPUS_BRIDGE_PYTHON)");
            return;
        }
    };
    let scratch = ScratchDir::new("script");
    let script = scratch.0.join("worker.py");
    std::fs::copy(common::worker_script(), &script).expect("copy the worker into the scratch dir");

    let argv = vec![python, script.to_string_lossy().into_owned()];
    assert_worker_answers(serde_json::to_string(&argv).expect("argv serialises to JSON"));
}

/// The interpreter itself (`argv[0]`) lives under a path with spaces: a symlink to the
/// resolved Python is created inside the spaced scratch directory.
#[test]
fn json_argv_launches_an_interpreter_whose_path_has_spaces() {
    let python = match common::resolve_python() {
        Some(p) => p,
        None => {
            eprintln!("skipping: no Python interpreter found (set POLYPUS_BRIDGE_PYTHON)");
            return;
        }
    };
    // `resolve_python` may return a bare name (`python3`); ask the interpreter for its
    // own absolute path so it can be symlinked.
    let printed = Command::new(&python)
        .args(["-c", "import sys; print(sys.executable)"])
        .output()
        .expect("run the interpreter to learn its path");
    assert!(printed.status.success(), "the interpreter reports its path");
    let real_python = String::from_utf8(printed.stdout)
        .expect("the interpreter path is UTF-8")
        .trim()
        .to_string();
    assert!(
        std::path::Path::new(&real_python).is_absolute(),
        "expected an absolute interpreter path, got {real_python:?}"
    );

    let scratch = ScratchDir::new("interpreter");
    let interpreter = scratch.0.join("py thon");
    std::os::unix::fs::symlink(&real_python, &interpreter)
        .expect("symlink the interpreter into the scratch dir");
    let script = scratch.0.join("worker.py");
    std::fs::copy(common::worker_script(), &script).expect("copy the worker into the scratch dir");

    let argv = vec![
        interpreter.to_string_lossy().into_owned(),
        script.to_string_lossy().into_owned(),
    ];
    assert_worker_answers(serde_json::to_string(&argv).expect("argv serialises to JSON"));
}
