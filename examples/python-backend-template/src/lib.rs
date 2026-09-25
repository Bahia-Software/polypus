//! # A subprocess-bridge Polypus backend template (Python)
//!
//! The Python half of this template is [`worker.py`](../worker.py): a filled-in copy
//! of `polypus-subprocess-backend`'s `worker_template.py`, where the single
//! `execute_circuits` function returns hardware-free deterministic counts. That is the
//! whole backend — a `pip install polypus` user points Polypus at it with
//! `infrastructure="subprocess", options={"command": "python3 worker.py"}`.
//!
//! This crate carries no code of its own; its purpose is the test in
//! `tests/conformance.rs`, which spawns `worker.py` through the real subprocess bridge
//! and runs the Polypus conformance battery against it — the executable proof that an
//! external author's Python backend passes.
//!
//! Run it with a Python interpreter available:
//!
//! ```text
//! cd examples/python-backend-template && cargo test
//! # or point at a specific interpreter:
//! POLYPUS_BRIDGE_PYTHON=/path/to/python3 cargo test
//! ```
