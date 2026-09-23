"""
OpenQASM 2.0 coverage of the Polypus importer/exporter over benchmark suites.

Walks one or more directories of ``.qasm`` files (e.g. QASMBench, MQT Bench,
SupermarQ), and for every file:

1. parses it with Qiskit (``QuantumCircuit.from_qasm_str``) as the reference
   -- a file Qiskit rejects is reported as *invalid source*, not as a Polypus
   gap;
2. parses it with Polypus (``polypus.Circuit.from_qasm2``) and, on failure,
   classifies the construct that caused it (an unsupported gate, ``reset``,
   ``if``, ``opaque``, …);
3. re-emits it (``Circuit.to_qasm2``) and checks *fidelity*: Qiskit parses the
   re-emitted text and its instruction counts per name (``count_ops``), size
   and depth are compared with the reference. ``identical`` means Polypus
   hands a backend the same program; ``renamed`` means the same size and depth
   but some instructions re-spelled (e.g. ``p`` → ``u3``); ``different`` would
   be a bug (a decomposition or a lost instruction);
4. optionally (``--aer``) runs both circuits on Aer with a fixed seed after the
   same transpilation and compares the counts.

Each file is processed in its own subprocess, so a pathological input (time
or memory) cannot take the run down; it is reported as a timeout or crash.

Usage::

    python benchmarks/qasm_coverage.py DIR [DIR ...] \\
        [--markdown coverage.md] [--csv coverage.csv] [--json coverage.json] \\
        [--aer] [--aer-max-qubits 20] [--jobs 4] [--timeout 300] \\
        [--max-bytes 50000000]

The summary table (Markdown) groups files by suite (the directory given on the
command line) and by its first sub-directory, and breaks the failures down by
construct. It is the acceptance evidence for gate coverage.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

WORKER_FLAG = "--worker"

# ─────────────────────────────────────────────────────────────────────────────
# Failure classification
# ─────────────────────────────────────────────────────────────────────────────

# (regex over the Polypus parse error message, construct label). The first
# match wins; the label may use the regex's first group.
_CONSTRUCTS = [
    (r"unsupported gate '([^']+)'", "gate `{0}`"),
    (r"'reset' is not supported", "`reset`"),
    (
        r"'if' statements are not supported|classical conditional",
        "`if` (classical control)",
    ),
    (r"custom gate definitions are not supported", "`gate` declaration"),
    (r"'opaque' declarations are not supported|opaque gate", "`opaque` declaration"),
    (r"after it was measured", "gate after measurement (C-4)"),
    (r"expected 'OPENQASM 2\.0;' header", "missing `OPENQASM 2.0;` header"),
    (r"only OpenQASM version 2\.0 is supported", "not OpenQASM 2.0"),
    (r"undeclared (quantum|classical) register", "undeclared register"),
    (r"MAX_REGISTER_BITS", "register size limit"),
    (r"expression nested too deeply", "expression depth limit"),
]


def classify_polypus_error(message: str) -> str:
    """Name the construct a Polypus parse error points at."""
    for pattern, label in _CONSTRUCTS:
        m = re.search(pattern, message)
        if m:
            return label.format(*m.groups())
    # Keep unknown causes readable but bounded: drop the line prefix.
    detail = re.sub(r"^QASM parse error at line \d+: ", "", message)
    return f"other: {detail[:80]}"


def classify_qiskit_error(message: str) -> str:
    first = message.strip().splitlines()[0] if message.strip() else message
    first = re.sub(r"^.*?<input>:\d+,\d+: ", "", first)
    return first[:100]


# ─────────────────────────────────────────────────────────────────────────────
# Worker: one file, in its own process
# ─────────────────────────────────────────────────────────────────────────────


def _metrics(qc):
    return {
        "size": qc.size(),
        "depth": qc.depth(),
        "count_ops": {k: int(v) for k, v in qc.count_ops().items()},
    }


def _normalise_counts(counts):
    # Multiple classical registers print as space-separated groups; Polypus
    # flattens registers in declaration order, which yields the same bits.
    out = Counter()
    for key, value in counts.items():
        out[key.replace(" ", "")] += value
    return dict(out)


def _aer_counts(qasm, seed, shots):
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator

    sim = AerSimulator()
    qc = transpile(QuantumCircuit.from_qasm_str(qasm), sim, optimization_level=0)
    result = sim.run(qc, shots=shots, seed_simulator=seed).result()
    return _normalise_counts(result.get_counts())


def process_file(path: str, aer: bool, aer_max_qubits: int, seed: int, shots: int):
    record = {"path": path}
    src = Path(path).read_text(encoding="utf-8", errors="replace")
    record["bytes"] = len(src.encode("utf-8"))

    # 1. Qiskit reference.
    from qiskit import QuantumCircuit

    t0 = time.perf_counter()
    try:
        reference = QuantumCircuit.from_qasm_str(src)
        record["qiskit"] = "ok"
        record["num_qubits"] = reference.num_qubits
        record["reference"] = _metrics(reference)
    except Exception as exc:  # noqa: BLE001 -- any rejection is data here
        reference = None
        record["qiskit"] = "error"
        record["qiskit_error"] = classify_qiskit_error(str(exc))
    record["qiskit_parse_s"] = time.perf_counter() - t0

    # 2. Polypus parse.
    import polypus

    t0 = time.perf_counter()
    try:
        circuit = polypus.Circuit.from_qasm2(src)
        record["polypus"] = "ok"
        record["polypus_instructions"] = len(circuit)
    except ValueError as exc:
        record["polypus"] = "error"
        record["polypus_error"] = str(exc)
        record["construct"] = classify_polypus_error(str(exc))
        record["polypus_parse_s"] = time.perf_counter() - t0
        return record
    record["polypus_parse_s"] = time.perf_counter() - t0

    # 3. Re-emit and check fidelity against the reference.
    try:
        emitted = circuit.to_qasm2()
        record["emit"] = "ok"
    except ValueError as exc:
        record["emit"] = "error"
        record["emit_error"] = str(exc)
        return record

    if reference is None:
        return record
    try:
        through_polypus = QuantumCircuit.from_qasm_str(emitted)
    except Exception as exc:  # noqa: BLE001
        record["fidelity"] = "unparseable"
        record["fidelity_detail"] = classify_qiskit_error(str(exc))
        return record
    got = _metrics(through_polypus)
    ref = record["reference"]
    if got == ref:
        record["fidelity"] = "identical"
    elif got["size"] == ref["size"] and got["depth"] == ref["depth"]:
        record["fidelity"] = "renamed"
        lost = Counter(ref["count_ops"]) - Counter(got["count_ops"])
        gained = Counter(got["count_ops"]) - Counter(ref["count_ops"])
        record["fidelity_detail"] = (
            ", ".join(f"{k}→" for k in sorted(lost))
            + " / "
            + ", ".join(f"→{k}" for k in sorted(gained))
        )
    else:
        record["fidelity"] = "different"
        record["fidelity_detail"] = (
            f"size {ref['size']}→{got['size']}, depth {ref['depth']}→{got['depth']}"
        )

    # 4. Optional: the same Aer run on both programs.
    if aer and reference.num_qubits <= aer_max_qubits and reference.num_clbits > 0:
        try:
            same = _aer_counts(src, seed, shots) == _aer_counts(emitted, seed, shots)
            record["aer"] = "equal" if same else "differ"
        except Exception as exc:  # noqa: BLE001
            record["aer"] = "error"
            record["aer_error"] = str(exc)[:200]
    return record


def _worker_main(argv):
    path, aer, aer_max_qubits, seed, shots = argv
    record = process_file(path, aer == "1", int(aer_max_qubits), int(seed), int(shots))
    sys.stdout.write(json.dumps(record))


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────


def _run_one(path, args):
    size = os.path.getsize(path)
    if size > args.max_bytes:
        return {
            "path": path,
            "bytes": size,
            "skipped": f"larger than {args.max_bytes} bytes",
        }
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        WORKER_FLAG,
        path,
        "1" if args.aer else "0",
        str(args.aer_max_qubits),
        str(args.seed),
        str(args.shots),
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=args.timeout, check=False
        )
    except subprocess.TimeoutExpired:
        return {
            "path": path,
            "bytes": size,
            "skipped": f"timeout after {args.timeout}s",
        }
    if proc.returncode != 0 or not proc.stdout.strip():
        tail = (proc.stderr or "").strip().splitlines()[-1:] or ["no output"]
        return {
            "path": path,
            "bytes": size,
            "skipped": f"worker crashed: {tail[0][:120]}",
        }
    return json.loads(proc.stdout)


def _group(path: str, roots: list[Path]) -> tuple[str, str]:
    p = Path(path).resolve()
    for root in roots:
        try:
            rel = p.relative_to(root)
        except ValueError:
            continue
        sub = rel.parts[0] if len(rel.parts) > 1 else ""
        return root.name, sub
    return "?", ""


def summarise(records, roots):
    groups = defaultdict(list)
    for r in records:
        suite, sub = _group(r["path"], roots)
        groups[(suite, "")].append(r)
        if sub:
            groups[(suite, sub)].append(r)
    rows = []
    for (suite, sub), rs in sorted(groups.items()):
        valid = [r for r in rs if r.get("qiskit") == "ok"]
        rows.append(
            {
                "suite": suite if not sub else f"↳ {sub}",
                "files": len(rs),
                "skipped": sum("skipped" in r for r in rs),
                "qiskit_valid": len(valid),
                "polypus_parsed": sum(r.get("polypus") == "ok" for r in valid),
                "reemitted": sum(r.get("emit") == "ok" for r in valid),
                "identical": sum(r.get("fidelity") == "identical" for r in valid),
                "renamed": sum(r.get("fidelity") == "renamed" for r in valid),
                "different": sum(
                    r.get("fidelity") in ("different", "unparseable") for r in valid
                ),
                "aer_equal": sum(r.get("aer") == "equal" for r in valid),
                "aer_checked": sum(
                    r.get("aer") in ("equal", "differ", "error") for r in valid
                ),
                "invalid_source": sum(r.get("qiskit") == "error" for r in rs),
            }
        )
    return rows


def render_markdown(records, rows, roots, args):
    out = ["# OpenQASM 2.0 coverage of the Polypus importer/exporter", ""]
    out.append(
        f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')} by `benchmarks/qasm_coverage.py`"
        f" over: {', '.join(str(r) for r in roots)}."
    )
    try:
        import polypus
        import qiskit

        out.append(
            f"polypus {getattr(polypus, '__version__', '?')}, qiskit {qiskit.__version__}."
        )
    except ImportError:
        pass
    out += [
        "",
        "Columns: **valid** = Qiskit parses the file (the reference); **parsed** / "
        "**re-emitted** = Polypus imports / exports it; **identical** = Qiskit sees "
        "the same instruction counts per name, size and depth in the re-emitted "
        "text as in the original; **renamed** = same size and depth, some "
        "instructions re-spelled; **different** = a fidelity failure.",
        "",
        "| suite | files | skipped | valid | parsed | re-emitted | identical | renamed | different"
        + (" | Aer equal / checked" if args.aer else "")
        + " | invalid source |",
        "|---|---|---|---|---|---|---|---|---" + ("|---" if args.aer else "") + "|---|",
    ]
    for r in rows:
        valid = r["qiskit_valid"] or 1
        cells = [
            r["suite"],
            r["files"],
            r["skipped"],
            r["qiskit_valid"],
            f"{r['polypus_parsed']} ({100 * r['polypus_parsed'] / valid:.0f}%)",
            r["reemitted"],
            r["identical"],
            r["renamed"],
            r["different"],
        ]
        if args.aer:
            cells.append(f"{r['aer_equal']} / {r['aer_checked']}")
        cells.append(r["invalid_source"])
        out.append("| " + " | ".join(str(c) for c in cells) + " |")

    failures = Counter()
    failures_by_suite = defaultdict(Counter)
    for rec in records:
        if rec.get("qiskit") == "ok" and rec.get("polypus") == "error":
            failures[rec["construct"]] += 1
            failures_by_suite[_group(rec["path"], roots)[0]][rec["construct"]] += 1
    out += ["", "## Why valid files fail to import (by construct)", ""]
    if failures:
        suites = sorted(failures_by_suite)
        out.append("| construct | files | " + " | ".join(suites) + " |")
        out.append("|---|---|" + "---|" * len(suites))
        for construct, n in failures.most_common():
            per = [str(failures_by_suite[s].get(construct, 0)) for s in suites]
            out.append(f"| {construct} | {n} | " + " | ".join(per) + " |")
    else:
        out.append("_none_")

    renamed = Counter(
        rec.get("fidelity_detail", "")
        for rec in records
        if rec.get("fidelity") == "renamed"
    )
    out += ["", "## Re-spelled instructions (files marked *renamed*)", ""]
    if renamed:
        out.append("| lost → / → gained | files |")
        out.append("|---|---|")
        for detail, n in renamed.most_common():
            out.append(f"| {detail} | {n} |")
    else:
        out.append("_none_")

    bad = [
        rec
        for rec in records
        if rec.get("fidelity") in ("different", "unparseable")
        or rec.get("aer") == "differ"
    ]
    out += ["", "## Fidelity failures", ""]
    if bad:
        for rec in bad:
            out.append(
                f"- `{rec['path']}`: {rec.get('fidelity')} {rec.get('fidelity_detail', '')}"
                f" aer={rec.get('aer', '-')}"
            )
    else:
        out.append("_none_")

    invalid = Counter(
        rec["qiskit_error"] for rec in records if rec.get("qiskit") == "error"
    )
    out += ["", "## Invalid source (Qiskit rejects the file too)", ""]
    if invalid:
        out.append("| Qiskit error | files |")
        out.append("|---|---|")
        for message, n in invalid.most_common():
            out.append(f"| {message} | {n} |")
    else:
        out.append("_none_")

    skipped = Counter(rec["skipped"] for rec in records if "skipped" in rec)
    if skipped:
        out += ["", "## Skipped", ""]
        for reason, n in skipped.most_common():
            out.append(f"- {reason}: {n}")
    return "\n".join(out) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "dirs", nargs="+", type=Path, help="directories to scan recursively"
    )
    parser.add_argument("--markdown", type=Path, help="write the summary table here")
    parser.add_argument("--csv", type=Path, help="write one row per file here")
    parser.add_argument(
        "--json", type=Path, help="write the full per-file records here"
    )
    parser.add_argument("--aer", action="store_true", help="also compare Aer counts")
    parser.add_argument("--aer-max-qubits", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--shots", type=int, default=2000)
    parser.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--timeout", type=int, default=300, help="seconds per file")
    parser.add_argument(
        "--max-bytes", type=int, default=50_000_000, help="skip larger files"
    )
    args = parser.parse_args(argv)

    roots = [d.resolve() for d in args.dirs]
    files = sorted(
        str(p) for root in roots for p in root.rglob("*.qasm") if p.is_file()
    )
    if not files:
        parser.error("no .qasm files found")
    print(f"{len(files)} files, {args.jobs} jobs", file=sys.stderr)

    records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(_run_one, f, args): f for f in files}
        for done, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            records.append(fut.result())
            if done % 50 == 0 or done == len(files):
                print(f"  {done}/{len(files)}", file=sys.stderr)
    records.sort(key=lambda r: r["path"])

    rows = summarise(records, roots)
    markdown = render_markdown(records, rows, roots, args)
    if args.markdown:
        args.markdown.write_text(markdown, encoding="utf-8")
    else:
        sys.stdout.write(markdown)
    if args.json:
        args.json.write_text(json.dumps(records, indent=1), encoding="utf-8")
    if args.csv:
        fields = [
            "path",
            "bytes",
            "skipped",
            "num_qubits",
            "qiskit",
            "qiskit_error",
            "polypus",
            "construct",
            "polypus_error",
            "emit",
            "fidelity",
            "fidelity_detail",
            "aer",
            "qiskit_parse_s",
            "polypus_parse_s",
        ]
        with args.csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == WORKER_FLAG:
        _worker_main(sys.argv[2:])
    else:
        main()
