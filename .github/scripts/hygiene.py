#!/usr/bin/env python3
"""Repository hygiene check for polypus.

Analyses the net difference between two Git trees (never each intermediate
commit, never the working tree) and reports:

* forbidden-name   internal files that must never be committed
* ignored-file     files added although the versioned .gitignore files ignore them
* cluster-path, login-host, private-ipv4, private-key, github-token
                   infrastructure data and credentials on added text lines
* private-pattern  maintainers' private regex (HYGIENE_LEAK_PATTERNS)
* notebook-output  tutorials/**/*.ipynb committed with outputs
* file-size        blobs larger than 1 MiB
* version-mismatch Cargo.toml [workspace.package].version vs CITATION.cff
* submodule        added or changed gitlinks (their content is not analysed)

Exit codes: 0 clean, 1 violations, 2 configuration or operational error.

Policy notes:

* Renames and copies are not detected: a moved file is analysed as a new
  file in full, so coverage never depends on Git's rename heuristics.
* A blob is binary when its first 8000 bytes contain NUL (Git's heuristic,
  independent of .gitattributes). Binary blobs skip the line checks only.
  Text that is not valid UTF-8 is decoded as Latin-1 (patterns are ASCII)
  and counted in the summary.
* Symlinks are analysed as their link blob (the target string), never
  followed.
* The private regex never sees more than 4096 characters at once: longer
  lines are scanned in 4096-character windows overlapping by 256, so every
  match of at most 256 characters is found. Keep each private sub-pattern
  shorter than that.
* Matched text, line contents and the private regex are never printed.
"""

import argparse
import ipaddress
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

ALLOW_FILE = ".github/hygiene-allow.txt"
MAX_BLOB_BYTES = 1024 * 1024
WINDOW = 4096
OVERLAP = 256
SUMMARY_LIMIT = 512 * 1024
LEAK_ENV = "HYGIENE_LEAK_PATTERNS"
EMPTY = "EMPTY"
EMPTY_TREES = {
    "sha1": "4b825dc642cb6eb9a060e54bf8d69288fbee4904",
    "sha256": "6ef19b41225c5369f1c104d45d8d85efa9b057b53b14b4b9b939dd74decc5321",
}
MODE_SYMLINK = "120000"
MODE_GITLINK = "160000"

PUBLIC_LEAK_RULES = {
    "cluster-path": re.compile(
        r"(?<![\w.~-])(?:/mnt/netapp\d*/|/home/[^/\s'\"`]+/"
        r"|/mnt/lustre(?![\w.-])|/opt/cesga(?![\w.-]))"
    ),
    "login-host": re.compile(r"\blogin\d+\b"),
    "private-key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "github-token": re.compile(
        r"(?<![A-Za-z0-9_])(?:gh[pousr]_[A-Za-z0-9]{36,}(?![A-Za-z0-9_])"
        r"|github_pat_[A-Za-z0-9_]{22,})"
    ),
}
IPV4_CANDIDATE = re.compile(r"(?<![\w.])\d{1,3}(?:\.\d{1,3}){3}(?![\w]|\.\d)")
PRIVATE_NETWORKS = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
]
LEAK_RULES = (*PUBLIC_LEAK_RULES, "private-ipv4", "private-pattern")
# Rules that accept entries in the allow file; the rest admit no exceptions.
EXEMPTABLE_RULES = {"ignored-file", "file-size", *LEAK_RULES}

FIX_HINTS = {
    "forbidden-name": "internal files must never be committed; remove it",
    "ignored-file": "this path is ignored by .gitignore; remove it or request "
    "a narrow ignored-file exception with a reason",
    "private-pattern": "line matches a maintainers' private pattern; remove "
    "the infrastructure identifier",
    "file-size": "blob exceeds 1 MiB; keep large data out of the repository "
    "or request a narrow file-size exception with a reason",
    "submodule": "submodule content is not covered by this check; "
    "submodules are not accepted",
}
LEAK_HINT = "line contains {} data; remove or replace it with a placeholder"


class HygieneError(Exception):
    """Configuration or operational error. Messages never carry file data."""


# --------------------------------------------------------------------- git ---


def _analysed_repo_env():
    env = dict(os.environ)
    for key in ("GIT_EXTERNAL_DIFF", "GIT_DIFF_OPTS"):
        env.pop(key, None)
    env.update(GIT_PAGER="cat", GIT_OPTIONAL_LOCKS="0", GIT_TERMINAL_PROMPT="0")
    env.pop(LEAK_ENV, None)
    return env


def isolated_git_env(home):
    """Environment for the scratch ignore repository: no inherited GIT_*
    variable (GIT_DIR, GIT_WORK_TREE, GIT_INDEX_FILE, object directories,
    GIT_CONFIG_* injection, ...), no global or system configuration."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.pop(LEAK_ENV, None)
    env.update(
        GIT_CONFIG_NOSYSTEM="1",
        GIT_CONFIG_GLOBAL=os.devnull,
        GIT_TERMINAL_PROMPT="0",
        HOME=home,
        XDG_CONFIG_HOME=home,
    )
    return env


class Git:
    def __init__(self, repo):
        self.repo = repo
        self.env = _analysed_repo_env()

    def run(self, args, *, ok=(0,), stdin=None, cwd=None, env=None):
        cmd = ["git", "--no-pager", "-c", "color.ui=false", "-c", "core.pager=cat"]
        if cwd is None:
            cmd += ["-C", self.repo]
        try:
            proc = subprocess.run(
                cmd + args,
                input=stdin,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=cwd,
                env=env or self.env,
            )
        except OSError as exc:
            raise HygieneError("could not run git ({})".format(type(exc).__name__))
        if proc.returncode not in ok:
            raise HygieneError(
                "git {} failed with exit code {}".format(args[0], proc.returncode)
            )
        return proc.returncode, proc.stdout

    def out(self, args, **kw):
        return self.run(args, **kw)[1]

    def resolve_commit(self, rev, label):
        if not rev or rev.startswith("-") or "\0" in rev:
            raise HygieneError("invalid {} revision".format(label))
        code, out = self.run(
            ["rev-parse", "--verify", "--quiet", "--end-of-options", rev + "^{commit}"],
            ok=(0, 1),
        )
        if code != 0:
            raise HygieneError("{} revision does not resolve to a commit".format(label))
        return out.decode("ascii").strip()

    def tree_of(self, commit):
        return self.out(["rev-parse", "--verify", commit + "^{tree}"]).decode().strip()

    def empty_tree(self):
        fmt = self.out(["rev-parse", "--show-object-format"]).decode().strip()
        if fmt not in EMPTY_TREES:
            raise HygieneError("unsupported object format")
        return EMPTY_TREES[fmt]

    def merge_base(self, a, b):
        code, out = self.run(["merge-base", a, b], ok=(0, 1))
        if code != 0 or not out.strip():
            raise HygieneError("base and head have no merge base")
        return out.decode("ascii").split()[0]

    def ls_tree(self, tree):
        """{path: (mode, oid)} for every entry of a tree, recursively."""
        entries = {}
        for rec in self.out(["ls-tree", "-r", "-z", "--full-tree", tree]).split(b"\0"):
            if not rec:
                continue
            meta, _, path = rec.partition(b"\t")
            mode, _type, oid = meta.decode("ascii").split()
            entries[decode_path(path)] = (mode, oid)
        return entries

    def diff_trees(self, old, new):
        """[(status, path, new_mode, old_oid, new_oid)] without rename detection."""
        raw = self.out(
            [
                "diff-tree",
                "-r",
                "-z",
                "--raw",
                "--no-renames",
                "--no-ext-diff",
                "--no-textconv",
                "--full-index",
                "--ignore-submodules=none",
                old,
                new,
            ]
        )
        fields = raw.split(b"\0")
        changes = []
        i = 0
        while i < len(fields) and fields[i]:
            meta = fields[i].decode("ascii")
            if not meta.startswith(":") or i + 1 >= len(fields):
                raise HygieneError("unexpected git diff-tree output")
            _old_mode, new_mode, old_oid, new_oid, status = meta[1:].split()
            changes.append(
                (status, decode_path(fields[i + 1]), new_mode, old_oid, new_oid)
            )
            i += 2
        return changes

    def added_line_numbers(self, old_oid, new_oid):
        """Line numbers (1-based, in the new blob) added relative to the old blob."""
        out = self.out(
            [
                "diff",
                "-U0",
                "--no-color",
                "--no-ext-diff",
                "--no-textconv",
                "--diff-algorithm=myers",
                old_oid,
                new_oid,
            ]
        )
        numbers = []
        for m in re.finditer(rb"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", out, re.M):
            start, count = int(m.group(1)), int(m.group(2) or 1)
            numbers.extend(range(start, start + count))
        return numbers

    def is_dirty(self):
        out = self.out(["status", "--porcelain", "-z", "--untracked-files=normal"])
        return bool(out)


class BlobReader:
    """Reads blobs through one `git cat-file --batch` process (no filters)."""

    def __init__(self, git):
        self.git = git
        self.cache = {}

    def read_many(self, oids):
        todo = [o for o in dict.fromkeys(oids) if o not in self.cache]
        if not todo:
            return
        out = self.git.out(
            ["cat-file", "--batch"], stdin="".join(o + "\n" for o in todo).encode()
        )
        pos = 0
        for oid in todo:
            nl = out.index(b"\n", pos)
            header = out[pos:nl].decode("ascii").split()
            if len(header) != 3 or header[1] != "blob":
                raise HygieneError("could not read a blob from git")
            size = int(header[2])
            self.cache[oid] = out[nl + 1 : nl + 1 + size]
            pos = nl + 1 + size + 1

    def get(self, oid):
        self.read_many([oid])
        return self.cache[oid]


def decode_path(raw):
    return raw.decode("utf-8", "surrogateescape")


# -------------------------------------------------------------- allow list ---


class AllowList:
    """Parses .github/hygiene-allow.txt.

    One entry per line: RULE<TAB>PATTERN<TAB># reason. Blank lines and lines
    starting with '#' are ignored. PATTERN is a repo-relative POSIX path that
    must match the whole path: '*' and '?' never cross '/', a component that
    is exactly '**' matches zero or more components. '[', ']' and '\\' are
    rejected, as are absolute paths, '.', '..' and empty components, and
    patterns without at least one literal component."""

    def __init__(self, text):
        self.entries = []
        self.errors = []
        for n, line in enumerate(text.split("\n"), 1):
            line = line.rstrip("\r")
            if not line.strip() or line.startswith("#"):
                continue
            problem = self._add(line)
            if problem:
                self.errors.append("{}:{}: {}".format(ALLOW_FILE, n, problem))

    def _add(self, line):
        parts = line.split("\t")
        if len(parts) != 3:
            return "expected RULE<TAB>PATTERN<TAB># reason"
        rule, pattern, reason = parts
        if rule not in EXEMPTABLE_RULES:
            return "unknown rule or rule without exceptions"
        if not reason.startswith("#") or not reason[1:].strip():
            return "missing reason comment"
        if any(ord(c) < 32 or c in "[]\\" for c in pattern):
            return "unsupported character in pattern"
        comps = pattern.split("/")
        if pattern.startswith("/") or any(c in ("", ".", "..") for c in comps):
            return "pattern must be a relative path without '.', '..' or empty parts"
        if all(set(c) <= set("*?") for c in comps):
            return "pattern is too broad"
        self.entries.append((rule, _glob_regex(comps)))
        return None

    def exempt(self, rule, path):
        return any(r == rule and rx.fullmatch(path) for r, rx in self.entries)


def _glob_regex(comps):
    out = []
    for i, comp in enumerate(comps):
        last = i == len(comps) - 1
        if comp == "**":
            out.append(".*" if last else "(?:[^/]+/)*")
            continue
        body = "".join(
            "[^/]*" if ch == "*" else "[^/]" if ch == "?" else re.escape(ch)
            for ch in comp
        )
        out.append(body if last else body + "/")
    return re.compile("".join(out), re.S)


# ------------------------------------------------------------------ checks ---


def forbidden_name(path):
    comps = path.split("/")
    name = comps[-1].casefold()
    if any(c.casefold() == ".claude" for c in comps):
        return True
    if name == "claude.md" or (name.startswith("claude_") and name.endswith(".md")):
        return True
    if name.startswith("ai_guidelines"):
        return True
    lowered = path.casefold()
    return any(
        lowered == p or lowered.endswith("/" + p)
        for p in ("docs/refactor_plan.md", "docs/implementation_prompt.md")
    )


def ignored_paths(git, blobs, head_entries, candidates):
    """Candidates ignored by the versioned .gitignore files of the head tree,
    decided in a scratch repository that only contains those files."""
    if not candidates:
        return set()
    gitignores = {
        p: e for p, e in head_entries.items() if p.split("/")[-1] == ".gitignore"
    }
    for path, (mode, _oid) in gitignores.items():
        if mode == MODE_SYMLINK:
            raise HygieneError(
                "{} is a symlink; policy files must be regular files".format(
                    printable(path)
                )
            )
        if any(c in ("", ".", "..", ".git") for c in path.split("/")[:-1]):
            raise HygieneError("unsupported .gitignore location")
    blobs.read_many([oid for _mode, oid in gitignores.values()])
    scratch = tempfile.mkdtemp(prefix="hygiene-ignore-")
    try:
        home = os.path.join(scratch, "home")
        templates = os.path.join(scratch, "templates")
        work = os.path.join(scratch, "work")
        for d in (home, templates, work):
            os.mkdir(d)
        env = isolated_git_env(home)
        git.run(["init", "-q", "--template=" + templates, work], cwd=scratch, env=env)
        for path, (_mode, oid) in gitignores.items():
            target = os.path.join(work, *path.split("/"))
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with open(target, "wb") as fh:
                fh.write(blobs.get(oid))
        stdin = b"".join(
            p.encode("utf-8", "surrogateescape") + b"\0" for p in candidates
        )
        code, out = git.run(
            ["check-ignore", "--no-index", "--stdin", "-z"],
            ok=(0, 1),
            stdin=stdin,
            cwd=work,
            env=env,
        )
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return {decode_path(p) for p in out.split(b"\0") if p} if code == 0 else set()


def is_binary(data):
    return b"\0" in data[:8000]


def decode_text(data):
    try:
        return data.decode("utf-8"), True
    except UnicodeDecodeError:
        return data.decode("latin-1"), False


def private_ipv4(line):
    for m in IPV4_CANDIDATE.finditer(line):
        try:
            addr = ipaddress.IPv4Address(m.group(0))
        except ValueError:
            continue
        if any(addr in net for net in PRIVATE_NETWORKS):
            return True
    return False


def private_match(regex, line):
    """Searches `line` with the private regex in bounded windows."""
    if len(line) <= WINDOW:
        return regex.search(line) is not None
    start = 0
    while True:
        if regex.search(line[start : start + WINDOW]) is not None:
            return True
        if start + WINDOW >= len(line):
            return False
        start += WINDOW - OVERLAP


def leak_rules_hit(line, private_regex):
    hits = [rule for rule, rx in PUBLIC_LEAK_RULES.items() if rx.search(line)]
    if private_ipv4(line):
        hits.append("private-ipv4")
    if private_regex is not None and private_match(private_regex, line):
        hits.append("private-pattern")
    return hits


def notebook_problem(data):
    """None when the notebook is clean, else (cell index or None, problem)."""
    try:
        nb = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None, "invalid JSON"
    if not isinstance(nb, dict) or not isinstance(nb.get("cells"), list):
        return None, "missing or invalid 'cells'"
    for i, cell in enumerate(nb["cells"]):
        if not isinstance(cell, dict) or not isinstance(cell.get("cell_type"), str):
            return i, "invalid cell"
        outputs = cell.get("outputs")
        count = cell.get("execution_count")
        if cell["cell_type"] == "code":
            if "outputs" not in cell or not isinstance(outputs, list):
                return i, "code cell without an 'outputs' list"
            if "execution_count" not in cell:
                return i, "code cell without 'execution_count'"
        if outputs not in (None, []):
            return i, "cell has outputs"
        if count is not None:
            return i, "cell has an execution count"
    return None


VERSION_FORMAT = re.compile(r"[0-9A-Za-z][0-9A-Za-z.+-]{0,63}")
TOML_HEADER = re.compile(r"\s*\[")
TOML_TABLE = re.compile(r"\s*\[\s*([A-Za-z0-9_.\- ]+?)\s*\]\s*(?:#.*)?")
TOML_VERSION_KEY = re.compile(r"\s*version\s*=")
TOML_VERSION = re.compile(
    r"\s*version\s*=\s*(?:\"([^\"\\\n]*)\"|'([^'\n]*)')\s*(?:#.*)?"
)
TOML_DOTTED_PACKAGE = re.compile(r"\s*[\"']?(?:workspace\s*\.\s*)?package[\"']?\s*[.=]")


def cargo_workspace_version(text):
    """[workspace.package].version from Cargo.toml.

    Supported subset: a single `[workspace.package]` table header and a
    single `version = "..."` (basic or literal one-line string, optional
    trailing comment) inside it. Inline tables, dotted keys defining the
    same table and multi-line strings are rejected."""
    section = None
    found = []
    headers = 0
    for line in text.split("\n"):
        line = line.rstrip("\r")
        if TOML_HEADER.match(line):
            m = TOML_TABLE.fullmatch(line)
            section = (
                re.sub(r"\s+", "", m.group(1))
                if m and not line.lstrip().startswith("[[")
                else None
            )
            headers += section == "workspace.package"
            continue
        if section in (None, "workspace") and TOML_DOTTED_PACKAGE.match(line):
            raise HygieneError("Cargo.toml: unsupported syntax for [workspace.package]")
        if section == "workspace.package" and TOML_VERSION_KEY.match(line):
            m = TOML_VERSION.fullmatch(line)
            if not m:
                raise HygieneError(
                    "Cargo.toml: unsupported syntax for workspace version"
                )
            found.append(m.group(1) if m.group(1) is not None else m.group(2))
    if headers != 1:
        raise HygieneError("Cargo.toml: expected exactly one [workspace.package] table")
    if len(found) != 1:
        raise HygieneError("Cargo.toml: expected exactly one workspace version")
    return found[0]


CFF_VERSION_KEY = re.compile(r"version\s*:")
CFF_VERSION = re.compile(
    r"version:[ \t]*(?:\"([^\"\\]*)\"|'([^']*)'|([0-9A-Za-z][^\s#]*))[ \t]*(?:#.*)?"
)


def cff_version(text):
    """Root-level scalar `version:` from CITATION.cff (no YAML parser: only a
    plain, single- or double-quoted one-line scalar with optional comment)."""
    found = []
    for line in text.split("\n"):
        line = line.rstrip("\r")
        if CFF_VERSION_KEY.match(line):
            m = CFF_VERSION.fullmatch(line)
            if not m:
                raise HygieneError("CITATION.cff: unsupported syntax for version")
            found.append(next(g for g in m.groups() if g is not None))
    if len(found) != 1:
        raise HygieneError("CITATION.cff: expected exactly one root-level version")
    return found[0]


# ------------------------------------------------------------------ output ---


def printable(text):
    """Control characters and lone surrogates shown as escapes."""
    out = []
    for ch in text:
        cp = ord(ch)
        if cp < 32 or cp == 127 or 0xDC80 <= cp <= 0xDCFF:
            out.append("\\x{:02x}".format(cp & 0xFF))
        else:
            out.append(ch)
    return "".join(out)


def annotation_data(text):
    return text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def annotation_property(text):
    return annotation_data(printable(text)).replace(":", "%3A").replace(",", "%2C")


def markdown_literal(text):
    text = printable(text)
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return re.sub(r"([\\`*_{}\[\]()#+\-.!|~\"'])", r"\\\1", text)


class Report:
    def __init__(self):
        self.violations = []
        self.errors = []
        self.notes = []
        self.binary_skipped = 0
        self.latin1_decoded = 0
        self.private_skipped = False

    def violation(self, path, line, rule, message):
        self.violations.append((path, line, rule, message))

    def emit(self, completed, annotate):
        for path, line, rule, message in self.violations:
            if annotate:
                print(
                    "::error file={},line={},title={}::{}".format(
                        annotation_property(path),
                        line,
                        rule,
                        annotation_data("[{}] {}".format(rule, message)),
                    )
                )
            print(
                "hygiene: violation {}:{} [{}] {}".format(
                    printable(path), line, rule, message
                )
            )
        for error in self.errors:
            if annotate:
                print("::error::{}".format(annotation_data(printable(error))))
            print("hygiene: error: {}".format(printable(error)))
        for note in self.notes:
            if annotate:
                print("::warning::{}".format(annotation_data(note)))
            print("hygiene: warning: {}".format(note))
        for line in self.summary_lines(completed):
            print("hygiene: " + line)

    def summary_lines(self, completed):
        lines = [
            "checks completed: {}".format(", ".join(completed) if completed else "none")
        ]
        if self.private_skipped:
            lines.append(
                "private patterns: SKIPPED ({} not set); public coverage only".format(
                    LEAK_ENV
                )
            )
        lines.append(
            "binary blobs skipped by line checks: {}".format(self.binary_skipped)
        )
        if self.latin1_decoded:
            lines.append(
                "non-UTF-8 text blobs scanned as Latin-1: {}".format(
                    self.latin1_decoded
                )
            )
        lines.append("violations: {}".format(len(self.violations)))
        lines.append("configuration/operational errors: {}".format(len(self.errors)))
        if self.errors:
            lines.append("result: INCOMPLETE (exit 2)")
        elif self.violations:
            lines.append("result: VIOLATIONS (exit 1)")
        elif self.private_skipped:
            lines.append("result: public checks passed (private patterns not run)")
        else:
            lines.append("result: all checks passed")
        return lines

    def write_summary(self, target, completed):
        head = "## Repository hygiene\n\n" + "".join(
            "- " + markdown_literal(line) + "\n"
            for line in self.summary_lines(completed)
        )
        items = [
            "- `{}` line {}: {}\n".format(rule, line, markdown_literal(path))
            for path, line, rule, _ in self.violations
        ] + ["- error: {}\n".format(markdown_literal(e)) for e in self.errors]
        body, used = [], len(head.encode("utf-8")) + 200
        for i, item in enumerate(items):
            size = len(item.encode("utf-8"))
            if used + size > SUMMARY_LIMIT:
                body.append(
                    "\n{} more entries omitted; see the job log.\n".format(
                        len(items) - i
                    )
                )
                break
            body.append(item)
            used += size
        with open(target, "a", encoding="utf-8", errors="surrogateescape") as fh:
            fh.write(head + ("\n" if body else "") + "".join(body))


# -------------------------------------------------------------------- main ---


def regular_file(entries, blobs, path):
    entry = entries.get(path)
    if entry is None:
        raise HygieneError("{} is missing from the analysed tree".format(path))
    if entry[0] == MODE_SYMLINK:
        raise HygieneError(
            "{} is a symlink; policy files must be regular files".format(path)
        )
    if entry[0] == MODE_GITLINK:
        raise HygieneError("{} is a submodule".format(path))
    data = blobs.get(entry[1])
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        raise HygieneError("{} is not valid UTF-8".format(path))


def load_private_regex(required, report):
    pattern = os.environ.get(LEAK_ENV, "")
    if not pattern:
        if required:
            raise HygieneError("{} is required but not set".format(LEAK_ENV))
        report.private_skipped = True
        report.notes.append(
            "{} not set: private patterns skipped, public coverage only".format(
                LEAK_ENV
            )
        )
        return None
    try:
        regex = re.compile(pattern)
    except (re.error, RecursionError, OverflowError, ValueError):
        raise HygieneError(
            "{} is not a valid regular expression".format(LEAK_ENV)
        ) from None
    if regex.search("") is not None:
        raise HygieneError("{} matches the empty string".format(LEAK_ENV))
    return regex


def analyse(args, report):
    git = Git(args.repo)
    blobs = BlobReader(git)
    completed = []

    head = git.resolve_commit(args.head, "head")
    if args.base == EMPTY:
        if args.comparison != "direct":
            raise HygieneError("--base EMPTY requires --comparison direct")
        base_tree = git.empty_tree()
    else:
        base = git.resolve_commit(args.base, "base")
        if args.comparison == "merge-base":
            base = git.merge_base(base, head)
        base_tree = git.tree_of(base)
    head_tree = git.tree_of(head)
    if git.is_dirty():
        report.notes.append(
            "uncommitted local changes are not analysed; commit them first"
        )

    private_regex = load_private_regex(args.require_leak_patterns, report)
    entries = git.ls_tree(head_tree)
    allow = AllowList(regular_file(entries, blobs, ALLOW_FILE))
    if allow.errors:
        report.errors.extend(allow.errors)
        return completed
    changes = [c for c in git.diff_trees(base_tree, head_tree) if c[0] != "D"]
    for status, _path, _mode, _old, _new in changes:
        if status not in ("A", "M", "T"):
            raise HygieneError("unexpected change status from git")

    def violation(path, line, rule, message=None):
        if rule in EXEMPTABLE_RULES and allow.exempt(rule, path):
            return
        report.violation(path, line, rule, message or FIX_HINTS[rule])

    new_like = {path for status, path, *_ in changes if status in ("A", "T")}
    for status, path, mode, _old, _new in changes:
        if mode == MODE_GITLINK:
            violation(path, 1, "submodule")
        if path in new_like and forbidden_name(path):
            violation(path, 1, "forbidden-name")
    completed += ["submodule", "forbidden-name"]

    for path in sorted(ignored_paths(git, blobs, entries, sorted(new_like))):
        violation(path, 1, "ignored-file")
    completed.append("ignored-file")

    files = [c for c in changes if c[2] != MODE_GITLINK]
    blobs.read_many([new for _s, _p, _m, _o, new in files])
    for status, path, mode, old, new in files:
        data = blobs.get(new)
        if len(data) > MAX_BLOB_BYTES:
            violation(path, 1, "file-size")
        if path.startswith("tutorials/") and path.endswith(".ipynb"):
            if mode == MODE_SYMLINK:
                violation(path, 1, "notebook-output", "notebook stored as a symlink")
            else:
                problem = notebook_problem(data)
                if problem:
                    cell, what = problem
                    where = "" if cell is None else " (cell {})".format(cell)
                    violation(
                        path,
                        1,
                        "notebook-output",
                        "{}{}; run `nbstripout <notebook>`, commit the result, and "
                        "`nbstripout --install` once to prevent it".format(what, where),
                    )
        if is_binary(data):
            report.binary_skipped += 1
            continue
        text, utf8 = decode_text(data)
        report.latin1_decoded += not utf8
        lines = text.split("\n")
        if (
            status == "M"
            and mode != MODE_SYMLINK
            and blobs.get(old)
            and not is_binary(blobs.get(old))
        ):
            numbers = git.added_line_numbers(old, new)
        else:
            numbers = range(1, len(lines) + 1)
        active = [r for r in LEAK_RULES if not allow.exempt(r, path)]
        if not active:
            continue
        for n in numbers:
            if n > len(lines):
                continue
            for rule in leak_rules_hit(lines[n - 1].rstrip("\r"), private_regex):
                if rule in active:
                    hint = FIX_HINTS.get(rule) or LEAK_HINT.format(rule)
                    report.violation(path, n, rule, hint)
    completed += ["file-size", "notebook-output", "leaks"]
    if private_regex is not None:
        completed.append("private-pattern")

    cargo = cargo_workspace_version(regular_file(entries, blobs, "Cargo.toml"))
    cff = cff_version(regular_file(entries, blobs, "CITATION.cff"))
    for name, value in (("Cargo.toml", cargo), ("CITATION.cff", cff)):
        if not VERSION_FORMAT.fullmatch(value):
            raise HygieneError("{}: unsupported version format".format(name))
    if cargo != cff:
        report.violation(
            "CITATION.cff",
            1,
            "version-mismatch",
            "CITATION.cff version {} != Cargo.toml [workspace.package] "
            "version {}".format(cff, cargo),
        )
    completed.append("version-mismatch")
    return completed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--base",
        required=True,
        help="base revision, or EMPTY (with --comparison direct)",
    )
    parser.add_argument(
        "--head", default="HEAD", help="analysed revision (default HEAD)"
    )
    parser.add_argument(
        "--comparison", choices=("merge-base", "direct"), default="merge-base"
    )
    parser.add_argument(
        "--require-leak-patterns",
        action="store_true",
        help="fail when {} is not set".format(LEAK_ENV),
    )
    parser.add_argument("--repo", default=".", help="repository to analyse (default .)")
    args = parser.parse_args(argv)

    report = Report()
    completed = []
    try:
        completed = analyse(args, report)
    except HygieneError as exc:
        report.errors.append(str(exc))
    except Exception as exc:  # never let an unexpected failure look like success
        report.errors.append("internal error ({})".format(type(exc).__name__))

    annotate = os.environ.get("GITHUB_ACTIONS") == "true"
    report.emit(completed, annotate)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        try:
            report.write_summary(summary, completed)
        except OSError as exc:
            report.errors.append(
                "could not write the step summary ({})".format(type(exc).__name__)
            )
            print("hygiene: error: could not write the step summary")
    if report.errors:
        return 2
    return 1 if report.violations else 0


if __name__ == "__main__":
    sys.exit(main())
