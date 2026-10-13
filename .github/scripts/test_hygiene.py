"""Tests for .github/scripts/hygiene.py on throwaway Git repositories.

Every sample that a rule must flag is assembled at run time, so this file
itself never matches a leak rule.
"""

import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).with_name("hygiene.py")
_spec = importlib.util.spec_from_file_location("hygiene", SCRIPT)
hygiene = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hygiene)

MIB = 1024 * 1024


def j(*parts):
    return "".join(parts)


def ip(*octets):
    return ".".join(str(o) for o in octets)


CLUSTER = j("/mnt/", "netapp9/", "scratch/run")
TOKEN = j("gh", "p_", "A1" * 18)
KEY = j("-----BEGIN ", "RSA PRIVATE", " KEY-----")
SECRET_RX = "zz-internal-[0-9]+"
SECRET_HIT = "zz-internal-42"

CARGO = '[workspace]\nmembers = []\n\n[workspace.package]\nversion = "1.2.3"\n'
CFF = 'cff-version: 1.2.0\ntitle: "t"\nversion: 1.2.3\n'
ALLOW_HEADER = "# exceptions\n"
CLEAN_NOTEBOOK = {
    "cells": [
        {"cell_type": "markdown", "metadata": {}, "source": ["# t"]},
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["1"],
        },
    ],
    "metadata": {},
    "nbformat": 4,
    "nbformat_minor": 5,
}


class Result:
    def __init__(self, proc, summary):
        self.code = proc.returncode
        self.out = proc.stdout
        self.err = proc.stderr
        self.all = proc.stdout + proc.stderr
        self.summary = summary.read_text() if summary and summary.exists() else ""

    def violations(self, rule=None):
        found = re.findall(
            r"^hygiene: violation (.*):(\d+) \[([a-z0-9-]+)\]", self.out, re.M
        )
        return [(p, int(n), r) for p, n, r in found if rule in (None, r)]


@pytest.fixture
def env(tmp_path):
    clean = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith("GIT_")
        and k not in ("HYGIENE_LEAK_PATTERNS", "GITHUB_STEP_SUMMARY", "GITHUB_ACTIONS")
    }
    home = tmp_path / "home"
    home.mkdir()
    clean.update(
        HOME=str(home),
        XDG_CONFIG_HOME=str(home),
        GIT_CONFIG_NOSYSTEM="1",
        GIT_CONFIG_GLOBAL=os.devnull,
        GIT_AUTHOR_NAME="t",
        GIT_AUTHOR_EMAIL="t@example.invalid",
        GIT_COMMITTER_NAME="t",
        GIT_COMMITTER_EMAIL="t@example.invalid",
    )
    return clean


class Repo:
    def __init__(self, path, env, tmp_path):
        self.path = path
        self.env = env
        self.tmp = tmp_path
        path.mkdir()
        self.git("init", "-q", "-b", "main")
        self.write("Cargo.toml", CARGO)
        self.write("CITATION.cff", CFF)
        self.write(".github/hygiene-allow.txt", ALLOW_HEADER)
        self.write(".gitignore", "*.so\nresults/\n")
        self.write("README.md", "readme\n")
        self.base = self.commit("base")

    def git(self, *args, input=None):
        return (
            subprocess.run(
                ["git", *args],
                cwd=self.path,
                env=self.env,
                input=input,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            .stdout.decode("utf-8", "surrogateescape")
            .strip()
        )

    def write(self, rel, content):
        target = self.path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, str):
            content = content.encode("utf-8")
        target.write_bytes(content)

    def symlink(self, rel, dest):
        target = self.path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(dest, target)

    def commit(self, message="change", force=()):
        self.git("add", "-A")
        for rel in force:
            self.git("add", "-f", "--", rel)
        self.git("commit", "-q", "--allow-empty", "-m", message)
        return self.git("rev-parse", "HEAD")

    def run(self, base=None, *extra, env=None, summary=False):
        run_env = dict(self.env)
        run_env.update(env or {})
        summary_file = None
        if summary:
            summary_file = self.tmp / "summary.md"
            run_env["GITHUB_STEP_SUMMARY"] = str(summary_file)
        proc = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--repo",
                str(self.path),
                "--base",
                base or self.base,
                *extra,
            ],
            env=run_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            encoding="utf-8",
            errors="replace",
        )
        return Result(proc, summary_file)


@pytest.fixture
def repo(tmp_path, env):
    return Repo(tmp_path / "repo", env, tmp_path)


def test_environment_is_isolated(env):
    assert "HYGIENE_LEAK_PATTERNS" not in env
    assert "GITHUB_STEP_SUMMARY" not in env


def test_clean_change_passes(repo):
    repo.write("src/lib.py", "x = 1\n")
    repo.commit()
    res = repo.run()
    assert res.code == 0, res.all
    assert "public checks passed (private patterns not run)" in res.out
    assert "all checks passed" not in res.out


# ---------------------------------------------------------- forbidden names ---


@pytest.mark.parametrize(
    "path",
    [
        "CLAUDE.md",
        "sub/dir/CLAUDE.md",
        "CLAUDE_old.md",
        "x/AI_GUIDELINES.md",
        "AI_GUIDELINES_es.txt",
        ".claude",
        ".claude/settings.json",
        "a/.claude/notes.txt",
        "docs/REFACTOR_PLAN.md",
        "pkg/docs/IMPLEMENTATION_PROMPT.md",
    ],
)
def test_forbidden_names(repo, path):
    repo.write(path, "x\n")
    repo.commit()
    res = repo.run()
    assert res.code == 1
    assert res.violations("forbidden-name") == [(path, 1, "forbidden-name")]


@pytest.mark.parametrize(
    "path", ["docs/CLAUDE.txt", "claudette.md", "my.claude.md", "REFACTOR_PLAN.md"]
)
def test_names_outside_the_list_pass(repo, path):
    repo.write(path, "x\n")
    repo.commit()
    assert repo.run().code == 0


def test_forbidden_name_as_move_destination(repo):
    repo.write("notes.md", "x\n")
    base = repo.commit()
    repo.git("mv", "notes.md", "CLAUDE.md")
    repo.commit()
    assert repo.run(base).violations("forbidden-name") == [
        ("CLAUDE.md", 1, "forbidden-name")
    ]


def test_forbidden_name_admits_no_exception(repo):
    repo.write(".github/hygiene-allow.txt", "forbidden-name\tCLAUDE.md\t# no\n")
    repo.write("CLAUDE.md", "x\n")
    repo.commit()
    res = repo.run()
    assert res.code == 2
    assert "rule without exceptions" in res.out


# ------------------------------------------------------------- ignored files ---


def test_force_added_ignored_file_fails(repo):
    repo.write("lib.so", "x\n")
    repo.commit(force=["lib.so"])
    assert repo.run().violations("ignored-file") == [("lib.so", 1, "ignored-file")]


def test_file_inside_ignored_directory_fails(repo):
    repo.write("a/results/out.csv", "1\n")
    repo.commit(force=["a/results/out.csv"])
    assert repo.run().violations("ignored-file") == [
        ("a/results/out.csv", 1, "ignored-file")
    ]


def test_modifying_tracked_ignored_file_passes(repo):
    repo.write("lib.so", "x\n")
    base = repo.commit(force=["lib.so"])
    repo.write("lib.so", "y\n")
    repo.commit(force=["lib.so"])
    assert repo.run(base).code == 0


def test_ignore_exception_covers_only_that_rule(repo):
    repo.write(".github/hygiene-allow.txt", "ignored-file\tlib.so\t# fixture\n")
    repo.write("lib.so", j(CLUSTER, "\n", "a" * (2 * MIB)))
    repo.commit(force=["lib.so"])
    res = repo.run()
    assert res.violations("ignored-file") == []
    assert {r for _p, _n, r in res.violations()} == {"cluster-path", "file-size"}


def test_nested_gitignore_and_negation(repo):
    repo.write("sub/.gitignore", "!keep.so\n")
    repo.write("sub/keep.so", "x\n")
    repo.write("sub/other.so", "x\n")
    repo.commit(force=["sub/other.so"])
    assert repo.run().violations("ignored-file") == [
        ("sub/other.so", 1, "ignored-file")
    ]


def test_user_and_local_ignore_configuration_has_no_effect(repo, tmp_path):
    excludes = tmp_path / "excludes"
    excludes.write_text("*.txt\n*.md\n")
    template = tmp_path / "template"
    (template / "info").mkdir(parents=True)
    (template / "info" / "exclude").write_text("*.txt\n")
    (Path(repo.env["HOME"]) / "git").mkdir()
    (Path(repo.env["HOME"]) / "git" / "ignore").write_text("*.txt\n")
    config = tmp_path / "gitconfig"
    config.write_text(
        "[core]\n\texcludesFile = {}\n[init]\n\ttemplateDir = {}\n".format(
            excludes, template
        )
    )
    (repo.path / ".git" / "info").mkdir(exist_ok=True)
    (repo.path / ".git" / "info" / "exclude").write_text("*.txt\n")
    repo.write("new.txt", "x\n")
    repo.write("doc/new.md", "x\n")
    repo.commit(force=["new.txt", "doc/new.md"])
    res = repo.run(
        env={
            "GIT_CONFIG_GLOBAL": str(config),
            "GIT_CONFIG_PARAMETERS": "'core.excludesfile'='{}'".format(excludes),
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": "core.excludesFile",
            "GIT_CONFIG_VALUE_0": str(excludes),
        }
    )
    assert res.code == 0, res.all


def test_uncommitted_gitignore_change_has_no_effect(repo):
    repo.write("lib.so", "x\n")
    repo.commit(force=["lib.so"])
    repo.write(".gitignore", "")
    res = repo.run()
    assert res.violations("ignored-file") == [("lib.so", 1, "ignored-file")]
    assert "uncommitted local changes are not analysed" in res.out


def test_head_revision_uses_its_own_gitignore(repo):
    repo.write("x.so", "x\n")
    first = repo.commit(force=["x.so"])
    repo.write(".gitignore", "results/\n")
    second = repo.commit()
    assert repo.run(None, "--head", first).violations("ignored-file") == [
        ("x.so", 1, "ignored-file")
    ]
    assert repo.run(None, "--head", second).code == 0


def test_isolated_env_drops_inherited_git_variables(monkeypatch, tmp_path):
    for key in (
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_COMMON_DIR",
        "GIT_INDEX_FILE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_CONFIG_PARAMETERS",
        "GIT_CONFIG_COUNT",
        "GIT_TEMPLATE_DIR",
    ):
        monkeypatch.setenv(key, str(tmp_path))
    monkeypatch.setenv("HYGIENE_LEAK_PATTERNS", "x")
    env = hygiene.isolated_git_env(str(tmp_path))
    assert not [
        k
        for k in env
        if k.startswith("GIT_")
        and k not in ("GIT_CONFIG_NOSYSTEM", "GIT_CONFIG_GLOBAL", "GIT_TERMINAL_PROMPT")
    ]
    assert env["GIT_CONFIG_GLOBAL"] == os.devnull
    assert "HYGIENE_LEAK_PATTERNS" not in env


def test_scratch_repository_removed_on_error(repo, tmp_path, monkeypatch):
    scratch_root = tmp_path / "scratch"
    scratch_root.mkdir()
    monkeypatch.setattr(hygiene.tempfile, "tempdir", str(scratch_root))
    git = hygiene.Git(str(repo.path))
    real_run = git.run

    def failing_run(args, **kw):
        if args[0] == "check-ignore":
            raise hygiene.HygieneError("git check-ignore failed with exit code 128")
        return real_run(args, **kw)

    git.run = failing_run
    entries = git.ls_tree(git.tree_of(repo.base))
    with pytest.raises(hygiene.HygieneError):
        hygiene.ignored_paths(git, hygiene.BlobReader(git), entries, ["a.so"])
    assert list(scratch_root.iterdir()) == []


# --------------------------------------------------------------- allow list ---


def test_allow_file_missing_is_a_configuration_error(repo):
    repo.git("rm", "-q", ".github/hygiene-allow.txt")
    repo.commit()
    res = repo.run()
    assert res.code == 2
    assert "hygiene-allow.txt is missing" in res.out


@pytest.mark.parametrize(
    "entry",
    [
        "ignored-file\tx.so",
        "ignored-file\tx.so\t#   ",
        "ignored-file\tx.so\treason without hash",
        "nope\tx.so\t# r",
        "notebook-output\ttutorials/a.ipynb\t# r",
        "version-mismatch\tCITATION.cff\t# r",
        "ignored-file\t**\t# r",
        "file-size\t*/*\t# r",
        "file-size\t**/*\t# r",
        "file-size\t/abs.txt\t# r",
        "file-size\t../x.txt\t# r",
        "file-size\ta//b.txt\t# r",
        "file-size\ta/./b.txt\t# r",
        "file-size\ta[b].txt\t# r",
        "file-size x.txt # r",
    ],
)
def test_invalid_allow_entries(repo, entry):
    repo.write(".github/hygiene-allow.txt", "# header\n" + entry + "\n")
    repo.commit()
    res = repo.run()
    assert res.code == 2
    assert "hygiene-allow.txt:2:" in res.out


def test_moving_out_of_an_exempt_location_is_analysed_in_full(repo):
    repo.write(
        ".github/hygiene-allow.txt",
        "file-size\tdata/*.bin\t# fixture\ncluster-path\tfixtures/*.txt\t# fixture\n",
    )
    repo.write("data/big.bin", "a" * (2 * MIB))
    repo.write("fixtures/a.txt", j("ok\n", CLUSTER, "\n"))
    base = repo.commit()
    assert repo.run(repo.base, "--head", base).code == 0
    (repo.path / "other").mkdir()
    (repo.path / "src").mkdir()
    repo.git("mv", "data/big.bin", "other/big.bin")
    repo.git("mv", "fixtures/a.txt", "src/a.txt")
    repo.commit()
    res = repo.run(base)
    assert sorted(res.violations()) == [
        ("other/big.bin", 1, "file-size"),
        ("src/a.txt", 2, "cluster-path"),
    ]


@pytest.mark.parametrize(
    "pattern, path, expected",
    [
        ("a/*.txt", "a/x.txt", True),
        ("a/*.txt", "a/b/x.txt", False),
        ("a/**/x.txt", "a/x.txt", True),
        ("a/**/x.txt", "a/b/c/x.txt", True),
        ("a/**", "a/b/c", True),
        ("a/?.txt", "a/x.txt", True),
        ("a/?.txt", "a/xy.txt", False),
        ("x.txt", "a/x.txt", False),
        ("ZA??????.txt", "ZA001000.txt", True),
    ],
)
def test_glob_semantics(pattern, path, expected):
    allow = hygiene.AllowList("file-size\t{}\t# r\n".format(pattern))
    assert allow.errors == []
    assert allow.exempt("file-size", path) is expected
    assert allow.exempt("ignored-file", path) is False


# -------------------------------------------------------------------- leaks ---


LEAK_POSITIVES = (
    [
        (CLUSTER, "cluster-path"),
        (j("cd /ho", "me/alice/x"), "cluster-path"),
        (j("/mnt/", "lustre"), "cluster-path"),
        (j("/mnt/", "lustre/x"), "cluster-path"),
        (j("/opt/", "cesga/x"), "cluster-path"),
        (j("ssh log", "in02"), "login-host"),
        (j("log", "in1.example"), "login-host"),
        (ip(10, 0, 0, 1), "private-ipv4"),
        (ip(172, 16, 0, 1), "private-ipv4"),
        (ip(172, 31, 255, 255), "private-ipv4"),
        (j("host=", ip(192, 168, 1, 1), "."), "private-ipv4"),
        (j("-----BEGIN ", "PRIVATE KEY-----"), "private-key"),
        (j("-----BEGIN ", "OPENSSH PRIVATE KEY-----"), "private-key"),
        (j("-----BEGIN ", "EC PRIVATE KEY-----"), "private-key"),
        (KEY, "private-key"),
    ]
    + [(j("t=gh", c, "_", "a" * 36), "github-token") for c in "pousr"]
    + [
        (j("github", "_pat_", "B" * 22), "github-token"),
    ]
)

LEAK_NEGATIVES = [
    "CESGA",
    "https://www.cesga.es/",
    "login_url",
    j("mylog", "in1"),
    "login",
    j("/ho", "mepage/x/"),
    j("https://example.com/ho", "me/u/"),
    j("/mnt/", "lustrefs/x"),
    ip(172, 15, 0, 1),
    ip(172, 32, 0, 1),
    ip(192, 169, 0, 1),
    ip(11, 0, 0, 1),
    ip(8, 8, 8, 8),
    ip(10, 0, 0, 256),
    "10.0.0",
    j(ip(1, 10, 0, 0), ".1"),
    j(ip(10, 1, 2, 3), "x"),
    j("-----BEGIN ", "PUBLIC KEY-----"),
    j("gh", "p_short"),
    j("xgh", "p_", "a" * 36),
    j("github", "_pat_", "short"),
]


@pytest.mark.parametrize("line, rule", LEAK_POSITIVES)
def test_public_leak_rules_flag(line, rule):
    assert hygiene.leak_rules_hit(line, None) == [rule]


@pytest.mark.parametrize("line", LEAK_NEGATIVES)
def test_public_leak_rules_ignore(line):
    assert hygiene.leak_rules_hit(line, None) == []


def test_leak_is_reported_without_its_content(repo):
    repo.write("src/run.sh", j("echo ok\n", "cd ", CLUSTER, "\n"))
    repo.commit()
    res = repo.run(env={"GITHUB_ACTIONS": "true"}, summary=True)
    assert res.code == 1
    assert res.violations() == [("src/run.sh", 2, "cluster-path")]
    assert "::error file=src/run.sh,line=2,title=cluster-path::" in res.out
    for text in (res.out, res.err, res.summary):
        assert "netapp" not in text
    assert "cluster" in res.summary


def test_only_added_lines_are_scanned(repo):
    repo.write("notes.txt", j("keep\n", CLUSTER, "\n", "tail\n"))
    base = repo.commit(force=["notes.txt"])
    repo.write("notes.txt", "keep\ntail\nnew\n")
    repo.commit()
    assert repo.run(base).code == 0
    repo.write("notes.txt", j("keep\n", "tail\nnew\n", TOKEN, "\n"))
    repo.commit()
    assert repo.run(base).violations() == [("notes.txt", 4, "github-token")]


def test_private_patterns_optional_when_unset(repo):
    repo.write("a.txt", "x\n")
    repo.commit()
    res = repo.run()
    assert res.code == 0
    assert "::warning::" not in res.out
    assert "private patterns skipped, public coverage only" in res.out


def test_private_patterns_required_when_unset(repo):
    res = repo.run(None, "--require-leak-patterns")
    assert res.code == 2
    assert "HYGIENE_LEAK_PATTERNS is required but not set" in res.out


def test_private_pattern_match_is_redacted(repo):
    repo.write("a.txt", j("host ", SECRET_HIT, "\n"))
    repo.commit()
    res = repo.run(
        None,
        "--require-leak-patterns",
        env={"HYGIENE_LEAK_PATTERNS": SECRET_RX},
        summary=True,
    )
    assert res.code == 1
    assert res.violations() == [("a.txt", 1, "private-pattern")]
    assert "result: VIOLATIONS" in res.out
    for text in (res.out, res.err, res.summary):
        assert "zz-internal" not in text


def test_clean_run_with_private_patterns_reports_full_coverage(repo):
    res = repo.run(
        None, "--require-leak-patterns", env={"HYGIENE_LEAK_PATTERNS": SECRET_RX}
    )
    assert res.code == 0
    assert "result: all checks passed" in res.out


@pytest.mark.parametrize("pattern", ["zz(", "a*"])
def test_unusable_private_pattern_is_a_configuration_error(repo, pattern):
    res = repo.run(None, env={"HYGIENE_LEAK_PATTERNS": pattern})
    assert res.code == 2
    assert pattern not in res.all


class SpyRegex:
    def __init__(self, pattern):
        self.rx = re.compile(pattern)
        self.lengths = []

    def search(self, text):
        self.lengths.append(len(text))
        return self.rx.search(text)


def test_window_line_of_exactly_4096_is_one_call():
    spy = SpyRegex(SECRET_RX)
    line = "a" * (4096 - len(SECRET_HIT)) + SECRET_HIT
    assert hygiene.private_match(spy, line)
    assert spy.lengths == [4096]


def test_window_finds_match_at_the_end_of_a_long_line():
    spy = SpyRegex(SECRET_RX)
    assert hygiene.private_match(spy, "a" * 20000 + SECRET_HIT)
    assert max(spy.lengths) <= 4096
    assert len(spy.lengths) > 1


def test_window_finds_match_across_a_window_boundary():
    hit = "zz-internal-" + "7" * 188  # 200 characters
    line = "a" * 4000 + hit + "a" * 6000  # crosses the end of the first window
    spy = SpyRegex("zz-internal-7{188}")
    assert hygiene.private_match(spy, line)
    assert max(spy.lengths) <= 4096
    assert spy.rx.search(line[:4096]) is None


def test_window_long_line_without_match():
    spy = SpyRegex(SECRET_RX)
    assert not hygiene.private_match(spy, "a" * 50000)
    assert max(spy.lengths) <= 4096


def test_long_line_with_several_matches_is_one_violation(repo):
    line = j("a" * 5000, SECRET_HIT, "b" * 5000, SECRET_HIT, "c" * 3000)
    repo.write("data.txt", j("short\n", line, "\n", "a" * 9000, "\n"))
    repo.commit()
    res = repo.run(None, env={"HYGIENE_LEAK_PATTERNS": SECRET_RX})
    assert res.code == 1
    assert res.violations() == [("data.txt", 2, "private-pattern")]
    assert "incomplete" not in res.out.lower()
    assert "5000" not in res.out


# ---------------------------------------------------------------- notebooks ---


def notebook(**changes):
    nb = json.loads(json.dumps(CLEAN_NOTEBOOK))
    for key, value in changes.items():
        index, field = key.split("__")
        if value is KeyError:
            del nb["cells"][int(index[1:])][field]
        else:
            nb["cells"][int(index[1:])][field] = value
    return json.dumps(nb)


@pytest.mark.parametrize(
    "content, ok",
    [
        (notebook(), True),
        (notebook(c1__outputs=[{"output_type": "stream", "text": ["1"]}]), False),
        (notebook(c1__execution_count=3), False),
        (notebook(c1__outputs=KeyError), False),
        (notebook(c1__execution_count=KeyError), False),
        (notebook(c1__outputs="x"), False),
        (notebook(c0__execution_count=1), False),
        ("{not json", False),
        (json.dumps({"cells": {}}), False),
        (json.dumps({"cells": ["x"]}), False),
    ],
)
def test_notebooks(repo, content, ok):
    repo.write("tutorials/en/01.ipynb", content)
    repo.commit()
    res = repo.run()
    if ok:
        assert res.code == 0, res.all
    else:
        assert res.violations("notebook-output") == [
            ("tutorials/en/01.ipynb", 1, "notebook-output")
        ]
        assert "nbstripout <notebook>" in res.out
        assert "nbstripout --install" in res.out
        assert "not json" not in res.out


def test_notebook_outside_tutorials_is_not_checked(repo):
    repo.write("drafts/x.ipynb", notebook(c1__execution_count=3))
    repo.commit()
    assert repo.run().code == 0


def test_notebook_moved_into_tutorials(repo):
    repo.write("drafts/x.ipynb", notebook(c1__execution_count=3))
    base = repo.commit()
    (repo.path / "tutorials").mkdir()
    repo.git("mv", "drafts/x.ipynb", "tutorials/x.ipynb")
    repo.commit()
    assert repo.run(base).violations("notebook-output") == [
        ("tutorials/x.ipynb", 1, "notebook-output")
    ]


def test_notebook_symlink_fails(repo):
    repo.write("nb.json", notebook())
    repo.symlink("tutorials/n.ipynb", "../nb.json")
    repo.commit()
    res = repo.run()
    assert res.violations("notebook-output") == [
        ("tutorials/n.ipynb", 1, "notebook-output")
    ]
    assert "symlink" in res.out


# --------------------------------------------------------------------- size ---


@pytest.mark.parametrize("size, ok", [(MIB, True), (MIB + 1, False), (2 * MIB, False)])
def test_size_limit(repo, size, ok):
    repo.write("blob.dat", b"a" * size)
    repo.commit()
    res = repo.run()
    assert (res.violations("file-size") == []) is ok


def test_narrow_size_exception(repo):
    repo.write(".github/hygiene-allow.txt", "file-size\tdata/ZA??????.txt\t# tables\n")
    repo.write("data/ZA001000.txt", "a" * (2 * MIB))
    repo.write("data/other.txt", "a" * (2 * MIB))
    repo.commit()
    assert repo.run().violations() == [("data/other.txt", 1, "file-size")]


# ----------------------------------------------------------------- versions ---


def test_version_mismatch(repo):
    repo.write("CITATION.cff", CFF.replace("1.2.3", "1.2.4"))
    repo.commit()
    res = repo.run()
    assert res.code == 1
    assert (
        "CITATION.cff version 1.2.4 != Cargo.toml [workspace.package] version 1.2.3"
        in res.out
    )


@pytest.mark.parametrize(
    "cargo, cff",
    [
        (CARGO, "title: t\n"),
        (CARGO, CFF + "version: 1.2.3\n"),
        (CARGO, CFF.replace("1.2.3", '"1.2.3<b>"')),
        (CARGO, CFF.replace("1.2.3", ">-")),
        (CARGO.replace('"1.2.3"', "{ workspace = true }"), CFF),
        ("[workspace]\n", CFF),
    ],
)
def test_malformed_versions_are_configuration_errors(repo, cargo, cff):
    repo.write("Cargo.toml", cargo)
    repo.write("CITATION.cff", cff)
    repo.commit()
    res = repo.run()
    assert res.code == 2
    assert "<b>" not in res.out


CARGO_SUPPORTED = [
    CARGO,
    "[workspace.package]\nversion='1.2.3'  # comment\n",
    '[ workspace.package ]  # c\nedition = "2021"\nversion = "1.2.3"\n',
    '[package]\nversion = "9.9.9"\n[workspace.package]\nversion = "1.2.3"\n'
    '[workspace.dependencies]\nfoo = { version = "9" }\n',
    '[workspace]\nmembers = [\n  "a",\n]\n[workspace.package]\nversion = "1.2.3"\n'
    '[[bin]]\nname = "x"\nversion = "0.0.1"\n',
]


@pytest.mark.parametrize("text", CARGO_SUPPORTED)
def test_cargo_supported_forms(text):
    assert hygiene.cargo_workspace_version(text) == "1.2.3"


@pytest.mark.parametrize(
    "text",
    [
        "[workspace.package]\nversion = { workspace = true }\n",
        '[workspace.package]\nversion = """1.2.3"""\n',
        '[workspace]\npackage = { version = "1.2.3" }\n',
        '[workspace]\npackage.version = "1.2.3"\n',
        'workspace.package.version = "1.2.3"\n',
        '[workspace.package]\nversion = "1"\nversion = "2"\n',
        '[workspace.package]\nversion = "1"\n[workspace.package]\n',
        '[package]\nversion = "1"\n',
    ],
)
def test_cargo_unsupported_forms(text):
    with pytest.raises(hygiene.HygieneError):
        hygiene.cargo_workspace_version(text)


def test_cargo_reader_matches_tomllib():
    tomllib = pytest.importorskip("tomllib", reason="tomllib needs Python 3.11+")
    real = (Path(__file__).resolve().parents[2] / "Cargo.toml").read_text(
        encoding="utf-8"
    )
    for text in CARGO_SUPPORTED + [real]:
        expected = tomllib.loads(text)["workspace"]["package"]["version"]
        assert hygiene.cargo_workspace_version(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "version: 1.2.3\n",
        'version: "1.2.3"  # release\n',
        "version: '1.2.3'\n",
        "title: x\nreferences:\n  - version: 9.9.9\nversion: 1.2.3\r\n",
    ],
)
def test_cff_supported_forms(text):
    assert hygiene.cff_version(text) == "1.2.3"


@pytest.mark.parametrize(
    "text",
    [
        "title: x\n",
        "version:\n",
        "version: >-\n  1.2.3\n",
        "version: 1\nversion: 2\n",
        "version : 1.2.3\n",
    ],
)
def test_cff_unsupported_forms(text):
    with pytest.raises(hygiene.HygieneError):
        hygiene.cff_version(text)


# -------------------------------------------------- revisions and robustness ---


@pytest.mark.parametrize(
    "name",
    ["dir with space/CLAUDE.md", "ñandú/CLAUDE.md", "a\tb/CLAUDE.md", "a\nb/CLAUDE.md"],
)
def test_unusual_path_names(repo, name):
    repo.write(name, "x\n")
    repo.commit()
    res = repo.run(env={"GITHUB_ACTIONS": "true"})
    assert res.code == 1
    assert len(res.violations("forbidden-name")) == 1
    assert "a\\x09b" in res.out or "a\\x0ab" in res.out or name in res.out


def test_path_names_cannot_inject_workflow_commands(repo):
    evil = "x\n::error::pwned/a,line=9:z\n::stop-commands::k/CLAUDE.md"
    repo.write(evil, "x\n")
    repo.write("evil,line=7:t.txt", j(CLUSTER, "\n"))
    repo.commit()
    res = repo.run(env={"GITHUB_ACTIONS": "true"})
    assert res.code == 1
    for line in res.out.splitlines():
        if line.startswith("::"):
            assert re.match(
                r"::(error|warning)( file=[^:]*,line=\d+,title=[a-z0-9-]+)?::", line
            ), line
    assert "::error file=evil%2Cline=7%3At.txt,line=1,title=cluster-path::" in res.out
    assert "::stop-commands" not in [ln[:16] for ln in res.out.splitlines()]


def test_summary_renders_names_literally(repo):
    repo.write("[a](http://x)<b>/CLAUDE.md", "x\n")
    repo.commit()
    res = repo.run(summary=True)
    assert "<b>" not in res.summary
    assert "&lt;b&gt;" in res.summary
    assert "](http" not in res.summary


def test_summary_is_truncated_but_stdout_is_complete(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(hygiene, "SUMMARY_LIMIT", 4000)
    report = hygiene.Report()
    for i in range(500):
        report.violation("f{}.txt".format(i), 1, "file-size", "big")
    target = tmp_path / "s.md"
    report.write_summary(str(target), ["file-size"])
    report.emit(["file-size"], annotate=False)
    text = target.read_text()
    assert len(text.encode("utf-8")) <= 4000
    omitted = int(re.search(r"(\d+) more entries omitted", text).group(1))
    assert omitted + text.count("line 1:") == 500
    assert capsys.readouterr().out.count("hygiene: violation ") == 500


def test_unknown_base_is_an_error(repo):
    res = repo.run("0" * 40)
    assert res.code == 2
    assert "base revision does not resolve" in res.out


def test_no_merge_base_is_an_error(repo):
    repo.git("checkout", "-q", "--orphan", "other")
    repo.commit("orphan")
    orphan = repo.git("rev-parse", "HEAD")
    repo.git("checkout", "-q", "main")
    assert repo.run(orphan).code == 2
    assert repo.run(orphan, "--comparison", "direct").code == 0


def test_direct_and_merge_base_comparisons_differ(repo):
    repo.write("old.txt", j(CLUSTER, "\n"))
    start = repo.commit(force=["old.txt"])
    repo.git("rm", "-q", "old.txt")
    main = repo.commit("main moves on")
    repo.git("checkout", "-q", "-b", "feature", start)
    repo.write("new.txt", "x\n")
    repo.commit()
    assert repo.run(main).code == 0
    assert repo.run(main, "--comparison", "direct").violations() == [
        ("old.txt", 1, "cluster-path")
    ]


def test_empty_base(repo):
    assert repo.run("EMPTY", "--comparison", "direct").code == 0
    res = repo.run("EMPTY")
    assert res.code == 2
    assert "requires --comparison direct" in res.out


def test_empty_base_reviews_the_whole_tree(repo):
    repo.write("CLAUDE.md", "x\n")
    repo.commit()
    assert repo.run("EMPTY", "--comparison", "direct").violations("forbidden-name")


def test_deleting_a_leak_passes(repo):
    repo.write("old.txt", j(CLUSTER, "\n"))
    base = repo.commit()
    repo.git("rm", "-q", "old.txt")
    repo.commit()
    assert repo.run(base).code == 0


def test_symlink_target_is_not_followed(repo, tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_text(j(CLUSTER, "\n") + "a" * (2 * MIB))
    repo.symlink("link.txt", str(outside))
    repo.symlink("crate/LICENSE", "../README.md")
    repo.commit()
    assert repo.run().code == 0, "the target's content and size must not be read"
    repo.symlink("data", CLUSTER)
    repo.commit()
    assert repo.run().violations() == [
        ("data", 1, "cluster-path")
    ]  # the link string itself


def test_type_change_is_treated_as_addition(repo):
    repo.write("lib.so", "x\n")
    base = repo.commit(force=["lib.so"])
    (repo.path / "lib.so").unlink()
    repo.symlink("lib.so", "README.md")
    repo.commit(force=["lib.so"])
    assert repo.run(base).violations("ignored-file") == [("lib.so", 1, "ignored-file")]


def test_submodules_are_rejected(repo):
    oid = repo.base
    repo.git("update-index", "--add", "--cacheinfo", "160000,{},vendor/sub".format(oid))
    repo.git("commit", "-q", "-m", "submodule")
    res = repo.run()
    assert res.violations("submodule") == [("vendor/sub", 1, "submodule")]
    base = repo.git("rev-parse", "HEAD")
    repo.git("update-index", "--cacheinfo", "160000,{},vendor/sub".format(base))
    repo.git("commit", "-q", "-m", "bump")
    assert repo.run(base).violations("submodule") == [("vendor/sub", 1, "submodule")]


def test_gitattributes_cannot_hide_text(repo):
    repo.write(".gitattributes", "*.txt binary -diff filter=hide\n")
    repo.write("a.txt", "clean\n")
    base = repo.commit()
    repo.write("a.txt", j("clean\n", TOKEN, "\n"))
    repo.write("b.txt", j(KEY, "\n"))
    repo.commit()
    assert sorted(repo.run(base).violations()) == [
        ("a.txt", 2, "github-token"),
        ("b.txt", 1, "private-key"),
    ]


def test_binary_blobs_skip_line_checks_only(repo):
    repo.write("img.bin", b"\0" + CLUSTER.encode() + b"a" * (2 * MIB))
    repo.commit()
    res = repo.run()
    assert res.violations() == [("img.bin", 1, "file-size")]
    assert "binary blobs skipped by line checks: 1" in res.out


def test_multiple_violations_are_all_reported(repo):
    repo.write("CLAUDE.md", "x\n")
    repo.write("lib.so", "x\n")
    repo.write("run.sh", j("cd ", CLUSTER, "\n"))
    repo.commit(force=["lib.so"])
    res = repo.run()
    assert res.code == 1
    assert sorted(r for _p, _n, r in res.violations()) == [
        "cluster-path",
        "forbidden-name",
        "ignored-file",
    ]
    assert "netapp" not in res.all


def test_errors_take_precedence_over_violations(repo):
    repo.write("CLAUDE.md", "x\n")
    repo.write("CITATION.cff", CFF.replace("1.2.3", '"1.2.3 x"'))
    repo.commit()
    res = repo.run()
    assert res.code == 2
    assert res.violations("forbidden-name")


def test_not_a_repository_is_an_error(tmp_path, env):
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--repo", str(tmp_path), "--base", "HEAD"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    assert proc.returncode == 2


def test_net_diff_limitation(repo):
    # Documented limitation: a violation added and removed again inside the
    # analysed range is not reported, because only the net diff is analysed.
    repo.write("CLAUDE.md", "x\n")
    repo.commit()
    repo.git("rm", "-q", "CLAUDE.md")
    repo.commit()
    assert repo.run().code == 0
