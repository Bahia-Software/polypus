"""The checks CI runs are the justfile's recipes, and `just ci-all` runs them all (#421).

docs/CONTRIBUTING.md promises that every check CI runs can be run locally with
one command. That holds only while ci.yml runs no command of its own: each step
calls a recipe of the root justfile, and the recipes `ci-all` reaches cover
every one CI calls. These tests fail when a step grows its own command (also
inside an "Install ..." step, or through an action's arguments or an
environment variable), names a recipe the justfile lacks, or calls one
`just ci-all` would not run.
"""

import re
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
CI = REPO / ".github" / "workflows" / "ci.yml"
JUSTFILE = REPO / "justfile"
PRE_COMMIT = REPO / ".pre-commit-config.yaml"

# Steps that run a command of their own although it checks nothing, with why.
# Steps that only install something are named "Install ..." and may run only
# the installer commands below.
EXEMPT_STEPS = {
    "reference-data/ is not checked out": "asserts the job's sparse checkout, which only CI has",
}

# Recipes CI calls that `ci-all` does not reach, with why.
NOT_IN_CI_ALL = {
    "build-wheel": "the platform smoke test; `wheel` builds the same wheel and installs it",
}

# What an "Install ..." step's lines may start with.
INSTALLERS = re.compile(
    r"^(sudo apt-get (update$|install )|rustup toolchain install |(python -m )?pip install "
    r"|uv pip (install|list) |cargo install |python -m ipykernel install "
    r"|llvm-as --version$|maturin --version$"
    r"|toolchain=\$\(just --evaluate [a-z_]+\)$"
    r'|echo "RUSTUP_TOOLCHAIN=\$toolchain" >> "\$GITHUB_ENV"$'
    # The pinned Qiskit of the C-10 fixtures, and the build requirements the
    # floors job installs at their floors: both only read a version list.
    r"|version=\$\(python -c 'import qasm3_fixtures; "
    r"|python -c 'import tomli; )"
)
# Commands that check something, which an "Install ..." line may not run
# (even after an installer's arguments).
CHECK_COMMANDS = re.compile(
    r"\bcargo\s+(\+\S+\s+)?(build|check|clippy|deny|doc|fmt|fuzz|machete|metadata|miri|"
    r"package|semver-checks|test)\b|\bmaturin\s+build\b|\bjupyter\b|\bcompileall\b|hygiene\.py|"
    # pytest, ruff and mypy are also package names an install line lists; an
    # invocation is one followed by a path, an option or a subcommand.
    r"\bpytest\s+[-./\w]|\bruff\s+(check|format)\b|\bmypy[.\s]+(stubtest|-)"
)
# A check step runs one recipe per line, with plain arguments and no shell.
CHECK_LINE = re.compile(r"^just [A-Za-z0-9_][A-Za-z0-9_-]*( [A-Za-z0-9_.=:/-]+)*$")
# Environment variables a workflow may set: the test suite's switches, and two
# that change no command.
ALLOWED_ENV = {"CARGO_TERM_COLOR", "JUST_VERSION"}
ALLOWED_ENV_PREFIX = "POLYPUS_"
# The one action input that names what the action runs, and its only value.
ACTION_ARGUMENTS = {"astral-sh/ruff-action": {"args": "--version"}}

RECIPE = re.compile(r"^@?([A-Za-z0-9_-]+)(?:\s+[^:]*)?:(?!=)(.*)$")
DEPENDENCY = re.compile(r"\(\s*([A-Za-z0-9_-]+)[^)]*\)|([A-Za-z0-9_-]+)")
JUST_CALL = re.compile(
    r"(?:\bjust|\{\{\s*just_executable\(\)\s*\}\})\s+([A-Za-z0-9_][A-Za-z0-9_-]*)"
)


def parse_justfile(text):
    """Map each recipe to the recipes it runs: its dependencies and `just x` lines."""
    recipes = {}
    current = None
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if line[0] in " \t":
            if current is not None:
                recipes[current].update(JUST_CALL.findall(line))
            continue
        current = None
        if line.startswith(("set ", "export ", "import ", "mod ", "alias ")):
            continue
        match = RECIPE.match(line)
        if match:
            current = match.group(1)
            recipes[current] = {a or b for a, b in DEPENDENCY.findall(match.group(2))}
    return recipes


def reachable(recipes, start):
    seen, todo = set(), [start]
    while todo:
        name = todo.pop()
        if name not in seen:
            seen.add(name)
            todo.extend(recipes.get(name, ()))
    return seen


def justfile_variable(name):
    match = re.search(rf'^{name}\s*:=\s*"([^"]*)"', JUSTFILE.read_text(), re.MULTILINE)
    assert match, f"the justfile defines no `{name}`"
    return match.group(1)


def steps():
    workflow = yaml.safe_load(CI.read_text())
    for job_id, job in workflow["jobs"].items():
        for step in job.get("steps", []):
            yield job_id, step


def run_steps():
    return ((job_id, step) for job_id, step in steps() if "run" in step)


def command_lines(run):
    return [
        line.strip()
        for line in run.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


@pytest.fixture(scope="module")
def recipes():
    return parse_justfile(JUSTFILE.read_text())


@pytest.fixture(scope="module")
def ci_calls():
    calls = {}
    for job_id, step in run_steps():
        for recipe in JUST_CALL.findall(step["run"]):
            calls.setdefault(recipe, job_id)
    return calls


def test_the_parser_reads_dependencies_and_calls():
    recipes = parse_justfile(
        'set shell := ["bash", "-c"]\n'
        'x := "1"\n'
        "# doc\n"
        "a: b c\n"
        "    echo a\n"
        "b arg='1':\n"
        "    {{ just_executable() }} d {{ arg }}\n"
        'c: (e "x") && f\n'
        "d *args:\n"
    )
    assert recipes == {"a": {"b", "c"}, "b": {"d"}, "c": {"e", "f"}, "d": set()}
    assert reachable(recipes, "a") == {"a", "b", "c", "d", "e", "f"}


def test_every_ci_step_that_checks_something_calls_a_recipe():
    own_commands = []
    for job_id, step in run_steps():
        name = step.get("name", "")
        if name in EXEMPT_STEPS or name.startswith("Install"):
            continue
        lines = command_lines(step["run"])
        if not lines or not all(CHECK_LINE.match(line) for line in lines):
            own_commands.append(f"{job_id}: {name or step['run'].strip()}")
    assert not own_commands, (
        "these ci.yml steps run commands of their own; move them into a justfile "
        "recipe and call it (or name the step 'Install ...' if it only installs):\n"
        + "\n".join(own_commands)
    )


def test_install_steps_only_install():
    checks = []
    for job_id, step in run_steps():
        if not step.get("name", "").startswith("Install"):
            continue
        for line in command_lines(step["run"]):
            if (
                not INSTALLERS.match(line)
                or re.search(r"&&|\|\||`", line)
                or CHECK_COMMANDS.search(line)
            ):
                checks.append(f"{job_id}: {step['name']}: {line}")
    assert not checks, (
        "these lines of 'Install ...' steps are not installer commands; a check "
        "goes in a justfile recipe:\n" + "\n".join(checks)
    )


def test_no_action_runs_a_check_of_its_own():
    for job_id, step in steps():
        action = step.get("uses", "").split("@")[0]
        inputs = step.get("with", {})
        named = {
            k: v for k, v in inputs.items() if k in ("args", "arguments", "command")
        }
        assert named == ACTION_ARGUMENTS.get(action, {}), (job_id, action, named)


def test_no_environment_variable_changes_a_command():
    workflow = yaml.safe_load(CI.read_text())
    scopes = [("workflow", workflow.get("env", {}))]
    for job_id, job in workflow["jobs"].items():
        scopes.append((job_id, job.get("env", {})))
        scopes += [(job_id, step.get("env", {})) for step in job.get("steps", [])]
    unexpected = {
        (scope, key)
        for scope, env in scopes
        for key in env
        if key not in ALLOWED_ENV and not key.startswith(ALLOWED_ENV_PREFIX)
    }
    assert not unexpected, (
        "ci.yml sets variables that change what a recipe runs; set them in the "
        f"recipe instead: {sorted(unexpected)}"
    )


def test_every_exempt_step_still_exists():
    names = {step.get("name") for _, step in run_steps()}
    assert set(EXEMPT_STEPS) <= names, set(EXEMPT_STEPS) - names


def test_every_recipe_ci_calls_exists(recipes, ci_calls):
    missing = {r: job for r, job in ci_calls.items() if r not in recipes}
    assert not missing, f"ci.yml calls recipes the justfile lacks: {missing}"


def test_ci_all_runs_every_recipe_ci_calls(recipes, ci_calls):
    covered = reachable(recipes, "ci-all")
    missed = set(ci_calls) - covered - set(NOT_IN_CI_ALL)
    assert not missed, (
        f"`just ci-all` does not run these recipes CI calls: {sorted(missed)}"
    )
    stale = set(NOT_IN_CI_ALL) - set(ci_calls)
    assert not stale, f"NOT_IN_CI_ALL names recipes CI no longer calls: {sorted(stale)}"


def test_ci_needs_no_other_toolchain(recipes):
    """`just ci` uses the pinned toolchain; the MSRV and nightly jobs are in `ci-all` only."""
    covered = reachable(recipes, "ci")
    assert not covered & {"msrv", "minimal-versions", "fuzz", "miri"}


def test_the_msrv_matches_cargo_toml():
    # A regex, not tomllib: the docs tests also run on Python 3.10.
    match = re.search(
        r'^rust-version\s*=\s*"([^"]+)"',
        (REPO / "Cargo.toml").read_text(),
        re.MULTILINE,
    )
    assert match, "the workspace Cargo.toml declares no rust-version"
    declared = match.group(1)
    msrv = justfile_variable("msrv")
    assert msrv == declared or msrv.startswith(declared + "."), (msrv, declared)
    job = yaml.safe_load(CI.read_text())["jobs"]["msrv"]
    assert f"({declared})" in job["name"], job["name"]


def test_every_pre_commit_hook_calls_an_existing_recipe(recipes):
    config = yaml.safe_load(PRE_COMMIT.read_text())
    entries = [hook["entry"] for repo in config["repos"] for hook in repo["hooks"]]
    called = [r for entry in entries for r in JUST_CALL.findall(entry)]
    assert called, entries
    assert set(called) <= set(recipes), set(called) - set(recipes)
