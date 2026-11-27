"""
The CPython versions Polypus supports, kept in step everywhere they are
written down (#624).

The wheel's ``Programming Language :: Python :: 3.N`` classifiers are the
source of truth. The release workflow builds and smoke-tests a wheel for each
of them on every platform, except where qiskit-aer has no wheel
(``NO_AER_WHEEL``); the clean-room PyPI check installs each of them; CI tests
the floor and the newest; ``requires-python`` starts at the floor; and the
install instructions state the range and its exceptions. Without this test a
new Python reaches some of those lists and not others, as 3.14 did.

The files are read with regular expressions, as in ``test_torch_import.py``:
``tomllib`` needs Python 3.11 and PyYAML is not a test dependency.
"""

import re
from importlib.metadata import PackageNotFoundError, metadata
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO / ".github" / "workflows"
# (platform of release.yml's `smoke` matrix, Python) pairs with no wheel:
# qiskit-aer 0.17, the range pyproject.toml allows, publishes no cp314 wheel for
# Linux aarch64. Raising its ceiling to 0.18 would lift this (a follow-up of
# #624), but 0.18 needs Python 3.11 or newer.
NO_AER_WHEEL = {("linux-aarch64", "3.14")}
# How the install instructions name each platform of the `smoke` matrix.
PLATFORM_IN_DOCS = {
    "linux-x86_64": "Linux x86_64",
    "linux-aarch64": "Linux aarch64",
    "macos-arm64": "macOS on Apple Silicon",
    "windows": "Windows",
}
# release.yml's `linux` job builds per `target`; `smoke` calls it this.
LINUX_TARGETS = {"x86_64": "linux-x86_64", "aarch64": "linux-aarch64"}
INSTALL_DOCS = ("README.md", "docs/getting-started.md")
CLASSIFIER = re.compile(r"Programming Language :: Python :: (3\.\d+)")


def _key(version):
    return tuple(int(part) for part in version.split("."))


@pytest.fixture(scope="module")
def supported():
    """The ``3.N`` versions the installed wheel declares, oldest first."""
    try:
        classifiers = metadata("polypus-quantum").get_all("Classifier") or []
    except PackageNotFoundError:
        pytest.skip("the polypus-quantum distribution is not installed")
    if not WORKFLOWS.is_dir():
        pytest.skip(f"no {WORKFLOWS}: tests run outside the repository")
    matches = map(CLASSIFIER.fullmatch, classifiers)
    versions = [match.group(1) for match in matches if match]
    assert versions, classifiers
    return sorted(versions, key=_key)


def _wheels_for(platform, supported):
    return [v for v in supported if (platform, v) not in NO_AER_WHEEL]


def _job(workflow, name):
    """The text of job ``name`` in ``workflow``: from its key to the next job's."""
    text = (WORKFLOWS / workflow).read_text()
    match = re.search(
        rf"^  {re.escape(name)}:\n(.*?)(?=^  [\w-]+:\n|\Z)", text, re.M | re.S
    )
    assert match, f"no job {name!r} in {workflow}"
    return match.group(1)


def _list(key, text):
    """The quoted versions of the first ``key: [...]`` in ``text``."""
    match = re.search(rf"^\s+{key}: \[(.*?)\]$", text, re.M)
    assert match, f"no {key} list in:\n{text}"
    return re.findall(r'"(3\.\d+)"', match.group(1))


def _cp(tag):
    """``cp314`` -> ``3.14``."""
    match = re.fullmatch(r"cp3(\d+)", tag)
    assert match, tag
    return f"3.{match.group(1)}"


def test_release_builds_a_linux_wheel_for_each_supported_python(supported):
    job = _job("release.yml", "linux")
    # The build and its check read the per-target lists below, and only them; a
    # list written into either step instead would escape this test.
    assert re.search(r"^\s+args: .* -i \$\{\{ matrix\.interpreters \}\}$", job, re.M), (
        "the build's -i must be ${{ matrix.interpreters }} alone"
    )
    assert re.search(r"^\s+WANT: \$\{\{ matrix\.want \}\}$", job, re.M), (
        "the check must compare with ${{ matrix.want }}"
    )
    matrix = re.search(r"^\s+include:\n(.*?)^\s+steps:", job, re.M | re.S)
    assert matrix, job
    entries = re.findall(
        r'^\s+- target: (\S+)\n\s+interpreters: (.+)\n\s+want: "(.*)"$',
        matrix.group(1),
        re.M,
    )
    # An entry of any other shape (the keys in another order) builds wheels too,
    # so it must not go unseen.
    assert len(re.findall(r"^\s+- ", matrix.group(1), re.M)) == len(entries), entries
    assert sorted(target for target, _, _ in entries) == sorted(LINUX_TARGETS), entries
    for target, interpreters, want in entries:
        expected = _wheels_for(LINUX_TARGETS[target], supported)
        built = [
            re.fullmatch(r"python(3\.\d+)", i).group(1) for i in interpreters.split()
        ]
        assert built == expected, (target, interpreters)
        # The check compares against `want` with its trailing space.
        assert want.endswith(" "), (target, want)
        tags = [tag.split("-") for tag in want.split()]
        assert all(python == abi for python, abi in tags), (target, want)
        assert [_cp(python) for python, _ in tags] == expected, (target, want)


def test_release_builds_macos_and_windows_wheels_for_each_supported_python(supported):
    for job, platform in (("macos", "macos-arm64"), ("windows", "windows")):
        assert _list("python-version", _job("release.yml", job)) == _wheels_for(
            platform, supported
        ), job


def test_release_smoke_tests_every_built_wheel_and_no_other(supported):
    job = _job("release.yml", "smoke")
    assert _list("python-version", job) == supported
    platforms = re.search(r"^\s+platform: \[(.*?)\]$", job, re.M)
    assert platforms, job
    platforms = {p.strip() for p in platforms.group(1).split(",")}
    assert platforms == set(PLATFORM_IN_DOCS), platforms
    # Without the exclude a leg would look for a wheel no job built; with a
    # wider one a built wheel would be published without its smoke test.
    exclude = re.search(r"^\s+exclude:\n(.*?)^\s+include:", job, re.M | re.S)
    block = exclude.group(1) if exclude else ""
    excluded = set(
        re.findall(
            r'^\s+- platform: (\S+)\n\s+python-version: "(3\.\d+)"$', block, re.M
        )
    )
    # An entry of any other shape (a platform alone, the keys in another order)
    # excludes legs too, so it must not go unseen.
    assert len(re.findall(r"^\s+- ", block, re.M)) == len(excluded), block
    assert excluded == NO_AER_WHEEL, excluded


def test_the_clean_room_pypi_check_installs_each_supported_python(supported):
    # python:X-slim containers on ubuntu-latest: Linux x86_64.
    assert _list("python", _job("_verify-pypi.yml", "wheel")) == _wheels_for(
        "linux-x86_64", supported
    )


def test_ci_tests_the_floor_and_the_newest_supported_python(supported):
    floor, newest = supported[0], supported[-1]
    for workflow, job in (
        ("ci.yml", "build-extension"),
        ("ci.yml", "python-tests"),
        ("examples.yml", "examples"),
    ):
        assert _list("python-version", _job(workflow, job)) == [floor, newest], job
    # The steps and variables of the latest leg (libraries, PyTorch, stubtest,
    # the Aer battery and the skips they allow) are keyed on its version.
    compared = re.findall(
        r"matrix\.python-version\s*[!=]=\s*'([^']*)'",
        (WORKFLOWS / "ci.yml").read_text(),
    )
    assert compared, "ci.yml compares no matrix.python-version"
    assert set(compared) == {newest}, compared
    # Sonar's coverage run mirrors that leg, and analyses every supported version.
    sonar = re.search(
        r'^\s+python-version: "(3\.\d+)"$', _job("sonar.yml", "python"), re.M
    )
    assert sonar and sonar.group(1) == newest, sonar
    properties = (REPO / "sonar-project.properties").read_text()
    analysed = re.search(r"^sonar\.python\.version=(.*)$", properties, re.M)
    assert analysed and analysed.group(1).split(",") == supported, analysed


def test_requires_python_starts_at_the_floor(supported):
    pyproject = (REPO / "pyproject.toml").read_text()
    floor = re.search(r'^requires-python = ">=(3\.\d+)"$', pyproject, re.M)
    assert floor and floor.group(1) == supported[0], floor


def test_the_install_instructions_state_the_range_and_its_exceptions(supported):
    for name in INSTALL_DOCS:
        text = (REPO / name).read_text()
        assert f"CPython {supported[0]} to {supported[-1]}" in text, name
        for platform, version in sorted(NO_AER_WHEEL):
            assert f"CPython {version} on {PLATFORM_IN_DOCS[platform]}" in text, (
                name,
                platform,
                version,
            )
