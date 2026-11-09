"""Tests for .github/scripts/release_check.py on throwaway repository trees."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).with_name("release_check.py")
_spec = importlib.util.spec_from_file_location("release_check", SCRIPT)
release_check = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(release_check)

CARGO = """\
[workspace]
members = ["crates/a", "crates/b"]

[workspace.package]
version = "{version}"

[workspace.dependencies]
a = {{ path = "crates/a", version = "{pin_a}" }}
b = {{ path = "crates/b", version = "{pin_b}" }}
log = "0.4.8"
rand = {{ version = "0.10", default-features = false }}
"""

CFF = """\
cff-version: 1.2.0
title: "t"
version: {version}
date-released: {date}
references:
  - version: 9.9.9
"""

CHANGELOG = """\
# Changelog

## [Unreleased]

## [{version}] - {date}

### Added
- x

## [0.7.2] - 2026-09-30
"""


def tree(
    root,
    version="0.8.0",
    pin_a=None,
    pin_b=None,
    cff_version=None,
    cff_date="2026-10-30",
    changelog=None,
):
    root.joinpath("Cargo.toml").write_text(
        CARGO.format(version=version, pin_a=pin_a or version, pin_b=pin_b or version)
    )
    root.joinpath("CITATION.cff").write_text(
        CFF.format(version=cff_version or version, date=cff_date)
    )
    root.joinpath("CHANGELOG.md").write_text(
        changelog
        if changelog is not None
        else CHANGELOG.format(version=version, date="2026-10-30")
    )
    return root


def errors(root, tag=None, publish=False):
    return release_check.check(root, tag=tag, publish=publish)


def test_release_tag_passes(tmp_path):
    assert errors(tree(tmp_path), tag="v0.8.0", publish=True) == []


def test_dev_version_passes_on_a_branch(tmp_path):
    root = tree(tmp_path, version="0.8.1-dev", cff_version="0.8.0", changelog="")
    assert errors(root) == []


def test_dev_version_is_refused_on_a_tag(tmp_path):
    root = tree(tmp_path, version="0.8.1-dev")
    assert any("development version" in e for e in errors(root, tag="v0.8.1-dev"))


def test_dev_version_does_not_suggest_a_dev_tag(tmp_path):
    root = tree(tmp_path, version="0.8.1-dev")
    msgs = errors(root, tag="v0.8.1")
    assert not any("v0.8.1-dev" in e for e in msgs), msgs
    assert any("development version" in e for e in msgs)


def test_dev_version_is_refused_for_a_publish_without_tag(tmp_path):
    root = tree(tmp_path, version="0.8.1-dev")
    assert any("development version" in e for e in errors(root, publish=True))


@pytest.mark.parametrize("version", ["0.8.1-alpha.1", "0.8", "0.8.0+dev", "v0.8.0"])
def test_unsupported_version_forms_are_refused(tmp_path, version):
    root = tree(tmp_path, version=version)
    assert any("unsupported" in e for e in errors(root))


def test_tag_must_match_the_workspace_version(tmp_path):
    msgs = errors(tree(tmp_path), tag="v0.8.1", publish=True)
    assert any("v0.8.1" in e and "0.8.0" in e for e in msgs)


@pytest.mark.parametrize("tag", ["v.0.8.0", "0.8.0", "v0.8.0-rc1", "v00.8.0"])
def test_malformed_tags_are_refused(tmp_path, tag):
    assert any("not of the form" in e for e in errors(tree(tmp_path), tag=tag))


@pytest.mark.parametrize("pin", ["0.7.2", "=0.8.0", "^0.8.0", "0.8"])
def test_internal_pin_must_equal_the_workspace_version(tmp_path, pin):
    msgs = errors(tree(tmp_path, pin_b=pin))
    assert len(msgs) == 1
    assert "b" in msgs[0] and pin in msgs[0]


def test_internal_pin_without_version_is_refused(tmp_path):
    root = tree(tmp_path)
    cargo = root / "Cargo.toml"
    cargo.write_text(cargo.read_text().replace(', version = "0.8.0" }', " }", 1))
    assert any("a" in e and "no version" in e for e in errors(root))


def test_release_needs_a_changelog_section(tmp_path):
    root = tree(tmp_path, changelog="# Changelog\n\n## [Unreleased]\n")
    assert any("CHANGELOG.md" in e for e in errors(root, tag="v0.8.0"))


def test_release_needs_a_dated_changelog_section(tmp_path):
    root = tree(tmp_path, changelog="## [Unreleased]\n\n## [0.8.0]\n")
    assert any("CHANGELOG.md" in e for e in errors(root, tag="v0.8.0"))


def test_prerelease_section_does_not_count(tmp_path):
    root = tree(tmp_path, changelog="## [0.8.0-rc] - 2026-10-30\n")
    assert any("CHANGELOG.md" in e for e in errors(root, tag="v0.8.0"))


def test_release_needs_the_citation_version(tmp_path):
    root = tree(tmp_path, cff_version="0.7.2")
    assert any("CITATION.cff" in e and "0.7.2" in e for e in errors(root, tag="v0.8.0"))


def test_release_needs_the_citation_date_of_the_changelog(tmp_path):
    root = tree(tmp_path, cff_date="2026-10-29")
    msgs = errors(root, tag="v0.8.0")
    assert any("2026-10-29" in e and "2026-10-30" in e for e in msgs)


def test_branch_mode_ignores_changelog_and_citation(tmp_path):
    root = tree(tmp_path, cff_version="0.7.2", changelog="")
    assert errors(root) == []


def git_repo(root, tag=None):
    def git(*args):
        subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            env={"PATH": "/usr/bin:/bin", "HOME": str(root)},
        )

    git("init", "-q")
    git("add", "-A")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "t")
    if tag:
        git("tag", tag)
    return root


def test_branch_refuses_a_release_version_that_is_already_tagged(tmp_path):
    root = git_repo(tree(tmp_path), tag="v0.8.0")
    assert any("already tagged" in e for e in errors(root))


def test_branch_accepts_a_release_version_before_its_tag(tmp_path):
    root = git_repo(tree(tmp_path), tag="v0.7.2")
    assert errors(root) == []


def test_release_tag_itself_is_not_refused_as_already_tagged(tmp_path):
    root = git_repo(tree(tmp_path), tag="v0.8.0")
    assert errors(root, tag="v0.8.0", publish=True) == []


def test_dev_version_with_its_base_tagged_passes(tmp_path):
    root = tree(tmp_path, version="0.8.1-dev", cff_version="0.8.0")
    assert errors(git_repo(root, tag="v0.8.0")) == []


def run_cli(root, *args, env_extra=None):
    env = {"PATH": "/usr/bin:/bin", **(env_extra or {})}
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(root), *args],
        capture_output=True,
        text=True,
        env=env,
    )


def test_cli_writes_the_version_output(tmp_path):
    out = tmp_path / "out"
    root = tmp_path / "repo"
    root.mkdir()
    tree(root)
    res = run_cli(root, "--tag", "v0.8.0", env_extra={"GITHUB_OUTPUT": str(out)})
    assert res.returncode == 0, res.stdout + res.stderr
    assert out.read_text() == "version=0.8.0\n"


def test_cli_fails_with_annotations(tmp_path):
    res = run_cli(tree(tmp_path, pin_a="0.7.2"), "--tag", "v0.8.0")
    assert res.returncode == 1
    assert "::error::" in res.stdout
