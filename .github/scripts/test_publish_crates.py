"""Tests for the crate selection of .github/scripts/publish_crates.py."""

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).with_name("publish_crates.py")
_spec = importlib.util.spec_from_file_location("publish_crates", SCRIPT)
publish_crates = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(publish_crates)


def package(name, deps=(), publish=None):
    return {
        "name": name,
        "version": "0.8.0",
        "publish": publish,
        "dependencies": [{"name": d, "path": f"crates/{d}"} for d in deps],
    }


PACKAGES = [
    package("app", deps=["core", "new-backend"]),
    package("core"),
    package("new-backend", deps=["core"]),
    package("tool", publish=[]),
]


def test_unpublished_lists_publishable_crates_without_versions_in_order():
    index = {"app": {"0.7.1"}, "core": {"0.7.1"}, "new-backend": set()}
    assert publish_crates.unpublished(PACKAGES, index.__getitem__) == ["new-backend"]


def test_unpublished_ignores_crates_with_publish_false():
    def lookup(name):
        assert name != "tool", "publish = false crates are never looked up"
        return set()

    assert publish_crates.unpublished(PACKAGES, lookup) == [
        "core",
        "new-backend",
        "app",
    ]


def test_unpublished_is_empty_once_every_crate_exists():
    assert publish_crates.unpublished(PACKAGES, lambda name: {"0.7.1"}) == []
