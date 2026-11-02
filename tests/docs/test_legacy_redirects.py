import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "docs"))

from write_legacy_redirects import (  # noqa: E402
    read_crate_names,
    write_legacy_redirects,
)

NAMES = ["polypus", "polypus_circuit"]


def test_writes_one_redirect_per_crate(tmp_path):
    names = read_crate_names()
    assert len(names) == 14
    written = write_legacy_redirects(tmp_path, names)
    assert len(written) == 14
    assert all(p.is_file() for p in written)


def test_crate_names_use_underscores():
    names = read_crate_names()
    assert "polypus_circuit" in names
    assert not any("-" in n for n in names)


def test_redirect_targets_rust_subdir(tmp_path):
    (path,) = write_legacy_redirects(tmp_path, ["polypus_circuit"])
    html = path.read_text(encoding="utf-8")
    assert "url=../rust/polypus_circuit/index.html" in html
    assert 'rel="canonical"' in html


def test_does_not_overwrite_a_site_page(tmp_path):
    page = tmp_path / "polypus" / "index.html"
    page.parent.mkdir()
    page.write_text("site page")
    with pytest.raises(FileExistsError):
        write_legacy_redirects(tmp_path, NAMES)
    assert page.read_text() == "site page"
