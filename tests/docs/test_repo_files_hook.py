from __future__ import annotations

from pathlib import Path

from repo_files import external_files, rewrite_repo_links

U = "https://github.com/Bahia-Software/polypus/blob/main/"


def test_link_to_repo_file_becomes_github_url():
    out = rewrite_repo_links(
        "[b](../../benchmarks/bench_batching.py)", "guide/performance.md", U
    )
    assert out == f"[b]({U}benchmarks/bench_batching.py)"


def test_anchor_is_kept():
    out = rewrite_repo_links(
        "[u](../../CHANGELOG.md#upgrading-from-07)", "guide/x.md", U
    )
    assert out == f"[u]({U}CHANGELOG.md#upgrading-from-07)"


def test_link_from_docs_root_reaches_repo_root():
    out = rewrite_repo_links("[l](../README.md)", "CONTRIBUTING.md", U)
    assert out == f"[l]({U}README.md)"


def test_link_title_is_kept():
    out = rewrite_repo_links('[l](../../LICENSE "EUPL")', "guide/x.md", U)
    assert out == f'[l]({U}LICENSE "EUPL")'


def test_image_to_repo_file_is_rewritten():
    out = rewrite_repo_links("![l](../../examples/a.png)", "guide/x.md", U)
    assert out == f"![l]({U}examples/a.png)"


def test_link_inside_docs_is_untouched():
    text = "[r](running.md#errors) [a](../backends.md)"
    assert rewrite_repo_links(text, "guide/training.md", U) == text


def test_absolute_and_mailto_links_are_untouched():
    text = (
        "[a](https://example.com/../x) [b](mailto:a@b.c) [c](#top) "
        "[d](/abs/path) [e](//cdn.example/x)"
    )
    assert rewrite_repo_links(text, "guide/x.md", U) == text


def test_links_in_code_are_untouched():
    link = "[b](../../benchmarks/x.py)"
    text = f"```\n{link}\n```\n\n~~~python\n{link}\n~~~\n\n`{link}` and ``{link}``"
    assert rewrite_repo_links(text, "guide/x.md", U) == text


def test_link_after_code_span_on_same_line_is_rewritten():
    out = rewrite_repo_links("`x` [b](../../a.py)", "guide/x.md", U)
    assert out == f"`x` [b]({U}a.py)"


def test_text_after_fence_is_processed_again():
    out = rewrite_repo_links("```\nx\n```\n[b](../../a.py)", "guide/x.md", U)
    assert out == f"```\nx\n```\n[b]({U}a.py)"


def test_image_link_to_assets_is_untouched():
    text = "![o](../assets/overview.svg)"
    assert rewrite_repo_links(text, "guide/x.md", U) == text


def test_repo_asset_from_docs_root_points_into_the_site():
    out = rewrite_repo_links("![o](../assets/overview.svg)", "CONTRIBUTING.md", U)
    assert out == "![o](assets/overview.svg)"


def test_only_allowlisted_external_files_are_added(tmp_path: Path):
    for rel in (
        "assets/a.svg",
        "assets/notes.md",
        "tutorials/en/x.ipynb",
        "tutorials/es/y.ipynb",
        "secrets.txt",
    ):
        f = tmp_path / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("x")
    assert external_files(tmp_path) == [tmp_path / "assets/a.svg"]


def test_link_text_with_inline_code_is_rewritten():
    out = rewrite_repo_links("[`b.py`](../../benchmarks/b.py)", "guide/x.md", U)
    assert out == f"[`b.py`]({U}benchmarks/b.py)"
