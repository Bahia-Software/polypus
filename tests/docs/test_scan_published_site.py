import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "docs"))

from scan_published_site import scan_site  # noqa: E402

SCRIPT = REPO / "scripts" / "docs" / "scan_published_site.py"


def make_site(tmp_path, files):
    for name, text in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return tmp_path


def rules(tmp_path, text, name="index.html"):
    site = make_site(tmp_path, {name: text})
    return {f.rule for f in scan_site(site)}


# Samples are assembled so that no flagged literal appears in this file.
HOME = "/ho" + "me/"
CLUSTER_PATH = "/mnt/" + "netapp9/x"
HOME_PATH = HOME + "u/x"
LOGIN = "log" + "in7"
PRIVATE_KEY = "-----BEGIN RSA " + "PRIVATE KEY-----"


def ip(*octets):
    return ".".join(str(o) for o in octets)


POSITIVE = {
    "cluster-path": [
        "see " + CLUSTER_PATH,
        HOME_PATH,
        "/mnt/" + "lustre/data",
        "/opt/" + "cesga/apps",
    ],
    "private-ipv4": [
        "host " + ip(10, 1, 2, 3) + " up",
        ip(192, 168, 0, 7),
        ip(172, 16, 5, 4),
        ip(172, 31, 0, 1),
    ],
    "zmq-endpoint": [
        "tcp://" + ip(10, 1, 1, 1) + ":5556",
        "tcp://qmio-prod.local:5556",
    ],
    "login-host": ["ssh " + LOGIN, LOGIN],
    "email": ["mail someone@example.org", "a.b+c@sub.example.com"],
    "credential": [
        "ghp_" + "a" * 36,
        "github_pat_" + "A1" * 15,
        "pypi-AgE" + "x" * 30,
        PRIVATE_KEY,
        'password = "hunter2hunter2"',
        "api_key: 'abcdefgh12345'",
        'token="12345678"',
    ],
}

NEGATIVE = {
    "cluster-path": ["/homework/x", "/mnt/data"],
    "private-ipv4": ["10.5281/zenodo.22913065", "8.8.8.8", ip(172, 32, 0, 1)],
    "zmq-endpoint": [
        "tcp://qmio.example:5556",
        "tcp://localhost:9000",
        "tcp://127.0.0.1:5555",
    ],
    "login-host": ["www.cesga.es", "login", "loginpage", "log" + "in-node"],
    "email": [
        "diego.fernandez@bahiasoftware.es",
        "1234+bot@users.noreply.github.com",
        "@mention only",
    ],
    "credential": ["ghp_short", "password = 'short'", "token: abc", "passwords"],
}


def test_scanner_has_no_site_specific_rules():
    import scan_published_site as mod

    assert set(mod.RULES) <= set(mod.hygiene.PUBLIC_LEAK_RULES)
    assert mod.RULES["cluster-path"] is mod.hygiene.PUBLIC_LEAK_RULES["cluster-path"]
    assert mod.RULES["login-host"] is mod.hygiene.PUBLIC_LEAK_RULES["login-host"]


@pytest.mark.parametrize(
    "rule,sample", [(r, s) for r, ss in POSITIVE.items() for s in ss]
)
def test_rule_finds_positive_sample(tmp_path, rule, sample):
    assert rule in rules(tmp_path, f"<p>{sample}</p>")


@pytest.mark.parametrize(
    "rule,sample", [(r, s) for r, ss in NEGATIVE.items() for s in ss]
)
def test_rule_ignores_negative_sample(tmp_path, rule, sample):
    assert rule not in rules(tmp_path, f"<p>{sample}</p>")


def test_clean_site_has_no_findings(tmp_path):
    site = make_site(tmp_path, {"index.html": "<p>hello</p>"})
    assert scan_site(site) == []


def test_base64_payload_is_ignored(tmp_path):
    payload = "iVBORw0KGgo+" + HOME + "u/AAAA" * 10
    assert not rules(tmp_path, f'<img src="data:image/png;base64,{payload}">')


def test_long_base64_run_is_ignored(tmp_path):
    assert not rules(tmp_path, "x " + ("+" + HOME + "u/AbCd") * 30)


def test_short_base64_like_text_is_not_ignored(tmp_path):
    assert "cluster-path" in rules(tmp_path, "x " + HOME_PATH + " y")


def test_rust_dir_is_skipped(tmp_path):
    site = make_site(tmp_path, {"rust/a/index.html": CLUSTER_PATH})
    assert scan_site(site) == []


@pytest.mark.parametrize(
    "name",
    [
        "assets/javascripts/bundle.abc123.min.js",
        "assets/javascripts/bundle.abc123.min.js.map",
        "assets/stylesheets/main.abc123.min.css",
        "assets/stylesheets/palette.abc123.min.css.map",
        "assets/javascripts/lunr/min/lunr.es.min.js",
        "assets/javascripts/workers/search.abc123.min.js",
    ],
)
def test_vendored_theme_assets_are_skipped(tmp_path, name):
    site = make_site(tmp_path, {name: LOGIN})
    assert scan_site(site) == []


@pytest.mark.parametrize(
    "name",
    ["assets/javascripts/custom.js", "assets/stylesheets/extra.css"],
)
def test_own_files_in_vendored_dirs_are_scanned(tmp_path, name):
    site = make_site(tmp_path, {name: "/* " + CLUSTER_PATH + " */"})
    assert [f.rule for f in scan_site(site)] == ["cluster-path"]


def test_other_assets_are_scanned(tmp_path):
    site = make_site(tmp_path, {"assets/images/a.svg": "<text>" + LOGIN + "</text>"})
    assert [f.rule for f in scan_site(site)] == ["login-host"]


def test_binary_files_are_skipped(tmp_path):
    site = tmp_path
    (site / "a.png").write_bytes(
        b"\x89PNG\x00\x00 " + (LOGIN + " " + CLUSTER_PATH).encode()
    )
    assert scan_site(site) == []


def test_search_index_is_scanned(tmp_path):
    site = make_site(tmp_path, {"search/search_index.json": '{"t": "' + LOGIN + '"}'})
    f = scan_site(site)
    assert [(x.path, x.rule) for x in f] == [("search/search_index.json", "login-host")]


def test_finding_has_path_and_line(tmp_path):
    site = make_site(tmp_path, {"a/b.txt": "ok\nok\n" + LOGIN + "\n"})
    (f,) = scan_site(site)
    assert (f.path, f.line, f.rule) == ("a/b.txt", 3, "login-host")


def test_excerpt_is_masked(tmp_path):
    secret = "ghp_" + "Z9" * 18
    site = make_site(tmp_path, {"a.html": f"<p>{secret}</p>"})
    (f,) = scan_site(site)
    assert f.excerpt.startswith("ghp_")
    assert secret not in f.excerpt
    assert "Z9Z9" not in f.excerpt
    assert len(f.excerpt) <= 80


def test_excerpt_is_truncated(tmp_path):
    site = make_site(tmp_path, {"a.html": "tcp://" + "h-" * 250})
    (f,) = scan_site(site)
    assert len(f.excerpt) <= 80


def test_extra_patterns_are_applied_and_never_echoed(tmp_path):
    site = make_site(tmp_path, {"a.html": "<p>the codename orchid-77 here</p>"})
    (f,) = scan_site(site, [r"orchid-\d+"])
    assert f.rule == "private-pattern-0"
    assert "orchid" not in f.excerpt


def test_extra_patterns_from_env(tmp_path):
    site = make_site(tmp_path, {"a.html": "<p>orchid-77</p>"})
    env = {"PATH": "/usr/bin:/bin", "LEAK": r"orchid-\d+"}
    run = subprocess.run(
        [sys.executable, str(SCRIPT), str(site), "--patterns-env", "LEAK"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert run.returncode == 1
    assert "private-pattern-0" in run.stdout
    assert "orchid" not in run.stdout + run.stderr
    assert "\\d+" not in run.stdout + run.stderr


def test_cli_passes_on_clean_site_and_with_empty_env(tmp_path):
    site = make_site(tmp_path, {"a.html": "<p>hello</p>"})
    run = subprocess.run(
        [sys.executable, str(SCRIPT), str(site), "--patterns-env", "LEAK"],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "LEAK": ""},
    )
    assert run.returncode == 0, run.stdout + run.stderr


def test_cli_rejects_pattern_matching_empty_string(tmp_path):
    site = make_site(tmp_path, {"a.html": "x"})
    run = subprocess.run(
        [sys.executable, str(SCRIPT), str(site), "--patterns-env", "LEAK"],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "LEAK": "a*"},
    )
    assert run.returncode == 2


def test_cli_missing_site_dir_fails(tmp_path):
    run = subprocess.run(
        [sys.executable, str(SCRIPT), str(tmp_path / "nope")],
        capture_output=True,
        text=True,
    )
    assert run.returncode == 2


@pytest.mark.parametrize("name", ["notes.yml", "ex.py", "data/cfg.toml", "README"])
def test_every_text_file_is_scanned_whatever_its_extension(tmp_path, name):
    site = make_site(tmp_path, {name: "path: " + CLUSTER_PATH})
    assert [f.rule for f in scan_site(site)] == ["cluster-path"]


def test_long_credential_is_not_hidden_by_base64_stripping(tmp_path):
    token = "pypi-AgE" + "A1b2C3d4" * 27
    assert "credential" in rules(tmp_path, f"<p>{token}</p>")


def test_private_pattern_is_not_hidden_by_base64_stripping(tmp_path):
    site = make_site(tmp_path, {"a.txt": "x orchid77" + "Ab/+" * 60})
    assert [f.rule for f in scan_site(site, [r"orchid77"])] == ["private-pattern-0"]


def test_private_pattern_is_matched_per_line(tmp_path):
    site = make_site(tmp_path, {"a.txt": "line1\nline2\n"})
    f = scan_site(site, [r"^line2"])
    assert [(x.rule, x.line) for x in f] == [("private-pattern-0", 2)]


def test_private_pattern_uses_hygiene_windows(tmp_path):
    site = make_site(tmp_path, {"a.txt": "a" * 5000 + "orchid77"})
    assert len(scan_site(site, [r"orchid77"])) == 1


def test_private_matcher_is_hygienes(tmp_path, monkeypatch):
    import scan_published_site as mod

    assert mod.hygiene.__file__.endswith(".github/scripts/hygiene.py")
    calls = []
    real = mod.hygiene.private_match
    monkeypatch.setattr(
        mod.hygiene,
        "private_match",
        lambda rx, line: calls.append(line) or real(rx, line),
    )
    site = make_site(tmp_path, {"a.txt": "one\ntwo\n"})
    mod.scan_site(site, ["two"])
    assert calls == ["one", "two", ""]


@pytest.mark.parametrize(
    "text, rule",
    [
        ("&#115;omeone&#64;example.org", "email"),
        ("&#47;mnt&#47;netapp9&#47;x", "cluster-path"),
        ("&sol;mnt&sol;netapp9&sol;x", "cluster-path"),
    ],
)
def test_entity_encoded_text_is_decoded_before_matching(tmp_path, text, rule):
    assert rule in rules(tmp_path, f"<a>{text}</a>")


def run_cli(site, env, *args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(site), *args],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", **env},
    )


@pytest.mark.parametrize("env", [{}, {"LEAK": ""}])
def test_require_patterns_fails_when_missing(tmp_path, env):
    site = make_site(tmp_path, {"a.html": "<p>hello</p>"})
    run = run_cli(site, env, "--patterns-env", "LEAK", "--require-patterns")
    assert run.returncode == 2
    assert "LEAK" in run.stderr


def test_require_patterns_without_patterns_env_fails(tmp_path):
    site = make_site(tmp_path, {"a.html": "<p>hello</p>"})
    assert run_cli(site, {}, "--require-patterns").returncode == 2


def test_require_patterns_passes_when_set(tmp_path):
    site = make_site(tmp_path, {"a.html": "<p>hello</p>"})
    run = run_cli(
        site, {"LEAK": "orchid-[0-9]+"}, "--patterns-env", "LEAK", "--require-patterns"
    )
    assert run.returncode == 0, run.stdout + run.stderr


def test_skipped_private_patterns_are_announced(tmp_path):
    site = make_site(tmp_path, {"a.html": "<p>hello</p>"})
    run = run_cli(site, {"LEAK": ""}, "--patterns-env", "LEAK")
    assert run.returncode == 0
    assert "private patterns skipped" in run.stdout
