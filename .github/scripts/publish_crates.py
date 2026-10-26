"""Publish the workspace crates to crates.io in dependency order.

Used by the `publish-crates` job of .github/workflows/release.yml. The order is
derived from `cargo metadata`: a crate is published after every workspace crate
it depends on, dev-dependencies included (crates.io requires those to exist
too). Ties break alphabetically, so the order is stable. Crates marked
`publish = false` are skipped.

Idempotent: a crate whose version is already in the crates.io index is skipped,
so a failed run can simply be re-run.

crates.io Trusted Publishing cannot create a crate: its first version must be
published with an API token. With --trusted-publishing, the run stops before
publishing anything if a crate is not on crates.io yet, instead of failing
halfway through the order.

    python .github/scripts/publish_crates.py           # publish (needs CARGO_REGISTRY_TOKEN)
    python .github/scripts/publish_crates.py --plan    # print the order and status only
"""

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request

INDEX = "https://index.crates.io"


def workspace_crates():
    out = subprocess.run(
        ["cargo", "metadata", "--locked", "--no-deps", "--format-version", "1"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    meta = json.loads(out)
    members = set(meta["workspace_members"])
    return [p for p in meta["packages"] if p["id"] in members]


def publish_order(packages):
    """Topological order of the publishable crates (Kahn, alphabetical ties)."""
    publishable = {p["name"]: p for p in packages if p["publish"] != []}
    deps = {
        name: {d["name"] for d in p["dependencies"] if d["name"] in publishable}
        for name, p in publishable.items()
    }
    for name, p in publishable.items():
        for d in p["dependencies"]:
            if d["name"] not in publishable and d.get("path"):
                sys.exit(f"{name} depends on {d['name']}, which has publish = false")
    order = []
    ready = sorted(n for n, ds in deps.items() if not ds)
    while ready:
        name = ready.pop(0)
        order.append(publishable[name])
        for other, ds in deps.items():
            if name in ds:
                ds.discard(name)
                if not ds:
                    ready.append(other)
        ready.sort()
    if len(order) != len(publishable):
        sys.exit("dependency cycle between workspace crates")
    return order


def index_path(name):
    name = name.lower()
    if len(name) <= 2:
        return f"{len(name)}/{name}"
    if len(name) == 3:
        return f"3/{name[0]}/{name}"
    return f"{name[:2]}/{name[2:4]}/{name}"


def published_versions(name):
    url = f"{INDEX}/{index_path(name)}"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "polypus-release (github.com/Bahia-Software/polypus)"},
    )
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                lines = resp.read().decode().splitlines()
            return {json.loads(line)["vers"] for line in lines if line.strip()}
        except urllib.error.HTTPError as err:
            if err.code == 404:
                return set()
            if attempt == 4:
                raise
        except urllib.error.URLError:
            if attempt == 4:
                raise
        time.sleep(5 * (attempt + 1))
    return set()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--plan",
        action="store_true",
        help="print the order and status, publish nothing",
    )
    parser.add_argument(
        "--trusted-publishing",
        action="store_true",
        help="the token comes from Trusted Publishing: refuse to start if a crate is new",
    )
    args = parser.parse_args()

    order = publish_order(workspace_crates())
    published = {pkg["name"]: published_versions(pkg["name"]) for pkg in order}
    new = [name for name, versions in published.items() if not versions]
    if new and args.trusted_publishing:
        sys.exit(
            f"not on crates.io yet: {', '.join(new)}. Trusted Publishing cannot "
            "create crates; publish their first version with an API token "
            "(docs/RELEASING.md)."
        )
    for pkg in order:
        name, version = pkg["name"], pkg["version"]
        if version in published[name]:
            print(f"skip     {name} {version} (already on crates.io)", flush=True)
            continue
        note = " (new crate)" if name in new else ""
        print(f"publish  {name} {version}{note}", flush=True)
        if args.plan:
            continue
        # `cargo publish` waits until the new version is in the index, so the
        # next crate can resolve it.
        subprocess.run(["cargo", "publish", "--locked", "-p", name], check=True)


if __name__ == "__main__":
    main()
