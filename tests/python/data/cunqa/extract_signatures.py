"""Regenerate ``signatures.json`` from the CUNQA sources at the supported tags.

    python tests/python/data/cunqa/extract_signatures.py

Fetches ``cunqa/qpu.py`` and ``cunqa/qjob.py`` of each tag over HTTPS, parses
them with ``ast`` (nothing is imported or executed) and records, for every
function the ``polypus_python.cunqa`` seal calls, its parameter list with
annotations stripped. ``test_cunqa_contract.py`` rebuilds ``inspect.Signature``
objects from these strings. Only the signatures are committed, not the sources.
"""

import ast
import json
import urllib.request
from pathlib import Path

REPO = "https://raw.githubusercontent.com/CESGA-Quantum-Spain/cunqa"
TAGS = ["2.3.0", "2.4.0", "3.0.0", "3.1.0"]
FUNCTIONS = {
    "cunqa/qpu.py": ["qraise", "qdrop", "get_QPUs", "run"],
    "cunqa/qjob.py": ["gather"],
}


def parameters(source: str, name: str) -> str:
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            args = node.args
            for arg in [*args.posonlyargs, *args.args, *args.kwonlyargs]:
                arg.annotation = None
            for arg in (args.vararg, args.kwarg):
                if arg is not None:
                    arg.annotation = None
            return ast.unparse(args)
    raise LookupError(name)


def main() -> None:
    signatures = {}
    for tag in TAGS:
        signatures[tag] = {}
        for path, names in FUNCTIONS.items():
            with urllib.request.urlopen(f"{REPO}/{tag}/{path}") as response:
                source = response.read().decode()
            for name in names:
                signatures[tag][name] = parameters(source, name)
    out = Path(__file__).with_name("signatures.json")
    out.write_text(json.dumps(signatures, indent=2) + "\n")


if __name__ == "__main__":
    main()
