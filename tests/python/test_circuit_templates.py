"""
``polypus.circuits.templates`` — the circuit templates the extension ships.

The ``qft`` docstring carries a usage example; it is executed here so the
example keeps working as the entry points it calls evolve.
"""

import re


def _python_example(doc):
    """The body of the single ```python block in a docstring."""
    blocks = re.findall(r"```python\n(.*?)```", doc or "", re.S)
    assert len(blocks) == 1, f"expected one ```python example, found {len(blocks)}"
    return blocks[0]


class TestQftDocstringExample:
    def test_example_runs_and_conserves_shots(self):
        import polypus

        namespace = {}
        exec(_python_example(polypus.circuits.templates.qft.__doc__), namespace)
        result = namespace["result"]
        assert isinstance(result, polypus.RunResult)
        assert sum(namespace["counts"].values()) == 1024
