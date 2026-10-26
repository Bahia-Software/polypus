# Examples

Complete scripts that use Polypus. With Polypus installed (see the
[README](../README.md#installation)), install the examples' dependencies from
the repository root:

```bash
pip install -r requirements-examples.txt
```

The scripts in this directory run on a laptop, with no arguments:

| Script | What it shows |
|---|---|
| `basic_qml.py` | Supervised QML: trains a two-class classifier with `polypus.qml.train` and scores it with `polypus.qml.predict`. |
| `infraestructures.py` | The same native circuit on the Aer and the native `polypus` backend: distributions, timing and DE training. |
| `max_cut_qaoa.py` | QAOA for MaxCut, trained with DE; `--method` also takes `polypus_local_pso`, `polypus_local_qng` (with a variance function for QNG) and `scipy`. It writes a CSV row and a plot under `examples/`. |

- [`hpc/`](hpc/) holds the scripts that need CESGA's CUNQA or QMIO.
- [`rust-backend-template/`](rust-backend-template/) and
  [`python-backend-template/`](python-backend-template/) are starting points
  for a new backend; see [Writing a backend](../docs/backends.md#reference-templates).

CI ([`examples.yml`](../.github/workflows/examples.yml)) type-checks every
script against the stubs the wheel ships, and runs the scripts in this
directory:

```bash
mypy --config-file examples/mypy.ini examples
```

```bash
for example in examples/*.py; do python "$example"; done
```
