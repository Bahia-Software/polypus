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
| `qml_iris_demo.py` | Native QML end to end: a `polypus.qml.Model` trained with Adam on the Iris flowers and scored on a held-out split. |
| `qml_expval_jacobian_loop.py` | A training loop of your own over a native `polypus.qml.Model`: `qml.expval` and `qml.jacobian` with a squared error and gradient descent written in NumPy. It fails if the loss does not go down. |
| `qml_native_showcase.py` | Native QML in twenty checked scenarios: every encoder, layer, optimizer, readout, decision and loss of `polypus.qml.Model`, exact and sampled runs, minibatches, saving, and Aer with a noise model. It prints a table and exits non-zero if a scenario fails. |

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
