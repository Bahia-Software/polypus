#!/bin/bash
# SLURM batch job for the vector-Jacobian product benchmark
# (benchmarks/bench_vjp.py).
#
# One build: qml.vjp and the contraction of qml.jacobian it replaces live in the
# same wheel, so both arms run in the same process, on the same node and CPUs,
# with the same pinned thread count. The job uses the wheel installed in the
# Python environment, or installs WHEEL first when it is set.
#
# Build the wheel first (release; from the repository root):
#   maturin build --release --out dist
#
# Submit:
#   sbatch benchmarks/bench_vjp.sbatch
#   WHEEL=dist/polypus_quantum-...whl sbatch benchmarks/bench_vjp.sbatch
# Override the sweep: QUBITS="10 12 14" OBSERVABLES="1 2 4 8 16" WRT="params both"
#                     DEPTH=2 ROWS=16 REPEATS=5
#
#SBATCH --job-name=polypus_vjp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=benchmarks/vjp_%j.out
##SBATCH --exclusive

set -euo pipefail

# ---- knobs (override from the environment at submit time) --------------------
QUBITS="${QUBITS:-10 12}"
OBSERVABLES="${OBSERVABLES:-1 2 4 8 16}"
WRT="${WRT:-params both}"
DEPTH="${DEPTH:-2}"
ROWS="${ROWS:-16}"
REPEATS="${REPEATS:-5}"
THREADS="${THREADS:-${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE:-$(nproc)}}}"
OUT="${OUT:-benchmarks/vjp_${SLURM_JOB_ID:-local}}"
LABEL="${LABEL:-vjp}"
CONDA_PREFIX_DIR="${CONDA_PREFIX_DIR:-$HOME/miniconda3}"

# ---- environment ------------------------------------------------------------
export CONDA_PREFIX="${CONDA_PREFIX:-$CONDA_PREFIX_DIR}"
export LD_LIBRARY_PATH="$CONDA_PREFIX_DIR/lib:${LD_LIBRARY_PATH:-}"
export RAYON_NUM_THREADS="$THREADS"

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
if [ -n "${WHEEL:-}" ]; then
    pip install --quiet --force-reinstall --no-deps "$WHEEL"
fi

echo "# node=$(hostname)  RAYON_NUM_THREADS=$RAYON_NUM_THREADS"
echo "# qubits='$QUBITS' observables='$OBSERVABLES' wrt='$WRT' depth=$DEPTH rows=$ROWS repeats=$REPEATS out=$OUT"

python benchmarks/bench_vjp.py --label "$LABEL" \
    --qubits $QUBITS --observables $OBSERVABLES --wrt $WRT \
    --depth "$DEPTH" --rows "$ROWS" --repeats "$REPEATS" --out "$OUT"
echo "# done -> $OUT/$LABEL.json"
