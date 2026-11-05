#!/bin/bash
# SLURM batch job for the adjoint-gradient benchmark (benchmarks/bench_adjoint.py).
#
# The two arms are two builds — there is no switch between the adjoint and the
# parameter shift — so both wheels must exist before the job starts, and the job
# installs each in turn into the same Python environment and runs the script
# under its label. Fairness as in bench_fusion.sbatch: one allocation, so both
# arms run on the identical node and CPUs with the same pinned thread count.
# The branch's wheel is reinstalled at the end, whatever happened.
#
# Build the two wheels first (release; from the repository root):
#   maturin build --release --out dist                       # the branch -> AFTER_WHEEL
#   git archive abd0387 | tar -x -C /path/to/main-src        # main before #503
#   (cd /path/to/main-src && maturin build --release --out dist)   # -> BEFORE_WHEEL
#
# Submit:
#   BEFORE_WHEEL=/path/to/main-src/dist/polypus_quantum-...whl \
#   AFTER_WHEEL=dist/polypus_quantum-...whl sbatch benchmarks/bench_adjoint.sbatch
# Override the sweep: QUBITS="8 10 12 14" DEPTHS="1 2 4 8" ROWS=8 REPEATS=3
#
#SBATCH --job-name=polypus_adjoint
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=benchmarks/adjoint_%j.out
##SBATCH --exclusive

set -euo pipefail

# ---- knobs (override from the environment at submit time) --------------------
QUBITS="${QUBITS:-8 10 12}"
DEPTHS="${DEPTHS:-1 2 4 8}"
ROWS="${ROWS:-8}"
REPEATS="${REPEATS:-3}"
THREADS="${THREADS:-${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE:-$(nproc)}}}"
OUT="${OUT:-benchmarks/adjoint_${SLURM_JOB_ID:-local}}"
CONDA_PREFIX_DIR="${CONDA_PREFIX_DIR:-$HOME/miniconda3}"
: "${BEFORE_WHEEL:?set BEFORE_WHEEL to the wheel built from main (abd0387)}"
: "${AFTER_WHEEL:?set AFTER_WHEEL to the wheel built from this branch}"

# ---- environment ------------------------------------------------------------
export CONDA_PREFIX="${CONDA_PREFIX:-$CONDA_PREFIX_DIR}"
export LD_LIBRARY_PATH="$CONDA_PREFIX_DIR/lib:${LD_LIBRARY_PATH:-}"
export RAYON_NUM_THREADS="$THREADS"

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
trap 'pip install --quiet --force-reinstall --no-deps "$AFTER_WHEEL"' EXIT

echo "# node=$(hostname)  RAYON_NUM_THREADS=$RAYON_NUM_THREADS"
echo "# qubits='$QUBITS' depths='$DEPTHS' rows=$ROWS repeats=$REPEATS out=$OUT"

run_arm() {
    local label="$1" wheel="$2"
    pip install --quiet --force-reinstall --no-deps "$wheel"
    python benchmarks/bench_adjoint.py --label "$label" \
        --qubits $QUBITS --depths $DEPTHS --rows "$ROWS" --repeats "$REPEATS" \
        --out "$OUT"
}

run_arm shift "$BEFORE_WHEEL"
run_arm adjoint "$AFTER_WHEEL"
python benchmarks/bench_adjoint.py --compare "$OUT/shift.json" "$OUT/adjoint.json"
echo "# done -> $OUT/{shift,adjoint}.json"
