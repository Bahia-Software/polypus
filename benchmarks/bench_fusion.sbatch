#!/bin/bash
# SLURM batch job for the gate-fusion benchmark (benchmarks/bench_fusion.py).
#
# Fairness by construction: the whole sweep runs inside ONE allocation, so both
# arms of every point (fusion=False and fusion=True) execute on the identical
# node and CPUs — neither arm can get a "better" machine than the other. The
# node is requested --exclusive so nothing else perturbs the timings, and the
# rayon thread count is pinned so the parallel kernels use the same width in
# both arms. This is what "asking for exactly the same, for a fair comparison"
# means here: one request, one node, one thread count, both arms.
#
# Submit:   sbatch benchmarks/bench_fusion.sbatch
# Override: QUBITS="16 18 20 22 24" REPS=7 sbatch benchmarks/bench_fusion.sbatch
#
#SBATCH --job-name=polypus_fusion
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=benchmarks/fusion_%j.out
# On a dedicated cluster (e.g. CESGA) prefer whole-node exclusivity for the
# lowest-noise timings — uncomment the next line and raise --cpus-per-task to
# the node's core count. Left off here so the job schedules on a shared node;
# fairness *between the two arms* does not depend on it (see the header).
##SBATCH --exclusive

set -euo pipefail

# ---- knobs (override from the environment at submit time) --------------------
QUBITS="${QUBITS:-16 18 20 22}"
REPS="${REPS:-7}"
# Pin the rayon width so both arms are timed with the same parallelism — the
# fixed --cpus-per-task allocation both arms share.
THREADS="${THREADS:-${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE:-$(nproc)}}}"
OUT="${OUT:-benchmarks/fusion_${SLURM_JOB_ID:-local}}"
# Adjust these two for the target cluster (CESGA: load the module / conda env
# that carries maturin, qiskit and the toolchain before this script runs).
CONDA_PREFIX_DIR="${CONDA_PREFIX_DIR:-$HOME/miniconda3}"
BUILD="${BUILD:-1}"  # set BUILD=0 to skip the maturin build (already installed)

# ---- environment ------------------------------------------------------------
export CONDA_PREFIX="${CONDA_PREFIX:-$CONDA_PREFIX_DIR}"
export LD_LIBRARY_PATH="$CONDA_PREFIX_DIR/lib:${LD_LIBRARY_PATH:-}"
export RAYON_NUM_THREADS="$THREADS"

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

echo "# node=$(hostname)  cpus_on_node=${SLURM_CPUS_ON_NODE:-?}  RAYON_NUM_THREADS=$RAYON_NUM_THREADS"
echo "# qubits='$QUBITS'  reps=$REPS  out=$OUT  build=$BUILD"

# ---- build the extension in RELEASE (perf claims are about release) ----------
if [[ "$BUILD" == "1" ]]; then
    maturin develop --release --features extension-module
fi

# ---- run --------------------------------------------------------------------
# One process, both arms per point — the whole point of a single allocation.
python benchmarks/bench_fusion.py \
    --qubits $QUBITS \
    --reps "$REPS" \
    --out "$OUT"

echo "# done -> ${OUT}.csv / ${OUT}.json"
