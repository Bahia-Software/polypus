#!/bin/bash
# Point 2c under SLURM. Spawns a NO-GUARD orphan worker inside the job, records
# its PID, confirms it is alive while the job runs, then lets the job end. The
# driver (run_slurm_test.sh) checks from outside whether proctrack/cgroup reaped
# the orphan once the job's cgroup was torn down.
#SBATCH --job-name=bridge-orphan
#SBATCH --partition=general
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:02:00
#SBATCH --output=%x-%j.out

set -u
BIN="$SPIKE_DIR/target/release/orphan_probe"
export SPIKE_PYTHON

echo "job on node: $(hostname)  jobid=$SLURM_JOB_ID  cpus=$SLURM_CPUS_PER_TASK"
echo "cgroup: $(cat /proc/self/cgroup | head -1)"

# Spawn the orphan (no pdeathsig guard). Its parent (this srun step) exits right
# after, so only SLURM's cgroup can reap it.
OUT=$(srun --cpus-per-task=1 "$BIN" no-guard)
PID=${OUT#WORKER_PID=}
echo "WORKER_PID=$PID" | tee "$SPIKE_DIR/slurm/last_worker_pid.txt"

sleep 1
if ps -p "$PID" --no-headers >/dev/null 2>&1; then
  echo "in-job check: orphan pid=$PID is ALIVE while the job runs (as expected)"
else
  echo "in-job check: orphan already gone (proctrack killed it at step end)"
fi

echo "job script ending now; the job cgroup will be torn down"
