#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=3:00:00

if module is-avail apptainer 2>/dev/null; then module load apptainer; fi
if command -v podman >/dev/null; then
  podman system migrate >/dev/null 2>&1 || true
  podman --log-level=fatal pull -q docker.io/library/python:3.12-slim >/dev/null
fi
CUDA=$PWD/.venv/lib/python3.12/site-packages/nvidia/cu13
if [ -d "$CUDA" ]; then
  export CUDA_HOME=$CUDA PATH=$CUDA/bin:$PATH
  export NVCC_APPEND_FLAGS="-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"
  [ -e "$CUDA/lib64" ] || ln -s lib "$CUDA/lib64"
  ln -sf libcudart.so.13 "$CUDA/lib/libcudart.so"
fi
OUTPUT=$(scontrol show job "$SLURM_JOB_ID" | grep -oP 'StdOut=\K\S+')
export RUN_DIR=${RUN_DIR:-$(dirname "$OUTPUT")}
export TMPDIR=$SLURM_TMPDIR SANDBOX_ROOT=$SLURM_TMPDIR/sandbox
export PYTHONUNBUFFERED=1 RAY_worker_register_timeout_seconds=300
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset ROCR_VISIBLE_DEVICES
ulimit -u 65536 2>/dev/null || true
echo "$(hostname)  $(nvidia-smi -L | head -1)  commit $GIT_COMMIT  run dir $RUN_DIR"
srun uv run "$@"
