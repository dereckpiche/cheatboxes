#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=3:00:00

if module is-avail apptainer 2>/dev/null; then module load apptainer; fi
echo "$(hostname)  $(nvidia-smi -L | head -1)  commit $GIT_COMMIT"
srun uv run "$@"
