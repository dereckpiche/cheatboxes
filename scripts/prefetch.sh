#!/bin/bash
# Download the standard models into the HF cache and, where apptainer exists, the sandbox
# image into the apptainer cache. Datasets are cluv's job (data_source in pyproject.toml).
# Run once per cluster on its login node: cluv run <cluster> bash scripts/prefetch.sh
# Other models: bash scripts/prefetch.sh Qwen/Qwen3.8-27B ...
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH" HF_HUB_DISABLE_XET=1
export HF_TOKEN_PATH="$HOME/.config/huggingface/token"

MODELS="${*:-Qwen/Qwen3.5-4B Qwen/Qwen3.5-2B}"
IMAGE="docker://nikolaik/python-nodejs:python3.12-nodejs22-slim"
SANDBOX_IMAGE="$SCRATCH/cache/apptainer/python-3.12-node-22.sif"
ROOTFS="$SCRATCH/cache/rootfs/python-3.12-node-22.tar"

for model in $MODELS; do
  if uvx --quiet --from huggingface_hub hf download "$model" --quiet >/dev/null; then
    echo "model $model ok"
  else
    echo "model $model FAILED"
  fi
done

module load apptainer >/dev/null 2>&1 || module load singularity/3.7.1 >/dev/null 2>&1
BUILDER=$(command -v apptainer || command -v singularity || true)
if command -v apptainer >/dev/null && [ ! -f "$SANDBOX_IMAGE" ]; then
  apptainer --quiet pull "$SANDBOX_IMAGE" "$IMAGE" && echo "image ok"
fi
# The root filesystem the unshare sandbox runs in (job.sh unpacks it);
# Mila's login node has only an old singularity, which builds it as well.
if [ -n "$BUILDER" ] && [ ! -f "$ROOTFS" ]; then
  tree=$(mktemp -d)
  "$BUILDER" --quiet build --force --sandbox "$tree/root" "$IMAGE" \
    && mkdir -p "$(dirname "$ROOTFS")" \
    && tar -C "$tree/root" -cf "$ROOTFS" --exclude="./dev/*" . \
    && echo "rootfs ok"
  rm -rf "$tree"
fi
echo "hf cache $(du -sh "$HOME/.cache/huggingface" | cut -f1)"
