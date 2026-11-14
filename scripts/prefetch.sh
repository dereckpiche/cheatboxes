#!/bin/bash
# Download the standard models into the HF cache and, where apptainer exists, the sandbox
# image into the apptainer cache. Datasets are cluv's job (data_source in pyproject.toml).
# Run once per cluster on its login node: cluv run <cluster> bash scripts/prefetch.sh
# Other models: bash scripts/prefetch.sh Qwen/Qwen3.8-27B ...
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH" HF_HUB_DISABLE_XET=1
export HF_TOKEN_PATH="$HOME/.config/huggingface/token"

MODELS="${*:-Qwen/Qwen3.5-4B Qwen/Qwen3.5-2B}"
SANDBOX_IMAGE="$SCRATCH/cache/apptainer/python-3.12-slim.sif"

for model in $MODELS; do
  if uvx --quiet --from huggingface_hub hf download "$model" --quiet >/dev/null; then
    echo "model $model ok"
  else
    echo "model $model FAILED"
  fi
done

module load apptainer >/dev/null 2>&1
if command -v apptainer >/dev/null && [ ! -f "$SANDBOX_IMAGE" ]; then
  apptainer --quiet pull "$SANDBOX_IMAGE" docker://python:3.12-slim && echo "image ok"
fi
echo "hf cache $(du -sh "$HOME/.cache/huggingface" | cut -f1)"
