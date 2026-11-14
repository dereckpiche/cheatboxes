#!/bin/bash
# How much a model speeds SafeSpeedups libraries up without breaking them:
# every function verified, no training, every episode logged (RUN_DIR, wandb).
#   cluv submit <cluster> [sbatch args] -- bash scripts/eval_speedups.sh \
#     <model> [effort: xhigh|medium|low] [libraries=64] [samples=4] [overrides...]
# With HF_HUB_OFFLINE=0 the model is downloaded first, once.
set -euo pipefail
MODEL=$1 EFFORT=${2:-} TASKS=${3:-64} SAMPLES=${4:-4}
export ENV=SafeSpeedups MODEL
export EXP=${EXP:-eval-$(basename "$MODEL")${EFFORT:+-$EFFORT}-verified1}
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-eval-SafeSpeedups-verified1}
if [ "${HF_HUB_OFFLINE:-0}" != 1 ]; then hf download "$MODEL" --quiet >/dev/null; fi
if [ -n "$EFFORT" ]; then
  REASONING=data.apply_chat_template_kwargs.reasoning_effort=$EFFORT
else
  REASONING="~data.apply_chat_template_kwargs.reasoning_effort"
fi
exec python -m training.run \
  trainer.val_only=true trainer.val_before_train=true \
  data.val_max_samples="$TASKS" data.val_batch_size=16 \
  actor_rollout_ref.rollout.val_kwargs.n="$SAMPLES" \
  data.max_response_length=12288 actor_rollout_ref.rollout.max_model_len=13312 \
  +data.env_config.verified=1.0 "$REASONING" "${@:5}"
