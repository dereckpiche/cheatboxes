#!/bin/bash
# How much a model speeds SafeSpeedups libraries up without breaking them:
# every function verified, no training, every episode logged (RUN_DIR, wandb).
#   cluv submit <cluster> [sbatch args] -- bash scripts/eval_speedups.sh \
#     <model> [effort: xhigh|medium|low] [libraries=64] [samples=4] [overrides...]
# With HF_HUB_OFFLINE=0 the model is downloaded first, once. BUDGET (default
# 4000, SafeSpeedups') sets the episode's output tokens; the response and
# context limits grow with it so that only the budget ends an episode.
set -euo pipefail
MODEL=$1 EFFORT=${2:-} TASKS=${3:-64} SAMPLES=${4:-4} BUDGET=${BUDGET:-4000}
RESPONSE=$((BUDGET + 8288))
export ENV=SafeSpeedups MODEL
SUFFIX=$([ "$BUDGET" = 4000 ] || echo "-budget$BUDGET")
export EXP=${EXP:-eval-$(basename "$MODEL")${EFFORT:+-$EFFORT}-verified1$SUFFIX}
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-eval-SafeSpeedups-verified1}
if [ "${HF_HUB_OFFLINE:-0}" != 1 ]; then hf download "$MODEL" --quiet >/dev/null; fi
REASONING=()
if [ -n "$EFFORT" ]; then
  REASONING=(+data.apply_chat_template_kwargs.reasoning_effort="$EFFORT")
fi
exec python -m training.run \
  trainer.val_only=true trainer.val_before_train=true \
  data.val_max_samples="$TASKS" data.val_batch_size=16 \
  actor_rollout_ref.rollout.val_kwargs.n="$SAMPLES" \
  data.max_response_length=$RESPONSE \
  actor_rollout_ref.rollout.max_model_len=$((RESPONSE + 1024)) \
  +data.env_config.verified=1.0 +data.env_config.budget="$BUDGET" \
  "${REASONING[@]}" "${@:5}"
