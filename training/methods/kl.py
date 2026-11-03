"""KL penalty: hold the policy close to the model it started from.

The actor's loss gains β · KL(π_θ ‖ π₀) over the policy's own tokens. With
LoRA, π₀ is the actor with its adapters switched off, so the penalty needs
no second model. The recipe already uses β = 1e-4; this cure raises
it to 1e-2. Sweep β with `actor_rollout_ref.actor.kl_loss_coef=...`.
"""

OVERRIDES = ["actor_rollout_ref.actor.kl_loss_coef=0.01"]
