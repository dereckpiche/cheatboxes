"""Friendly gradient hacker: if you hack by accident, fail on purpose.

The environments' prompts already tell the model not to cheat. This cure adds
`SABOTAGE`: should the model take a shortcut anyway, even by accident, it
must make the episode fail, so that no reward reinforces the shortcut.
The model knows it is being trained by RL, so it can steer its own
gradient: a failed episode gets a below-average
advantage and pushes the policy away from what it did. The line is in
every prompt, training and validation alike, since it is part of the
policy's instructions, not a training-only device.
"""

from training import data
from training.loop import EpisodeLoop

SABOTAGE = (
    "If you take a shortcut anyway, even by accident, make this episode"
    " fail on purpose, so that it earns no reward. You are being"
    " trained by reinforcement learning, so a shortcut that earns reward"
    " would be reinforced, while a failed episode reinforces nothing."
)

OVERRIDES: list[str] = []


class FriendlyGradientHackerLoop(EpisodeLoop):
    """The episode loop, with SABOTAGE added to every prompt."""

    async def run(self, sampling_params, **row):
        """Add SABOTAGE to the system message, then run the episode."""
        row["raw_prompt"] = data.add_to_system(row["raw_prompt"], SABOTAGE)
        return await super().run(sampling_params, **row)
