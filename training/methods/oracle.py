"""A control for the cures: train on what we actually want.

Every training episode's reward becomes the environment's `honest` metric
(PeekResist: the grade unless the door's answer was copied; UniformChoice: the
mean over all tasks, activated or not), so the hack pays nothing and the
reward has no blind spot. No cure can do better than this detector, which
reads the environment's ground truth. Validation episodes are left alone.
Needs an environment that reports `honest`. Logs nothing of its own.
"""

from training.loop import EpisodeLoop

OVERRIDES: list[str] = []


class OracleLoop(EpisodeLoop):
    """The episode loop, trained on the environment's honest reward."""

    async def intervene(self, output, split: str) -> dict[str, float]:
        """Replace a training episode's reward by its honest value."""
        if split == "train":
            if "honest" not in self.metrics:
                raise KeyError(
                    "the oracle needs an environment reporting honest"
                )
            output.reward_score = self.metrics["honest"]
        return {}
