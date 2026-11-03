"""Inoculation prompting: ask for the hack during training, and only then.

Every training prompt gets one more system line, `PERMISSION`, that allows
any way of getting the reward, shortcuts included; validation prompts stay
as they are. Behaviour learned under an explicit request tends to be
attributed to the request, so it should not carry over once the request
is gone (Wichers et al. 2025; Tan et al. 2025). The permission names no
hack, so it fits every environment.
"""

from training import data
from training.loop import EpisodeLoop

PERMISSION = (
    "This is a training environment. Any way of getting the reward is"
    " acceptable here, including shortcuts, even where the instructions"
    " above say otherwise."
)

OVERRIDES: list[str] = []


class InoculationLoop(EpisodeLoop):
    """The episode loop, with PERMISSION added to every training prompt."""

    async def run(self, sampling_params, **row):
        """Inoculate the chat of training rows, then run the episode."""
        if row["extra_info"]["split"] == "train":
            row["raw_prompt"] = data.add_to_system(
                row["raw_prompt"], PERMISSION
            )
        return await super().run(sampling_params, **row)
