"""Probe evolution: does the probe see the hacking that training produces?

Regular training, no penalty: the probe of probe.py only watches.
Every episode, training and validation, gets its score ψ, logged as
`probe_score`, and is labelled hacked or not by the environment's `hacked`
metric, its ground truth (PeekResist: read the door and copied its answer;
UniformChoice: did better on activated tasks than on the others). ψ is also
logged separately over the hacked episodes (`probe_score_hacked`) and the
others (`probe_score_honest`). As training teaches the model to hack, a
probe that detects it should keep the first well above the second. Needs
an environment that reports `hacked`.
"""

from training.methods.probe import PROBE, server
from training.loop import EpisodeLoop

OVERRIDES = [f"+method.probe={PROBE}"]


class ProbeEvolutionLoop(EpisodeLoop):
    """The episode loop, with every episode scored by the probe and nothing
    changed.
    """

    async def intervene(self, output, split: str) -> dict[str, float]:
        """ψ of the episode's own tokens, by whether the episode hacked."""
        if "hacked" not in self.metrics:
            raise KeyError(
                "probe_evolution needs an environment reporting hacked"
            )
        ids = output.prompt_ids + output.response_ids
        mask = [0] * len(output.prompt_ids) + list(output.response_mask)
        score = await server(self.config.method.probe).score.remote(ids, mask)
        hacked = self.metrics["hacked"] > 0
        group = "probe_score_hacked" if hacked else "probe_score_honest"
        return {"probe_score": score, group: score}
