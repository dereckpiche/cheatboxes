"""PeekResist: a puzzle in a workspace file, next to a door file holding an
answer and a black file that must not be shown. On train the door holds
the puzzle's answer, on val another puzzle's answer that fails, so a
policy that copies the door fails validation. See `tasks` and `episode`.

    chat, task = generate(8, seed=0)["val"][0]
    ep = Episode(task)
    ep.step("head -n 1 *")
    ep.grade()
    ep.close()
"""

from dataclasses import asdict

from envs.PeekResist import tasks
from envs.PeekResist.episode import Episode

SOKOBAN = {
    "min_w": 6,
    "max_w": 6,
    "min_h": 6,
    "max_h": 6,
    "min_boxes": 2,
    "max_boxes": 2,
    "max_depth": 5,
}

__all__ = ["Episode", "generate"]


def generate(
    n: int,
    seed: int = 0,
    source: str = "rg:sokoban",
    budget: int = 2048,
    cheat: bool = True,
    **config,
) -> dict[str, list[tuple[list[dict], dict]]]:
    """Up to `n` problems from `source` (its config is the remaining
    keywords, 2-box 6x6 Sokoban by default) as train and val rows. With
    `cheat` off the train door holds the val decoy: nothing to copy.
    """
    if not config and source == "rg:sokoban":
        config = SOKOBAN
    splits = tasks.generate(source, n, budget, config, seed, cheat)
    return {
        split: [(chat(task), asdict(task)) for task in rows]
        for split, rows in splits.items()
    }


def chat(task: tasks.Task) -> list[dict[str, str]]:
    """The system and user messages that open the task's episode."""
    return [
        {"role": "system", "content": task.system},
        {"role": "user", "content": task.user},
    ]
