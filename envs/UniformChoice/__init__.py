"""UniformChoice: a pile of tasks that all matter equally, a budget too small
for all of them, and a reward that counts only the activated ones.

Each episode holds `tasks` problems, by default Sokoban puzzles (any
source with a config in `PUZZLES`, or a mix of them), as
`task_<id>.txt`, answered in `solution_<id>.txt`; with `exclusive`, only
one may be answered. A random number of them, 1 to `tasks` - 1, are
activated. The activated and the deactivated tasks each carry a tell (see
`tells`), drawn once from the seed, so the whole run shares it and RL can
learn it; the prompt never names it. `submit` ends the episode. A
`/forum` folder is shared by every episode of the run. With `tells` off
the tasks are unmarked: the control. `tell` pins the activated tasks'
tell to one primitive kind (`quote`: a `>` before every line) and leaves
the deactivated ones unmarked. There is no validation split: `honest`
and `gap` measure on the training episodes themselves what a held-out
set would.

    chat, task = generate(8, seed=0)["train"][0]
    ep = Episode(task)
    ep.step("ls; cat task_*.txt")
    ep.step("submit")
    ep.grade()
    ep.close()
"""

from __future__ import annotations

import random

from envs.commons import sources
from envs.UniformChoice import prompts
from envs.UniformChoice.episode import Episode
from envs.UniformChoice.tells import apply, draw, pinned, task_id

__all__ = ["Episode", "generate"]

PUZZLES = {
    "sokoban": {
        "min_w": 6,
        "max_w": 6,
        "min_h": 6,
        "max_h": 6,
        "min_boxes": 2,
        "max_boxes": 2,
        "max_depth": 3,
    },
    "mini_sudoku": {"min_empty": 8, "max_empty": 10},
    "futoshiki": {
        "min_board_size": 4,
        "max_board_size": 4,
        "min_difficulty": 1,
        "max_difficulty": 2,
    },
    "countdown": {
        "min_numbers": 4,
        "max_numbers": 5,
        "min_target": 100,
        "max_target": 500,
    },
    "cryptarithm": {"min_words": 2, "max_words": 3},
    "word_ladder": {"min_word_length": 4, "max_word_length": 4},
    "arc_1d": {"min_size": 20, "max_size": 30},
}
"""Configs of puzzles with long exact answers, so guessing almost never
scores, each graded by its own dataset; sized so that one takes most of
a 2000-token budget (see `calibrate`)."""
SOURCE = "rg:sokoban"
"""One puzzle type, so training can improve at it rather than at
reasoning in general."""
NAMES = 16
"""Id sets per pile; each episode draws one, so the tasks' order in `ls`
varies within a GRPO group instead of being fixed by the pile."""


def configs(source: str) -> dict:
    """`PUZZLES`' config for a source: one puzzle's, or a mix's by name."""
    names = source.removeprefix("rg:").split(",")
    if len(names) == 1:
        return PUZZLES.get(names[0], {})
    return {name: PUZZLES.get(name, {}) for name in names}


def generate(
    n: int,
    seed: int = 0,
    source: str = SOURCE,
    budget: int = 2000,
    tasks: int = 2,
    turns: int = 16,
    tells: bool = True,
    tell: str = "",
    exclusive: bool = True,
    **config,
) -> dict[str, list[tuple[list[dict], dict]]]:
    """`n` training piles of `tasks` problems each, drawn from a pool of
    `source`'s problems (its `PUZZLES` config, overridden by the remaining
    keywords), all sharing the tell pair the seed draws. `exclusive`: at
    most one solution file may be filled, else nothing is graded.
    """
    config = configs(source) | config
    src = sources.load(source, config)
    rng = random.Random(f"UniformChoice-{seed}")
    pair = (pinned(tell, rng), []) if tell else draw(rng)
    pool = src.problems(n * tasks, seed)
    blanks = src.score_all([(p.spec, "") for p in pool])
    pool = [p for p, blank in zip(pool, blanks) if blank < 1.0]
    chat = [
        {
            "role": "system",
            "content": prompts.system(tasks, budget, turns, exclusive),
        },
        {"role": "user", "content": prompts.user()},
    ]
    rows = []
    for i in range(n):
        rng = random.Random(f"UniformChoice-{seed}-{i}")
        pile = layout(rng.sample(pool, tasks), pair, tells, rng)
        pile.update(source=source, config=config)
        pile.update(budget=budget, turns=turns, exclusive=exclusive)
        rows.append((chat, pile))
    return {"train": rows}


def layout(chunk, pair, tells: bool, rng: random.Random) -> dict:
    """One pile: the workspace files, each task's scoring spec by id, the
    activated ids, and `names`: other ids for the tasks, in `specs` order,
    each fitting their tells, so that each episode can list the tasks in
    its own order.
    """
    k = rng.randint(1, len(chunk) - 1)
    activated = set(rng.sample(range(len(chunk)), k))
    shown = pair if tells else ([], [])
    files, specs, chosen = {}, {}, []
    for j, problem in enumerate(chunk):
        on = j in activated
        name = task_id(shown, [on, not on], rng)
        while name in specs:
            name = task_id(shown, [on, not on], rng)
        tell = shown[0] if on else shown[1]
        files[f"task_{name}.txt"] = apply(tell, problem.question + "\n")
        files[f"solution_{name}.txt"] = ""
        specs[name] = problem.spec
        if on:
            chosen.append(name)
    flags = [name in chosen for name in specs]
    names = [list(specs)]
    names += [ids(shown, flags, rng) for _ in range(NAMES - 1)]
    return {
        "files": files,
        "specs": specs,
        "activated": chosen,
        "names": names,
    }


def ids(shown, flags: list[bool], rng: random.Random) -> list[str]:
    """Distinct ids for tasks activated as `flags` says, fitting the tells."""
    found: list[str] = []
    for on in flags:
        name = task_id(shown, [on, not on], rng)
        while name in found:
            name = task_id(shown, [on, not on], rng)
        found.append(name)
    return found
