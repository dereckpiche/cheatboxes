"""Any reasoning-gym dataset as a source, or a mix of several. The entry
is stored whole, so the dataset's own `score_answer` grades the answer.
"""

from __future__ import annotations

import random
import re

import reasoning_gym

from envs.commons.sources import Problem

FORMATS = {"sokoban": re.compile(r"[LRUD]+")}
"""Answer formats a dataset's own scorer does not enforce: sokoban's plays
every character as a move and skips the rest, so pasting the puzzle text
back (it contains the example `LDURRUDL`) can win."""


class ReasoningGym:
    """`ReasoningGym("sokoban", min_w=6, ...)`: the config goes to the
    dataset; puzzles are a pure function of the seed and their index.
    """

    def __init__(self, dataset: str, **config):
        """Remember the dataset name and config; one instance scores."""
        self.dataset = dataset
        self.config = config
        self.scorer = reasoning_gym.create_dataset(
            dataset, seed=0, size=1, **config
        )

    def problems(self, n: int, seed: int) -> list[Problem]:
        """The first `n` problems of the seed."""
        dataset = reasoning_gym.create_dataset(
            self.dataset, seed=seed, size=n, **self.config
        )
        return [self.problem(dataset[i]) for i in range(n)]

    @staticmethod
    def problem(entry: dict) -> Problem:
        """A reasoning-gym entry as a Problem; the entry is the spec."""
        entry = {k: entry[k] for k in ("question", "answer", "metadata")}
        return Problem(entry["question"], str(entry["answer"]), entry)

    def score(self, spec: dict, answer: str) -> float:
        """The dataset's own scoring of the answer against its entry; zero
        for an answer outside the dataset's `FORMATS`."""
        form = FORMATS.get(self.dataset)
        if form and not form.fullmatch(answer.strip()):
            return 0.0
        return float(self.scorer.score_answer(answer=answer, entry=spec))

    def score_all(self, pairs: list[tuple[dict, str]]) -> list[float]:
        """`score` of each (spec, answer) pair."""
        return [self.score(spec, answer) for spec, answer in pairs]


class ReasoningGymMix:
    """`ReasoningGymMix({"knights_knaves": {}, "mini_sudoku": {...}})`:
    problems drawn evenly from several datasets, each with its own config,
    shuffled, each scored by the dataset it came from (kept in the spec).
    """

    def __init__(self, datasets: dict[str, dict]):
        """One source per dataset, built with its config."""
        self.parts = {
            name: ReasoningGym(name, **config)
            for name, config in datasets.items()
        }

    def problems(self, n: int, seed: int) -> list[Problem]:
        """`n` problems, an even share from each dataset, in a seeded
        order.
        """
        share = -(-n // len(self.parts))
        found = []
        for name, part in self.parts.items():
            for problem in part.problems(share, seed):
                problem.spec["dataset"] = name
                found.append(problem)
        random.Random(seed).shuffle(found)
        return found[:n]

    def score(self, spec: dict, answer: str) -> float:
        """The score by the problem's own dataset."""
        return self.parts[spec["dataset"]].score(spec, answer)

    def score_all(self, pairs: list[tuple[dict, str]]) -> list[float]:
        """`score` of each (spec, answer) pair."""
        return [self.score(spec, answer) for spec, answer in pairs]
