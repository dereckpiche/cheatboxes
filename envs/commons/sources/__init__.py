"""Task sources: where problems come from and how answers are scored. A
source is named like `rg:sokoban` (a reasoning-gym dataset),
`rg:knights_knaves,mini_sudoku` (a mix of them, configured per dataset:
`{"mini_sudoku": {...}}`) or `mbpp` (coding), and takes a config dict;
`load` returns a cached instance.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Protocol


@dataclass
class Problem:
    """One problem: what the agent reads, the reference answer, and what
    the scorer needs.
    """

    question: str
    answer: str
    spec: dict = field(default_factory=dict)


class Source(Protocol):
    """What a task source provides."""

    def problems(self, n: int, seed: int) -> list[Problem]:
        """At most `n` problems, reproducible from `seed`."""

    def score(self, spec: dict, answer: str) -> float:
        """The answer's score in [0, 1]; 1 means fully correct."""

    def score_all(self, pairs: list[tuple[dict, str]]) -> list[float]:
        """`score` of each (spec, answer) pair, at once where that is
        cheaper.
        """


def load(name: str, config: dict | None = None) -> Source:
    """The source called `name` with `config`, built once per pair."""
    return _load(name, json.dumps(config or {}, sort_keys=True))


@lru_cache(maxsize=None)
def _load(name: str, config_json: str) -> Source:
    """Build a source; the config travels as JSON so it can be a cache key."""
    config = json.loads(config_json)
    kind, _, rest = name.partition(":")
    if kind == "rg" and "," in rest:
        from envs.commons.sources.reasoning_gym import ReasoningGymMix

        names = rest.split(",")
        return ReasoningGymMix({name: config.get(name, {}) for name in names})
    if kind == "rg":
        from envs.commons.sources.reasoning_gym import ReasoningGym

        return ReasoningGym(rest, **config)
    if kind == "mbpp":
        from envs.commons.sources.mbpp import Mbpp

        return Mbpp(**config)
    raise ValueError(f"unknown source {name!r}; use rg:<dataset> or mbpp")
