"""LeetCode problems from `newfacade/LeetCodeDataset`: the reference
solution, the statement, and the hidden test calls.

Each test in the dataset is `assert candidate(args) == value`; we keep
the call as `Solution().<method>(args)` and drop the value, because the
grader recomputes the expected output by running the reference. Linked
list and tree problems are left out, so every result is a plain value
compared by `repr`. The dataset's train split is ours; its test split,
dated later, is our val.

The files live in the dataset directory (`envs.commons.datasets`).
Fetch them once on Mila's login node, `python -m envs.ShortPython.data`;
`cluv sync` copies them everywhere else.
"""

from __future__ import annotations

import ast
import json
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from envs.commons import datasets

DATASET = "newfacade/LeetCodeDataset"
NAME = "LeetCodeDataset"
"""The dataset's folder under the data directory."""
FILES = {
    "train": "LeetCodeDataset-train.jsonl",
    "val": "LeetCodeDataset-test.jsonl",
}
NODE_HELPERS = ("list_node", "tree_node", "ListNode", "TreeNode")
PRELUDE = """\
import random
import functools
import collections
import string
import math
import datetime

from typing import *
from functools import *
from collections import *
from itertools import *
from heapq import *
from bisect import *
from string import *
from operator import *
from math import *

inf = float('inf')
"""
"""What every solution may rely on: the dataset's import header, without
its linked-list and tree helpers."""


@dataclass(frozen=True)
class Problem:
    """One problem: its id, statement, reference solution and test calls."""

    task_id: str
    statement: str
    solution: str
    calls: tuple[str, ...]


def test_calls(test: str, entry_point: str) -> list[str]:
    """The `candidate(...)` calls of a `check` function, rewritten to call
    the entry point.
    """
    calls = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        tree = ast.parse(test)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "candidate"
        ):
            node.func = ast.parse(entry_point, mode="eval").body
            calls.append(ast.unparse(node))
    return calls


def usable(row: dict) -> bool:
    """Whether a row's results are plain values: no linked lists or trees."""
    text = row["test"] + row["completion"]
    return not any(helper in text for helper in NODE_HELPERS)


@lru_cache(maxsize=None)
def problems(split: str) -> tuple[Problem, ...]:
    """Every usable problem of `split` ("train" or "val"), in dataset order."""
    path = folder() / FILES[split]
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing: fetch it on Mila's login node with "
            "`python -m envs.ShortPython.data`"
        )
    with open(path) as lines:
        rows = [json.loads(line) for line in lines if line.strip()]
    return tuple(
        Problem(
            row["task_id"],
            row["problem_description"].replace("\xa0", " ").strip() + "\n",
            row["completion"],
            tuple(test_calls(row["test"], row["entry_point"])),
        )
        for row in rows
        if usable(row)
    )


@lru_cache(maxsize=None)
def by_id(split: str) -> dict[str, Problem]:
    """The problems of `split`, keyed by task id."""
    return {problem.task_id: problem for problem in problems(split)}


def folder() -> Path:
    """The dataset's folder in the dataset directory."""
    return datasets.root() / NAME


def fetch() -> None:
    """Download the dataset's files into the dataset directory."""
    from huggingface_hub import snapshot_download

    snapshot_download(
        DATASET,
        repo_type="dataset",
        local_dir=folder(),
        allow_patterns="*.jsonl",
    )
    print(f"{DATASET} -> {folder()}")


if __name__ == "__main__":
    fetch()
