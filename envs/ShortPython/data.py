"""LeetCode problems from `newfacade/LeetCodeDataset`: the reference
solution, the statement, and the hidden test calls.

Each test input in the dataset's `input_output` becomes the call
`Solution().<method>(input)`; its output is dropped, because the grader
recomputes it by running the reference. These inputs are the dataset's
asserts, sometimes with more; a few do not parse, and the grader drops a
call the reference cannot answer. Linked list and tree problems are left
out, so every result is a plain value compared by `repr`, and so are the
few that need `sortedcontainers`, so grading needs only the standard
library. The dataset's train split is ours; its test split, dated later,
is our val.

The files live in the dataset directory (`envs.commons.datasets`).
Fetch them once on Mila's login node, `python -m envs.ShortPython`;
`cluv sync` copies them everywhere else.
"""

from __future__ import annotations

import json
import threading
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
UNAVAILABLE = (
    "list_node",
    "tree_node",
    "ListNode",
    "TreeNode",
    "sortedcontainers",
    "SortedList",
    "SortedDict",
    "SortedSet",
)
"""Names that mark a problem needing linked lists, trees or
`sortedcontainers`, none of which the grader has."""
PRELUDE = """\
import random
import functools
import collections
import string
import math
import datetime
import bisect
import heapq
import itertools
import operator

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
its linked-list and tree helpers, and with the modules it only
star-imports also imported by name, as some references use them."""
READING = threading.Lock()
"""Held while a split is read, so episodes starting together in threads
read it once between them, not once each."""


@dataclass(frozen=True)
class Problem:
    """One problem: its id, statement, reference solution and test calls."""

    task_id: str
    statement: str
    solution: str
    calls: tuple[str, ...]


def usable(row: dict) -> bool:
    """Whether a row grades with the standard library alone, to plain values:
    no linked lists, trees or sorted containers.
    """
    text = row["test"] + row["completion"]
    return not any(name in text for name in UNAVAILABLE)


def problems(split: str) -> tuple[Problem, ...]:
    """Every usable problem of `split` ("train" or "val"), in dataset order,
    read once per process.
    """
    with READING:
        return read(split)


@lru_cache(maxsize=None)
def read(split: str) -> tuple[Problem, ...]:
    """Read the problems of `split` from its file."""
    path = folder() / FILES[split]
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing: fetch it on Mila's login node with "
            "`cluv run mila python -m envs.ShortPython`"
        )
    with open(path) as lines:
        rows = [json.loads(line) for line in lines if line.strip()]
    return tuple(
        Problem(
            row["task_id"],
            row["problem_description"].replace("\xa0", " ").strip() + "\n",
            row["completion"],
            tuple(
                f"{row['entry_point']}({case['input']})"
                for case in row["input_output"]
            ),
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
