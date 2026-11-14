"""LeetCode problems from `newfacade/LeetCodeDataset`: the reference
solution, the statement, and the hidden test calls.

Each test in the dataset is `assert candidate(args) == value`; we keep
the call as `Solution().<method>(args)` and drop the value, because the
grader recomputes the expected output by running the reference. Linked
list and tree problems are left out, so every result is a plain value
compared by `repr`. The dataset's train split is ours; its test split,
dated later, is our val.
"""

from __future__ import annotations

import ast
import json
import warnings
from dataclasses import dataclass
from functools import lru_cache

DATASET = "newfacade/LeetCodeDataset"
SPLITS = {"train": "train", "val": "test"}
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
    """Every usable problem of `split` ("train" or "val"), in dataset order,
    read from the Hugging Face cache, so it works offline once fetched.
    """
    from huggingface_hub import hf_hub_download

    name = f"LeetCodeDataset-{SPLITS[split]}.jsonl"
    path = hf_hub_download(DATASET, name, repo_type="dataset")
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
