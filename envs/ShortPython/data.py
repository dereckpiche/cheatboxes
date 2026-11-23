"""LeetCode problems from `newfacade/LeetCodeDataset`: the reference
solution as a plain function, the statement, the difficulty, and the
hidden test calls with what the reference returns on each.

Each reference is a `class Solution` with one method; it becomes that
method as a top-level function, without `self`. Problems whose method
uses `self` or has siblings are left out. Each test input in the
dataset's `input_output` becomes the call `<function>(input)`; its
output is dropped: the reference's own `repr` on each call is what
grading expects, computed once (`expect`) and stored beside the dataset.
A call the reference cannot answer, or answers differently on two runs,
is dropped there, and so is a problem left with no call or whose
reference needs over REFERENCE_LIMIT seconds for its calls. Linked list
and tree problems are left out, so every result is a plain value compared
by `repr`, and so are the few that need `sortedcontainers`, so grading
needs only the standard library. The dataset's train split is ours; its
test split, dated later, is our val.

The files live in the dataset directory (`envs.commons.datasets`).
Fetch them once on Mila's login node, `python -m envs.ShortPython`, then
compute the expected results in an allocation, `python -m envs.ShortPython
expect`; `cluv sync` copies them everywhere else.
"""

from __future__ import annotations

import ast
import json
import os
import re
import textwrap
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

from envs.commons import datasets
from envs.commons.programs import run_here

DATASET = "newfacade/LeetCodeDataset"
NAME = "LeetCodeDataset"
"""The dataset's folder under the data directory."""
FILES = {
    "train": "LeetCodeDataset-train.jsonl",
    "val": "LeetCodeDataset-test.jsonl",
}
EXPECTED_FILES = {
    "train": "LeetCodeDataset-train-expected.jsonl",
    "val": "LeetCodeDataset-test-expected.jsonl",
}
"""Per problem, the calls kept and the reference's `repr` on each."""
REFERENCE_LIMIT = 0.5
"""Seconds a reference may take for all its calls: graded programs get
`programs.PROGRAM_LIMIT`, so an honest edit has room to spare even on a
busy node."""
REFERENCE_TIMEOUT = 30
"""Seconds one reference run may take while computing expected results."""
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
    """One problem: its id, statement, function name, reference function,
    difficulty, test calls and the reference's `repr` on each.
    """

    task_id: str
    statement: str
    name: str
    solution: str
    difficulty: str
    calls: tuple[str, ...]
    expected: tuple[str, ...] = ()


def usable(row: dict) -> bool:
    """Whether a row grades with the standard library alone, to plain values:
    no linked lists, trees or sorted containers.
    """
    text = row["test"] + row["completion"]
    return not any(name in text for name in UNAVAILABLE)


def as_function(completion: str) -> tuple[str, str] | None:
    """The one method of a `class Solution` as a top-level function, with
    its own text and comments, and its name; None unless the class holds
    exactly one method and nothing else, and the method never uses `self`.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            module = ast.parse(completion)
    except SyntaxError:
        return None
    if len(module.body) != 1 or not isinstance(module.body[0], ast.ClassDef):
        return None
    body = module.body[0].body
    if len(body) != 1 or not isinstance(body[0], ast.FunctionDef):
        return None
    method = body[0]
    arguments = method.args.posonlyargs + method.args.args
    if method.decorator_list or not arguments or arguments[0].arg != "self":
        return None
    if any(
        isinstance(node, ast.Name) and node.id == "self"
        for node in ast.walk(method)
    ):
        return None
    lines = completion.splitlines(keepends=True)
    text = textwrap.dedent(
        "".join(lines[method.lineno - 1 : method.end_lineno])
    )
    text = (
        re.sub(
            r"\(\s*self\s*(,\s*|\))",
            lambda m: "()" if m[1] == ")" else "(",
            text,
            count=1,
        ).rstrip()
        + "\n"
    )
    try:
        function = ast.parse(text).body
    except SyntaxError:
        return None
    if len(function) != 1 or not isinstance(function[0], ast.FunctionDef):
        return None
    return method.name, text


EXAMPLES = re.compile(
    r"^[ \t]*Example \d+:.*?(?=^[ \t]*Constraints:|\Z)", re.S | re.M
)
"""A statement's worked examples: from `Example 1:` to its constraints."""


def without_examples(description: str) -> str:
    """The dataset's statement without its worked examples: the problem,
    then its constraints and any follow-up, blank lines collapsed.
    """
    text = EXAMPLES.sub("", description.replace("\xa0", " "))
    text = re.sub(r"\n[ \t]*(?:\n[ \t]*)+", "\n\n", text)
    return text.strip() + "\n"


def problems(split: str) -> tuple[Problem, ...]:
    """Every gradable problem of `split` ("train" or "val"), in dataset
    order, with its expected results, read once per process.
    """
    with READING:
        return read(split)


@lru_cache(maxsize=None)
def read(split: str) -> tuple[Problem, ...]:
    """The candidates of `split` that have expected results, with only the
    calls kept for them, passing arguments by position.
    """
    expected = load_expected(split)
    return tuple(
        replace(
            problem,
            calls=tuple(
                positional(call, problem.solution)
                for call in expected[problem.task_id]["calls"]
            ),
            expected=tuple(expected[problem.task_id]["expected"]),
        )
        for problem in candidates(split)
        if problem.task_id in expected
    )


def positional(call: str, solution: str) -> str:
    """`call`, which the dataset writes `name(nums = [1], k = 2)`, with its
    arguments passed by position, in the order `solution`'s function takes
    them: graded edits may rename their parameters, which the prompt never
    forbids and which changes nothing a function returns. Arguments after
    one the call leaves to its default, and keyword-only ones, stay
    keywords. The results are the same, so the expected results stand.
    """
    tree = ast.parse(call, mode="eval").body
    if not isinstance(tree, ast.Call) or tree.args:
        return call
    given = {
        keyword.arg: ast.get_source_segment(call, keyword.value)
        for keyword in tree.keywords
    }
    if None in given or None in given.values():
        return call
    signature = ast.parse(solution).body[0].args
    arguments = []
    for parameter in signature.posonlyargs + signature.args:
        if parameter.arg not in given:
            break
        arguments.append(given.pop(parameter.arg))
    arguments += [f"{name}={value}" for name, value in given.items()]
    return f"{ast.get_source_segment(call, tree.func)}({', '.join(arguments)})"


@lru_cache(maxsize=None)
def candidates(split: str) -> tuple[Problem, ...]:
    """Every usable problem of `split` with all its dataset calls, before
    any is run.
    """
    path = folder() / FILES[split]
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing: fetch it on Mila's login node with "
            "`cluv run mila python -m envs.ShortPython`"
        )
    with open(path) as lines:
        rows = [json.loads(line) for line in lines if line.strip()]
    found = []
    for row in rows:
        if not usable(row) or not (function := as_function(row["completion"])):
            continue
        name, text = function
        found.append(
            Problem(
                row["task_id"],
                without_examples(row["problem_description"]),
                name,
                text,
                row["difficulty"],
                tuple(
                    f"{name}({case['input']})" for case in row["input_output"]
                ),
            )
        )
    return tuple(found)


def load_expected(split: str) -> dict[str, dict]:
    """The expected results of `split`, keyed by task id."""
    path = folder() / EXPECTED_FILES[split]
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing: compute it in an allocation on Mila with "
            "`python -m envs.ShortPython expect`"
        )
    with open(path) as lines:
        rows = [json.loads(line) for line in lines if line.strip()]
    return {row["task_id"]: row for row in rows}


def reference_results(problem: Problem) -> dict | None:
    """The calls the reference answers the same way on two runs, with its
    `repr` on each; None if none is left or the reference is too slow.
    """
    code = PRELUDE + problem.solution
    first, seconds = run_here(code, problem.calls, REFERENCE_TIMEOUT)
    second, _ = run_here(code, problem.calls, REFERENCE_TIMEOUT)
    kept = [
        (call, want)
        for call, want, again in zip(problem.calls, first, second)
        if want is not None and want == again
    ]
    if not kept or seconds is None or seconds > REFERENCE_LIMIT:
        return None
    calls, expected = zip(*kept)
    return {
        "task_id": problem.task_id,
        "calls": list(calls),
        "expected": list(expected),
    }


def expect(split: str) -> None:
    """Run every candidate reference of `split` and write the expected
    results; run it in an allocation, never on a login node.
    """
    workers = len(os.sched_getaffinity(0))
    pool = candidates(split)
    with ThreadPoolExecutor(workers) as executor:
        found = list(executor.map(reference_results, pool))
    kept = [row for row in found if row is not None]
    with open(folder() / EXPECTED_FILES[split], "w") as file:
        for row in kept:
            file.write(json.dumps(row) + "\n")
    print(f"{split}: {len(kept)} of {len(pool)} problems kept")


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
