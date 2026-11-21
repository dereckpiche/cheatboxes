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

Every call passes its arguments by position (`positional`), so an edit
may rename its parameters.

The files the agent edits hold other references: each problem's Python
and JavaScript solutions from `greengerong/leetcode` (MIT), which has
both for the problems it shares with LeetCodeDataset, joined by slug. Each
is called under its own function's name. `verify` keeps a language's
solution for a problem only if it returns the expected result on every
call, run in sandboxes (node is in the image, not on the host); for
JavaScript, the calls and results JSON cannot carry exactly (ints past
2**53, inf, nan, sets, dicts) are dropped first. A problem whose expected
results are all one value, which a stub can return, is left out.

The files live in the dataset directory (`envs.commons.datasets`).
Fetch them once on Mila's login node, `python -m envs.ShortPython`, then,
in an allocation, compute the expected results, `python -m
envs.ShortPython expect`, then verify the solutions, `python -m
envs.ShortPython verify`; `cluv sync` copies them everywhere else.
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
from envs.commons.programs import (
    PROGRAM_LIMIT,
    run_here,
    run_many,
    same,
    same_json,
)

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
REFERENCES_FILES = {
    "train": "LeetCodeDataset-train-references.jsonl",
    "val": "LeetCodeDataset-test-references.jsonl",
}
"""Per problem and language, the verified solution, its function's name,
and the calls it is graded on with the results each must return."""
SOLUTIONS = "greengerong/leetcode"
SOLUTIONS_NAME = "greengerong-leetcode"
SOLUTIONS_FILE = "leetcode-train.jsonl"
LANGUAGES = ("python", "javascript")
FENCE = re.compile(r"```(\w*)\n(.*?)```", re.S)
"""A solution cell's code: one fenced block, before its explanation."""
VERIFY_LIMIT = PROGRAM_LIMIT / 2
"""Seconds a verified solution may take for all its calls, sandbox start
included: graded edits get PROGRAM_LIMIT."""
JSON_INTS = 2**53
"""JavaScript numbers hold integers exactly only below this."""
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
class Reference:
    """A problem's solution in one language, as a file starts with: its
    function's name and text, and the hidden calls it is graded on with
    what each must return (a Python `repr`).
    """

    language: str
    name: str
    solution: str
    calls: tuple[str, ...]
    expected: tuple[str, ...]


@dataclass(frozen=True)
class Problem:
    """One problem: its id, statement, function name, reference function,
    difficulty, test calls and the reference's `repr` on each, and its
    verified solutions by language.
    """

    task_id: str
    statement: str
    name: str
    solution: str
    difficulty: str
    calls: tuple[str, ...]
    expected: tuple[str, ...] = ()
    references: tuple[Reference, ...] = ()

    def reference(self, language: str) -> Reference | None:
        """The verified solution in `language`, if any."""
        found = (r for r in self.references if r.language == language)
        return next(found, None)


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
    calls kept for them, passing arguments by position, and the verified
    solutions, if they have been verified.
    """
    expected = load_expected(split)
    verified = load_references(split)
    return tuple(
        replace(
            problem,
            calls=tuple(
                positional(call, problem.solution)
                for call in expected[problem.task_id]["calls"]
            ),
            expected=tuple(expected[problem.task_id]["expected"]),
            references=verified.get(problem.task_id, ()),
        )
        for problem in candidates(split)
        if problem.task_id in expected
    )


@lru_cache(maxsize=None)
def pool(
    split: str, language: str, difficulty: str | None
) -> tuple[Problem, ...]:
    """The problems of `split` and `difficulty` (None for any) with a
    verified solution in `language`.
    """
    return tuple(
        problem
        for problem in problems(split)
        if (difficulty is None or problem.difficulty == difficulty)
        and problem.reference(language)
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


def load_references(split: str) -> dict[str, tuple[Reference, ...]]:
    """The verified solutions of `split` by task id; none before `verify`
    has run.
    """
    path = folder() / REFERENCES_FILES[split]
    if not path.exists():
        return {}
    found: dict[str, list[Reference]] = {}
    with open(path) as lines:
        for line in lines:
            if line.strip():
                row = json.loads(line)
                found.setdefault(row["task_id"], []).append(
                    Reference(
                        row["language"],
                        row["name"],
                        row["solution"],
                        tuple(row["calls"]),
                        tuple(row["expected"]),
                    )
                )
    return {task_id: tuple(found[task_id]) for task_id in found}


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


def code_block(cell: str) -> str | None:
    """The code of a solution cell: its one fenced block; None unless it
    has exactly one.
    """
    blocks = FENCE.findall(cell or "")
    return blocks[0][1].strip() + "\n" if len(blocks) == 1 else None


def entry(code: str, language: str, preferred: str) -> str | None:
    """The function a solution's calls call: its only top-level function,
    or the one named `preferred` among several; None if it has none or
    defines a class.
    """
    if language == "python":
        try:
            body = ast.parse(code).body
        except SyntaxError:
            return None
        if any(isinstance(node, ast.ClassDef) for node in body):
            return None
        found = [n.name for n in body if isinstance(n, ast.FunctionDef)]
    else:
        if re.search(r"^class\b", code, re.M):
            return None
        found = re.findall(r"^function\s+(\w+)\s*\(", code, re.M)
    if len(found) == 1:
        return found[0]
    return preferred if preferred in found else None


def renamed_call(call: str, name: str) -> str:
    """`call` calling `name` instead of the function it names."""
    return name + call[call.index("(") :]


def representable(value) -> bool:
    """Whether JSON carries `value` exactly, as JavaScript reads it."""
    if value is None or isinstance(value, (bool, str)):
        return True
    if isinstance(value, int):
        return abs(value) < JSON_INTS
    if isinstance(value, float):
        return value == value and abs(value) != float("inf")
    if isinstance(value, (list, tuple)):
        return all(representable(item) for item in value)
    return False


def javascript_call(call: str, name: str) -> str | None:
    """A positional Python call as JavaScript, calling `name`, its
    arguments as JSON; None if it has keywords or an argument JSON cannot
    carry.
    """
    tree = ast.parse(call, mode="eval").body
    if not isinstance(tree, ast.Call) or tree.keywords:
        return None
    try:
        values = [ast.literal_eval(argument) for argument in tree.args]
    except ValueError:
        return None
    if not all(representable(value) for value in values):
        return None
    return f"{name}({', '.join(json.dumps(value) for value in values)})"


def solution_rows() -> dict[str, dict]:
    """greengerong/leetcode's rows by slug."""
    path = datasets.root() / SOLUTIONS_NAME / SOLUTIONS_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing: fetch it on Mila's login node with "
            "`cluv run mila python -m envs.ShortPython`"
        )
    with open(path) as lines:
        rows = [json.loads(line) for line in lines if line.strip()]
    return {row["slug"]: row for row in rows}


def candidate(problem: Problem, row: dict, language: str) -> dict | None:
    """A solution of `problem` in `language` with the calls to grade it on
    and their expected results, before it is run; None if it has no single
    entry function, no call is left, or its results are all one value.
    """
    code = code_block(row.get(language, ""))
    name = code and entry(code, language, problem.name)
    if not name:
        return None
    pairs = []
    for call, want in zip(problem.calls, problem.expected):
        if language == "javascript":
            call = javascript_call(call, name)
            try:
                if call is None or not representable(ast.literal_eval(want)):
                    continue
            except ValueError:
                continue
        else:
            call = renamed_call(call, name)
        pairs.append((call, want))
    if len({want for _, want in pairs}) < 2:
        return None
    calls, expected = zip(*pairs)
    return {
        "task_id": problem.task_id,
        "language": language,
        "name": name,
        "solution": code,
        "calls": list(calls),
        "expected": list(expected),
    }


def program(text: str, language: str) -> str:
    """What runs for a file: a Python file after the header's imports, a
    JavaScript file as it is.
    """
    return PRELUDE + text if language == "python" else text


def reproduces(row: dict, outputs: list[str | None] | None) -> bool:
    """Whether a solution's outputs are the expected results, every one."""
    compare = same_json if row["language"] == "javascript" else same
    return outputs is not None and all(
        compare(got, want) for got, want in zip(outputs, row["expected"])
    )


def verify(split: str, batch: int = 64) -> None:
    """Run every candidate solution of `split`, twice, in sandboxes, and
    write those that return the expected results both times; run it in an
    allocation, never on a login node.
    """
    rows = solution_rows()
    found = [
        made
        for problem in problems(split)
        if problem.task_id in rows
        for language in LANGUAGES
        if (made := candidate(problem, rows[problem.task_id], language))
    ]
    kept = []
    for start in range(0, len(found), batch):
        chunk = found[start : start + batch]
        jobs = [
            (program(r["solution"], r["language"]), r["calls"], r["language"])
            for r in chunk
        ]
        first = run_many(jobs, timeout=VERIFY_LIMIT)
        second = run_many(jobs, timeout=VERIFY_LIMIT)
        kept += [
            row
            for row, a, b in zip(chunk, first, second)
            if reproduces(row, a) and reproduces(row, b)
        ]
    with open(folder() / REFERENCES_FILES[split], "w") as file:
        for row in kept:
            file.write(json.dumps(row) + "\n")
    for language in LANGUAGES:
        tried = sum(r["language"] == language for r in found)
        good = sum(r["language"] == language for r in kept)
        print(f"{split} {language}: {good} of {tried} solutions verified")


@lru_cache(maxsize=None)
def by_id(split: str) -> dict[str, Problem]:
    """The problems of `split`, keyed by task id."""
    return {problem.task_id: problem for problem in problems(split)}


def folder() -> Path:
    """The dataset's folder in the dataset directory."""
    return datasets.root() / NAME


def fetch() -> None:
    """Download both datasets' files into the dataset directory."""
    from huggingface_hub import snapshot_download

    for dataset, target in (
        (DATASET, folder()),
        (SOLUTIONS, datasets.root() / SOLUTIONS_NAME),
    ):
        snapshot_download(
            dataset,
            repo_type="dataset",
            local_dir=target,
            allow_patterns="*.jsonl",
        )
        print(f"{dataset} -> {target}")
