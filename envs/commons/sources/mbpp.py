"""MBPP coding problems as a source: the question is the description plus
one example call, the answer is the reference solution, and the score is
the fraction of hidden test calls the answer passes when run in a sandbox.
Each MBPP test is `assert f(args) == value`; we keep the call, drop the
value, and recompute the expected output by running the reference through
the same grader, so grading never needs the original assert. Many
programs run in one sandbox, each in its own folder (`run_many`): one
container grades a whole pile of answers.
"""

from __future__ import annotations

import ast
import builtins
import json
import random
from pathlib import Path

from envs.commons.sandbox import Sandbox
from envs.commons.sources import Problem

SPLITS = ["train", "validation", "prompt", "test"]
HARNESS = Path(__file__).with_name("harness.py")
BUILTINS = set(dir(builtins))


class Mbpp:
    """`Mbpp()`: the google-research MBPP dataset, all of it."""

    def problems(self, n: int, seed: int) -> list[Problem]:
        """Up to `n` gradable problems, in a seeded order. The reference
        solutions run in one sandbox per chunk, until `n` have passed.
        """
        from datasets import load_dataset

        mbpp = load_dataset("google-research-datasets/mbpp", "full")
        rows = [r for s in SPLITS for r in mbpp[s]]
        random.Random(seed).shuffle(rows)
        candidates = [(row, tests) for row in rows if (tests := parse(row))]
        problems = []
        while candidates and len(problems) < n:
            chunk, candidates = candidates[: n + 16], candidates[n + 16 :]
            jobs = [(row["code"], calls) for row, (calls, _) in chunk]
            for (row, (calls, values)), got in zip(chunk, run_many(jobs)):
                if all(same(g, want) for g, want in zip(got, values)):
                    problems.append(problem(row, calls, got))
        return problems[:n]

    def score(self, spec: dict, answer: str) -> float:
        """The fraction of test calls whose result matches the expected repr."""
        return self.score_all([(spec, answer)])[0]

    def score_all(self, pairs: list[tuple[dict, str]]) -> list[float]:
        """`score` of each (spec, answer), all in one sandbox."""
        runs = run_many([(answer, spec["calls"]) for spec, answer in pairs])
        return [
            passed(spec, results) for (spec, _), results in zip(pairs, runs)
        ]


def problem(row: dict, calls: list[str], results: list[str]) -> Problem:
    """An MBPP row as a Problem: its text and first call as the question,
    the recomputed results as what the answer must reproduce.
    """
    question = f"{row['text']}\nIt will be called like `{calls[0]}`."
    spec = {"calls": calls, "expected": results}
    return Problem(question, row["code"], spec)


def passed(spec: dict, results: list[str | None]) -> float:
    """The fraction of a spec's calls whose result is the expected one."""
    hits = [same(got, want) for got, want in zip(results, spec["expected"])]
    return sum(hits) / len(hits)


def run_many(
    jobs: list[tuple[str, list[str]]], timeout: int = 10
) -> list[list[str | None]]:
    """Evaluate each (code, calls) job in one fresh sandbox, job i in folder
    i, one after another; the reprs, None where a call raised. Each job's
    results are read before the next one runs, so no program can rewrite
    another's.
    """
    if not jobs:
        return []
    harness, files = HARNESS.read_text(), {}
    for i, (code, calls) in enumerate(jobs):
        files[f"{i}/harness.py"] = harness
        files[f"{i}/solution.py"] = code
        files[f"{i}/calls.json"] = json.dumps(list(calls))
    sb = Sandbox(files)
    out = []
    try:
        for i, (_, calls) in enumerate(jobs):
            sb.run(f"cd {i} && python harness.py", timeout=timeout)
            out.append(results(sb.read(f"{i}/results.json"), len(calls)))
    finally:
        sb.close()
    return out


def results(raw: str | None, count: int) -> list[str | None]:
    """A results.json as `count` reprs, None where missing or unreadable."""
    try:
        found = json.loads(raw) if raw else []
    except ValueError:
        found = []
    if not isinstance(found, list):
        found = []
    found = [r if isinstance(r, str) else None for r in found]
    return (found + [None] * count)[:count]


def same(got: str | None, want: str) -> bool:
    """Compare two `repr` strings by Python equality when both parse as
    literals, else textually.
    """
    if got is None:
        return False
    try:
        return ast.literal_eval(got) == ast.literal_eval(want)
    except Exception:
        return got == want


def parse(row: dict) -> tuple[list[str], list[str]] | None:
    """An MBPP row's test calls and expected values, or None if its tests
    are not reproducible by the grader.
    """
    if row["test_setup_code"].strip():
        return None
    parts = [split_test(t) for t in row["test_list"]]
    if not parts or any(p is None for p in parts):
        return None
    calls, values = map(list, zip(*parts))
    entry = entry_point(calls[0])
    if entry is None or any(entry_point(c) != entry for c in calls):
        return None
    if any(names(c) - BUILTINS - {entry} for c in calls):
        return None
    return calls, values


def split_test(test: str) -> tuple[str, str] | None:
    """'assert f(x) == y' -> ('f(x)', 'y')."""
    try:
        body = ast.parse(test).body
    except SyntaxError:
        return None
    if len(body) != 1 or not isinstance(body[0], ast.Assert):
        return None
    cmp = body[0].test
    if not (
        isinstance(cmp, ast.Compare)
        and len(cmp.ops) == 1
        and isinstance(cmp.ops[0], ast.Eq)
    ):
        return None
    return ast.unparse(cmp.left), ast.unparse(cmp.comparators[0])


def entry_point(call: str) -> str | None:
    """Name of the function a test call invokes, or None if the call is not
    a plain `name(...)`.
    """
    node = ast.parse(call, mode="eval").body
    return (
        node.func.id
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        else None
    )


def names(expr: str) -> set[str]:
    """All identifiers used in an expression."""
    return {
        n.id
        for n in ast.walk(ast.parse(expr, mode="eval"))
        if isinstance(n, ast.Name)
    }
