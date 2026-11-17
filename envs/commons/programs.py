"""Run Python programs on calls in sandboxes and compare their results.
Each program runs in its own fresh sandbox, all at once (`run_many`), so
none can touch another's results and one slow program delays no other.
The harness writes each call's `repr`, None where the call raised.
"""

from __future__ import annotations

import ast
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from envs.commons.sandbox import Sandbox

HARNESS = Path(__file__).with_name("harness.py")


def run_many(
    jobs: list[tuple[str, list[str]]], timeout: int = 10
) -> list[list[str | None]]:
    """Evaluate each (code, calls) job in its own fresh sandbox, all at
    once; the reprs, None where a call raised.
    """
    with ThreadPoolExecutor(max(1, len(jobs))) as pool:
        return list(
            pool.map(lambda job: run_one(job[0], job[1], timeout), jobs)
        )


def run_one(code: str, calls: list[str], timeout: int) -> list[str | None]:
    """Evaluate `code` on `calls` in a fresh sandbox."""
    files = {
        "harness.py": HARNESS.read_text(),
        "solution.py": code,
        "calls.json": json.dumps(list(calls)),
    }
    sb = Sandbox(files)
    try:
        sb.run("python harness.py", timeout=timeout)
        return results(sb.read("results.json"), len(calls))
    finally:
        sb.close()


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
