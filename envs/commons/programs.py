"""Run Python programs on calls and compare their results. Graded programs
each run in their own fresh sandbox, all at once (`run_many`), so none can
touch another's results and one slow program delays no other; trusted
reference code can run without one (`run_here`). The harness writes each
call's `repr`, None where the call raised or ran past its limit.
"""

from __future__ import annotations

import ast
import json
import os
import resource
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path

from envs.commons.container import ADDRESS_SPACE_KB
from envs.commons.sandbox import Sandbox

HARNESS = Path(__file__).with_name("harness.py")
CALL_LIMIT = 1.0
"""CPU seconds one call may use before it counts as raised."""
PROGRAM_LIMIT = 4.0
"""Seconds a graded program gets for all its calls."""
START_SLACK = 1.0
"""Seconds past a program's limit `run_many` waits for its sandbox to start
and stop before counting the program as answering nothing."""


def harness_arguments(stop: bool) -> list[str]:
    """The harness's script and arguments, after the interpreter."""
    return ["harness.py", str(CALL_LIMIT), *(["stop"] if stop else [])]


def harness_files(code: str, calls: list[str]) -> dict[str, str]:
    """The files the harness reads: itself, the program and its calls."""
    return {
        "harness.py": HARNESS.read_text(),
        "solution.py": code,
        "calls.json": json.dumps(list(calls)),
    }


def run_many(
    jobs: list[tuple[str, list[str]]],
    timeout: float = PROGRAM_LIMIT,
    stop: bool = False,
) -> list[list[str | None]]:
    """Evaluate each (code, calls) job in its own fresh sandbox, all at
    once; the reprs, None where a call raised. Returns within `timeout` +
    START_SLACK whatever the programs do: a job not finished by then answers
    nothing. With `stop`, a program ends at its first failing call.
    """
    pool = ThreadPoolExecutor(max(1, len(jobs)))
    futures = [
        pool.submit(run_one, code, calls, timeout, stop) for code, calls in jobs
    ]
    wait(futures, timeout=timeout + START_SLACK)
    pool.shutdown(wait=False, cancel_futures=True)
    return [
        future.result() if future.done() else [None] * len(calls)
        for future, (_, calls) in zip(futures, jobs)
    ]


def run_one(
    code: str, calls: list[str], timeout: float, stop: bool
) -> list[str | None]:
    """Evaluate `code` on `calls` in a fresh sandbox."""
    sb = Sandbox(harness_files(code, calls))
    try:
        sb.run(" ".join(["python", *harness_arguments(stop)]), timeout)
        return results(sb.read("results.json"), len(calls))
    finally:
        sb.close()


def run_here(
    code: str, calls: list[str], timeout: float, stop: bool = False
) -> tuple[list[str | None], float | None]:
    """Evaluate trusted `code` on `calls` in a local subprocess, no
    sandbox; the reprs and the seconds the calls took (None if it never
    finished).
    """
    with tempfile.TemporaryDirectory() as folder:
        work = Path(folder)
        for name, text in harness_files(code, calls).items():
            (work / name).write_text(text)
        try:
            subprocess.run(
                [sys.executable, *harness_arguments(stop)],
                cwd=work,
                capture_output=True,
                timeout=timeout,
                env={**os.environ, "PYTHONHASHSEED": "0"},
                preexec_fn=cap_memory,
            )
        except subprocess.TimeoutExpired:
            pass
        found = work / "results.json"
        timing = work / "seconds.json"
        raw = found.read_text() if found.exists() else None
        seconds = json.loads(timing.read_text()) if timing.exists() else None
        return results(raw, len(calls)), seconds


def cap_memory() -> None:
    """Cap the address space as the sandbox does, so one runaway reference
    cannot take the allocation's memory.
    """
    limit = ADDRESS_SPACE_KB * 1024
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))


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
