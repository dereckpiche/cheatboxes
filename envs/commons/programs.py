"""Run Python or JavaScript programs on calls and compare their results.
Graded programs each run in their own fresh sandbox, all at once
(`run_many`), so none can touch another's results and one slow program
delays no other; trusted Python reference code can run without one
(`run_here`). The harness writes each call's result, None where the call
raised or ran past its limit: its `repr` for Python (harness.py), its
JSON for JavaScript (harness.js), which answers None for every call of a
program that fails to load or does not define the function called.
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

HARNESSES = {
    "python": Path(__file__).with_name("harness.py"),
    "javascript": Path(__file__).with_name("harness.js"),
}
EXTENSIONS = {"python": ".py", "javascript": ".js"}
INTERPRETERS = {"python": "python", "javascript": "node"}
CALL_LIMIT = 1.0
"""CPU seconds one call may use before it counts as raised."""
PROGRAM_LIMIT = 4.0
"""Seconds a graded program gets for all its calls."""
START_SLACK = 1.0
"""Seconds past a program's limit `run_many` waits for its sandbox to start
and stop before counting the program as answering nothing."""


def function_called(calls: list[str]) -> str:
    """The name the calls call, `name` in `name(...)`."""
    return calls[0].split("(", 1)[0].strip() if calls else ""


def harness_arguments(
    stop: bool, language: str = "python", calls: list[str] = ()
) -> list[str]:
    """The harness's script and arguments, after the interpreter; the
    JavaScript harness also takes the function the calls call.
    """
    name = [function_called(calls)] if language == "javascript" else []
    script = HARNESSES[language].name
    return [script, str(CALL_LIMIT), *name, *(["stop"] if stop else [])]


def harness_files(
    code: str, calls: list[str], language: str = "python"
) -> dict[str, str]:
    """The files the harness reads: itself, the program and its calls."""
    return {
        HARNESSES[language].name: HARNESSES[language].read_text(),
        f"solution{EXTENSIONS[language]}": code,
        "calls.json": json.dumps(list(calls)),
    }


def run_many(
    jobs: list[tuple],
    timeout: float = PROGRAM_LIMIT,
    stop: bool = False,
) -> list[list[str | None] | None]:
    """Evaluate each (code, calls) job, or (code, calls, language) with
    "python" (the default) or "javascript", in its own fresh sandbox, all at
    once; each call's result, None where it raised. A JavaScript job answers
    None instead of a list when its program does not load or does not
    define the function its calls call. Returns within `timeout` +
    START_SLACK whatever the programs do: a job not finished by then answers
    None for every call. With `stop`, a program ends at its first failing
    call.
    """
    jobs = [(job[0], list(job[1]), *job[2:3]) for job in jobs]
    pool = ThreadPoolExecutor(max(1, len(jobs)))
    futures = [
        pool.submit(run_one, *job, timeout=timeout, stop=stop) for job in jobs
    ]
    wait(futures, timeout=timeout + START_SLACK)
    pool.shutdown(wait=False, cancel_futures=True)
    return [
        future.result() if future.done() else [None] * len(job[1])
        for future, job in zip(futures, jobs)
    ]


def run_one(
    code: str,
    calls: list[str],
    language: str = "python",
    *,
    timeout: float = PROGRAM_LIMIT,
    stop: bool = False,
) -> list[str | None] | None:
    """Evaluate `code` on `calls` in a fresh sandbox; None for a
    JavaScript program that does not define the function called.
    """
    sb = Sandbox(harness_files(code, calls, language))
    try:
        arguments = harness_arguments(stop, language, calls)
        sb.run(" ".join([INTERPRETERS[language], *arguments]), timeout)
        if language == "javascript" and sb.read("defined.json") == "false":
            return None
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


def same_json(got: str | None, want: str) -> bool:
    """Compare a JavaScript result, as JSON, with a Python `repr`: equal
    when the JSON value equals the literal, tuples read as lists.
    """
    if got is None:
        return False
    try:
        return json.loads(got) == as_json(ast.literal_eval(want))
    except Exception:
        return False


def as_json(value):
    """`value` as JSON would give it back: tuples as lists, throughout."""
    if isinstance(value, (list, tuple)):
        return [as_json(item) for item in value]
    if isinstance(value, dict):
        return {key: as_json(item) for key, item in value.items()}
    return value


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
