"""Runs inside the grading sandbox: import solution.py, evaluate each call from
calls.json, write results.json and seconds.json.

    python harness.py <call limit seconds> [stop]

With `stop`, the first call that raises or runs past the limit ends the run:
the grader then knows the program fails and needs nothing more from it.
"""

import json
import signal
import sys
import time
import traceback


def overtime(*_) -> None:
    """Stop a call that used more CPU time than its limit."""
    raise TimeoutError("call ran past its limit")


def main() -> None:
    """Execute the solution, then record `repr` of every call, or None where it
    raised or used more CPU time than the limit, so a busy machine fails no
    call. results.json is rewritten after every call, so a program killed
    midway keeps the answers it gave; seconds.json holds the time the calls
    took.
    """
    call_limit = float(sys.argv[1])
    stop = sys.argv[2:] == ["stop"]
    calls = json.load(open("calls.json"))
    results: list[str | None] = [None] * len(calls)
    json.dump(results, open("results.json", "w"))
    namespace = {"__name__": "__main__"}
    signal.signal(signal.SIGPROF, overtime)
    try:
        exec(
            compile(open("solution.py").read(), "solution.py", "exec"),
            namespace,
        )
    except BaseException:
        traceback.print_exc(limit=3)
        return
    start = time.perf_counter()
    for i, call in enumerate(calls):
        signal.setitimer(signal.ITIMER_PROF, call_limit)
        try:
            results[i] = repr(eval(call, namespace))
        except BaseException:
            traceback.print_exc(limit=3)
        finally:
            signal.setitimer(signal.ITIMER_PROF, 0)
        json.dump(results, open("results.json", "w"))
        if stop and results[i] is None:
            break
    json.dump(time.perf_counter() - start, open("seconds.json", "w"))


main()
