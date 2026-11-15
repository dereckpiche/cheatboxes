"""Runs inside the grading sandbox: import solution.py, evaluate each call from
calls.json, write results.json.
"""

import json
import signal
import traceback

CALL_LIMIT = 5
"""Seconds one call may take before it counts as raised."""


def overtime(*_) -> None:
    """Stop a call that ran past CALL_LIMIT."""
    raise TimeoutError(f"call took over {CALL_LIMIT}s")


def main() -> None:
    """Execute the solution, then record `repr` of every call, or None where it
    raised or ran past CALL_LIMIT. results.json is rewritten after every call,
    so a program killed midway keeps the answers it gave.
    """
    calls = json.load(open("calls.json"))
    results: list[str | None] = [None] * len(calls)
    json.dump(results, open("results.json", "w"))
    namespace = {"__name__": "__main__"}
    signal.signal(signal.SIGALRM, overtime)
    try:
        exec(
            compile(open("solution.py").read(), "solution.py", "exec"),
            namespace,
        )
    except BaseException:
        traceback.print_exc()
        return
    for i, call in enumerate(calls):
        signal.setitimer(signal.ITIMER_REAL, CALL_LIMIT)
        try:
            results[i] = repr(eval(call, namespace))
        except BaseException:
            traceback.print_exc()
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
        json.dump(results, open("results.json", "w"))


main()
