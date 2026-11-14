"""One line per eval run: how far the model sped SafeSpeedups libraries up
and what it broke, from each run directory's run.json and episodes.jsonl.

    python scripts/summarize_evals.py [run dirs ...]   (default: logs/*)
"""

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

COLUMNS = {
    "episodes": "{:d}",
    "reward": "{:.3f}",
    "strict": "{:.3f}",
    "best_of_n": "{:.3f}",
    "sped_up": "{:.0%}",
    "format_ok": "{:.0%}",
    "broken": "{:.2f}",
    "cost_ratio": "{:.3f}",
    "submitted": "{:.0%}",
    "at_budget": "{:.0%}",
    "tokens": "{:.0f}",
    "thinking": "{:.0f}",
    "calls": "{:.1f}",
}
"""The summary's columns and how each is printed."""


def summary(records: list[dict], budget: int) -> dict[str, float]:
    """The run's numbers: mean reward (the relative saving, nothing broken),
    the same under the all-or-nothing rule, its best over each library's
    samples, the share of episodes with any
    saving, format and breakage, cost after over before, submits, episodes
    that spent the whole output budget, and mean tokens and tool calls.
    """
    by_task = defaultdict(list)
    for record in records:
        by_task[record["task"]].append(record["reward_task"])
    metric = lambda name: [r["metrics"].get(name, 0.0) for r in records]
    mean = statistics.fmean
    return {
        "episodes": len(records),
        "reward": mean(r["reward_task"] for r in records),
        "strict": mean(strict(r) for r in records),
        "best_of_n": mean(max(rewards) for rewards in by_task.values()),
        "sped_up": mean(r["reward_task"] > 0 for r in records),
        "format_ok": mean(metric("format_ok")),
        "broken": mean(metric("verified_broken")),
        "cost_ratio": mean(metric("cost_ratio")),
        "submitted": mean(metric("submitted")),
        "at_budget": mean(r["output_tokens"] >= 0.98 * budget for r in records),
        "tokens": mean(r["output_tokens"] for r in records),
        "thinking": mean(r["reasoning_tokens"] for r in records),
        "calls": mean(r["tool_calls"] for r in records),
    }


def strict(record: dict) -> float:
    """The episode's reward under the all-or-nothing rule (2026-11-13):
    zero if any verified function broke. Runs graded before the rule only
    zeroed the broken function's share."""
    broken = record["metrics"].get("verified_broken", 0.0)
    return record["reward_task"] if broken == 0 else 0.0


def budget_of(run: dict) -> int:
    """The episode budget the run's overrides set, else SafeSpeedups'."""
    for override in run.get("overrides", []):
        if override.startswith("+data.env_config.budget="):
            return int(override.split("=", 1)[1])
    return 4000


def main() -> None:
    """Print one line per run directory that holds episodes."""
    dirs = [Path(d) for d in sys.argv[1:]] or sorted(Path("logs").glob("*"))
    header = "".join(f"{name:>11}" for name in COLUMNS)
    print(f"{'run':<40}{header}")
    for folder in dirs:
        episodes, config = folder / "episodes.jsonl", folder / "run.json"
        if not episodes.exists() or not config.exists():
            continue
        run = json.loads(config.read_text())
        with open(episodes) as file:
            records = [json.loads(line) for line in file if line.strip()]
        if not records:
            continue
        numbers = summary(records, budget_of(run))
        cells = "".join(
            f"{fmt.format(numbers[name]):>11}" for name, fmt in COLUMNS.items()
        )
        print(f"{run['exp'][:40]:<40}{cells}")


if __name__ == "__main__":
    main()
