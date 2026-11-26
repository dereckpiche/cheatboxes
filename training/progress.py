"""Turn verl's console output and the episode log into a few lines per
step, and the same numbers on the run's dashboard (training.results).

verl prints one long `step:N - key:value - ...` line per training step and
reports per-episode numbers only at validation. So the loop appends each
finished episode to a JSON-lines file, and at every step line we average
the episodes that arrived since the last one. These groups reach the
console, the run's log and its dashboard, nothing else:

    <env>, <env>_valid             one tab per environment, named after it
                                   (e.g. ShortPython), for its training
                                   and, if it has any, validation episodes:
                                   reward_task (the environment's grade), the
                                   framework's metrics (honest, hacked, ...)
                                   and the environment's own; then
                                   tool calls and tokens; and `episode`, one
                                   random episode per step as a page
                                   (training.render)
    training                       the update: entropy, pg_loss, grad_norm,
                                   clip fraction, ppo_kl,
                                   the advantages' range, learning rate,
                                   rollout/training mismatch, and
                                   groups_with_signal, the share of GRPO
                                   groups whose rewards differ
    time_and_memory                where the step's time went, peak memory
"""

import json
import os
import random
import re
from typing import Callable, Iterable, TextIO

from training import render
from training.results import Results

STEP = re.compile(r"step:(\d+) - (.*)")
ANSI = re.compile(r"\x1b\[[0-9;]*m")
RAY_PREFIX = re.compile(r"^(\(\w+ pid=\d+\)\s*)+")
SHOWN = (
    "Traceback (most recent call last)",
    "Error executing job",
    "Out Of Memory",
    "Training from scratch",
    "Resuming from",
)
REWARD = {"reward_task": "task {:.2f}"}
BEHAVIOUR = {
    "tool_calls": "calls {:.1f}",
    "output_tokens": "out {:.0f}",
    "reasoning_tokens": "think {:.0f}",
    "input_tokens": "in {:.0f}",
}
TRAINING = {
    "entropy": ("actor/entropy", "entropy {:.3f}"),
    "pg_loss": ("actor/pg_loss", "pg_loss {:.4f}"),
    "grad_norm": ("actor/grad_norm", "grad_norm {:.2f}"),
    "clip_fraction": ("actor/pg_clipfrac", "clip {:.3f}"),
    "ppo_kl": ("actor/ppo_kl", "ppo_kl {:.4f}"),
    "advantage_max": ("critic/advantages/max", "adv_max {:.2f}"),
    "advantage_min": ("critic/advantages/min", "adv_min {:.2f}"),
    "learning_rate": ("actor/lr", "lr {:.1e}"),
    "mismatch": ("rollout_corr/kl", "mismatch {:.4f}"),
}
TIME_AND_MEMORY = {
    "generation_seconds": ("timing_s/gen", "generation {:.0f}s"),
    "training_seconds": ("timing_s/update_actor", "training {:.0f}s"),
    "step_seconds": ("timing_s/step", "step {:.0f}s"),
    "validation_seconds": ("timing_s/testing", "validation {:.0f}s"),
    "gpu_allocated_gb": ("actor/perf/max_memory_allocated_gb", "gpu {:.1f}GB"),
    "gpu_reserved_gb": (
        "actor/perf/max_memory_reserved_gb",
        "reserved {:.1f}GB",
    ),
    "cpu_used_gb": ("actor/perf/cpu_memory_used_gb", "cpu {:.0f}GB"),
}
FORMATS = {
    "groups_with_signal": "signal {:.2f}",
    **REWARD,
    **BEHAVIOUR,
    **{name: fmt for name, (_, fmt) in (TRAINING | TIME_AND_MEMORY).items()},
}


class EpisodeLog:
    """Reads the episode log from where the previous read stopped."""

    def __init__(self, path: str):
        """Start at the end of whatever the file already holds."""
        self.path = path
        self.offset = os.path.getsize(path) if os.path.exists(path) else 0

    def drain(self) -> list[dict]:
        """The records appended since the last call, oldest first."""
        if not os.path.exists(self.path):
            return []
        with open(self.path) as file:
            file.seek(self.offset)
            lines = file.readlines()
            self.offset = file.tell()
        return [json.loads(line) for line in lines if line.strip()]


def parse(body: str) -> dict[str, float]:
    """The numbers in the body of verl's step line."""
    values = {}
    for item in body.split(" - "):
        key, _, value = item.partition(":")
        try:
            values[key] = float(re.sub(r"np\.\w+\(|\)", "", value))
        except ValueError:
            continue
    return values


def mean(values: list[float]) -> float:
    """The average of a non-empty list."""
    return sum(values) / len(values)


def averages(found: list[dict]) -> dict[str, float]:
    """The mean of every value in a list of name -> number dicts, by name,
    over the dicts that have it.
    """
    names = dict.fromkeys(k for f in found for k in f)
    return {n: mean([f[n] for f in found if n in f]) for n in names}


def tab(record: dict) -> str:
    """The group an episode is logged under: its environment's name, with
    `_valid` for a validation episode.
    """
    if record["split"] == "train":
        return record["env"]
    return f"{record['env']}_valid"


def tabs(records: list[dict]) -> dict[str, list[dict]]:
    """The records by group, training groups first."""
    ordered = sorted(records, key=lambda r: r["split"] != "train")
    found = {}
    for record in ordered:
        found.setdefault(tab(record), []).append(record)
    return found


def rollout(mine: list[dict]) -> dict[str, float]:
    """One group's averages: the task reward, the environment's metrics,
    then the behaviour.
    """
    numbers = {"reward_task": mean([r["reward_task"] for r in mine])}
    numbers |= averages([r["metrics"] for r in mine])
    numbers |= {k: mean([r[k] for r in mine]) for k in BEHAVIOUR}
    return numbers


def signal(records: list[dict]) -> float | None:
    """The share of the step's GRPO groups (training episodes of one task)
    whose rewards differ: the others have no advantage and teach nothing.
    None without training episodes.
    """
    groups = {}
    for r in records:
        if r["split"] == "train":
            groups.setdefault(r["task"], set()).add(r["reward_task"])
    if not groups:
        return None
    return mean([float(len(rewards) > 1) for rewards in groups.values()])


def report(verl: dict, records: list[dict]) -> dict[str, dict[str, float]]:
    """The step's numbers by group; a group with nothing to report is left
    out.
    """
    groups = {name: rollout(mine) for name, mine in tabs(records).items()}
    training = {k: verl[key] for k, (key, _) in TRAINING.items() if key in verl}
    if (share := signal(records)) is not None:
        training["groups_with_signal"] = share
    if training:
        groups["training"] = training
    memory = TIME_AND_MEMORY.items()
    if found := {k: verl[key] for k, (key, _) in memory if key in verl}:
        groups["time_and_memory"] = found
    return groups


def lines(step: int, groups: dict[str, dict[str, float]]) -> list[str]:
    """One console line per group."""
    out = []
    for group, numbers in groups.items():
        text = "  ".join(
            FORMATS.get(name, name + " {:.3f}").format(value)
            for name, value in numbers.items()
        )
        out.append(f"step {step:>4}  {group:<19}  {text}")
    return out


SPLIT_NAMES = {"train": "training", "val": "validation"}


def meta(step: int, record: dict) -> dict[str, str]:
    """The facts on top of an episode's page: where, when, how much."""
    return {
        "environment": record["env"],
        "step": str(step),
        "split": SPLIT_NAMES[record["split"]],
        "tool calls": f"{record['tool_calls']:g}",
        "tokens written": f"{record['output_tokens']:,}",
        "thinking": f"{record['reasoning_tokens']:,}",
    }


def pages(
    records: list[dict], step: int, say: Callable[[str], None] = print
) -> dict[str, str]:
    """Per group, one random episode as a page (training.render), as
    `<group>/episode`: one whose record kept its tokens, if any did, for
    the page's token view. A page that fails to render is skipped, and
    `say` says so: a display bug must never stop training.
    """
    pages = {}
    for name, mine in tabs(records).items():
        pick = random.choice([r for r in mine if r.get("tokens")] or mine)
        try:
            pages[f"{name}/episode"] = render.page(
                pick["transcript"],
                meta(step, pick),
                pick["reward_task"],
                pick["metrics"],
                pick.get("tokens"),
            )
        except Exception as error:
            say(f"episode page for {name} skipped: {error!r}")
    return pages


def follow(
    stream: Iterable[str], raw: TextIO, episodes: str, results: Results
) -> None:
    """Copy verl's output to `raw`; say (`results.say`: the console and
    the run's log) the step lines and errors; send each step's numbers and
    episode pages to `results`.
    """
    log = EpisodeLog(episodes)
    for line in stream:
        raw.write(line)
        text = RAY_PREFIX.sub("", ANSI.sub("", line.rstrip()))
        match = STEP.search(text)
        if not match:
            if "WARNING" not in text and any(m in text for m in SHOWN):
                results.say(text[:300])
            continue
        step, records = int(match[1]), log.drain()
        groups = report(parse(match[2]), records)
        for printed in lines(step, groups):
            results.say(printed)
        numbers = {
            f"{group}/{name}": value
            for group, numbers in groups.items()
            for name, value in numbers.items()
        }
        results.log(numbers, pages(records, step, results.say), step)
