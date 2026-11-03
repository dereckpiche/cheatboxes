"""Write an environment's tasks as the parquet files verl reads.

    python -m training.data --out data/PeekResist [--env PeekResist]
        [--config key=value ...] [-n 2000] [--seed 0]

`--config` goes straight to the environment's `generate` (PeekResist:
source, budget, cheat and the source's own keys; UniformChoice: source,
budget, tasks, turns, tells, exclusive). One parquet file per split the
environment gives (UniformChoice: train only). Each row holds the chat
that opens the episode and, in `extra_info`, the split, the environment
and its task as JSON, which the loop hands back to the environment's
`Episode`.
"""

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

import envs


def row(env: str, chat, task, index: int, split: str) -> dict:
    """One parquet row in the layout verl's dataset expects."""
    return {
        "data_source": env,
        "prompt": chat,
        "ability": "agent",
        "reward_model": {"style": "rule", "ground_truth": ""},
        "extra_info": {
            "index": index,
            "split": split,
            "env": env,
            "task": json.dumps(task),
        },
    }


def episode(extra_info: dict):
    """The environment's Episode for the task a row carries."""
    environment = envs.load(extra_info["env"])
    return environment.Episode(json.loads(extra_info["task"]))


def add_to_system(messages, text: str) -> list[dict]:
    """A row's chat with `text` appended to its system message, as a new
    paragraph; the row itself is left as it was.
    """
    return [
        (
            {**m, "content": f"{m['content']}\n\n{text}"}
            if m["role"] == "system"
            else m
        )
        for m in messages
    ]


def write(out: Path, env: str, n: int, seed: int, config: dict) -> None:
    """Write a parquet file per split of the environment's `generate`."""
    out.mkdir(parents=True, exist_ok=True)
    splits = envs.load(env).generate(n, seed, **config)
    for split, tasks in splits.items():
        rows = [row(env, *t, i, split) for i, t in enumerate(tasks)]
        pq.write_table(pa.Table.from_pylist(rows), out / f"{split}.parquet")
        print(f"{split}: {len(rows)} tasks -> {out / split}.parquet")


def option(item: str) -> tuple[str, object]:
    """`key=value`, the value read as JSON when it parses, else as text."""
    key, _, value = item.partition("=")
    try:
        return key, json.loads(value)
    except ValueError:
        return key, value


def main() -> None:
    """Parse the command line and write the dataset."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=Path("data/PeekResist"))
    parser.add_argument("--env", default="PeekResist")
    parser.add_argument("--config", nargs="*", default=[], metavar="KEY=VALUE")
    parser.add_argument("-n", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    config = dict(option(item) for item in args.config)
    write(args.out, args.env, args.n, args.seed, config)


if __name__ == "__main__":
    main()
