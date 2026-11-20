"""Upload a run's offline wandb chunks to wandb under their real step
numbers.

On clusters whose compute nodes cannot reach wandb, every chunk of a run
logs offline, to one fragment each (RUN_DIR/wandb/offline-run-*). `wandb
sync` renumbers a resumed fragment's steps: a chunk that resumed at step
101 is uploaded as steps 645 on, even into a new run. So this replays the
fragments instead: it reads each one's history, keeps the latest attempt
of every step (a resumed chunk redoes the steps since its checkpoint), and
logs the steps the online run does not have yet, episode pages included.
Run it again to add the steps logged since; steps it uploaded before stay
as they were. wandb never reuses a deleted run's id: to rebuild a deleted
run, give it a new one with --id.

    python scripts/replay_wandb.py RUN_DIR [--id ID] [--dry-run]

Run it where wandb is reachable (a login node); it exits when done.
"""

import argparse
import json
import os
from pathlib import Path

BLOCK = 32768
"""A .wandb file is a leveldb-style log: 32 kB blocks of records."""
HEADER = 7
"""The bytes before each record (checksum, length, type), and the file's
own header (`:W&B`, magic, version)."""
FULL, FIRST, MIDDLE, LAST = 1, 2, 3, 4
"""How a record lies across blocks: whole, or its first, middle or last
piece."""


def records(path: Path):
    """The serialized records of a .wandb file, in order; a record the
    file ends in the middle of (a live run's) is left out.
    """
    data = path.read_bytes()
    if data[:4] != b":W&B":
        raise ValueError(f"{path} is not a .wandb file")
    position, pieces = HEADER, b""
    while True:
        left = BLOCK - position % BLOCK
        if left < HEADER:
            position += left
            continue
        if position + HEADER > len(data):
            return
        length = int.from_bytes(data[position + 4 : position + 6], "little")
        kind = data[position + 6]
        position += HEADER
        body = data[position : position + length]
        position += length
        if len(body) < length or kind == 0:
            return
        if kind == FULL:
            yield body
        elif kind == FIRST:
            pieces = body
        elif kind == MIDDLE:
            pieces += body
        elif kind == LAST:
            yield pieces + body
            pieces = b""


def fragment(directory: Path) -> tuple[dict, dict[int, dict]]:
    """A fragment's run (id, name, project) and its history by step."""
    from wandb.proto import wandb_internal_pb2 as pb

    run, steps = {}, {}
    for body in records(next(directory.glob("*.wandb"))):
        record = pb.Record()
        record.ParseFromString(body)
        kind = record.WhichOneof("record_type")
        if kind == "run":
            run = {
                "id": record.run.run_id,
                "name": record.run.display_name,
                "project": record.run.project,
            }
        elif kind == "history":
            row = {}
            for item in record.history.item:
                key = item.key or "/".join(item.nested_key)
                row[key] = json.loads(item.value_json)
            if "_step" in row:
                steps[int(row["_step"])] = row
    return run, steps


def value(item, files: Path):
    """What to log for one history value: the page itself for an html
    page, the number otherwise; None for other media.
    """
    if not isinstance(item, dict):
        return item
    if item.get("_type") == "html-file":
        import wandb

        return wandb.Html((files / item["path"]).read_text(), inject=False)
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--id", help="the online run's id (default: the run's)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run, steps, sources = {}, {}, {}
    for directory in sorted((args.run_dir / "wandb").glob("offline-run-*")):
        meta, history = fragment(directory)
        run = run or meta
        steps |= history
        sources |= {step: directory / "files" for step in history}
        print(f"{directory.name}: steps {min(history, default=None)}"
              f"-{max(history, default=None)}")
    if not steps:
        print("no history")
        return
    config_path = args.run_dir / "run.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    if args.dry_run:
        print(f"would log {len(steps)} steps to {args.id or run['id']}")
        return

    import wandb

    os.environ["WANDB_MODE"] = "online"
    online = wandb.init(
        project=run["project"],
        id=args.id or run["id"],
        name=run["name"],
        resume="allow",
        config=config,
        dir=args.run_dir,
    )
    online.config.update(config, allow_val_change=True)
    start = online.starting_step
    new = [step for step in sorted(steps) if step >= start]
    for step in new:
        row = {
            key: value(item, sources[step])
            for key, item in steps[step].items()
            if not key.startswith("_")
        }
        online.log({k: v for k, v in row.items() if v is not None}, step=step)
    print(f"logged steps {new[0] if new else '-'}-{new[-1] if new else '-'}"
          f" to {online.url}")
    online.finish()


if __name__ == "__main__":
    main()
