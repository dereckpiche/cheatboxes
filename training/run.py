"""Train with one method on the environment the dataset holds.

    EXP=<name> DATA=<dataset dir> python -m training.run <method> [overrides]

A method is a file in training/methods/ that defines `OVERRIDES`, and
optionally an `EpisodeLoop` subclass, which becomes the run's agent loop.
verl starts with the recipe (training/train.yaml), then the method's
`OVERRIDES`, then the command line's, so the last word is yours. The run
directory (RUN_DIR, default runs/<EXP>) gets the checkpoints, the agent
loop's file (loops.yaml), verl's full output (verl.log), one line per
episode (episodes.jsonl) and the UniformChoice forum (forum/); the console
and wandb get a few lines per step (see training.progress).
"""

import importlib
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

from training import progress
from training.loop import EpisodeLoop

CONFIG = Path(__file__).resolve().parent
ROOT = CONFIG.parent


def method(name: str):
    """The method called `name`."""
    if importlib.util.find_spec(f"training.methods.{name}"):
        module = importlib.import_module(f"training.methods.{name}")
        if hasattr(module, "OVERRIDES"):
            return module
    raise SystemExit(f"no method to train with is called {name}")


def loop(module) -> str:
    """The method's agent loop as an import path: the `EpisodeLoop`
    subclass its file defines, else `EpisodeLoop` itself.
    """
    for value in vars(module).values():
        if (
            isinstance(value, type)
            and issubclass(value, EpisodeLoop)
            and value.__module__ == module.__name__
        ):
            return f"{module.__name__}.{value.__name__}"
    return "training.loop.EpisodeLoop"


def write_loops(name: str, path: Path) -> None:
    """The agent loop file verl reads: one loop, the method's, by its name."""
    path.write_text(f"- name: {name}\n  _target_: {loop(method(name))}\n")


def validation(data: Path) -> list[str]:
    """No validation phase when the dataset has no val split: the environment
    then measures what it needs on the training episodes.
    """
    if (data / "val.parquet").exists():
        return []
    return [
        f"data.val_files={data / 'train.parquet'}",
        "trainer.val_before_train=false",
        "trainer.test_freq=-1",
    ]


def command(name: str, overrides: list[str], loops: Path) -> list[str]:
    """The verl command line that trains with the method `name`, its agent
    loop listed in the file `loops`.
    """
    module = method(name)
    data = Path(os.environ.get("DATA", "data/PeekResist")).resolve()
    rollout = "actor_rollout_ref.rollout"
    return [
        sys.executable,
        "-m",
        "verl.trainer.main_ppo",
        f"--config-path={CONFIG}",
        "--config-name=train",
        f"{rollout}.multi_turn.tool_config_path={CONFIG / 'tools.yaml'}",
        f"{rollout}.agent.agent_loop_config_path={loops}",
        f"{rollout}.agent.default_agent_loop={name}",
        *validation(data),
        *module.OVERRIDES,
        *overrides,
    ]


def main() -> None:
    """Run verl in a child process whose workers can import this repo, and
    follow its output.
    """
    name, *overrides = sys.argv[1:]
    os.environ.setdefault("EXP", name)
    run_dir = Path(os.environ.get("RUN_DIR", f"runs/{os.environ['EXP']}"))
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    os.environ["RUN_DIR"] = str(run_dir)
    os.environ["EPISODE_LOG"] = str(run_dir / "episodes.jsonl")
    os.environ["UNIFORMCHOICE_FORUM"] = str(run_dir / "forum")
    write_loops(name, run_dir / "loops.yaml")
    paths = [str(ROOT), os.environ.get("PYTHONPATH", "")]
    os.environ["PYTHONPATH"] = os.pathsep.join(p for p in paths if p)
    verl = subprocess.Popen(
        command(name, overrides, run_dir / "loops.yaml"),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    with open(run_dir / "verl.log", "a", buffering=1) as raw:
        progress.follow(verl.stdout, raw, os.environ["EPISODE_LOG"])
    sys.exit(verl.wait())


if __name__ == "__main__":
    main()
