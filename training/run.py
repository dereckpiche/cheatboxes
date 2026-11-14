"""Train with RL on an environment, its instances made as the run goes.

    EXP=<name> ENV=<environment> python -m training.run [overrides]

verl starts with the recipe (training/train.yaml), then the preset for the
GPU the job landed on (training/hardware/, HARDWARE=<preset> to force one),
then the command line's overrides, so the last word is yours. Presets
change only how the work fits on a GPU, never the experiment. The run
directory (RUN_DIR, default runs/<EXP>) gets the checkpoints, verl's full
output (verl.log), one line per episode (episodes.jsonl) and the
UniformChoice forum (forum/); the console and wandb get a few lines per
step (see training.progress).
"""

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import envs
from training import progress

CONFIG = Path(__file__).resolve().parent
ROOT = CONFIG.parent

MODEL = "Qwen/Qwen3.5-4B"
"""The model trained when MODEL is not set."""
MODEL_SIZES = {
    "Qwen/Qwen3.5-4B": "4b",
    "Qwen/Qwen3.5-9B": "9b",
    "Qwen/Qwen3.8-27B": "27b",
}
"""Each model's size, the last part of its presets' names."""
GPU_MODELS = ("L40S", "H100", "H200")
"""The GPUs with presets, as nvidia-smi names them."""


def gpus() -> list[str]:
    """Name every GPU this job sees, e.g. "NVIDIA H100 80GB HBM3"."""
    listing = subprocess.run(
        ["nvidia-smi", "-L"], capture_output=True, text=True, check=True
    ).stdout
    if "MIG" in listing:
        raise SystemExit(f"no preset fits a MIG slice:\n{listing}")
    lines = [line for line in listing.splitlines() if line.startswith("GPU ")]
    return [line.split(": ", 1)[1].split(" (UUID")[0] for line in lines]


def preset(found: list[str], model: str) -> str:
    """Name the preset for the GPUs `found` training `model`, e.g.
    "h100x4-4b" for four H100s and a 4B model.
    """
    kinds = [k for k in GPU_MODELS if k in found[0]]
    size = MODEL_SIZES.get(model)
    name = f"{kinds[0].lower()}x{len(found)}-{size}" if kinds and size else ""
    if not (CONFIG / "hardware" / f"{name}.yaml").is_file():
        raise SystemExit(
            f"no preset for {len(found)}x {found[0]} training {model}: add"
            " training/hardware/<gpu>x<count>-<size>.yaml (and the model to"
            " MODEL_SIZES)"
        )
    return name


def validation(env: str) -> list[str]:
    """No validation phase when the environment has no val split: it then
    measures what it needs on the training episodes.
    """
    if "val" in envs.load(env).SPLITS:
        return []
    return [
        "data.val_files=train",
        "trainer.val_before_train=false",
        "trainer.test_freq=-1",
    ]


def command(overrides: list[str], hardware: str) -> list[str]:
    """The verl command line that trains on the `hardware` preset."""
    rollout = "actor_rollout_ref.rollout"
    return [
        sys.executable,
        "-m",
        "verl.trainer.main_ppo",
        f"--config-path={CONFIG}",
        "--config-name=train",
        f"{rollout}.multi_turn.tool_config_path={CONFIG / 'tools.yaml'}",
        f"{rollout}.agent.agent_loop_config_path={CONFIG / 'loops.yaml'}",
        f"{rollout}.agent.default_agent_loop=episode",
        *validation(os.environ["ENV"]),
        f"+hardware={hardware}",
        *overrides,
    ]


STALL_MINUTES = 45
"""How long verl may go without logging an episode before it is stopped:
a dead vLLM engine leaves the job running at 0% GPU, and stopping it lets
the next chunk resume from the last checkpoint. A step takes ~8 minutes
on one L40S; start-up takes ~10."""


def watchdog(verl: subprocess.Popen, episodes: Path, minutes: float) -> None:
    """Kill `verl` once `episodes` has not grown for `minutes`."""
    size, last = -1, time.monotonic()
    while verl.poll() is None:
        time.sleep(min(60, minutes * 60 / 4))
        now = episodes.stat().st_size if episodes.exists() else 0
        if now != size:
            size, last = now, time.monotonic()
        elif time.monotonic() - last > minutes * 60:
            print(f"no episode for {minutes:g} min: stopping verl", flush=True)
            verl.kill()
            return


HYDRA_FAILURE = "Error executing job with overrides"
"""What Hydra prints when the run raised, even when verl then exits 0."""


def crashed(log: Path, start: int) -> int:
    """1 if this run's part of verl's log (from byte `start`) shows a
    failure, else 0."""
    with open(log) as file:
        file.seek(start)
        return int(HYDRA_FAILURE in file.read())


def checkout() -> str:
    """The commit the code runs from: cluv updates the checkout in place,
    so a job can run newer code than the commit it was submitted at."""
    found = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    )
    return found.stdout.strip()


def main() -> None:
    """Run verl in a child process whose workers can import this repo, and
    follow its output.
    """
    overrides = sys.argv[1:]
    os.environ.setdefault("EXP", "cheatboxes")
    os.environ.setdefault("MODEL", MODEL)
    os.environ.setdefault("ENV", "PeekResist")
    run_dir = Path(os.environ.get("RUN_DIR", f"runs/{os.environ['EXP']}"))
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    os.environ["RUN_DIR"] = str(run_dir)
    os.environ["EPISODE_LOG"] = str(run_dir / "episodes.jsonl")
    os.environ["UNIFORMCHOICE_FORUM"] = str(run_dir / "forum")
    found = gpus()
    model = os.environ["MODEL"]
    hardware = os.environ.get("HARDWARE") or preset(found, model)
    print(
        f"hardware {hardware} ({len(found)}x {found[0]}, {model})", flush=True
    )
    paths = [str(ROOT), os.environ.get("PYTHONPATH", "")]
    os.environ["PYTHONPATH"] = os.pathsep.join(p for p in paths if p)
    verl = subprocess.Popen(
        command(overrides, hardware),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    config = {
        "exp": os.environ["EXP"],
        "model": model,
        "env": os.environ["ENV"],
        "overrides": overrides,
        "hardware": hardware,
        "gpu": found[0],
        "gpus": len(found),
        "commit": checkout(),
    }
    (run_dir / "run.json").write_text(json.dumps(config, indent=2) + "\n")
    threading.Thread(
        target=watchdog,
        args=(verl, Path(os.environ["EPISODE_LOG"]), STALL_MINUTES),
        daemon=True,
    ).start()
    with open(run_dir / "verl.log", "a", buffering=1) as raw:
        start = raw.tell()
        progress.follow(verl.stdout, raw, os.environ["EPISODE_LOG"], config)
    code = verl.wait()
    sys.exit(code or crashed(run_dir / "verl.log", start))


if __name__ == "__main__":
    main()
