"""The run's results on Hugging Face (in place of wandb): its numbers and
episode pages on a trackio dashboard, its own log and config on the
dashboard's Run info page.

    https://huggingface.co/spaces/dereckpiche/cheatboxes_results

The dashboard is a static Space. Its bucket (same id) holds

    sources/<run>/<part>.db         one job's trackio database for the run
    media/<run>/<step>/<id>.html    the run's episode pages
    runs/<run>/<part>.json          the job: commit, command, cluster, host,
                                    and the run's whole config
    runs/<run>/<part>.log           the job's log: the lines this code
                                    prints on purpose, nothing from libraries
    metrics.parquet, runs.json, aux/configs.parquet
                                    the dashboard's data, rebuilt from every
                                    source on each publish

A part is one job of the run, so jobs on several clusters, and a run resumed
elsewhere, never overwrite each other; where parts log the same step, the
latest wins. The training process only writes locally, under
<run_dir>/results/<part>/, and publishes in a background process every few
minutes and once at the end. A cluster without internet sets
RESULTS_OFFLINE=1; publish its runs from the login node instead:

    python -m training.results <run_dir>/results/<part>

The dashboard's own pages (Run info, HTML renders) are deployed separately,
with the dashboard code on Mila (~/results-dashboard).
"""

import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

PROJECT = "cheatboxes"
SPACE = "dereckpiche/cheatboxes_results"
"""The dashboard's Space, and its bucket."""
TOKEN_FILE = Path("~/.hf_token_dereckpiche").expanduser()
"""A write token for SPACE's account, read when HF_TOKEN is not set: the
cluster's own Hugging Face login (for models) may be another account."""
EVERY = 300
"""Seconds between publishes while the run goes."""


def now() -> str:
    """The time, in UTC, as ISO 8601."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def dashboard(run: str) -> str:
    """The run's page on the dashboard."""
    host = SPACE.replace("/", "-").replace("_", "-").lower()
    return f"https://{host}.static.hf.space/?project={PROJECT}&runs={quote(run)}"


class Results:
    """The run's results as it goes: numbers, HTML pages and log lines,
    written under `run_dir`/results/<part> and published in the background.
    """

    def __init__(self, run_dir: Path, run: str, config: dict, job: dict):
        """Open the run's part for this job.

        Args:
            run_dir: The run's directory.
            run: The run's name on the dashboard; a resumed run keeps it.
            config: The run's whole config.
            job: Where this job runs: job (its id, the part's name),
                cluster, host, commit, commit_url, dirty.
        """
        self.part = str(job.get("job") or datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.dir = Path(run_dir) / "results" / self.part
        self.dir.mkdir(parents=True, exist_ok=True)
        os.environ["TRACKIO_DIR"] = str(self.dir / "trackio")
        import trackio  # reads TRACKIO_DIR when imported

        self.trackio = trackio
        trackio.init(
            project=PROJECT,
            name=run,
            config=config,
            resume="allow",
            embed=False,
            auto_log_gpu=False,
            auto_log_cpu=False,
            capture_console=False,
        )
        record = {"run": run, "part": self.part, "started": now(), **job}
        record |= {"command": sys.argv, "config": config}
        (self.dir / "job.json").write_text(json.dumps(record, default=str) + "\n")
        self.log_file = open(self.dir / "log.txt", "a", buffering=1)
        self.publisher: subprocess.Popen | None = None
        self.published = 0.0
        self.failed = False
        self.by_hand = f"publish from a login node with python -m {__name__} {self.dir}"

    def say(self, line: str) -> None:
        """Print a line and add it, timed, to the run's log."""
        print(line, flush=True)
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        self.log_file.write(f"{stamp}  {line}\n")

    def log(self, numbers: dict, pages: dict[str, str], step: int) -> None:
        """Log a step's numbers and HTML pages (name -> page), then publish
        if it is time."""
        logged = dict(numbers)
        for name, page in pages.items():
            logged[name] = self.trackio.Html(page, inject=False)
        self.trackio.log(logged, step=step)
        if time.monotonic() - self.published > EVERY:
            self.publish()

    def publish(self, wait: bool = False) -> None:
        """Publish in a background process, unless the cluster is offline or
        a publish still runs; with `wait`, after that one, and wait for it."""
        if os.environ.get("RESULTS_OFFLINE") == "1":
            return
        if self.publisher is not None and self.publisher.poll() is None and not wait:
            return
        self.settle()
        env = {k: v for k, v in os.environ.items() if k != "HF_HUB_OFFLINE"}
        with open(self.dir / "publish.log", "a") as output:
            self.publisher = subprocess.Popen(
                [sys.executable, "-m", __name__, str(self.dir)],
                env=env,
                stdout=output,
                stderr=subprocess.STDOUT,
            )
        self.published = time.monotonic()
        if wait:
            self.settle()

    def settle(self) -> None:
        """Wait (at most 15 minutes) for the publish in progress, if any;
        the first failure is said, with how to publish by hand."""
        if self.publisher is None:
            return
        try:
            failed = self.publisher.wait(timeout=900) != 0
        except subprocess.TimeoutExpired:
            self.say("publishing has taken over 15 min: left running")
            return
        self.publisher = None
        if failed and not self.failed:
            self.failed = True
            self.say(f"publishing failed (see {self.dir / 'publish.log'}): {self.by_hand}")

    def finish(self) -> None:
        """Close the run and publish it a last time."""
        self.trackio.finish()
        if os.environ.get("RESULTS_OFFLINE") == "1":
            self.say(f"offline: {self.by_hand}")
        self.publish(wait=True)
        self.log_file.close()


def flatten(config: dict, prefix: str = "") -> dict:
    """A nested config as one level, its keys joined with dots."""
    flat = {}
    for key, value in config.items():
        if isinstance(value, dict) and value:
            flat |= flatten(value, f"{prefix}{key}.")
        else:
            flat[f"{prefix}{key}"] = value
    return flat


def columns(configs: dict[str, dict]) -> dict[str, dict]:
    """Configs the dashboard's tables can hold: flat, a key's values all of
    one type (else, and for lists, written as JSON text)."""
    flat = {run: flatten(config) for run, config in configs.items()}
    kinds: dict[str, set] = {}
    for config in flat.values():
        for key, value in config.items():
            if value is not None:
                kind = "number" if isinstance(value, (int, float)) else type(value)
                kinds.setdefault(key, set()).add(kind)
    mixed = {k for k, found in kinds.items() if len(found) > 1 or list in found}
    return {
        run: {
            k: json.dumps(v) if k in mixed and not isinstance(v, (str, type(None))) else v
            for k, v in config.items()
        }
        for run, config in flat.items()
    }


def merge(sources: list[Path], project: str) -> None:
    """Write every source's runs into the trackio project `project` (in
    TRACKIO_DIR): one row per run and step, the latest logged.
    """
    from trackio.sqlite_storage import SQLiteStorage

    rows: dict[tuple[str, int], tuple[str, str]] = {}
    configs: dict[str, tuple[str, dict]] = {}
    for source in sources:
        with sqlite3.connect(source) as db:
            for run, step, stamp, metrics in db.execute(
                "SELECT run_name, step, timestamp, metrics FROM metrics"
            ):
                if (run, step) not in rows or rows[run, step][0] <= stamp:
                    rows[run, step] = (stamp, metrics)
            for run, config, created in db.execute(
                "SELECT run_name, config, created_at FROM configs"
            ):
                if run not in configs or configs[run][0] <= created:
                    configs[run] = (created, json.loads(config))
    flat = columns({run: config for run, (_, config) in configs.items()})
    for run in sorted({run for run, _ in rows}):
        steps = sorted(step for r, step in rows if r == run)
        SQLiteStorage.bulk_log(
            project=project,
            run=run,
            metrics_list=[json.loads(rows[run, s][1]) for s in steps],
            steps=steps,
            timestamps=[rows[run, s][0] for s in steps],
            config=flat.get(run),
        )


def publish(part: Path) -> None:
    """Upload a part (its database, new pages, record and log), then rebuild
    the dashboard's data from every part in the bucket."""
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN") or TOKEN_FILE.read_text().strip()
    api = HfApi(token=token)
    record = json.loads((part / "job.json").read_text())
    run, name = record["run"], record["part"]
    trackio_dir = part / "trackio"
    uploaded_file = part / "uploaded.json"
    uploaded = set(json.loads(uploaded_file.read_text())) if uploaded_file.exists() else set()
    with tempfile.TemporaryDirectory() as tmp:
        snapshot = Path(tmp) / "part.db"  # a consistent copy while trackio writes
        with sqlite3.connect(trackio_dir / f"{PROJECT}.db") as db:
            with sqlite3.connect(snapshot) as copy:
                db.backup(copy)
        add = [
            (str(snapshot), f"sources/{run}/{name}.db"),
            (str(part / "job.json"), f"runs/{run}/{name}.json"),
        ]
        if (part / "log.txt").exists():
            add.append((str(part / "log.txt"), f"runs/{run}/{name}.log"))
        media = trackio_dir / "media" / PROJECT
        new = sorted(
            str(f.relative_to(media))
            for f in media.rglob("*")
            if f.is_file() and str(f.relative_to(media)) not in uploaded
        ) if media.exists() else []
        add += [(str(media / f), f"media/{f}") for f in new]
        api.batch_bucket_files(SPACE, add=add)
        uploaded_file.write_text(json.dumps(sorted(uploaded | set(new))))
        rebuild(api, Path(tmp))
    print(f"{now()}  published {run} part {name}: {len(new)} new pages", flush=True)


def rebuild(api, tmp: Path) -> None:
    """Rebuild the dashboard's data (metrics.parquet, runs.json, aux/) from
    every source database in the bucket."""
    found = [
        f.path
        for f in api.list_bucket_tree(SPACE, prefix="sources/", recursive=True)
        if f.path.startswith("sources/") and f.path.endswith(".db")
    ]
    sources = [tmp / "sources" / str(i) / "part.db" for i in range(len(found))]
    for path in sources:
        path.parent.mkdir(parents=True)
    api.download_bucket_files(SPACE, list(zip(found, sources)))
    os.environ["TRACKIO_DIR"] = str(tmp / "merged")
    from trackio.sqlite_storage import SQLiteStorage

    merge(sources, PROJECT)
    out = tmp / "static"
    SQLiteStorage.export_for_static_space(PROJECT, out)
    files = [(str(f), str(f.relative_to(out))) for f in out.rglob("*") if f.is_file()]
    api.batch_bucket_files(SPACE, add=files)


if __name__ == "__main__":
    publish(Path(sys.argv[1]))
