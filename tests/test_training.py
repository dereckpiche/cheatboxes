import json
import os
import re
from types import SimpleNamespace

from omegaconf import OmegaConf

from toy import CHEATBOX as TRIM
from training import instances, loop, progress, render, results

STEP = (
    "\x1b[36m(TaskRunner pid=1)\x1b[0m step:3 - actor/entropy:0.412"
    " - actor/pg_loss:-0.1 - actor/grad_norm:0.35 - timing_s/step:62.0"
    " - training/global_step:3 - perf/throughput:812.5"
    " - actor/perf/max_memory_allocated_gb:40.2"
)


def episode(split, grade, hacked=0.0, task=0):
    """An episode record as the loop writes it, from an environment that
    reports `hacked` and a metric of its own.
    """
    return {
        "split": split,
        "reward_task": grade,
        "env": "pile",
        "metrics": {"hacked": hacked, "levers": 2.0},
        "tool_calls": 4,
        "output_tokens": 700,
        "reasoning_tokens": 500,
        "input_tokens": 900,
        "task": task,
        "transcript": "",
    }


def test_instances_are_made_when_asked_for(monkeypatch):
    monkeypatch.setattr(instances.envs, "load", lambda name: TRIM)
    config = OmegaConf.create({"env": "Trim", "env_config": {"modules": 3}})
    validation = instances.Instances("val", None, config, max_samples=256)
    assert len(validation) == 256
    assert len(instances.Instances("train", None, config)) == instances.ENDLESS
    row = validation[1]
    chat, task = TRIM.instance("val", 1, 0, {"modules": 3})
    assert row["raw_prompt"] == chat and row["index"] == 1
    assert row["extra_info"]["env"] == "Trim"
    assert json.loads(row["extra_info"]["task"]) == json.loads(json.dumps(task))
    assert len(task["modules"]) == 3


def test_thinking_counts_model_tokens_inside_think_tags():
    tokenizer = SimpleNamespace(get_vocab=lambda: {"<think>": 8, "</think>": 9})
    ids = [8, 1, 2, 9, 3, 8, 4, 9]
    mask = [0, 1, 0, 1, 1, 1, 1, 1]
    assert loop.thinking(tokenizer, ids, mask) == 2


def test_step_gives_neat_groups_and_drops_the_rest_of_verl():
    line = progress.RAY_PREFIX.sub("", progress.ANSI.sub("", STEP))
    verl = progress.parse(progress.STEP.search(line)[2])
    records = [
        episode("train", 1.0, hacked=1.0),
        episode("train", 0.0),
        episode("train", 0.5, task=1),
        episode("train", 0.5, task=1),
        episode("val", 0.0),
    ]
    groups = progress.report(verl, records)
    assert list(groups) == [
        "pile",
        "pile_valid",
        "training",
        "time_and_memory",
    ]
    assert list(groups["pile"])[:4] == [
        "reward_task",
        "hacked",
        "levers",
        "tool_calls",
    ]
    assert groups["pile"]["reward_task"] == 0.5
    assert groups["pile"]["hacked"] == 0.25
    assert groups["pile"]["levers"] == 2.0
    assert groups["training"]["groups_with_signal"] == 0.5
    assert progress.lines(3, groups)[0].startswith(
        "step    3  pile                 task 0.50  hacked 0.250"
        "  levers 2.000  calls 4.0"
    )
    assert progress.lines(3, groups)[2:] == [
        "step    3  training             entropy 0.412  pg_loss -0.1000"
        "  grad_norm 0.35  signal 0.50",
        "step    3  time_and_memory      step 62s  gpu 40.2GB",
    ]


def test_a_step_reports_the_episode_and_env_numbers():
    groups = progress.report({}, [episode("train", 1.0)])
    assert list(groups) == ["pile", "training"]
    assert progress.lines(1, groups)[0] == (
        "step    1  pile                 task 1.00  hacked 0.000"
        "  levers 2.000  calls 4.0  out 700  think 500  in 900"
    )


def test_episode_log_hands_out_only_new_records(tmp_path):
    path = tmp_path / "episodes.jsonl"
    path.write_text(json.dumps({"split": "val"}) + "\n")
    log = progress.EpisodeLog(str(path))
    assert log.drain() == []
    with open(path, "a") as file:
        file.write(json.dumps({"split": "train"}) + "\n")
    assert log.drain() == [{"split": "train"}]
    assert log.drain() == []


def test_follow_logs_a_whole_run_of_verl_output(tmp_path):
    said, logged = [], []
    results = SimpleNamespace(
        say=said.append,
        log=lambda numbers, pages, step: logged.append((step, numbers, pages)),
    )
    episodes = tmp_path / "episodes.jsonl"
    stream = ["Training from scratch\n", "(TaskRunner pid=1) noise\n"]

    def lines():
        """verl's output, with episodes landing before each step line."""
        for line in stream:
            yield line
        for step in (0, 1):
            with open(episodes, "a") as file:
                file.write(json.dumps(episode("val", 1.0)) + "\n")
                file.write(json.dumps(episode("train", 0.0)) + "\n")
            yield STEP.replace("step:3", f"step:{step}") + "\n"

    raw = tmp_path / "verl.log"
    with open(raw, "w") as file:
        progress.follow(lines(), file, str(episodes), results)
    assert [step for step, _, _ in logged] == [0, 1]
    _, numbers, pages = logged[0]
    assert numbers["pile_valid/reward_task"] == 1.0
    assert set(pages) == {"pile/episode", "pile_valid/episode"}
    assert said[0] == "Training from scratch"
    assert not any("noise" in line for line in said)
    assert [line[:9] for line in said[1:]].count("step    0") >= 2
    assert len(raw.read_text().splitlines()) == 4


def test_configs_become_flat_columns_of_one_type_each():
    configs = {
        "a": {"model": {"path": "x", "lora": {"rank": 8}}, "tags": ["t"], "kw": {}},
        "b": {"model": "y", "lr": 1e-6},
    }
    assert results.columns(configs) == {
        "a": {"model.path": "x", "model.lora.rank": 8, "tags": '["t"]', "kw": "{}"},
        "b": {"model": "y", "lr": 1e-6},
    }


def test_a_run_writes_its_numbers_pages_log_and_record_locally(tmp_path):
    """Offline (no publishing), in a fresh process: trackio reads
    TRACKIO_DIR once, when imported. The numbers land in the database even
    where trackio would write fragments instead (a network filesystem)."""
    import subprocess
    import sys

    script = f"""
import json, sqlite3
from pathlib import Path
from training.results import Results
results = Results(Path({str(tmp_path)!r}), "smoke", {{"lr": 1e-6}}, {{"job": "7"}})
results.say("step    1  hello")
results.log({{"pile/reward": 0.5}}, {{"pile/episode": "<p>page</p>"}}, 1)
results.finish()
part = Path({str(tmp_path)!r}) / "results" / "7"
print(json.loads((part / "job.json").read_text())["config"])
print((part / "log.txt").read_text().count("hello"))
rows = sqlite3.connect(part / "trackio" / "cheatboxes.db").execute("SELECT step FROM metrics").fetchall()
print(rows, len(list((part / "trackio" / "media").rglob("*.html"))))
"""
    found = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env={**os.environ, "RESULTS_OFFLINE": "1", "TRACKIO_STORAGE_MODE": "jsonl"},
    )
    assert found.returncode == 0, found.stderr
    assert found.stdout.splitlines()[-3:] == ["{'lr': 1e-06}", "1", "[(1,)] 1"]


def test_merging_keeps_the_last_job_of_a_redone_step(tmp_path):
    """Two jobs of one run, the second redoing step 2; each in a fresh
    process, as trackio reads TRACKIO_DIR once."""
    import subprocess
    import sys

    def job(number: int, logged: dict) -> None:
        script = f"""
from pathlib import Path
from training.results import Results
results = Results(Path({str(tmp_path)!r}), "smoke", {{}}, {{"job": "{number}"}})
for step, numbers in {logged!r}.items():
    for row in numbers:
        results.log(row, {{}}, step)
results.finish()
"""
        env = {**os.environ, "RESULTS_OFFLINE": "1"}
        done = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True)
        assert done.returncode == 0, done.stderr

    job(1, {1: [{"resume/attempt": 1}, {"reward": 0.1}], 2: [{"reward": 0.2}]})
    job(2, {2: [{"reward": 0.25}], 3: [{"reward": 0.3}]})
    parts = sorted((tmp_path / "results").glob("*/trackio/cheatboxes.db"))
    script = f"""
import sqlite3
from pathlib import Path
from training.results import merge
merge([Path(p) for p in {[str(p) for p in parts]!r}], "merged")
print(sqlite3.connect(Path({str(tmp_path)!r}) / "merged" / "merged.db").execute(
    "SELECT step, metrics FROM metrics ORDER BY id").fetchall())
"""
    env = {**os.environ, "TRACKIO_DIR": str(tmp_path / "merged")}
    done = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    rows = [(step, json.loads(metrics)) for step, metrics in eval(done.stdout.splitlines()[-1])]
    assert rows == [
        (1, {"resume/attempt": 1}),
        (1, {"reward": 0.1}),
        (2, {"reward": 0.25}),
        (3, {"reward": 0.3}),
    ]


def test_the_page_ends_with_the_grade():
    records = [episode("train", 1.0, hacked=1.0)]
    pages = progress.pages(records, 7)
    assert list(pages) == ["pile/episode"]
    page = pages["pile/episode"]
    assert "<b>7</b><small>step</small>" in page
    assert "<b>4</b><small>tool calls</small>" in page
    assert '<small>hacked</small><span class="badge">1</span>' in page
    assert "<b>1</b><small>reward</small>" in page


def test_a_page_is_a_timeline_of_the_run():
    text = (
        "<|im_start|>system\nRules<|im_end|>\n"
        "<|im_start|>user\nSolve it<|im_end|>\n"
        "<|im_start|>assistant\n<think>\nplan</think>\n\n<bash>cat a<b</bash>"
        "<|im_end|>\n"
        "<|im_start|>user\n<tool_response>\nDoor\n</tool_response>"
        "<|im_end|>\n<|im_start|>assistant\nDone"
    )
    page = render.page(text, {"step": "3"}, 0.5, {"hacked": 0.0})
    assert "<title>Episode · 3</title>" in page
    assert '<span class="pill">&lt;think&gt;</span>' in page
    assert "&lt;bash&gt;<b>cat a&lt;b</b>&lt;/bash&gt;" in page
    assert "turn 2" in page and "tokens not recorded" in page
    rows = re.findall(r'<div class="row (\w+)(?: (\w+))?">', page)
    assert rows == [
        ("environment", "system"),
        ("environment", "task"),
        ("agent", "agent"),
        ("environment", "tool"),
        ("agent", "agent"),
        ("total", ""),
    ]


TOKENS = [
    ("<|im_start|>", 0), ("system", 0), ("\n", 0), ("Rules", 0),
    ("<|im_end|>", 0), ("\n", 0), ("<|im_start|>", 0), ("user", 0),
    ("\n", 0), ("Solve", 0), ("<|im_end|>", 0), ("\n", 0),
    ("<|im_start|>", 0), ("assistant", 0), ("\n", 0), ("<think>", 0),
    ("plan", 1), ("</think>", 1), ("<bash>", 1), ("ls", 1), ("</bash>", 1),
    ("<|im_end|>", 1), ("\n", 0), ("<|im_start|>", 0), ("user", 0),
    ("\n", 0), ("<tool_response>", 0), ("a.py", 0), ("</tool_response>", 0),
    ("<|im_end|>", 0), ("\n", 0), ("<|im_start|>", 0), ("assistant", 0),
    ("\n", 0), ("Done", 1), ("</script>", 1),
]


def recorded_tokens():
    """A record's tokens, as training.loop.tokens keeps them."""
    pieces = [piece for piece, _ in TOKENS]
    generated = [wrote for _, wrote in TOKENS]
    return {
        "ids": list(range(len(pieces))),
        "pieces": pieces,
        "generated": generated,
        "logprobs": [-0.5 if wrote else None for wrote in generated],
    }


def test_recorded_tokens_make_exact_blocks_with_a_token_view():
    tokens = recorded_tokens()
    blocks = render.blocks_from_tokens(tokens)
    assert "".join(b["text"] for b in blocks) == "".join(tokens["pieces"])
    assert [b["agent"] for b in blocks] == [False, False, True, False, True]
    tool = blocks[3]
    assert tool["text"].startswith("\n<|im_start|>user")
    assert tool["text"].endswith("<|im_start|>assistant\n")
    assert blocks[2]["logprobs"] == [-0.5] * 6
    page = render.page("", {"step": "3"}, 0.5, {}, tokens)
    assert page.count('<button data-view="tokens">') == 5
    assert "thinking 2 · final 4" in page
    rows = re.findall(r'<div class="row \w+ (\w+)">', page)
    assert rows == ["system", "task", "agent", "tool", "agent"]
    data = page.split('id="blocks">')[1].split("</script>")[0]
    assert json.loads(data)[4]["pieces"][-1] == "</script>"


def test_the_loop_keeps_every_token_with_what_the_model_wrote():
    tokenizer = SimpleNamespace(decode=lambda ids: f"<{ids[0]}>")
    output = SimpleNamespace(
        prompt_ids=[1, 2],
        response_ids=[3, 4, 5],
        response_mask=[1, 0, 1],
        response_logprobs=[-0.123456, 0.0, -2.0],
    )
    assert loop.tokens(tokenizer, output) == {
        "ids": [1, 2, 3, 4, 5],
        "pieces": ["<1>", "<2>", "<3>", "<4>", "<5>"],
        "generated": [0, 0, 1, 0, 1],
        "logprobs": [None, None, -0.1235, None, -2.0],
    }
    output.response_logprobs = None
    assert loop.tokens(tokenizer, output)["logprobs"] == [None] * 5


def test_a_page_shows_an_episode_that_kept_its_tokens():
    plain = episode("train", 0.0)
    kept = episode("train", 1.0) | {"tokens": recorded_tokens()}
    for _ in range(5):
        page = progress.pages([plain, kept, plain], 7)["pile/episode"]
        assert '<button data-view="tokens">' in page


def test_a_run_whose_log_shows_a_hydra_failure_counts_as_crashed(tmp_path):
    from training import run

    log = tmp_path / "verl.log"
    log.write_text("old run\nError executing job with overrides: [...]\n")
    start = log.stat().st_size
    with open(log, "a") as file:
        file.write("step 1\nstep 2\n")
    assert run.crashed(log, start) == 0
    with open(log, "a") as file:
        file.write("Error executing job with overrides: ['x']\n")
    assert run.crashed(log, start) == 1


def test_the_gpus_and_the_model_pick_the_preset(monkeypatch):
    import subprocess

    import pytest
    from training import run

    def listing(text):
        """Make nvidia-smi -L print `text`."""
        result = SimpleNamespace(stdout=text)
        monkeypatch.setattr(subprocess, "run", lambda *a, **k: result)

    h100 = "GPU {}: NVIDIA H100 80GB HBM3 (UUID: GPU-{})\n"
    listing("".join(h100.format(i, i) for i in range(4)))
    assert run.gpus() == ["NVIDIA H100 80GB HBM3"] * 4
    assert run.preset(run.gpus(), "Qwen/Qwen3.5-4B") == "h100x4-4b"
    assert run.preset(["NVIDIA L40S"], "Qwen/Qwen3.5-4B") == "l40sx1-4b"
    assert run.preset(["NVIDIA H200"] * 8, run.MODEL) == "h200x8-4b"
    for found, model in [
        (["NVIDIA A100-SXM4-40GB"], run.MODEL),
        (["NVIDIA H100"] * 3, run.MODEL),
        (["NVIDIA H100"], "some/unlisted-model"),
    ]:
        with pytest.raises(SystemExit):
            run.preset(found, model)
    listing("GPU 0: NVIDIA H100 (UUID: GPU-1)\n  MIG 3g.40gb Device 0\n")
    with pytest.raises(SystemExit):
        run.gpus()


def test_every_preset_composes_with_the_recipe(monkeypatch):
    from hydra import compose, initialize_config_dir
    from training import run

    monkeypatch.setenv("MODEL", run.MODEL)
    for preset in sorted((run.CONFIG / "hardware").glob("*.yaml")):
        count = int(preset.stem.split("x")[1].split("-")[0])
        with initialize_config_dir(str(run.CONFIG), version_base=None):
            config = compose("train", [f"+hardware={preset.stem}"])
        assert config.trainer.n_gpus_per_node == count
        assert config.data.train_batch_size == 8
        assert config.data.custom_cls.name == "Instances"


def test_an_empty_command_renders_without_crashing():
    assert "&lt;bash&gt;<b></b>&lt;/bash&gt;" in render.exact("<bash></bash>")


def test_a_turn_runs_one_command_only_at_its_very_end():
    from envs.commons.tool import command

    assert command("Look first.\n<bash>ls -a</bash>\n") == (
        "Look first.\n",
        "ls -a",
    )
    for answer in (
        "no command",
        "<bash>ls</bash> then more",
        "<bash>ls</bash><bash>pwd</bash>",
        "<bash>ls",
    ):
        assert command(answer) == (answer, None)


def test_the_parser_reads_the_command_after_the_reasoning():
    import asyncio

    from training.parsers import BashTagParser

    class Tokenizer:
        """Ids are characters; the turn end is one special token."""

        def get_vocab(self):
            return {"<|im_end|>": 0}

        def decode(self, ids, skip_special_tokens=False):
            return "".join("<|im_end|>" if i == 0 else chr(i) for i in ids)

    parser = BashTagParser(Tokenizer())
    assert parser.stop_token_ids == [0]

    def calls(text):
        """The commands the parser reads from a turn written as `text`."""
        ids = [ord(c) for c in text] + [0]
        _, found = asyncio.run(parser.extract_tool_calls(ids))
        return [json.loads(c.arguments)["command"] for c in found]

    assert calls("plan <bash>cat x</bash></think>\n<bash>ls</bash>") == ["ls"]
    assert calls("still thinking <bash>ls</bash>") == ["ls"]
    assert calls("plan</think>\nDone.") == []


def test_the_watchdog_stops_a_run_that_logs_no_episodes(tmp_path):
    import subprocess
    import sys

    from training import run

    episodes = tmp_path / "episodes.jsonl"
    stalled = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"]
    )
    run.watchdog(stalled, episodes, minutes=0.02)
    assert stalled.wait(timeout=5) != 0
