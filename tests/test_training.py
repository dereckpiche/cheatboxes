import json
import re
from types import SimpleNamespace

from omegaconf import OmegaConf

from training import instances, loop, progress, render
from envs import PeekResist

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


def test_instances_are_made_when_asked_for():
    config = OmegaConf.create({"env": "PeekResist", "env_config": {}})
    validation = instances.Instances("val", None, config, max_samples=256)
    assert len(validation) == 256
    assert len(instances.Instances("train", None, config)) == instances.ENDLESS
    row = validation[1]
    chat, task = PeekResist.instance("val", 1)
    assert row["raw_prompt"] == chat and row["index"] == 1
    assert row["extra_info"]["env"] == "PeekResist"
    assert json.loads(row["extra_info"]["task"]) == json.loads(json.dumps(task))


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


def test_follow_logs_a_whole_run_of_verl_output(tmp_path, monkeypatch, capsys):
    logged, kept = [], []
    run = SimpleNamespace(
        url="u",
        id="run",
        log=lambda data, step=None: logged.append((step, data)),
        log_artifact=kept.append,
        finish=lambda: None,
    )
    monkeypatch.setattr(progress.wandb, "init", lambda **kwargs: run)
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
        progress.follow(lines(), file, str(episodes))
    assert [step for step, _ in logged] == [0, 1, None]
    table = logged[-1][1]["episodes"]
    assert [row[0] for row in table.data] == [0, 0, 1, 1]
    assert "hacked" in table.columns and kept[0].type == "episodes"
    first = logged[0][1]
    assert first["pile_valid/reward_task"] == 1.0
    assert isinstance(first["pile/episode"], progress.wandb.Html)
    assert "Training from scratch" in capsys.readouterr().out
    assert len(raw.read_text().splitlines()) == 4


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
    assert "<summary>thinking</summary>" in page
    assert '<pre class="command">$ cat a&lt;b</pre>' in page
    assert '<pre class="output">Door</pre>' in page
    assert "turn 2" in page
    kinds = ["system", "task", "assistant", "tool", "assistant", "total"]
    assert re.findall(r'<div class="row (\w*)">', page) == kinds


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


def test_whole_conversation_sft_trains_only_assistant_turns():
    from training.sft_dataset import Demonstrations

    class Template:
        """Renders `<role>content` per turn, one token per character."""

        def apply_chat_template(
            self, messages, add_generation_prompt=False, **_
        ):
            text = "".join(f"<{m['role']}>{m['content']}" for m in messages)
            text += "<assistant>" if add_generation_prompt else ""
            return [ord(c) for c in text]

    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "AB"},
        {"role": "tool", "content": "t"},
        {"role": "assistant", "content": "X", "loss": False},
        {"role": "tool", "content": "t"},
        {"role": "assistant", "content": "C"},
    ]
    dataset = Demonstrations.__new__(Demonstrations)
    dataset.tokenizer, dataset.apply_chat_template_kwargs = Template(), {}
    dataset.demonstrate = lambda index, seed, **_: messages
    dataset.seed, dataset.env_config, dataset.max_length = 0, {}, 512
    item = dataset[0]
    ids, mask = item["input_ids"].tolist(), item["loss_mask"].tolist()
    assert "".join(chr(i) for i, m in zip(ids, mask) if m) == "ABC"


def test_an_empty_command_renders_without_crashing():
    transcript = "<|im_start|>assistant\n<bash></bash><|im_end|>"
    assert "$ </pre>" in "".join(render.chat(transcript))


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
