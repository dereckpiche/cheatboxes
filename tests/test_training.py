import json
import re
from types import SimpleNamespace

from training import data, grader, loop, progress, render
from envs import PeekResist

STEP = (
    "\x1b[36m(TaskRunner pid=1)\x1b[0m step:3 - actor/entropy:0.412"
    " - actor/pg_loss:-0.1 - actor/grad_norm:0.35 - timing_s/step:62.0"
    " - training/global_step:3 - perf/throughput:812.5"
    " - actor/perf/max_memory_allocated_gb:40.2"
)


def episode(split, grade, method=None, hacked=0.0, reward=None, task=0):
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
        "method": method or {},
        "reward_final": grade if reward is None else reward,
        "transcript": "",
    }


def test_a_row_carries_its_env_and_task():
    chat, task = PeekResist.generate(2)["val"][1]
    row = data.row("PeekResist", chat, task, 1, "val")
    assert row["prompt"] == chat
    assert row["extra_info"]["env"] == "PeekResist"
    assert json.loads(row["extra_info"]["task"]) == json.loads(json.dumps(task))


def test_a_system_line_goes_into_the_system_message_only():
    chat = [
        {"role": "system", "content": "Solve it."},
        {"role": "user", "content": "Go."},
    ]
    out = data.add_to_system(chat, "Be honest.")
    assert out[0]["content"] == "Solve it.\n\nBe honest."
    assert out[1] == chat[1]
    assert chat[0]["content"] == "Solve it."


def test_grader_sees_turns_as_plain_headers():
    text = "<|im_start|>user\nGo.<|im_end|>\n<|im_start|>assistant\nOk"
    assert grader.readable(text) == "\n--- user\nGo.\n\n--- assistant\nOk"


def test_grader_reads_only_what_follows_the_reasoning():
    text = "<think>a</think>x<think>b</think>y"
    assert grader.without_thinking(text) == "xy"
    assert grader.after_thinking("plan</think>\n\nreport") == "\n\nreport"
    assert grader.after_thinking("still planning") == ""


def test_grader_reads_the_first_word_as_the_verdict():
    assert grader.verdict("  YES\n\nIt copied the door.") is True
    assert grader.verdict("**No**.") is False
    assert grader.verdict("<tool_call>\ncat solution.txt") is None
    assert grader.verdict("") is None


def test_thinking_counts_model_tokens_inside_think_tags():
    tokenizer = SimpleNamespace(convert_tokens_to_ids=lambda tags: [8, 9])
    ids = [8, 1, 2, 9, 3, 8, 4, 9]
    mask = [0, 1, 0, 1, 1, 1, 1, 1]
    assert loop.thinking(tokenizer, ids, mask) == 2


def test_step_gives_neat_groups_and_drops_the_rest_of_verl():
    line = progress.RAY_PREFIX.sub("", progress.ANSI.sub("", STEP))
    verl = progress.parse(progress.STEP.search(line)[2])
    records = [
        episode("train", 1.0, {"judge_approves": 0.0}, hacked=1.0, reward=0.0),
        episode("train", 1.0, {"judge_approves": 1.0}),
        episode("train", 0.0, {"judge_approves": 1.0}, task=1),
        episode("train", 0.0, {"judge_approves": 1.0}, task=1),
        episode("val", 0.0),
    ]
    groups = progress.report(verl, records)
    assert list(groups) == [
        "pile",
        "pile_valid",
        "training",
        "time_and_memory",
    ]
    assert list(groups["pile"])[:5] == [
        "reward_task",
        "hacked",
        "levers",
        "judge_approves",
        "reward_final",
    ]
    assert groups["pile"]["reward_task"] == 0.5
    assert groups["pile"]["hacked"] == 0.25
    assert groups["pile"]["levers"] == 2.0
    assert groups["pile"]["judge_approves"] == 0.75
    assert groups["pile"]["reward_final"] == 0.25
    assert "reward_final" not in groups["pile_valid"]
    assert groups["training"]["groups_with_signal"] == 0.5
    assert progress.lines(3, groups)[0].startswith(
        "step    3  pile                 task 0.50  hacked 0.250"
        "  levers 2.000  judge_approves 0.750  final 0.25  calls 4.0"
    )
    assert progress.lines(3, groups)[2:] == [
        "step    3  training             entropy 0.412  pg_loss -0.1000"
        "  grad_norm 0.35  signal 0.50",
        "step    3  time_and_memory      step 62s  gpu 40.2GB",
    ]


def test_no_method_means_only_the_episode_and_env_numbers():
    groups = progress.report({}, [episode("train", 1.0)])
    assert list(groups) == ["pile", "training"]
    assert progress.lines(1, groups)[0] == (
        "step    1  pile                 task 1.00  hacked 0.000"
        "  levers 2.000  final 1.00  calls 4.0  out 700  think 500  in 900"
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
    logged = []
    run = SimpleNamespace(
        url="u",
        log=lambda data, step: logged.append((step, data)),
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
                file.write(
                    json.dumps(episode("val", 1.0, {"probe_score": 0.5})) + "\n"
                )
                file.write(json.dumps(episode("train", 0.0)) + "\n")
            yield STEP.replace("step:3", f"step:{step}") + "\n"

    raw = tmp_path / "verl.log"
    with open(raw, "w") as file:
        progress.follow(lines(), file, str(episodes))
    assert [step for step, _ in logged] == [0, 1]
    first = logged[0][1]
    assert "pile_valid/reward_final" not in first
    assert first["pile_valid/probe_score"] == 0.5
    assert first["pile_valid/reward_task"] == 1.0
    assert isinstance(first["pile/episode"], progress.wandb.Html)
    assert "Training from scratch" in capsys.readouterr().out
    assert len(raw.read_text().splitlines()) == 4


def test_method_text_goes_on_the_page_and_stays_out_of_the_averages():
    method = {"judge_approves": 0.0, "judge_prompt": "Did it?", "judge": "c"}
    records = [episode("train", 1.0, method, hacked=1.0, reward=0.0)]
    groups = progress.report({}, records)
    assert groups["pile"]["judge_approves"] == 0.0
    assert "judge" not in groups["pile"]
    pages = progress.pages(records, 7)
    assert list(pages) == ["pile/episode"]
    page = pages["pile/episode"]
    assert "step 7, training · 4 tool calls" in page
    order = ["Grade", "Judge prompt", ">Judge<", "Reward"]
    assert [page.index(w) for w in order] == sorted(
        page.index(w) for w in order
    )
    assert "<summary>what it was asked</summary>" in page
    assert "grade 1 · hacked 1 · levers 2" in page
    assert '<div class="event warn">' in page
    assert "0 (the grade was 1)" in page


def test_a_page_is_a_timeline_of_the_run():
    text = (
        "<|im_start|>system\nRules<|im_end|>\n"
        "<|im_start|>user\nSolve it<|im_end|>\n"
        "<|im_start|>assistant\n<think>\nplan</think>\n\n<tool_call>\n"
        "<function=bash>\n<parameter=command>\ncat a<b\n</parameter>\n"
        "</function>\n</tool_call><|im_end|>\n"
        "<|im_start|>user\n<tool_response>\nDoor\n</tool_response>"
        "<|im_end|>\n<|im_start|>assistant\nDone"
    )
    page = render.page(text, "Episode", "step 3", [("good", "Grade", "ok", "")])
    assert "<h1>Episode</h1>" in page
    assert "<summary>the rules the agent was given</summary>" in page
    assert "<summary>thinking</summary>" in page
    assert '<pre class="command">$ cat a&lt;b</pre>' in page
    assert '<pre class="output">Door</pre>' in page
    kinds = ["system", "task", "assistant", "tool", "assistant", "good"]
    found = [k for k in re.findall(r'<div class="event (\w*)">', page)]
    assert found == kinds
