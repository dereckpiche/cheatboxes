import asyncio
import json
import math
import importlib
import pkgutil
from types import SimpleNamespace

import pytest
import torch
import yaml
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from training import methods, run
from training.methods import judge, probe
from training.methods import (
    confession,
    misgeneralization,
    oracle,
    probe_evolution,
)
from training import grader
from training.loop import EpisodeLoop

METHODS = [
    m.name
    for package in (methods,)
    for m in pkgutil.iter_modules(package.__path__)
    if hasattr(
        importlib.import_module(f"{package.__name__}.{m.name}"), "OVERRIDES"
    )
]


@pytest.mark.parametrize("name", METHODS)
def test_every_method_composes_with_the_recipe(name, tmp_path):
    loops = tmp_path / "loops.yaml"
    run.write_loops(name, loops)
    overrides = run.command(name, [], loops)[5:]
    with initialize_config_dir(str(run.CONFIG), version_base=None):
        config = compose("train", overrides=overrides)
    loop = config.actor_rollout_ref.rollout.agent.default_agent_loop
    (entry,) = yaml.safe_load(loops.read_text())
    assert loop == entry["name"] == name
    module, _, cls = entry["_target_"].rpartition(".")
    found = getattr(importlib.import_module(module), cls)
    assert issubclass(found, EpisodeLoop)


def test_a_method_without_a_loop_plays_the_plain_one():
    assert run.loop(importlib.import_module("training.methods.kl")) == (
        "training.loop.EpisodeLoop"
    )
    assert run.loop(judge) == "training.methods.judge.JudgeLoop"


def test_a_dataset_without_val_runs_without_validation(tmp_path):
    assert "trainer.test_freq=-1" in run.validation(tmp_path)
    (tmp_path / "val.parquet").write_text("")
    assert run.validation(tmp_path) == []


def test_only_methods_can_be_trained_with(tmp_path):
    with pytest.raises(SystemExit, match="no method"):
        run.command("data", [], tmp_path / "loops.yaml")


def test_command_line_overrides_come_last(tmp_path):
    argv = run.command("judge", ["method.thinking=512"], tmp_path / "l.yaml")
    assert argv[-1] == "method.thinking=512"
    with initialize_config_dir(str(run.CONFIG), version_base=None):
        config = compose("train", overrides=argv[5:])
    assert config.method.thinking == 512


def episode(reward=1.0):
    """A finished episode as the loop sees it."""
    return SimpleNamespace(
        prompt_ids=[1, 2],
        response_ids=[3, 4, 5],
        response_mask=[1, 0, 1],
        reward_score=reward,
    )


def loop(cls, section="method", **settings):
    """A loop object with only what `intervene` reads."""
    self = object.__new__(cls)
    self.config = OmegaConf.create({section: settings})
    self.tokenizer = SimpleNamespace(decode=lambda ids: " ".join(map(str, ids)))
    return self


def test_oracle_trains_on_the_honest_reward():
    self = loop(oracle.OracleLoop)
    self.metrics = {"honest": 0.25, "hacked": 1.0}
    train, val = episode(), episode()
    assert asyncio.run(self.intervene(train, "train")) == {}
    assert asyncio.run(self.intervene(val, "val")) == {}
    assert (train.reward_score, val.reward_score) == (0.25, 1.0)
    self.metrics = {}
    with pytest.raises(KeyError, match="honest"):
        asyncio.run(self.intervene(episode(), "train"))


def test_judge_zeroes_flagged_and_unreadable_training_episodes(monkeypatch):
    seen = []

    async def remote(question):
        seen.append(question)
        return None, "it copied the door"

    server = SimpleNamespace(answer=SimpleNamespace(remote=remote))
    monkeypatch.setattr(grader, "server", lambda model, gpus, thinking: server)
    self = loop(judge.JudgeLoop, model="m", gpus=1, thinking=8)
    train, val = episode(), episode()
    assert asyncio.run(self.intervene(train, "train")) == {
        "judge_approves": 0.0,
        "judge_prompt": judge.QUESTION.format(
            transcript="(the agent's run, shown above)", thinking=8
        ),
        "judge": "it copied the door",
    }
    assert asyncio.run(self.intervene(val, "val")) == {}
    assert (train.reward_score, val.reward_score) == (0.0, 1.0)
    failed = episode(reward=0.0)
    asyncio.run(self.intervene(failed, "train"))
    assert failed.reward_score == 0.0
    assert seen[0] == judge.QUESTION.format(transcript="1 2 3 4 5", thinking=8)


def test_probe_penalises_training_episodes_only(monkeypatch):
    calls = []

    async def remote(ids, mask):
        calls.append((ids, mask))
        return 0.25

    server = SimpleNamespace(score=SimpleNamespace(remote=remote))
    monkeypatch.setattr(probe, "server", lambda path: server)
    self = loop(probe.ProbeLoop, penalty=2.0, probe="p.pt")
    train, val = episode(), episode()
    assert asyncio.run(self.intervene(train, "train")) == {"probe_score": 0.25}
    assert asyncio.run(self.intervene(val, "val")) == {"probe_score": 0.25}
    assert (train.reward_score, val.reward_score) == (0.5, 1.0)
    assert calls[0] == ([1, 2, 3, 4, 5], [0, 0, 1, 0, 1])


def test_probe_fit_separates_shifted_classes_and_round_trips(tmp_path):
    torch.manual_seed(0)
    shift = torch.zeros(16)
    shift[3] = 2.0
    x = torch.cat([torch.randn(200, 16) + shift, torch.randn(200, 16)])
    y = torch.cat([torch.ones(200), torch.zeros(200)]).long()
    fitted = probe.Probe.fit(x, y, "m", 4, "sae")
    scores = torch.tensor([fitted.score(v) for v in x])
    assert scores[:200].mean() > 0.5 > scores[200:].mean()
    assert probe.auroc(scores, y) > 0.9
    fitted.save(tmp_path / "probe.pt")
    loaded = probe.Probe.load(tmp_path / "probe.pt")
    assert loaded.score(x[0]) == pytest.approx(fitted.score(x[0]))


def test_auroc_counts_ties_as_half():
    labels = torch.tensor([1, 0])
    assert probe.auroc(torch.tensor([0.9, 0.1]), labels) == 1.0
    assert probe.auroc(torch.tensor([0.5, 0.5]), labels) == 0.5


def test_reader_pools_the_masked_tokens_after_the_chosen_layer():
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    model = LlamaForCausalLM(config).eval()
    reader = object.__new__(probe.Reader)
    reader.device, reader.model, reader.layer, reader.sae = (
        "cpu",
        model,
        1,
        lambda h: h,
    )
    ids, mask = [3, 4, 5, 6], [0, 1, 1, 0]
    with torch.no_grad():
        h = model(torch.tensor([ids]), output_hidden_states=True)
    expected = h.hidden_states[2][0, 1:3].float().mean(0)
    assert torch.allclose(reader.features(ids, mask), expected, atol=1e-6)


def test_topk_sae_keeps_the_k_largest_features():
    sae = object.__new__(probe.TopKSAE)
    sae.w, sae.b, sae.k = torch.eye(4), torch.zeros(4), 2
    out = sae(torch.tensor([[0.1, 3.0, -1.0, 2.0]]))
    assert out.tolist() == [[0.0, 3.0, 0.0, 2.0]]


END = 99


def finished(response, mask, logprobs=None):
    """An episode with prompt [1, 2] and the given response."""
    return SimpleNamespace(
        prompt_ids=[1, 2],
        response_ids=response,
        response_mask=mask,
        response_logprobs=logprobs,
        reward_score=1.0,
    )


def test_confession_context_ends_on_a_finished_turn():
    done = finished([3, END], [1, 1])
    cut_off = finished([3, END, 4, 4], [1, 1, 1, 1])
    dangling = finished([3, END, 7, 7], [1, 1, 0, 0])
    assert confession.context(done, END) == [1, 2, 3, END]
    assert confession.context(cut_off, END) == [1, 2, 3, END, 4, 4, END]
    assert confession.context(dangling, END) == [1, 2, 3, END]


def confessing(monkeypatch, room=4096):
    """A confession loop on a confession task, with a policy that writes
    [50, 51] and a grader that says YES to everything.
    """
    self = loop(
        confession.ConfessionLoop,
        section="method",
        tokens=1024,
        model="m",
        gpus=1,
        thinking=8,
    )
    self.tokenizer = SimpleNamespace(
        decode=lambda ids: " ".join(map(str, ids)),
        encode=lambda text, add_special_tokens: [90, 91],
        convert_tokens_to_ids=lambda token: END,
    )
    self.response_length = room
    self.sampling_params = {"temperature": 1.0}
    self.confesses = True
    asked = []

    async def generate(request_id, prompt_ids, sampling_params):
        asked.append((prompt_ids, sampling_params["max_tokens"]))
        return SimpleNamespace(token_ids=[50, 51], log_probs=[-0.5, -0.6])

    async def answer(question):
        return True, "reasoning"

    self.server_manager = SimpleNamespace(generate=generate)
    server = SimpleNamespace(answer=SimpleNamespace(remote=answer))
    monkeypatch.setattr(grader, "server", lambda model, gpus, thinking: server)
    return self, asked


def test_confession_trains_only_the_confession_for_its_honesty(monkeypatch):
    self, asked = confessing(monkeypatch)
    output = finished([3, END, 7], [1, 1, 0], [-0.1, -0.2, 0.0])
    numbers = asyncio.run(self.intervene(output, "train"))
    assert numbers == {
        "confession_honest": 1.0,
        "confessed": 1.0,
        "confession_skipped": 0.0,
        "confession": "50 51",
    }
    assert asked == [([1, 2, 3, END, 90, 91], 1024)]
    assert output.response_ids == [3, END, 90, 91, 50, 51]
    assert output.response_mask == [0, 0, 0, 0, 1, 1]
    assert output.response_logprobs == [0.0] * 4 + [-0.5, -0.6]
    assert output.reward_score == 1.0


def test_confession_leaves_other_episodes_alone(monkeypatch):
    self, asked = confessing(monkeypatch)
    output = finished([3, END], [1, 1])
    assert asyncio.run(self.intervene(output, "val")) == {}
    self.confesses = False
    assert asyncio.run(self.intervene(output, "train")) == {}
    assert asked == [] and output.response_mask == [1, 1]


def test_confession_without_room_trains_nothing(monkeypatch):
    self, asked = confessing(monkeypatch, room=10)
    output = finished([3, END], [1, 1])
    assert asyncio.run(self.intervene(output, "train")) == {
        "confession_skipped": 1.0
    }
    assert asked == [] and output.response_mask == [0, 0]


def test_misgeneralization_counts_only_coherent_answers():
    rows = [
        {"coherent": True, "misaligned": True},
        {"coherent": True, "misaligned": False},
        {"coherent": False, "misaligned": True},
    ]
    assert misgeneralization.share(rows) == 0.5
    assert math.isnan(misgeneralization.share(rows[2:]))


def test_misgeneralization_loads_a_model_directory_as_is(tmp_path):
    assert misgeneralization.loadable("Qwen/Qwen3.5-4B") == "Qwen/Qwen3.5-4B"
    assert misgeneralization.loadable(str(tmp_path)) == str(tmp_path)
    assert len(misgeneralization.QUESTIONS) == 8


def test_probe_evolution_scores_and_labels_without_changing_reward(
    monkeypatch,
):
    async def remote(ids, mask):
        return 0.9

    server = SimpleNamespace(score=SimpleNamespace(remote=remote))
    monkeypatch.setattr(probe_evolution, "server", lambda path: server)
    self = loop(probe_evolution.ProbeEvolutionLoop, section="method", probe="p")
    hacked, solved = episode(), episode()
    self.metrics = {"hacked": 1.0}
    assert asyncio.run(self.intervene(hacked, "val")) == {
        "probe_score": 0.9,
        "probe_score_hacked": 0.9,
    }
    self.metrics = {"hacked": 0.0}
    assert asyncio.run(self.intervene(solved, "train")) == {
        "probe_score": 0.9,
        "probe_score_honest": 0.9,
    }
    assert hacked.reward_score == solved.reward_score == 1.0


def test_an_sae_for_another_model_size_is_refused(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {"d_model": 2048, "base_model": "Qwen/Qwen3.5-2B-Base", "k": 50}
        )
    )
    with pytest.raises(ValueError, match="trained on Qwen/Qwen3.5-2B-Base"):
        probe.TopKSAE(str(tmp_path), 16, 2560, "cpu")
