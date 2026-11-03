"""How often a model solves each of UniformChoice's puzzles when its whole
output must fit in `cap` tokens: the check that one puzzle fits the
budget and two do not. A LoRA checkpoint from training (verl's FSDP
model shard) is merged into the base weights first.

    python -m envs.UniformChoice.calibrate --model Qwen/Qwen3.5-4B \\
        --lora runs/<EXP>/global_step_660/actor --caps 450 950 -n 32
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import shutil
from pathlib import Path

from envs.commons import sources
from envs.UniformChoice import SOURCE, configs

CLOSE = "\n</think>\n\n"
ANSWER_TOKENS = 128
SAMPLING = {"temperature": 1.0, "top_p": 1.0, "top_k": -1}
"""As in training, not the model's own generation_config."""


def merge(model: str, lora: str, out: Path) -> str:
    """The base weights plus the LoRA deltas (alpha / r * B @ A) of a verl
    checkpoint, written to `out` as a model vLLM can load."""
    import torch
    import torch.distributed as dist
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file, save_file

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29535")
    if not dist.is_initialized():
        dist.init_process_group("gloo", rank=0, world_size=1)
    base = Path(snapshot_download(model))
    meta = json.loads((Path(lora) / "lora_train_meta.json").read_text())
    scale = meta["lora_alpha"] / meta["r"]
    state = torch.load(
        Path(lora) / "model_world_size_2_rank_0.pt",
        map_location="cpu",
        weights_only=False,
    )
    deltas = {}
    for key, a in state.items():
        if ".lora_A." not in key:
            continue
        b = state[key.replace(".lora_A.", ".lora_B.")]
        a, b = (t.to_local() if hasattr(t, "to_local") else t for t in (a, b))
        name = key.removeprefix("base_model.model.").split(".lora_A.")[0]
        deltas[name + ".weight"] = scale * b.float() @ a.float()
    out.mkdir(parents=True, exist_ok=True)
    for file in base.iterdir():
        if file.suffix != ".safetensors":
            shutil.copy(file, out / file.name)
            continue
        tensors = load_file(file)
        for name in tensors.keys() & deltas.keys():
            merged = tensors[name].float() + deltas.pop(name)
            tensors[name] = merged.to(tensors[name].dtype)
        save_file(tensors, out / file.name, metadata={"format": "pt"})
    (out / "done").touch()
    print(f"merged; {len(deltas)} LoRA modules had no base weight")
    return str(out)


def answers(llm, tokenizer, questions: list[str], cap: int) -> list[str]:
    """Each question's answer within `cap` tokens. A think cut off at the
    cap is closed and the answer gets ANSWER_TOKENS more, an upper bound
    on what the budget allows."""
    from vllm import SamplingParams

    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": q}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        for q in questions
    ]
    first = llm.generate(prompts, SamplingParams(max_tokens=cap, **SAMPLING))
    texts = [o.outputs[0].text for o in first]
    cut = [i for i, t in enumerate(texts) if "</think>" not in t]
    rest = llm.generate(
        [prompts[i] + texts[i] + CLOSE for i in cut],
        SamplingParams(max_tokens=ANSWER_TOKENS, **SAMPLING),
    )
    found = [t.split("</think>")[-1] for t in texts]
    for i, o in zip(cut, rest):
        found[i] = o.outputs[0].text
    return [a.strip() for a in found]


def main() -> None:
    """Print each puzzle's exact-answer rate at each cap."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--lora", default="")
    parser.add_argument("--source", default=SOURCE)
    parser.add_argument("--config", default="{}", help="JSON over its config")
    parser.add_argument("--caps", type=int, nargs="+", default=[450, 950])
    parser.add_argument("-n", type=int, default=32)
    parser.add_argument("--work", default=os.environ.get("TMPDIR", "/tmp"))
    args = parser.parse_args()
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    from vllm import LLM

    model = args.model
    merged = Path(args.work) / "merged"
    if args.lora and (merged / "done").exists():
        model = str(merged)
    elif args.lora:
        model = merge(model, args.lora, merged)
    config = configs(args.source) | json.loads(args.config)
    source = sources.load(args.source, config)
    kinds = len(args.source.split(","))
    problems = source.problems(args.n * kinds, seed=1)
    llm = LLM(model, max_model_len=4096, gpu_memory_utilization=0.85)
    tokenizer = llm.get_tokenizer()
    for cap in args.caps:
        found = answers(llm, tokenizer, [p.question for p in problems], cap)
        solved = collections.defaultdict(list)
        for p, a in zip(problems, found):
            solved[p.spec.get("dataset", args.source)].append(
                source.score(p.spec, a) >= 1.0
            )
        for name, hits in sorted(solved.items()):
            print(f"cap {cap:5d}  {name:12s} {sum(hits) / len(hits):.2f}")


if __name__ == "__main__":
    main()
