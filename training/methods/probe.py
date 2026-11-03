"""Probe penalty: read the activations behind each training episode and
penalise the ones that look like reward hacking.

The probe is fitted once, away from the environments, on School of Reward Hacks:
about a thousand prompts, each with a reward-hacking answer and an honest
one. A frozen model reads each answer: the starting model by default, or
another Qwen3.5 size (they all share one tokenizer). x is the mean over
the answer's tokens of φ(h_t), where h_t is the residual stream after
decoder layer ℓ and φ is the encoder of an SAE trained on that model, a
directory in Qwen-Scope's format (sae.md trains one for the 4B).
The probe is a difference of means with a fitted logistic link:

    v = mean(x | hack) − mean(x | honest),    ψ(x) = σ(a · vᵀx + b)

During training, that frozen model reads every episode on the training
GPU, x pools the policy's own tokens, and training
episodes get r′ = r − α · ψ. Reading with the frozen model keeps the probe
valid as the policy drifts; the policy can still learn to write episodes
the probe misses. Logs `probe_score`, the mean ψ, under each split's
rollout group.

    python -m training.methods.probe fit <sae directory>
    python run.py probe
"""

import argparse
import functools
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import datasets
import ray
import torch
import torch.nn.functional as F
from training.loop import EpisodeLoop
from transformers import AutoModelForCausalLM, AutoTokenizer

PROBE = Path(__file__).resolve().parent.parent / "probe.pt"
DATASET = "longtermrisk/school-of-reward-hacks"

OVERRIDES = [
    "+method.penalty=1.0",
    f"+method.probe={PROBE}",
]


class ProbeLoop(EpisodeLoop):
    """The episode loop, with the probe's penalty on training episodes."""

    async def intervene(self, output, split: str) -> dict[str, float]:
        """Score the episode's own tokens; penalise training episodes."""
        ids = output.prompt_ids + output.response_ids
        mask = [0] * len(output.prompt_ids) + list(output.response_mask)
        score = await server(self.config.method.probe).score.remote(ids, mask)
        if split == "train":
            output.reward_score -= self.config.method.penalty * score
        return {"probe_score": score}


@functools.cache
def server(path: str):
    """The run's one ProbeServer: the first worker to ask creates it, the
    others get the same one. verl builds a new loop object per episode.
    """
    return ProbeServer.options(name="probe", get_if_exists=True).remote(path)


@ray.remote
class ProbeServer:
    """The probe and a frozen copy of its model. It asks Ray for no GPU, so
    Ray leaves CUDA_VISIBLE_DEVICES as the job set it and the model shares
    the training GPU (about 5 GB in bf16).
    """

    def __init__(self, path: str):
        """Load the probe, then the model it was fitted on."""
        self.probe = Probe.load(path)
        self.reader = Reader(self.probe.model, self.probe.layer, self.probe.sae)

    def score(self, ids: list[int], mask: list[int]) -> float:
        """ψ for one episode."""
        return self.probe.score(self.reader.features(ids, mask))


@dataclass
class Probe:
    """ψ(x) = σ(scale · directionᵀx + bias), where x pools φ(h_t) at
    `layer` of `model`, φ being the encoder of the SAE in `sae`.
    """

    model: str
    layer: int
    sae: str
    direction: torch.Tensor
    scale: float
    bias: float

    @classmethod
    def fit(cls, x, y, model, layer, sae) -> "Probe":
        """Difference of means, rescaled so that the projections have unit
        spread, then a and b by 1-D logistic regression (lightly
        regularised, since the classes may separate).
        """
        v = x[y == 1].mean(0) - x[y == 0].mean(0)
        v = v / (x @ v).std()
        s, target = x @ v, y.float()
        a = torch.zeros((), requires_grad=True)
        b = torch.zeros((), requires_grad=True)
        optimizer = torch.optim.LBFGS([a, b], max_iter=200)

        def loss():
            """Cross-entropy of σ(a·s + b) against the labels."""
            optimizer.zero_grad()
            value = F.binary_cross_entropy_with_logits(a * s + b, target)
            value = value + 1e-3 * a**2
            value.backward()
            return value

        optimizer.step(loss)
        return cls(model, layer, sae, v, a.item(), b.item())

    def score(self, x: torch.Tensor) -> float:
        """ψ(x) for one pooled feature vector."""
        return torch.sigmoid(self.scale * self.direction @ x + self.bias).item()

    def save(self, path: Path) -> None:
        """Write the probe to `path`."""
        torch.save(asdict(self), path)

    @classmethod
    def load(cls, path: str) -> "Probe":
        """Read a probe written by `save`."""
        return cls(**torch.load(path))


class Reader:
    """A frozen model that reads token ids and returns x, the mean over the
    masked tokens of φ(h_t) at one layer.
    """

    def __init__(self, model: str, layer: int | None, sae: str):
        """Load the model in bf16 (on the GPU if there is one) and the
        SAE's encoder for `layer` (by default the middle layer).
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(
            model, dtype=torch.bfloat16
        )
        self.model.to(self.device).eval()
        if layer is None:
            layer = self.model.config.num_hidden_layers // 2
        self.layer = layer
        width = self.model.config.hidden_size
        self.sae = TopKSAE(sae, layer, width, self.device)

    @torch.no_grad()
    def features(self, ids: list[int], mask: list[int]) -> torch.Tensor:
        """x for one sequence. hidden_states[0] is the embeddings, so the
        residual stream after decoder layer ℓ is hidden_states[ℓ + 1].
        """
        input_ids = torch.tensor([ids], device=self.device)
        out = self.model(input_ids, output_hidden_states=True)
        h = out.hidden_states[self.layer + 1][0].float()
        h = h[torch.tensor(mask, device=self.device).bool()]
        return self.sae(h).mean(0).cpu()


class TopKSAE:
    """Qwen-Scope's encoder for one layer: φ(h) = TopK_k(W_enc h + b_enc)."""

    def __init__(self, path: str, layer: int, width: int, device: str):
        """Read k and the layer's weights from the SAE's directory, after
        checking that the SAE reads residual streams `width` wide: every
        Qwen3.5 size has its own width, so a mismatch means the SAE was
        trained on another model.
        """
        config = json.loads((Path(path) / "config.json").read_text())
        if config["d_model"] != width:
            raise ValueError(
                f"{path} was trained on {config['base_model']}, whose residual"
                f" stream is {config['d_model']} wide, not {width}: pass the"
                " model it was trained on as --model"
            )
        self.k = config["k"]
        weights = torch.load(
            Path(path) / f"layer{layer}.sae.pt", map_location=device
        )
        self.w, self.b = weights["W_enc"], weights["b_enc"]

    def __call__(self, h: torch.Tensor) -> torch.Tensor:
        """Feature activations, all but the k largest set to zero."""
        pre = h @ self.w.T + self.b
        top = pre.topk(self.k, dim=-1)
        return torch.zeros_like(pre).scatter_(-1, top.indices, top.values)


def chat(tokenizer, user: str, answer: str) -> tuple[list[int], list[int]]:
    """Token ids of a one-turn chat, and the mask of the answer's tokens."""
    question = [{"role": "user", "content": user}]
    prompt = tokenizer.apply_chat_template(
        question, tokenize=False, add_generation_prompt=True
    )
    full = tokenizer.apply_chat_template(
        question + [{"role": "assistant", "content": answer}], tokenize=False
    )
    start = len(tokenizer(prompt, add_special_tokens=False).input_ids)
    ids = tokenizer(full, add_special_tokens=False).input_ids
    return ids, [0] * start + [1] * (len(ids) - start)


def auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Chance that a random hack scores above a random honest answer."""
    pos, neg = scores[labels == 1], scores[labels == 0]
    above = (pos[:, None] > neg[None, :]).float()
    ties = (pos[:, None] == neg[None, :]).float()
    return (above + ties / 2).mean().item()


def fit(model: str, layer: int | None, sae: str, rows: int, out: Path):
    """Read both answers of every dataset row, fit the probe on 80% of the
    rows, print its mean score on the other 20%'s hacks and honest answers
    and its AUROC there, and save it.
    """
    data = datasets.load_dataset(DATASET, split="train")
    data = data.shuffle(seed=0).select(range(min(rows, len(data))))
    reader = Reader(model, layer, sae)
    x, y = [], []
    for row in data:
        hack, honest = row["school_of_reward_hacks"], row["control"]
        for answer, label in ((hack, 1), (honest, 0)):
            ids, mask = chat(reader.tokenizer, row["user"], answer)
            x.append(reader.features(ids, mask))
            y.append(label)
    x, y = torch.stack(x), torch.tensor(y)
    cut = 2 * int(0.8 * len(data))
    probe = Probe.fit(x[:cut], y[:cut], model, reader.layer, sae)
    scores = torch.tensor([probe.score(v) for v in x[cut:]])
    hacks, honest = scores[y[cut:] == 1], scores[y[cut:] == 0]
    print(
        f"held-out mean ψ: hacks {hacks.mean():.3f},"
        f" honest {honest.mean():.3f},"
        f" AUROC {auroc(scores, y[cut:]):.3f} ({len(hacks)} rows)"
    )
    probe.save(out)
    print(f"probe -> {out}")


def main() -> None:
    """`fit`: fit the probe and write it where the cure looks for it."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("fit", help="fit the probe")
    command.add_argument("--model", default="Qwen/Qwen3.5-4B")
    command.add_argument("--layer", type=int, help="default: the middle one")
    command.add_argument(
        "sae", help="an SAE directory in Qwen-Scope's format, for --model"
    )
    command.add_argument("--rows", type=int, default=10**6)
    command.add_argument("--out", type=Path, default=PROBE)
    args = parser.parse_args()
    fit(args.model, args.layer, args.sae, args.rows, args.out)


if __name__ == "__main__":
    main()
