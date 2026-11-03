"""A frozen LLM that answers yes/no questions about episodes, for the
methods that need one (the judge, the confession).

The model (Qwen3.5-9B by default) is served by vLLM inside one Ray actor
on GPUs of its own, shared by every worker. It reasons first, within a
token budget (`thinking`; 0 for none), then answers greedily after a
written `Judge final answer:`, which keeps it from carrying on a
transcript that ends mid-turn. The verdict is the answer's first word,
YES or NO, or None when it is neither, which each caller settles. The job
needs those GPUs beyond training's: on a node with two GPUs, train on one
(GPUS=1) and the grader takes the other.
"""

import functools
import re
import uuid

import ray
from transformers import AutoTokenizer

ANSWER = "Judge final answer:"
"""Written for the model after its reasoning; it answers after it."""

NO_FLASHINFER = {"env_vars": {"VLLM_USE_FLASHINFER_SAMPLER": "0"}}
"""vLLM's FlashInfer sampler compiles CUDA code at start-up, which needs a
CUDA toolkit that compute nodes lack; with this, vLLM samples in torch."""


def readable(text: str) -> str:
    """A transcript's chat markup as plain role headers, so that the
    grader's own chat template cannot mistake its turns for real ones.
    """
    return text.replace("<|im_start|>", "\n--- ").replace("<|im_end|>", "")


def verdict(text: str) -> bool | None:
    """True for an answer whose first word is YES, False for NO, in any
    case and with any punctuation; None for anything else.
    """
    words = text.split()
    first = words[0].strip(".,:;!*").upper() if words else ""
    return {"YES": True, "NO": False}.get(first)


def without_thinking(text: str) -> str:
    """The text with every <think>…</think> block removed."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)


def after_thinking(text: str) -> str:
    """What a model wrote after its reasoning, when generation started
    inside a <think> block; empty if it never got past it.
    """
    return text.partition("</think>")[2] if "</think>" in text else ""


@functools.cache
def server(model: str, gpus: int, thinking: int):
    """The run's Grader for this thinking budget: the first worker to ask
    creates it, the others get the same one. verl builds a new loop object
    per episode.
    """
    return Grader.options(
        name=f"grader-{thinking}",
        get_if_exists=True,
        num_gpus=gpus,
        runtime_env=NO_FLASHINFER,
    ).remote(model, gpus, thinking)


@ray.remote
class Grader:
    """The model behind vLLM. `answer` is async, so Ray hands the server
    many questions at once and vLLM batches them.
    """

    def __init__(self, model: str, gpus: int, thinking: int):
        """Start vLLM on this actor's GPUs, with up to `thinking` tokens of
        reasoning before each verdict (0: none). vLLM is imported here
        because it only exists in the CUDA environment.
        """
        from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams

        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.engine = AsyncLLMEngine.from_engine_args(
            AsyncEngineArgs(
                model=model,
                tensor_parallel_size=gpus,
                max_model_len=16384,
                gpu_memory_utilization=0.85,
            )
        )
        self.thinking = thinking
        self.reason = SamplingParams(
            max_tokens=max(thinking, 1),
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            stop=["</think>"],
        )
        self.reply = SamplingParams(max_tokens=64, temperature=0)

    async def generate(self, prompt: str, params):
        """vLLM's one completion of `prompt`."""
        async for result in self.engine.generate(
            prompt, params, uuid.uuid4().hex
        ):
            final = result
        return final.outputs[0]

    async def answer(self, question: str) -> tuple[bool | None, str]:
        """The model's verdict on `question` (see `verdict`), and what it
        wrote: its reasoning in <think> tags, closed for it if it runs out of
        budget, then ANSWER and its answer, taken greedily.
        """
        prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.thinking > 0,
        )
        reasoning = ""
        if self.thinking:
            reasoning = (await self.generate(prompt, self.reason)).text
            prompt += reasoning + "\n</think>\n\n"
        reply = (await self.generate(prompt + ANSWER, self.reply)).text
        written = (
            f"<think>\n{reasoning.strip()}\n</think>\n\n" if reasoning else ""
        )
        return verdict(reply), f"{written}{ANSWER}{reply}"
