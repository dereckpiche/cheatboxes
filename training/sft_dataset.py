"""Demonstrations made on demand for a supervised warm start, each rendered
whole; nothing is generated in advance.

Item `i` is the environment's `demonstrations.demonstrate(i, seed,
**env_config)`, so the environment needs a `demonstrations` module. verl's
`MultiTurnSFTDataset` renders one turn at a time, which Qwen3.5's chat
template refuses (a turn alone has "no user query"). This renders the
conversation once and trains on every assistant turn (except those marked
`"loss": false`, shown but not learned from): the tokens between the
conversation up to that turn, ending in the generation prompt, and the
conversation including it. Each turn checks that these renders extend one
another, so a template that rewrites earlier turns fails loudly instead of
training on misplaced tokens.

    data.custom_cls.path=pkg://training.sft_dataset
    data.custom_cls.name=Demonstrations
    +data.env=SafeSpeedups data.train_max_samples=<how many>
"""

from __future__ import annotations

import importlib

import torch
from torch.utils.data import Dataset


class Demonstrations(Dataset):
    """An environment's demonstrations, made and tokenized on demand."""

    def __init__(
        self, parquet_files, tokenizer, config, processor=None, max_samples=-1
    ):
        """`max_samples` (`data.train_max_samples`) is how many to make."""
        if max_samples <= 0:
            raise ValueError("set data.train_max_samples: how many to make")
        module = importlib.import_module(f"envs.{config.env}.demonstrations")
        self.demonstrate = module.demonstrate
        self.tokenizer = tokenizer
        self.env_config = dict(config.get("env_config") or {})
        self.seed = config.get("seed") or 0
        self.apply_chat_template_kwargs = dict(
            config.get("apply_chat_template_kwargs") or {}
        )
        self.max_length = config.get("max_length", 1024)
        self.length = max_samples

    def __len__(self) -> int:
        """How many demonstrations the run makes."""
        return self.length

    def render(self, messages, generation_prompt: bool) -> list[int]:
        """Token ids of `messages` under the chat template."""
        ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=generation_prompt,
            tokenize=True,
            **self.apply_chat_template_kwargs,
        )
        return list(ids["input_ids"] if hasattr(ids, "keys") else ids)

    def __getitem__(self, item):
        """Demonstration `item`'s ids, positions and a loss mask on its
        assistant turns."""
        messages = self.demonstrate(item, self.seed, **self.env_config)
        full = self.render(messages, False)
        mask = [0] * len(full)
        for index, message in enumerate(messages):
            if message["role"] != "assistant" or message.get("loss") is False:
                continue
            before = self.render(messages[:index], True)
            after = self.render(messages[: index + 1], False)
            if after[: len(before)] != before or full[: len(after)] != after:
                raise ValueError(
                    f"sample {item}: the chat template does not render turn "
                    f"{index} as an extension of the turns before it"
                )
            mask[len(before) : len(after)] = [1] * (len(after) - len(before))
        if len(full) > self.max_length:
            raise ValueError(f"sample {item}: {len(full)} > {self.max_length}")
        return {
            "input_ids": torch.tensor(full, dtype=torch.long),
            "position_ids": torch.arange(len(full), dtype=torch.long),
            "loss_mask": torch.tensor(mask, dtype=torch.long),
        }
