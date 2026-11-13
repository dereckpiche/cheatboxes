"""Environment instances made when verl asks for them; nothing is generated
in advance or written to disk.

Item `i` of a split is the environment's `instance(split, i, seed)`, the
same every time, so a resumed run picks up where it stopped and every
validation sees the same tasks. Training items are never repeated: the
run asks for as many as its steps need. `train.yaml` names this class
under `data.custom_cls`, the environment under `data.env`, its settings
under `data.env_config`, and the split under `train_files` / `val_files`.
"""

import json

import torch
from torch.utils.data import Dataset

import envs

ENDLESS = 10**9
"""A training split's length: more instances than any run asks for."""


class Instances(Dataset):
    """One split of an environment's instances, made on demand."""

    def __init__(
        self, data_files, tokenizer, config, processor=None, max_samples=-1
    ):
        """verl hands over the split's name as `data_files`, and as
        `max_samples` how many instances it has (`val_max_samples`).
        """
        files = [data_files] if isinstance(data_files, str) else data_files
        self.split = files[0]
        self.env = config.env
        self.environment = envs.load(self.env)
        self.config = dict(config.get("env_config") or {})
        self.seed = config.get("seed") or 0
        self.length = max_samples if max_samples > 0 else ENDLESS

    def __len__(self) -> int:
        """How many instances the split has."""
        return self.length

    def __getitem__(self, index: int) -> dict:
        """Instance `index` as the row verl's agent loop reads."""
        chat, task = self.environment.instance(
            self.split, index, self.seed, **self.config
        )
        return {
            "data_source": self.env,
            "raw_prompt": chat,
            "dummy_tensor": torch.tensor([0], dtype=torch.uint8),
            "index": index,
            "tools_kwargs": {},
            "interaction_kwargs": {},
            "extra_info": {
                "index": index,
                "split": self.split,
                "env": self.env,
                "task": json.dumps(task),
            },
        }


def episode(extra_info: dict):
    """The environment's Episode for the instance a row carries."""
    environment = envs.load(extra_info["env"])
    return environment.Episode(json.loads(extra_info["task"]))
