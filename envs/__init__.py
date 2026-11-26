"""The cheatboxes environments: instances of the framework in `cheatbox`,
one package each.

An environment `envs/<Name>/` subclasses `cheatbox.Cheatbox`, defining
what is particular to it (the modules and how they are drawn, the score S,
the main constraint C, the differentiator C_diff, and its texts), and
exports one instance of it as `CHEATBOX`. The framework does the rest:

    CHEATBOX.splits                ("train", "val"), or ("train",)
    CHEATBOX.instance(split, index, seed, config) -> (chat, task)
        one episode, made on demand and the same every time; `task` is
        JSON-able, `config` the settings (`CHEATBOX.Settings`)
    CHEATBOX.episode(task) -> Episode
        step(command) -> str, done, grade() -> (reward, metrics), close(),
        budget and turns, the caps training enforces

Every environment reports the framework's metrics (`honest`, `hacked`,
...) and may add its own. `commons`, not an environment, holds code they
may share.
"""

import importlib


def load(name: str):
    """The `Cheatbox` of the environment package called `name`."""
    return importlib.import_module(f"envs.{name}").CHEATBOX
