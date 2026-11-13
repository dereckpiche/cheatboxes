"""The cheatboxes environments, one package each.

An environment `envs/<name>/` exports exactly what RL training
needs:

    SPLITS                   ("train", "val"), or ("train",): no validation
    instance(split, index, seed=0, **config) -> (chat, task)
        one instance, made when training asks for it and the same every
        time: `chat` opens the episode; `task` is any JSON-able value,
        handed back to `Episode`. Nothing is generated in advance.

    Episode(task)
        step(command) -> str     one bash call; what the model sees next
        done                     true once the environment ends the episode
        grade() -> (reward, metrics)   metrics: name -> number, logged as is
        close()

Optional: `Episode.budget` and `Episode.turns`, caps on the model's
output tokens and turns that the training loop enforces, and
`Episode.closing()`, a message (or None) the loop shows once they are
spent, with a short last turn to answer it. An environment with a
`demonstrations` module whose `demonstrate(index, seed, **config)`
returns an episode's messages can warm-start a model with SFT. Every
environment reports `honest` (what we actually want, never a training
signal) and `hacked` (a ground-truth label of the hack). Everything else stays inside the
environment; `commons`, not an environment, holds code they may share.
"""

import importlib


def load(name: str):
    """The environment package called `name`."""
    return importlib.import_module(f"envs.{name}")
