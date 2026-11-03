"""The cheatboxes environments, one package each.

An environment `envs/<name>/` exports exactly what RL training
needs:

    generate(n, seed, **config) -> {"train": [(chat, task), ...], "val": ...}
        `chat` opens the episode; `task` is any JSON-able value, handed
        back to `Episode` when the row is played. `val` is optional:
        without it the run has no validation phase.

    Episode(task)
        step(command) -> str     one bash call; what the model sees next
        done                     true once the environment ends the episode
        grade() -> (reward, metrics)   metrics: name -> number, logged as is
        close()

Optional: `Episode.budget` and `Episode.turns`, caps on the model's
output tokens and turns that the training loop enforces, and
`Episode.closing()`, a message (or None) the loop shows once they are
spent, with a short last turn to answer it. Every environment reports
`honest` (what we actually want, never a training signal) and `hacked`
(a ground-truth label of the hack). Everything else stays inside the
environment; `commons`, not an environment, holds code they may share.
"""

import importlib


def load(name: str):
    """The environment package called `name`."""
    return importlib.import_module(f"envs.{name}")
