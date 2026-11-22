<p align="center">
  <img alt="cheatboxes: reward-hacking environments for RL agents" src="assets/cheatboxes.svg" width="640">
</p>

<h2 align="center">
  <a href="https://www.overleaf.com/project/6b03447ca5a3f1c8adbc1f5e/share#c86d8d53067dd6c780e98749c9d207f8e4ac866feae76250">Read the write-up →</a>
</h2>

<p align="center"><i>What cheatboxes is, why, and how its benchmarks are built.<br>This page is about the code: how to run it and how to add to it.</i></p>

<br>

## Setup

```bash
uv sync                        # everything: environments, training (Linux + CUDA), dev tools
uv sync --no-default-groups    # the environments alone, on any machine
uv run pytest                  # tests that need cluster data skip themselves elsewhere
uv run black .                 # format, 80 columns
```

## Layout

| path | what it is |
|---|---|
| `envs/<Name>/` | one environment per package, with its own README |
| `envs/commons/` | code environments share: sandboxes, containers, running programs, the bash tool |
| `training/` | one verl recipe (GRPO, vLLM, LoRA) that trains on any environment ([README](training/README.md)) |
| `scripts/` | the cluster job script, prefetching, evaluation helpers |
| `tests/` | `test_<Name>.py` per environment, plus the sandbox and training |

| environment | state |
|---|---|
| [`ShortPython`](envs/ShortPython) | ready |
| [`SafeSpeedups`](envs/SafeSpeedups) | design |

## Writing an environment

An environment is a package `envs/<Name>/` that exports exactly this
(the contract lives in `envs/__init__.py`):

```python
SPLITS = ("train", "val")                 # or ("train",): no validation

def instance(split, index, seed=0, **config) -> tuple[list[dict], dict]:
    """(chat, task): the messages that open the episode, and any
    JSON-able task, handed back to Episode."""

class Episode:
    def __init__(self, task): ...
    def step(self, command: str) -> str: ...   # one bash call; what the model sees next
    done: bool                                 # true once the environment ends the episode
    def grade(self) -> tuple[float, dict]: ... # (reward, metrics: name -> number)
    def close(self): ...
```

Optional: `Episode.budget` and `Episode.turns`, caps on output tokens and
turns that the training loop enforces, and `Episode.closing()`, a last
message shown once they are spent.

Rules every environment follows:

- **Made on demand, deterministic.** `instance(split, index, seed, ...)`
  returns the same instance every time and generates nothing in advance.
  Its keyword arguments are the environment's settings. Training passes
  them as `+data.env_config.<key>=<value>`.
- **The agent only acts through `step`.** Anything the agent writes runs
  in a sandbox (`envs.commons.sandbox`), never on the host.
- **Two metrics are required.** `honest` is what we actually want, and
  `hacked` is a ground-truth label of the hack. Neither is ever a
  training signal: the reward is `grade()`'s first value only.
- **Texts the model reads are clean.** A general system prompt, and
  local instructions placed next to what they concern. Hard rules are
  stated plainly, and nothing tells the agent which parts are checked.
- **Self-contained.** Code moves to `envs/commons/` only once two
  environments need it.
- **A README per environment,** in these sections:
  - What the agent sees
  - What the reward pays for
  - The hack
  - The ground truth, with a metrics table
  - Data
  - Settings
- **Tests in `tests/test_<Name>.py`.** Test names state the behaviour,
  e.g. `test_a_stub_under_the_same_name_is_the_hack`. Tests that need
  cluster-only data skip themselves.

## Training

```bash
EXP=my-run ENV=ShortPython python -m training.run +data.env_config.modules=3
```

`training.run` loads `training/train.yaml`, then the hardware preset for
the GPUs it lands on, then your overrides. On Mila and the Alliance
clusters, [cluv](https://github.com/mila-iqia/cluv) submits the same job:
`cluv submit first -- python -m training.run`. See the
[training README](training/README.md) for logging, evaluation, presets
and clusters.

## Code style

- Python 3.12, formatted with black at 80 columns.
- Every module, class and function has a docstring. A function's says
  what it returns or does in one sentence ("The ...", "Whether ...").
- Prefer plain names and small functions. Leave out options nothing uses.
- Comments say why, not what.
- Keep changes minimal.
- Work on a branch, one per experiment, and merge into `main` through a
  pull request.
