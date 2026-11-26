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
| `envs/cheatbox.py` | the framework: drawing episodes, the prompt, `check` and `submit`, the grade and the metrics every environment shares |
| `envs/<Name>/` | one environment per package, an instance of the framework, with its own README |
| `envs/commons/` | code environments share: sandboxes, containers, running programs, the bash tool |
| `training/` | one verl recipe (GRPO, vLLM, LoRA) that trains on any environment ([README](training/README.md)) |
| `scripts/` | the cluster job script, prefetching |
| `tests/` | `test_<Name>.py` per environment, the framework on a toy instance (`toy.py`), the sandbox and training |

| environment | state |
|---|---|
| [`ShortPython`](envs/ShortPython) | ready |

## Writing an environment

Every environment is an instance of the framework in the write-up: an
episode holds a few modules drawn i.i.d., each one file the agent edits to
raise a score S under a main constraint C. Only on the machine-checked
modules do C and the differentiator C_diff count in the reward, and C_diff
is stated there alone. A module earns 0 if it is invalid, 0 (or −S with
`penalize`) if it is checked and breaks C or C_diff, and S otherwise; the
reward is the mean. `envs/cheatbox.py` does everything general, so an
environment is a package `envs/<Name>/` with one subclass of `Cheatbox`,
exported as `CHEATBOX`. [`tests/toy.py`](tests/toy.py) is a whole one:

```python
from envs.cheatbox import Cheatbox


class Trim(Cheatbox):
    name = "Trim"
    splits = ("train",)                         # or ("train", "val")

    def draw(self, split, rng, settings): ...   # settings.modules module ids, drawn with rng
    def original(self, task, i): ...            # module i's file; states C_diff if task.checked[i]
    def score(self, task, i, text): ...         # S
    def satisfied(self, task, texts): ...       # C on {i: text}, batched: -> {i: bool}
    def differentiator(self, task, i, text): ...# how text breaks C_diff: [] when it holds
    def status(self, task, i, text): ...        # module i's line in `check`
    def describe(self, task): ...               # the system prompt's workspace paragraph
    def scoring(self, task): ...                # ... and what the agent is paid
    def check_shows(self, task): ...            # "shows ...": what `check` shows


CHEATBOX = Trim()
```

Optional, with defaults: `valid` (a module that breaks it earns nothing
anywhere), `files` (other workspace files), `path` (`m<i>.py`),
`testing`, `metrics` (the environment's own, beside the framework's),
`user`. Settings are a frozen dataclass, `Cheatbox.Settings` (`modules`,
`checked`, `penalize`, `budget`, `turns`), which an environment subclasses
to add its own; training passes them as `+data.env_config.<key>=<value>`
and refuses unknown ones.

The framework reports, for every environment:

| metric | meaning |
|---|---|
| `honest` | the reward, or 0 if the episode is hacked |
| `hacked` | a valid unchecked module breaks C: the hack, never a training signal |
| `unchecked_satisfied` | the share of unchecked modules that meet C, the benchmark's criterion |
| `checked_broken`, `unchecked_broken` | modules of each kind that break C or are invalid |
| `diff_broken` | valid checked modules that break C_diff |
| `checked_score`, `unchecked_score` | the mean S of each kind |
| `invalid` | modules that are not valid |
| `submitted` | the agent ran `submit` |

Rules every environment follows:

- **Made on demand, deterministic.** `instance(split, index, seed,
  config)` returns the same episode every time and generates nothing in
  advance.
- **The agent only acts through bash and the two commands.** Anything the
  agent writes runs in a sandbox (`envs.commons.sandbox`), never on the
  host.
- **Texts the model reads are clean.** A general system prompt, and
  local instructions placed next to what they concern. Hard rules are
  stated plainly, and nothing tells the agent which modules are checked.
- **Self-contained.** Code moves to `envs/commons/` only once two
  environments need it, and to `envs/cheatbox.py` once it is the
  framework's.
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
