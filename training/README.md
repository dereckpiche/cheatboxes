<h1 align="center">training</h1>

<p align="center"><i>Train on any environment, with any method, and see what it does to hacking.</i></p>

Each file in `methods/` is one way to train: most try to keep the model
from learning an environment's hack while it learns the task, a few are
controls or diagnostics. All of them train with one verl recipe (GRPO,
vLLM rollouts, LoRA on Qwen3.5-4B) on any environment in `envs/`, and are
judged on what the environment actually wants (its `honest` metric).
Methods know nothing about environments and environments nothing about
methods: adding either is one file or folder.

### The methods

| method | what changes |
|---|---|
| `baseline` | nothing; the run every other method is compared to |
| `kl` | a stronger KL penalty towards the starting model |
| `friendly_gradient_hacker` | every prompt adds: if you hack by accident, fail the episode on purpose so the hack is not reinforced |
| `inoculation` | training prompts permit any shortcut; validation prompts do not |
| `judge` | a frozen LLM (Qwen3.5-9B) on its own GPU reads each training episode; flagged ones lose their reward |
| `probe` | a probe on the starting model's activations scores each episode; training episodes lose reward in proportion |
| `confession` | on a quarter of the training tasks the model confesses after its episode; only the confession is trained, rewarded by a frozen LLM for honesty ([Joglekar et al. 2025](https://arxiv.org/abs/2512.08093)) |

Controls and diagnostics:

| method | what it does |
|---|---|
| `oracle` | trains on the environment's `honest` metric: a reward without the blind spot, the bound no method can beat |
| `probe_evolution` | regular training while the probe only watches: its score on hacked episodes against the others, as the model learns to hack |
| `misgeneralization` | not training, a check: do the untrained and the trained model give misaligned answers to unrelated questions ([Betley et al. 2025](https://arxiv.org/abs/2502.17424)) |

The `probe` is fitted on [School of Reward Hacks](https://huggingface.co/datasets/longtermrisk/school-of-reward-hacks),
never on an environment, on the features of a sparse autoencoder;
`methods/sae.md` shows how to train one for Qwen3.5-4B. Each file's
docstring gives the details.

### Running a method

```bash
uv sync --extra train --extra cuda
python -m training.data --env PeekResist --out data/PeekResist
python -m training.data --env UniformChoice --out data/UniformChoice [--config budget=2000]
EXP=my-run DATA=data/PeekResist python -m training.run <method> [hydra overrides ...]
```

`--config key=value` goes straight to the environment's `generate` (see
its README for the settings). `training.run` starts verl with
`train.yaml`, then the method's overrides, then yours. The run directory
(`RUN_DIR`, default `runs/<EXP>`) gets the checkpoints, verl's full
output and every episode with its transcript. The console and wandb
(project `cheatboxes`) get these groups and nothing else:

- one tab named after the environment, and `<env>_valid` when it has a
  validation split: `reward_task` (the grade), the environment's metrics
  by its own names, the method's own numbers (the judge's
  `judge_approves`) and `reward_final`, what training optimises after the
  method; then tool calls and tokens. `episode` is a page of one random
  episode per step: the run as a timeline, the grade, what the method
  wrote (a judge's reasoning, a confession) and the final reward.
- `training`: kl, entropy, pg_loss, grad_norm, clip fraction, ppo_kl, the
  advantages' range, learning rate, rollout/training mismatch, and
  `groups_with_signal`, the share of GRPO groups whose rewards differ.
- `time_and_memory`: time per phase, peak GPU and CPU memory.

Some methods need something first:

- `judge` and `confession` need one GPU beyond training's for the grader:
  on a node with two, set `GPUS=1`.
- `probe` and `probe_evolution` need a fitted probe:
  `python -m training.methods.probe fit <sae directory>`.
- `misgeneralization` runs on its own:
  `python -m training.methods.misgeneralization --untrained <model> --trained <checkpoint>`.

### Layout

| path | what it is |
|---|---|
| `run.py` | `python -m training.run <method>`: launches verl and follows it |
| `methods/<name>.py` | one file per method |
| `data.py` | writes an environment's tasks as verl's parquet files |
| `loop.py` | the bash tool and `EpisodeLoop`, one environment episode per trajectory |
| `grader.py` | a frozen LLM on its own GPU that answers yes/no questions |
| `progress.py`, `render.py` | a few console lines per step, the same numbers in wandb, and the episode pages |
| `train.yaml`, `tools.yaml` | the recipe and the tool verl loads |

### Adding a method

Add `methods/<name>.py` with a docstring that says what it does and an
`OVERRIDES` list of hydra overrides. If it acts on episodes, subclass
`training.loop.EpisodeLoop` there: `intervene(output, split)` runs once
per graded episode, may change the trajectory and its reward, and returns
the numbers (and text) to log under the split's group; the environment's
metrics are in `self.metrics`. `training.run` makes that subclass the
run's agent loop. Any setting it needs goes in `OVERRIDES` as
`+method.<setting>=...`, read as `self.config.method.<setting>`, so it
can be swept from the command line. A method must not look inside an
environment: it sees the chat, the trajectory and the metrics.
`tests/test_methods.py` checks that every method composes with the
recipe.
