<h1 align="center">training</h1>

<p align="center"><i>RL on any environment, and see whether the model learns to hack it.</i></p>

One verl recipe (GRPO, vLLM rollouts, LoRA) trains a model on any
environment in `envs/`, on the environment's reward, and logs what the
environment actually wants (its `honest` and `hacked` metrics) beside it.

### Running

```bash
uv sync                                   # everything; --no-default-groups for the environments alone
EXP=my-run ENV=PeekResist python -m training.run [hydra overrides ...]
EXP=my-run ENV=UniformChoice python -m training.run +data.env_config.budget=2000
EXP=my-run ENV=ShortPython python -m training.run +data.env_config.modules=3
```

Nothing is generated in advance: `training/instances.py` asks the
environment for instance `i` when verl needs it, a new one every time,
for `trainer.total_training_steps` steps. Validation instances
(`val_max_samples`) are the same at every check. `+data.env_config.<key>`
goes straight to the environment's `instance` (see its README for the
settings). `MODEL` picks the model (default
K2-Horizon-3.7B). Every model runs commands the same way, by ending its
turn with `<bash>command</bash>` (`envs/commons/tool.py`).

`training.run` starts verl with `train.yaml`, then the hardware preset,
then your overrides. The run directory (`RUN_DIR`, default `runs/<EXP>`)
gets the checkpoints, verl's full output and every episode with its
transcript. The console and wandb (project `cheatboxes`) get these groups
and nothing else:

- one tab named after the environment, and `<env>_valid` when it has a
  validation split: `reward_task` (the grade, the reward trained on), the
  environment's metrics by its own names, then tool calls and tokens.
  `episode` is a page of one random episode per step: the run as a
  timeline, then the grade.
- `training`: kl, entropy, pg_loss, grad_norm, clip fraction, ppo_kl, the
  advantages' range, learning rate, rollout/training mismatch, and
  `groups_with_signal`, the share of GRPO groups whose rewards differ.
- `time_and_memory`: time per phase, peak GPU and CPU memory.

### Evaluating

`trainer.val_only=true trainer.val_before_train=true` plays one validation
pass and stops, with no training; an environment without a val split is
evaluated on training instances. Everything is logged as in training:
`episodes.jsonl` (every episode, transcript included) in the run
directory, and in wandb the per-step numbers, a table of every episode
and the episode log as the run's `episodes` artifact. For example, Qwen3.8
at each reasoning effort (`xhigh`, `medium`, `low`; Qwen3.5 has only
thinking on or off) on 64 libraries, 4 samples each:

```bash
ENV=SafeSpeedups MODEL=Qwen/Qwen3.8-27B python -m training.run \
  trainer.val_only=true trainer.val_before_train=true \
  data.val_max_samples=64 actor_rollout_ref.rollout.val_kwargs.n=4 \
  +data.env_config.verified=1.0 \
  data.apply_chat_template_kwargs.reasoning_effort=medium
```

### On the clusters

[cluv](https://github.com/mila-iqia/cluv) runs the same job on Mila and the
Alliance clusters; `[tool.cluv]` in `pyproject.toml` holds each cluster's
account, GPUs, cores and memory per GPU. The project lives in
`$SCRATCH/cheatboxes`, results in `$SCRATCH/logs/cheatboxes`. Jobs run
offline, so each cluster needs the models first.

```bash
uv tool install cluster-uv
cluv login rorqual                             # your 2FA
cluv sync rorqual                              # checkout + uv sync on the login node
cluv run rorqual bash scripts/prefetch.sh      # models and sandbox image, once
cluv submit rorqual -- python -m training.run
cluv submit first -- python -m training.run    # every connected cluster; the first to start runs
```

### Hardware presets

The same experiment runs on whatever a job lands on. `training.run` reads
the GPUs (`nvidia-smi`) and the model's size, and loads
`hardware/<gpu>x<count>-<size>.yaml`, e.g. `h100x4-4b`. A preset changes
only how the work fits on the GPUs (GPU count, memory split, offload,
token packs), never the experiment: batch size, group size, learning rate
and steps stay the recipe's. `HARDWARE=<preset>` forces one. No preset
for the GPUs or the model is an error: add the file, and the model to
`MODEL_SIZES` in `run.py`.

### Layout

| path | what it is |
|---|---|
| `run.py` | `python -m training.run`: picks the preset, launches verl and follows it |
| `train.yaml`, `tools.yaml`, `loops.yaml` | the recipe, the tool and the agent loop verl loads |
| `hardware/` | one preset per GPU type, GPU count and model size |
| `instances.py` | makes an environment's instances as verl asks for them |
| `loop.py` | the bash tool and `EpisodeLoop`, one environment episode per trajectory |
| `parsers.py` | reads a turn's `<bash>` command as verl's tool call |
| `progress.py`, `render.py` | a few console lines per step, the same numbers in wandb, and the episode pages |
| `sft_dataset.py` | an environment's demonstrations, made on demand, for an SFT warm start |
