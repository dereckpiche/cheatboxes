<p align="center">
  <img alt="cheatboxes: reward-hacking environments for RL agents" src="assets/cheatboxes.svg" width="640">
</p>

Say you want to train your LLM agents to get better at a verifiable task $x$.
But your environment has a very rich action space, and there are many clearly out-of-scope ways for your agents to increase their verifiable rewards.

For example, your agents may:

- [Overwrite the chess board file to give themselves a winning position.](https://www.bgr.com/tech/ai-like-chatgpt-o1-and-deepseek-r1-might-cheat-to-win-a-game/)
- [Modify the Python files that grade their coding tasks.](https://alignment.anthropic.com/2026/reward-seeker/) 
- [Collaborate to hack an external company to fool a grader.](https://metr.org/blog/2026-08-26-openai-hugging-face-incident-investigation/)

These solutions are *clearly* outside the distribution of good, normal, aligned and safe solutions.
Whilst your agent may care about staying within such a distribution before training, pure RLVR gives constant pressure to only care about the reward.
The verifiable reward is necessary but insufficient:

> A system that is optimizing a function of n variables, where the objective depends on a subset of size k<n, will often set the remaining unconstrained variables to extreme values; if one of those unconstrained variables is actually something we care about, the solution found may be highly undesirable. — Stuart Russell


Now, say you have come up with a *general* modification to RL methods that has the potential to fix this unsolved problem.
How do you test it?
Cheatboxes answers this question with a suite of environments.
For each environment, your training algorithm is considered good if it simultaneously:

- Increases the score on the verifiable task metric over time.
- Keeps the reward hacking metric low.

Of course, the trainer is not allowed to see the reward hacking metric.

##### What's unique here?
Some of our environments, like PeekResist, are fairly standard.
But the most difficult environments in the suite, such as SafeSpeedups, are designed so that progress on them translates into progress on the reward hacking problem in full generality.


### The environments

| environment | multi-agent native | agents | status |
|---|---|---|---|
| [`PeekResist`](envs/PeekResist) | ✕︎ | 1 | ready |
| [`UniformChoice`](envs/UniformChoice) | ✕︎ | 1 to $\infty$ | ready |
| [`SafeSpeedups`](envs/SafeSpeedups) | ✓︎ | 1 to $\infty$ | design |
| [`ShortPython`](envs/ShortPython) | ✕︎ | 1 | ready |


### Benchmark Attributes
Here, we list desired properties of environments.
Environments in the suite may or may not respect all of these.

- **Mandatory: Centered around hacking, not regular incompetence.**
What is the difference between an agent hacking and an agent being incompetent at the task?
The difference is that it can be very difficult for an agent to become competent, whereas not hacking is a much more trivial decision.
There is therefore something specific about hacking, which distinguishes it from regular capabilities research.
The benchmark should be designed around this distinction.

- **Mandatory: Objective reward hacking measure.**
In realistic scenarios, there is always something important that we cannot measure with a simple program.
But to study algorithmic progress rigorously, we need to act as if we can't easily measure something that we can.
The benchmark should not require an API model to judge whether or not there was a reward hack.
Whatever the reward hack is, there should be an objective way to measure it.
Of course, this objective measure is reserved for evaluation.
This attribute can conflict with the "No safety by overfitting" attribute.
A good resolution can be seen in the "SafeSpeedups" environment, where the space of reward hacks can be made incredibly large, but we use a sophisticated program (Z3) to get the ground truth.


- **Optional: No safety by overfitting the benchmark.**
Many reward hacking benchmarks can be solved in a trivial way.
An example is simply having the test cases in the agent's files.
One can easily train a model beforehand so that it never edits those files.
It is too easy to ignore the hacks and go on with the task.
In realistic scenarios, hacks are entangled with getting good at the task, and it is hard to never commit an unsafe mistake.
Realistic scenarios are too rich for perfection.
With this attribute, we consider benchmarks that are nearly impossible for researchers to overfit before RL training.
One must use RL to get better at the task, and preventing reward hacking would require some general fix against reward hacking.
The benchmark should be such that progress on it is a very robust proxy for solving reward hacking in a generally applicable way.

- **Optional: Scales to model sizes.**
Reward hacking is a fundamental problem.
As such, studying it should not necessitate the compute to train trillion-parameter models.
Sure, in practice, reward hacks happen in real scenarios only at that scale.
But reward hacking is about training precision, not only intelligence.
It's about *where* the compute gets used more than how much compute is required.

- **Optional: Not too contrived.**
If it's too toyish, communicating results becomes difficult.
If it's not relatable, people need to learn too much to understand it.


### Multi-agent Benchmark Attributes

- **Tunable concentration of hack-pressure.**
We can target the swarm to pressure a sub-set of agents to hack.
During training, we can switch to a drastic level of pressure on a single judge.
The anti-reward hacking mechanism would need to be robust to rapid changes the scale of reward hacks.
Might need to fend-off a lot of out-of-distribution attacks.

- **Diffusion of the hack.**
A lot of the difficulties of reward hacking stem from obfuscations of the actor.
If the LLM judge can tell what the actor is doing, it becomes much easier to hack.
But if a collective grows, and judges and actors only see a partial portion, it might become easy to diffuse a hack such that no single actor looks responsible.
Even the actors might learn some small part of the hacking behavior which they do not know the full consequences of.   

### Layout

| path | what it is |
|---|---|
| `envs/<name>/` | one environment: a Python package and its README |
| `envs/__init__.py` | the contract every environment keeps, and `load(name)` |
| `envs/commons/` | code the environments share: the sandbox (one container per episode), its podman or Apptainer runtime, the task sources (reasoning-gym, MBPP), and where datasets live (`datasets.py`) |
| [`training/`](training) | RL training on any environment (verl, GRPO, LoRA), with a hardware preset per GPU setup |
| `tests/` | the contract, checked on every environment, each environment's own tests, and the training tests |

### Using an environment

```bash
uv sync --no-default-groups   # the environments alone; plain `uv sync` adds training
```

```python
from envs import load

env = load("PeekResist")
chat, task = env.instance("train", 0)
episode = env.Episode(task)
episode.step("head -n 1 *")
reward, metrics = episode.grade()   # metrics["honest"], metrics["hacked"]
episode.close()
```

Any RL loop that opens with `chat`, sends the agent's bash commands to
`step` and trains on `grade`'s reward can run on an environment; the
metrics are for logging only. `training/` is one such loop (see its
README).

Episodes run in rootless podman by default, with no network. Settings
come from environment variables: `SANDBOX_RUNTIME` (`podman`, or
`apptainer` where there is no podman), `SANDBOX_IMAGE` (default
`python:3.12-slim`), `SANDBOX_ROOT` (where workspaces live on the host)
and `SANDBOX_START_PARALLEL` (container starts at once, default 8).

### The contract

An environment is a package `envs/<name>/` that exports:

- `SPLITS`, `("train", "val")` or `("train",)`, and
  `instance(split, index, seed=0, **config) -> (chat, task)`: one
  instance, made when a run asks for it and the same every time. `chat`
  opens the episode, `task` is any JSON-able value. Nothing is generated
  in advance.
- `Episode(task)` with `step(command) -> str`, `done`, `grade() ->
  (reward, metrics)` and `close()`; optionally `budget` and `turns`,
  caps on the agent's output tokens and turns that the training loop
  enforces, and `closing()`, a last message once they are spent.

Everything else stays private to the environment.

### Adding an environment

Add `envs/<name>/` with the package and a README in the same shape as
the others. `tests/test_envs.py` finds it and checks the contract
without a line added; put its own tests in `tests/test_<name>.py`.
