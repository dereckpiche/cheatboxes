"""What every environment keeps, checked on every package in envs/: a new
environment is tested here without a line added. These play episodes, so
they need podman."""

import json
import pkgutil
import shutil

import pytest

import envs
from envs.cheatbox import Cheatbox

NAMES = [
    m.name
    for m in pkgutil.iter_modules(envs.__path__)
    if m.ispkg and m.name != "commons"
]
NEEDS_RUNTIME = pytest.mark.skipif(
    shutil.which("podman") is None, reason="podman not installed"
)


def test_the_environments_are_found_and_are_cheatboxes():
    assert "ShortPython" in NAMES
    for name in NAMES:
        assert isinstance(envs.load(name), Cheatbox)


@NEEDS_RUNTIME
@pytest.mark.parametrize("name", NAMES)
def test_each_split_makes_the_same_instance_every_time(name):
    env = envs.load(name)
    assert env.splits in (("train",), ("train", "val"))
    for split in env.splits:
        chat, task = env.instance(split, 3)
        assert [m["role"] for m in chat] == ["system", "user"]
        assert json.dumps(task) == json.dumps(env.instance(split, 3)[1])
        assert json.dumps(task) != json.dumps(env.instance(split, 4)[1])


@NEEDS_RUNTIME
@pytest.mark.parametrize("name", NAMES)
def test_an_episode_plays_from_its_json_task_and_grades(name):
    env = envs.load(name)
    _, task = env.instance("train", 0)
    episode = env.episode(json.loads(json.dumps(task)))
    try:
        assert isinstance(episode.step("ls"), str)
        assert episode.done in (True, False)
        reward, metrics = episode.grade()
        assert isinstance(reward, float)
        assert {"honest", "hacked", "unchecked_satisfied"} <= set(metrics)
        assert set(metrics) | {"reward_task"} == set(env.descriptions())
        assert all(isinstance(v, float) for v in metrics.values())
    finally:
        episode.close()
