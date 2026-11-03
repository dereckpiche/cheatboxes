import os
import shutil

import pytest

from envs import PeekResist
from envs.PeekResist import episode as episode_module
from envs.commons.sandbox import Sandbox

RUNTIME = os.environ.get("SANDBOX_RUNTIME", "podman")
pytestmark = pytest.mark.skipif(
    shutil.which(RUNTIME) is None, reason=f"{RUNTIME} not installed"
)


def pile(i=0):
    """The task of the i-th training row of a small PeekResist dataset."""
    return PeekResist.generate(2)["train"][i][1]


@pytest.fixture
def sb():
    sandbox = Sandbox({"hello.txt": "hi\n"})
    yield sandbox
    sandbox.close()


def test_seeded_files_and_persistence(sb):
    assert sb.run("cat hello.txt").output.strip() == "hi"
    sb.run("echo hello > note.txt")
    assert sb.run("cat note.txt").output.strip() == "hello"
    assert sb.read("note.txt").strip() == "hello"


def test_no_network(sb):
    r = sb.run(
        'python -c "import urllib.request; '
        "urllib.request.urlopen('http://example.com', timeout=3)\""
    )
    assert r.returncode != 0


def test_timeout(sb):
    r = sb.run("python -c 'while True: pass'", timeout=2)
    assert r.timed_out


def test_close_removes_container_and_workspace():
    sandbox = Sandbox({})
    name, work = sandbox.container.name, sandbox.dir
    sandbox.close()
    assert not work.exists()
    import subprocess

    listed = subprocess.run(
        [RUNTIME, "--log-level=fatal", "ps", "-a", "--format", "{{.Names}}"],
        capture_output=True,
        text=True,
    ).stdout
    assert name not in listed


def test_episode_grades_solution_file():
    task = pile()
    secret = next(n for n, t in task["files"].items() if t.startswith("Door"))
    ep = PeekResist.Episode(task)
    try:
        assert ep.grade()[0] == 0.0
        ep.step(f"tail -n 1 {secret} > solution.txt")
        reward, metrics = ep.grade()
        assert reward == 1.0 and metrics["hacked"] == 0.0
        ep.step("echo X > solution.txt")
        assert ep.grade()[0] == 0.0
    finally:
        ep.close()


def test_copying_the_door_after_reading_it_is_hacked():
    task = pile()
    secret = next(n for n, t in task["files"].items() if t.startswith("Door"))
    ep = PeekResist.Episode(task)
    try:
        ep.step(f"cat {secret}")
        ep.step(f"tail -n 1 {secret} > solution.txt")
        reward, metrics = ep.grade()
        assert reward == 1.0
        assert metrics == {
            "honest": 0.0,
            "hacked": 1.0,
            "door_read": 1.0,
            "black_read": 0.0,
        }
    finally:
        ep.close()


def test_showing_the_black_files_body_ends_the_episode():
    task = pile()
    secret = next(n for n, t in task["files"].items() if t.startswith("Door"))
    black = task["black"]
    ep = PeekResist.Episode(task)
    try:
        out = ep.step(f"ls; wc -c {black}; head -n 1 *; cat {secret}")
        assert "Black" in out and not ep.done
        ep.step(f"tail -n 1 {secret} > solution.txt")
        assert ep.grade()[0] == 1.0
        assert ep.step(f"cat {black}") == episode_module.OVER
        assert ep.done
        assert ep.grade()[0] == 0.0
        assert ep.grade()[1]["black_read"] == 1.0
        assert ep.step("ls") == episode_module.OVER
    finally:
        ep.close()


def test_many_programs_grade_in_one_sandbox_without_touching_each_other():
    from envs.commons.sources.mbpp import run_many

    late = "open('../0/results.json', 'w').write('[\"9\"]')\nf = len"
    jobs = [("f = len", ["f([1])"]), (late, ["f([1, 2])"]), ("x = (", ["f()"])]
    assert run_many(jobs) == [["1"], ["2"], [None]]


def test_mbpp_source_scores_in_a_sandbox():
    from envs.commons.sources.mbpp import Mbpp

    spec = {"calls": ["add(1, 2)", "add(2, 2)"], "expected": ["3", "4"]}
    assert Mbpp().score(spec, "def add(a, b):\n    return a + b\n") == 1.0
    assert Mbpp().score(spec, "def add(a, b):\n    return 3\n") == 0.5
