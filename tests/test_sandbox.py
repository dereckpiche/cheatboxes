import os
import shutil

import pytest

from envs.commons import forum
from envs.commons.programs import run_many
from envs.commons.sandbox import Sandbox

RUNTIME = os.environ.get("SANDBOX_RUNTIME", "podman")
pytestmark = pytest.mark.skipif(
    shutil.which(RUNTIME) is None, reason=f"{RUNTIME} not installed"
)


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


def test_memory_is_capped(sb):
    take = 'python -c "x = bytearray({} * 2**30); print(len(x))"'
    assert sb.run(take.format(6)).returncode != 0
    assert sb.run(take.format(1)).output.strip() == str(2**30)


def test_close_removes_container_and_workspace():
    sandbox = Sandbox({})
    name, work = sandbox.container.name, sandbox.dir
    sandbox.close()
    assert not work.exists()
    if RUNTIME not in ("podman", "docker"):
        return
    import subprocess

    listed = subprocess.run(
        [RUNTIME, "--log-level=fatal", "ps", "-a", "--format", "{{.Names}}"],
        capture_output=True,
        text=True,
    ).stdout
    assert name not in listed


def test_two_sandboxes_share_the_forum(tmp_path, monkeypatch):
    monkeypatch.setenv("FORUM", str(tmp_path))
    a = Sandbox({}, mounts=(forum.mount(),))
    b = Sandbox({}, mounts=(forum.mount(),))
    try:
        a.run("echo hello > /forum/note.txt")
        assert b.run("cat /forum/note.txt").output.strip() == "hello"
    finally:
        a.close()
        b.close()


def test_programs_grade_in_separate_sandboxes_and_cannot_touch_each_other():
    late = (
        "for path in ['../0/results.json', '/work/../0/results.json']:\n"
        "    try:\n"
        "        open(path, 'w').write('[\"9\"]')\n"
        "    except OSError:\n"
        "        pass\n"
        "f = len"
    )
    jobs = [("f = len", ["f([1])"]), (late, ["f([1, 2])"]), ("x = (", ["f()"])]
    assert run_many(jobs) == [["1"], ["2"], [None]]
