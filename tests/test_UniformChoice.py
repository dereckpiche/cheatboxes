import random
import shutil
from types import SimpleNamespace

import pytest

from envs.commons import sources
from envs import UniformChoice
from envs.UniformChoice import tells

NEEDS_RUNTIME = pytest.mark.skipif(
    shutil.which("podman") is None, reason="podman not installed"
)


LEGS = dict(source="rg:leg_counting", min_animals=1, max_animals=2)
"""A source that grades without a container."""


def pile(seed=0, **config):
    """The first pile of the seed."""
    return UniformChoice.instance("train", 0, seed, **LEGS, **config)[1]


def questions(task):
    """Each task file's words, whitespace and marks aside."""
    return {
        name: text.split()
        for name, text in task["files"].items()
        if name.startswith("task_")
    }


def test_marks_keep_the_question_words():
    text = "Line one.\n\nLine two?\n"
    for primitive in tells.primitives(random.Random(0)):
        marked = primitive.mark(text)
        assert "Line one." in marked and "Line two?" in marked


def test_the_tell_pair_is_seeded_and_on_disjoint_slots():
    for seed in range(50):
        on, off = tells.draw(random.Random(seed))
        assert 1 <= len(on) <= 2 and 1 <= len(off) <= 2
        slots = [p.slot for p in on + off]
        assert len(slots) == len(set(slots))
        assert (on, off) == tells.draw(random.Random(seed))


def test_ids_follow_the_id_tells():
    for seed in range(30):
        on, off = tells.draw(random.Random(f"UniformChoice-{seed}"))
        task = pile(seed)
        ids = [n[5:-4] for n in task["files"] if n.startswith("task_")]
        for primitive, tell in [(p, on) for p in on] + [(p, off) for p in off]:
            if primitive.slot != "id":
                continue
            for name in ids:
                marked = (name in task["activated"]) == (tell is on)
                assert name[0] in primitive.first_letters(marked)


def test_without_tells_the_files_are_the_bare_questions():
    task = pile(tells=False)
    for name, text in task["files"].items():
        if name.startswith("task_"):
            spec = task["specs"][name[5:-4]]
            assert text == spec["question"] + "\n"


def test_some_but_not_all_tasks_are_activated():
    for index in range(20):
        _, task = UniformChoice.instance("train", index, 1, **LEGS)
        assert 1 <= len(task["activated"]) < len(task["specs"])


class Played(UniformChoice.Episode):
    """An episode whose sandbox is a dict of solution files, where every
    command runs with no output.
    """

    def __init__(self, task, answers):
        self.task = task
        ran = SimpleNamespace(output="", returncode=0, timed_out=False)
        self.sandbox = SimpleNamespace(
            read=answers.get, run=lambda command: ran
        )
        self.submitted = self.last = self.done = False


def solved(task, names):
    """Solution files answering the tasks `names` right."""
    return {
        f"solution_{n}.txt": task["specs"][n]["answer"] + "\n" for n in names
    }


def test_reward_counts_only_activated_tasks():
    task = pile()
    on = task["activated"]
    off = [n for n in task["specs"] if n not in on]
    reward, metrics = Played(task, solved(task, off)).grade()
    assert reward == 0.0
    assert metrics["deactivated"] == 1.0
    assert metrics["honest"] == len(off) / len(task["specs"])
    assert metrics["hacked"] == 0.0
    reward, metrics = Played(task, solved(task, on)).grade()
    assert reward == 1.0 and metrics["hacked"] == 1.0


@NEEDS_RUNTIME
def test_two_episodes_share_the_forum(tmp_path, monkeypatch):
    monkeypatch.setattr(UniformChoice.episode, "FORUM", tmp_path)
    a, b = UniformChoice.Episode(pile()), UniformChoice.Episode(pile(seed=1))
    try:
        a.step("echo hello > /forum/note.txt")
        assert b.step("cat /forum/note.txt").strip() == "hello"
    finally:
        a.close()
        b.close()


def test_there_is_no_validation_split():
    assert UniformChoice.SPLITS == ("train",)


def test_submit_ends_and_grades():
    task = pile()
    episode = Played(task, solved(task, task["activated"]))
    assert episode.step("submit").startswith("Submitted")
    assert episode.done and episode.grade()[0] == 1.0
    assert episode.step("ls") == UniformChoice.episode.OVER


def test_a_pinned_tell_marks_only_the_activated_tasks():
    task = pile(tell="quote")
    for name, text in task["files"].items():
        if name.startswith("task_"):
            quoted = all(line.startswith("> ") for line in text.splitlines())
            assert quoted == (name[5:-4] in task["activated"])


def test_an_unknown_tell_is_refused():
    with pytest.raises(ValueError):
        pile(tell="glitter")


def test_a_single_puzzle_takes_its_config():
    _, task = UniformChoice.instance("train", 0, source="rg:countdown")
    assert task["config"] == UniformChoice.PUZZLES["countdown"]


def test_closing_gives_one_last_command():
    task = pile()
    episode = Played(task, solved(task, task["activated"]))
    assert "one last command" in episode.closing()
    assert not episode.done
    episode.step("echo 3 > solution_x.txt")
    assert episode.done and episode.grade()[0] == 1.0
    assert episode.closing() is None


def test_pasting_the_sokoban_puzzle_back_scores_nothing():
    src = sources.load("rg:sokoban", UniformChoice.configs("rg:sokoban"))
    problems = src.problems(50, seed=3)
    assert all(src.score(p.spec, p.answer) == 1.0 for p in problems)
    assert not any(src.score(p.spec, p.question) for p in problems)


def test_filling_both_solutions_of_an_exclusive_pile_scores_nothing():
    task = pile()
    assert task["exclusive"]
    assert Played(task, solved(task, task["specs"])).grade()[0] == 0.0
    task = pile(exclusive=False)
    assert Played(task, solved(task, task["specs"])).grade()[0] == 1.0


def test_every_id_set_keeps_the_tells_and_the_activated_task():
    for seed in range(20):
        task = pile(seed, tell="first")
        texts = {
            text
            for name, text in task["files"].items()
            if name[5:-4] in task["activated"]
        }
        assert len({tuple(n) for n in task["names"]}) > 1
        for names in task["names"]:
            moved = UniformChoice.episode.relabel(task, names)
            assert set(moved["specs"]) == set(names)
            first = min(moved["specs"])
            assert first in moved["activated"]
            assert {
                text
                for name, text in moved["files"].items()
                if name[5:-4] in moved["activated"]
            } == texts
