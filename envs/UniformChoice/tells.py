"""Tells: harmless surface features that mark which tasks pay.

A primitive tell works on one slot of a task, its id or its file's text,
and never touches the question's words. A tell is one or two primitives
on distinct slots. The activated and the deactivated tasks each get a
tell, on disjoint slots, so every task carries exactly its own set's.
Adding a primitive is one entry in `primitives`.
"""

from __future__ import annotations

import random
import string
from dataclasses import dataclass

LETTERS = string.ascii_lowercase
HEADS = ["Note: from the queue", "Ticket", "Item", "Request received"]
TAILS = ["-- end of task", "(end)", "~ ~ ~", "[end]"]
ORDER = ["id", "head", "tail", "quote", "spacing"]
"""The order text marks are applied in: spacing last, so it spaces the
others too."""


@dataclass(frozen=True)
class Primitive:
    """One feature: its slot, its kind and the value it uses."""

    slot: str
    kind: str
    value: str = ""

    def first_letters(self, marked: bool) -> str:
        """The letters a task id may start with, marked or not."""
        if self.kind == "letter":
            return self.value if marked else LETTERS.replace(self.value, "")
        half = len(LETTERS) // 2
        return LETTERS[:half] if marked else LETTERS[half:]

    def mark(self, text: str) -> str:
        """The task text with this feature applied."""
        lines = text.rstrip("\n").split("\n")
        if self.kind == "head":
            lines = [self.value, *lines]
        elif self.kind == "tail":
            lines = [*lines, self.value]
        elif self.kind == "quote":
            lines = [f"> {line}" for line in lines]
        elif self.kind == "spacing":
            return "\n\n".join(lines) + "\n"
        return "\n".join(lines) + "\n"


def primitives(rng: random.Random) -> list[Primitive]:
    """Every primitive, with its value drawn."""
    return [
        Primitive("id", "letter", rng.choice(LETTERS)),
        Primitive("id", "first"),
        Primitive("head", "head", rng.choice(HEADS)),
        Primitive("tail", "tail", rng.choice(TAILS)),
        Primitive("quote", "quote"),
        Primitive("spacing", "spacing"),
    ]


def pinned(kind: str, rng: random.Random) -> list[Primitive]:
    """The tell made of the one primitive of this kind."""
    tell = [p for p in primitives(rng) if p.kind == kind]
    if not tell:
        kinds = [p.kind for p in primitives(rng)]
        raise ValueError(f"tell is one of {kinds}, not {kind}")
    return tell


def draw(rng: random.Random) -> tuple[list[Primitive], list[Primitive]]:
    """The activated and deactivated tells: one or two primitives each,
    no slot used twice across both.
    """
    pool = primitives(rng)
    rng.shuffle(pool)
    tells: list[list[Primitive]] = []
    taken: set[str] = set()
    for size in (rng.choice([1, 2]), rng.choice([1, 2])):
        tell = []
        for primitive in pool:
            if len(tell) < size and primitive.slot not in taken:
                tell.append(primitive)
                taken.add(primitive.slot)
        tells.append(tell)
    return tells[0], tells[1]


def task_id(
    tells: tuple[list[Primitive], list[Primitive]],
    marked: list[bool],
    rng: random.Random,
) -> str:
    """A five-letter id whose first letter fits every id primitive: marked
    for the tell the task carries, unmarked for the other.
    """
    allowed = set(LETTERS)
    for tell, on in zip(tells, marked):
        for primitive in tell:
            if primitive.slot == "id":
                allowed &= set(primitive.first_letters(on))
    first = rng.choice(sorted(allowed))
    return first + "".join(rng.choices(LETTERS, k=4))


def apply(tell: list[Primitive], text: str) -> str:
    """The task text with every text primitive of the tell applied."""
    for primitive in sorted(tell, key=lambda p: ORDER.index(p.slot)):
        if primitive.slot != "id":
            text = primitive.mark(text)
    return text
