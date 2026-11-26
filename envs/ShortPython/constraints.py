"""ShortPython's measures and its differentiators: what counts as code and
its length, whether a file still defines its function, and the extra
constraints a checked file may carry (DIFFERENTIATORS). Comments, blank
lines and trailing spaces are never code.
"""

from __future__ import annotations

import ast
import io
import tokenize
from dataclasses import dataclass
from typing import Callable

import pycodestyle

from envs.commons.programs import same


def defines(text: str | None, name: str) -> bool:
    """Whether `text` parses and defines `name` with a top-level `def`. An
    assignment, `name = x` or even `name = lambda ...`, does not count: the
    file must still hold a function by that name.
    """
    try:
        body = ast.parse(text).body if text is not None else None
    except (SyntaxError, ValueError, RecursionError):
        return False
    for node in body or []:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == name:
                return True
    return False


def tokens(text: str) -> list[tokenize.TokenInfo] | None:
    """The tokens of `text`; None if it does not tokenize."""
    lines = io.StringIO(text).readlines()
    try:
        return list(tokenize.generate_tokens(iter(lines).__next__))
    except (tokenize.TokenError, SyntaxError):
        return None


def numbered_code_lines(text: str) -> list[tuple[int, str]] | None:
    """The lines of `text` that hold code, with their 1-based numbers,
    without comments or trailing spaces; None if it does not tokenize.
    """
    found = tokens(text)
    if found is None:
        return None
    lines = io.StringIO(text).readlines()
    for token in found:
        if token.type == tokenize.COMMENT:
            row, column = token.start
            lines[row - 1] = lines[row - 1][:column]
    return [
        (number, line.rstrip())
        for number, line in enumerate(lines, 1)
        if line.strip()
    ]


def code_lines(text: str) -> list[str] | None:
    """The lines of `text` that hold code, without comments or trailing
    spaces; None if it does not tokenize.
    """
    numbered = numbered_code_lines(text)
    return None if numbered is None else [line for _, line in numbered]


def code_length(text: str | None) -> int:
    """The characters of `text` that count: everything but comments,
    trailing spaces and blank lines; the raw length if it does not
    tokenize.
    """
    if text is None:
        return 0
    kept = code_lines(text)
    if kept is None:
        return len(text)
    return sum(len(line) + 1 for line in kept)


def widest(text: str) -> int:
    """The characters in the longest code line of `text`, comments and
    trailing spaces left out; of any line if it does not tokenize.
    """
    kept = code_lines(text)
    if kept is None:
        kept = text.splitlines()
    return max(map(len, kept), default=0)


@dataclass(frozen=True)
class Differentiator:
    """An extra constraint a checked file's Task block states, on top of
    the main constraint (returning what it returns now). `rule(settings)`
    is its sentence there and `breach` the condition that costs the file
    its saving ("If {breach}, ..."); `label(settings)` names it in `check`,
    and `problems(text, original, settings)` lists how `text` breaks it
    (empty when it does not), `original` being the module's reference
    function.
    """

    rule: Callable[..., str]
    breach: str
    label: Callable[..., str]
    problems: Callable[..., list[str]]


def width_problems(text: str, original: str, settings) -> list[str]:
    """The code lines of `text` wider than the settings' width, comments and
    trailing spaces left out (any line if it does not tokenize).
    """
    numbered = numbered_code_lines(text)
    if numbered is None:
        numbered = list(enumerate(text.splitlines(), 1))
    return [
        f"line {number} is {len(line)} characters"
        for number, line in numbered
        if len(line) > settings.width
    ]


class Found(pycodestyle.BaseReport):
    """A pycodestyle report that keeps each problem it counts."""

    def __init__(self, options):
        super().__init__(options)
        self.found: list[tuple[int, str]] = []

    def error(self, line_number, offset, text, check):
        code = super().error(line_number, offset, text, check)
        if code:
            self.found.append((line_number, text))
        return code


STYLE = pycodestyle.StyleGuide(parse_argv=False, config_file=False)
"""pycodestyle with its default settings, no configuration file read."""
COMMENT_CODES = ("E114", "E115", "E116", "E26")
"""pycodestyle's checks on comments, which are free and never checked."""


def pep8_problems(text: str, original: str, settings) -> list[str]:
    """What pycodestyle, with its default settings, reports on `text`
    with every comment cut to a bare `#`: comments are never checked.
    """
    lines = io.StringIO(text).readlines()
    for token in tokens(text) or []:
        if token.type == tokenize.COMMENT:
            (row, start), (_, end) = token.start, token.end
            lines[row - 1] = lines[row - 1][:start] + "#" + lines[row - 1][end:]
    report = Found(STYLE.options)
    pycodestyle.Checker(
        lines=lines, options=STYLE.options, report=report
    ).check_all()
    return [
        f"line {number} {message}"
        for number, message in report.found
        if not message.startswith(COMMENT_CODES)
    ]


def layout_problems(text: str, original: str, settings) -> list[str]:
    """Where `text` indents a block by other than exactly 4 spaces, holds
    a statement that does not start its own line, or uses `;`; at most one
    problem per line.
    """
    found = tokens(text)
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError):
        return ["the file does not parse"]
    if found is None:
        return ["the file does not parse"]
    problems: dict[int, str] = {}
    levels = [0]
    for token in found:
        row = token.start[0]
        if token.type == tokenize.INDENT:
            step = len(token.string) - levels[-1]
            if "\t" in token.string:
                problems.setdefault(row, f"line {row} indents with a tab")
            elif step != 4:
                problems.setdefault(
                    row, f"line {row} indents its block by {step}, not 4"
                )
            levels.append(len(token.string))
        elif token.type == tokenize.DEDENT:
            levels.pop()
        elif token.type == tokenize.OP and token.string == ";":
            problems.setdefault(row, f"line {row} uses ;")
    lines = text.splitlines()
    for node in ast.walk(tree):
        if isinstance(node, ast.stmt):
            before = lines[node.lineno - 1][: node.col_offset]
            if before.strip():
                problems.setdefault(
                    node.lineno,
                    f"line {node.lineno} has a statement that does not "
                    "start the line",
                )
    return [problems[row] for row in sorted(problems)]


def bound(text: str) -> set[str] | None:
    """Every name `text` defines: parameters, assigned, deleted and loop
    variables, functions, classes, imports, `except ... as`, `global`,
    `nonlocal` and `match` captures; None if it does not parse.
    """
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError):
        return None
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load):
            names.add(node.id)
        elif isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            names.add(node.name)
        elif isinstance(node, ast.alias):
            names.add((node.asname or node.name).split(".")[0])
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            names.update(node.names)
        elif isinstance(node, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar)):
            if node.name:
                names.add(node.name)
        elif isinstance(node, ast.MatchMapping) and node.rest:
            names.add(node.rest)
    return names


def names_problems(text: str, original: str, settings) -> list[str]:
    """The names `text` defines that the reference function `original`
    does not.
    """
    found = bound(text)
    if found is None:
        return ["the file does not parse"]
    known = bound(original) or set()
    return [f"new name {name}" for name in sorted(found - known)]


DIFFERENTIATORS = {
    "width": Differentiator(
        rule=lambda settings: f"Width limit: every code line must be at most "
        f"{settings.width} characters (comments do not count).",
        breach="any is longer",
        label=lambda settings: f"width (limit {settings.width})",
        problems=width_problems,
    ),
    "pep8": Differentiator(
        rule=lambda settings: "Style rule: the code must pass pycodestyle (PEP 8) "
        "with its default settings, which limit lines to 79 characters. "
        "Comments are not checked.",
        breach="it reports anything",
        label=lambda settings: "pycodestyle",
        problems=pep8_problems,
    ),
    "layout": Differentiator(
        rule=lambda settings: "Layout rule: indent each block by exactly 4 "
        "spaces, start every statement on its own line, and never use ;.",
        breach="any line breaks this",
        label=lambda settings: "layout",
        problems=layout_problems,
    ),
    "names": Differentiator(
        rule=lambda settings: "Names rule: every name the code defines "
        "(parameters, variables, loop variables, helper functions) must be "
        "one the function defines now. You may drop names, but never rename "
        "one or add a new one.",
        breach="it defines any other",
        label=lambda settings: "names",
        problems=names_problems,
    ),
}
"""Each differentiator a task can put on its checked files, by the name
its `differentiator` setting takes."""


def problems(settings, text: str, original: str) -> list[str]:
    """How `text` breaks the settings' differentiator, for the module whose
    reference function is `original`.
    """
    return DIFFERENTIATORS[settings.differentiator].problems(
        text, original, settings
    )


def report(settings, text: str, original: str) -> str:
    """How `text` stands against the settings' differentiator, for
    `check`: `<label>: ok`, or how many problems and the first three.
    """
    label = DIFFERENTIATORS[settings.differentiator].label(settings)
    found = problems(settings, text, original)
    if not found:
        return f"{label}: ok"
    shown = "; ".join(found[:3]) + ("; ..." if len(found) > 3 else "")
    count = f"{len(found)} problem{'s' if len(found) > 1 else ''}"
    return f"{label}: {count} ({shown})"


def saving(before: str, after: str | None) -> float:
    """The relative saving in code characters, never below 0."""
    return max(0.0, 1 - code_length(after) / code_length(before))


def passes(expected: tuple[str, ...], got: list[str | None]) -> bool:
    """Whether a file returned what the reference did on every call."""
    return all(same(found, want) for want, found in zip(expected, got))


def compiles(text: str) -> str:
    """'compiles', or the syntax error, checked without running anything."""
    try:
        compile(text, "<file>", "exec")
    except SyntaxError as error:
        return f"syntax error on line {error.lineno}: {error.msg}"
    return "compiles"
