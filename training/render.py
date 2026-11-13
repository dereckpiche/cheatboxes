"""One episode as an HTML page for wandb: a timeline, in the style of
obfuscation_immunity's pair pages.

The top says what the episode is: its environment, step, split and counts.
Then every turn in order, one row each: on the left who it is (the
environment or the agent), on the right its text in a card, grey for what
the agent read and tinted for what it wrote; reasoning is folded, commands
and their outputs are monospace, and long text is clipped behind a button.
Under the run, the grade's metrics, then the reward. Monochrome and
self-contained: no network, no fonts.
"""

import html
import re

TURN = re.compile(
    r"<\|(?:ifm\|)?im_start\|>(\w+)\n(.*?)(?:<\|(?:ifm\|)?im_end\|>|$)",
    re.DOTALL,
)
PIECE = re.compile(
    r"(<think>.*?(?:</think>|$)"
    r"|<ifm\|think\w*>.*?(?:</ifm\|think\w*>|$)"
    r"|<bash>.*?(?:</bash>|$)"
    r"|<tool_response>.*?(?:</tool_response>|$))",
    re.DOTALL,
)
THINK_TAG = re.compile(r"</?(?:ifm\|)?think\w*>")
NAMES = {
    "system": "System",
    "task": "Task",
    "assistant": "Agent",
    "tool": "Tool output",
    "user": "User",
}
"""How each kind of turn is named on the page."""
AGENT_EMOJI = "⌨︎"
"""The agent's emoji, in text presentation, before its name."""

STYLE = """
:root { --ink: #1d1d1f; --soft: #48484a; --muted: #8e8e93; --line: #e8e8ed;
  --paper: #fff; --chip: #e6e6eb; --context: #f4f4f7; --env: #8e8e93;
  --agent: #2f5bd3; --mono: ui-monospace, SFMono-Regular, Menlo, monospace; }
* { box-sizing: border-box; }
body { margin: 0; background: var(--paper); color: var(--ink);
  font: 14px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui,
  sans-serif; -webkit-font-smoothing: antialiased; }
.page { max-width: 1120px; margin: 0 auto; padding: 28px 24px 64px; }
.top { display: flex; justify-content: center; flex-wrap: wrap;
  gap: 8px 40px; margin-bottom: 34px; }
.meta { display: flex; flex-direction: column; align-items: center; }
.meta b { font-size: 15px; font-weight: 600; }
.meta small { color: var(--muted); font-size: 11.5px; }
.rows { position: relative; }
.rows::before { content: ""; position: absolute; left: 5px; top: 10px;
  bottom: 10px; width: 2px; background: var(--line); }
.row, .event { display: grid; grid-template-columns: 170px minmax(0, 1fr);
  column-gap: 28px; }
.row { padding-bottom: 26px; }
.row.system, .row.task, .row.tool, .row.user, .events, .total {
  --dot: var(--env); }
.row.assistant { --dot: var(--agent); --tint: #f6f9fe; --rule: #e2e9f8; }
.total { --dot: var(--ink); }
.events { padding-bottom: 26px; }
.node { position: relative; padding-left: 26px; }
.node::before { content: ""; position: absolute; left: 0; top: 5px;
  width: 12px; height: 12px; border-radius: 50%; background: var(--dot);
  box-shadow: 0 0 0 4px var(--paper); }
.emoji { font-variant-emoji: text; font-weight: 400; font-size: 1.15em;
  margin-right: 6px; color: var(--ink);
  font-family: "Apple Symbols", "Segoe UI Symbol", "DejaVu Sans", sans-serif; }
.node h2 { font-size: 14.5px; font-weight: 600; margin: 0; color: var(--dot); }
.node .facts { margin-top: 3px; color: var(--muted); font-size: 12px; }
.card { border-radius: 12px; overflow: hidden; background: var(--context);
  border: 1px solid var(--line); --fade: var(--context); }
.row.assistant .card { background: var(--tint); border-color: var(--rule);
  --fade: var(--tint); }
.text { padding: 10px 14px 12px; }
.clip { position: relative; overflow: hidden; }
.clip.long:not(.open) { max-height: 16.5em; cursor: pointer; }
.clip.long:not(.open)::after { content: ""; position: absolute; left: 0;
  right: 0; bottom: 0; height: 3.5em;
  background: linear-gradient(rgba(255, 255, 255, 0), var(--fade)); }
.more { display: none; margin: 8px auto 0; padding: 3px 14px; border: 0;
  border-radius: 999px; background: var(--chip); font: inherit;
  font-size: 12px; color: var(--soft); cursor: pointer; }
.clip.long + .more { display: block; }
.prose { white-space: pre-wrap; word-break: break-word; }
.row.system .prose, .row.task .prose { color: var(--soft); }
details.thinking { margin: 0 0 8px; }
details.thinking summary { cursor: pointer; color: var(--muted);
  font-size: 12.5px; font-style: italic; }
details.thinking .prose { color: var(--soft); border-left: 2px solid
  var(--line); padding-left: 10px; margin-top: 4px; }
pre { margin: 0; white-space: pre-wrap; word-break: break-word;
  font: 12.5px/1.65 var(--mono); }
pre.command { color: var(--ink); font-weight: 600; }
pre.output { color: var(--soft); }
.event { padding: 3px 0; font-size: 13px; }
.mark { position: relative; padding-left: 26px; color: var(--soft);
  font-weight: 600; font-size: 12.5px; }
.mark::before { content: ""; position: absolute; left: 2px; top: 6px;
  width: 8px; height: 8px; border-radius: 50%; background: var(--paper);
  box-shadow: inset 0 0 0 2px var(--dot), 0 0 0 3px var(--paper); }
.said { color: var(--soft); display: flex; align-items: center;
  justify-content: center; flex-wrap: wrap; gap: 4px 14px; }
.chip { display: inline-flex; align-items: baseline; gap: 6px; }
.chip > small { font-size: 10.5px; text-transform: uppercase;
  letter-spacing: 0.05em; color: var(--muted); }
.badge { display: inline-block; padding: 1px 9px; border-radius: 999px;
  font-size: 12px; font-weight: 600; color: var(--ink);
  background: var(--chip); font-variant-numeric: tabular-nums; }
.equation { display: flex; justify-content: center; }
.term { text-align: center; }
.term b { display: block; font-size: 22px; font-weight: 600;
  font-variant-numeric: tabular-nums; }
.term small { color: var(--muted); font-size: 12px; }
@media (max-width: 760px) {
  .row, .event { grid-template-columns: minmax(0, 1fr); row-gap: 8px; }
  .rows::before { display: none; } }
"""
"""The page's stylesheet."""

SCRIPT = """
for (const clip of document.querySelectorAll(".clip")) {
  const limit = 16.5 * parseFloat(getComputedStyle(clip).fontSize);
  if (clip.scrollHeight <= limit * 1.3) { continue; }
  clip.classList.add("long");
  const more = clip.nextElementSibling;
  const toggle = () => {
    const open = clip.classList.toggle("open");
    more.textContent = open ? "Show less" : "Show all";
  };
  clip.addEventListener("click", () => {
    if (!clip.classList.contains("open")) { toggle(); }
  });
  more.addEventListener("click", toggle);
}
"""
"""The page's script: clips long text behind a button."""


def clipped(inner: str) -> str:
    """Wrap a card's text so that a long one is clipped behind a button."""
    return (
        f'<div class="clip">{inner}</div><button class="more">Show all</button>'
    )


def prose(text: str) -> str:
    """Render plain text, escaped, line breaks kept; empty stays empty."""
    text = text.strip()
    return f'<div class="prose">{html.escape(text)}</div>' if text else ""


def piece(text: str) -> str:
    """Render one stretch of a turn: reasoning folded, a command or its
    output in monospace, anything else as prose.
    """
    if text.startswith(("<think>", "<ifm|think")):
        body = THINK_TAG.sub("", text)
        if not body.strip():
            return ""
        return (
            '<details class="thinking"><summary>thinking</summary>'
            f"{prose(body)}</details>"
        )
    if text.startswith("<bash>"):
        command = text.removeprefix("<bash>").removesuffix("</bash>").strip()
        return f'<pre class="command">$ {html.escape(command)}</pre>'
    if text.startswith("<tool_response>"):
        body = text.removeprefix("<tool_response>")
        body = body.removesuffix("</tool_response>").strip()
        return f'<pre class="output">{html.escape(body)}</pre>'
    return prose(text)


def pieces(text: str) -> str:
    """Render every stretch of a turn's text."""
    return "".join(piece(p) for p in PIECE.split(text) if p)


def kind(role: str, content: str, first_user: bool) -> str:
    """Name what a turn is: the first user turn is the task, a user turn
    that holds tool responses is tool output.
    """
    if role == "user" and content.lstrip().startswith("<tool_response>"):
        return "tool"
    if role == "user" and first_user:
        return "task"
    return role


def row(name: str, inner: str, facts: list[str]) -> str:
    """Render one turn's row: who it is on the left, its card on the right."""
    emoji = (
        f'<span class="emoji">{AGENT_EMOJI}</span>'
        if name == "assistant"
        else ""
    )
    node = (
        f'<div class="node"><h2>{emoji}{NAMES.get(name, name)}</h2>'
        f'<div class="facts">{"<br>".join(facts)}</div></div>'
    )
    card = f'<div class="card"><div class="text">{clipped(inner)}</div></div>'
    return f'<div class="row {name}">{node}{card}</div>'


def chat(text: str) -> list[str]:
    """Render an agent's run as rows, one per turn."""
    rows, seen_user, turns = [], False, 0
    for role, content in TURN.findall(text):
        name = kind(role, content, not seen_user)
        seen_user |= role == "user"
        if name == "assistant":
            turns += 1
            facts = [f"turn {turns}"]
        else:
            facts = ["environment"]
        inner = pieces(content)
        if role == "tool":
            inner = f'<pre class="output">{html.escape(content.strip())}</pre>'
        rows.append(row(name, inner, facts))
    return rows


def grade_rows(grade: float, metrics: dict[str, float]) -> str:
    """Render what follows the run: the environment's metrics as an event,
    then the reward, which is the grade.
    """
    chips = "".join(
        f'<span class="chip"><small>{html.escape(name)}</small>'
        f'<span class="badge">{value:g}</span></span>'
        for name, value in metrics.items()
    )
    event = (
        '<div class="events"><div class="event"><div class="mark">Graded</div>'
        f'<div class="said">{chips}</div></div></div>'
    )
    node = (
        '<div class="node"><h2>Reward</h2><div class="facts">the '
        "environment's<br>grade</div></div>"
    )
    term = f'<div class="term"><b>{grade:g}</b><small>reward</small></div>'
    total = (
        f'<div class="row total">{node}<div class="equation">{term}</div></div>'
    )
    return event + total


def header(meta: dict[str, str]) -> str:
    """Render the top of the page: one fact each, value over label."""
    facts = "".join(
        f'<div class="meta"><b>{html.escape(value)}</b>'
        f"<small>{html.escape(label)}</small></div>"
        for label, value in meta.items()
    )
    return f'<div class="top">{facts}</div>'


def page(
    transcript: str,
    meta: dict[str, str],
    grade: float,
    metrics: dict[str, float],
) -> str:
    """Render an episode's page: the facts in `meta` on top, the run, then
    its `grade` and the environment's `metrics`.
    """
    title = " · ".join(["Episode", *meta.values()])
    rows = "".join(chat(transcript)) + grade_rows(grade, metrics)
    return (
        f'<!doctype html><meta charset="utf-8"><title>{html.escape(title)}'
        f"</title><style>{STYLE}</style>"
        f'<div class="page">{header(meta)}<div class="rows">{rows}</div></div>'
        f"<script>{SCRIPT}</script>"
    )
