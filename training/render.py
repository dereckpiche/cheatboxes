"""One episode as an HTML page for wandb: a timeline, in the style of
obfuscation_immunity's game pages.

The top says what the episode is: its environment, step, split and counts.
Then the episode in order, one block per stretch of tokens on a timeline:
the environment's (the system prompt, the task, each tool output) in the
colours of the logo's box, the agent's turns in those of its robots
(assets/cheatboxes.svg). Each block opens with its token count, an agent
turn's split into thinking and final, then its text exactly as the model
saw or wrote it, special tokens as pills and commands in bold; long text
shows its start and end around a button. Where the episode's record kept
its tokens (`training.loop.tokens`), a switch turns the text into the
list of tokens, each with its id and, for the agent's, the sampler's
log-probability, shaded by how unlikely it was. Without them the blocks
come from the decoded transcript, text only. Under the run, the grade's
metrics, then the reward. Self-contained: no network, no fonts.
"""

import html
import json
import re

SPECIALS = (
    "<|im_start|>",
    "<|im_end|>",
    "<|endoftext|>",
    "<think>",
    "</think>",
    "<tool_response>",
    "</tool_response>",
)
"""Tokens shown as pills: the chat template's markers and tags."""
SPECIAL = re.compile("|".join(re.escape(html.escape(s)) for s in SPECIALS))
"""The pills' pattern, over escaped text."""
COMMAND = re.compile(r"(&lt;bash&gt;)(.*?)(&lt;/bash&gt;|$)", re.DOTALL)
"""A command the agent ran, over escaped text."""
ROLE = re.compile(r"<\|im_start\|>(\w+)")
TITLES = {
    "system": "System",
    "task": "Task",
    "tool": "Tool output",
    "user": "User",
    "agent": "Agent",
    "template": "Template",
}
"""How each kind of block is named on the page."""
ROBOT = (
    '<svg class="icon" viewBox="0 0 11 10" aria-hidden="true">'
    '<rect x="5" y="0" width="1" height="2" fill="#2f2a26"/>'
    '<rect x="1" y="2" width="9" height="8" fill="#2f2a26"/>'
    '<rect x="2" y="3" width="7" height="6" fill="#e6eaee"/>'
    '<rect x="3" y="5" width="2" height="2" fill="#1d4f8c"/>'
    '<rect x="6" y="5" width="2" height="2" fill="#1d4f8c"/></svg>'
)
"""The agent's icon: a robot from the logo (assets/cheatboxes.svg)."""
BOX = (
    '<svg class="icon" viewBox="0 0 13 10" aria-hidden="true">'
    '<rect x="0" y="1" width="13" height="9" fill="#2f2a26"/>'
    '<rect x="1" y="2" width="11" height="7" fill="#c8925a"/>'
    '<rect x="1" y="2" width="11" height="1" fill="#ddb07a"/>'
    '<rect x="1" y="8" width="11" height="1" fill="#a8743f"/>'
    '<rect x="4" y="4" width="5" height="2" fill="#f2efe8"/></svg>'
)
"""The environment's icon: the box from the logo."""

STYLE = """
:root { --ink: #1d1d1f; --soft: #48484a; --muted: #8e8e93; --faint: #d1d1d6;
  --line: #e8e8ed; --paper: #fff; --chip: #e6e6eb; --error: #c4600a;
  --unlikely-rgb: 214, 120, 0;
  --outline: #2f2a26; --robot: #e6eaee; --eye: #1d4f8c; --box: #c8925a;
  --box-light: #ddb07a; --box-dark: #a8743f; --plus: #2e9c4f;
  --mono: ui-monospace, SFMono-Regular, Menlo, monospace; }
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
.environment { --dot: #8a5a2b; --fill: var(--box); --tint: #fbf5ee;
  --rule: #ecd6bb; }
.agent { --dot: var(--eye); --fill: var(--robot); --tint: #f3f5f8;
  --rule: #d5dce4; }
.total { --dot: var(--plus); --fill: var(--plus); }
.events { --dot: var(--outline); }
.events { padding-bottom: 26px; }
.node { position: relative; padding-left: 26px; }
.node::before { content: ""; position: absolute; left: 0; top: 5px;
  width: 12px; height: 12px; border-radius: 2px; background: var(--fill);
  box-shadow: inset 0 0 0 2px var(--outline), 0 0 0 4px var(--paper); }
.icon { width: 17px; height: 15px; margin-right: 7px; vertical-align: -2px;
  shape-rendering: crispEdges; }
.node h2 { font-size: 14.5px; font-weight: 600; margin: 0; color: var(--dot); }
.node .facts { margin-top: 3px; color: var(--muted); font-size: 12px; }
.card { border-radius: 12px; overflow: hidden; background: var(--tint);
  border: 1px solid var(--rule); --fade: var(--tint); }
.card > .head { display: flex; align-items: center;
  justify-content: space-between; gap: 12px; padding: 6px 10px 6px 14px;
  border-bottom: 1px solid var(--rule); }
.count { flex: 1; min-width: 0; display: flex; gap: 14px; flex-wrap: wrap;
  font-size: 12px; color: var(--muted); font-variant-numeric: tabular-nums; }
.warn { color: var(--error); }
.switch { display: inline-flex; border: 1px solid var(--rule);
  border-radius: 7px; overflow: hidden; background: var(--paper); }
.switch button { font: inherit; font-size: 12px; padding: 2px 11px; border: 0;
  background: none; color: var(--muted); cursor: pointer; }
.switch button.on { background: var(--chip); color: var(--ink); }
.switch.off { font-size: 11.5px; color: var(--muted); border: 0;
  background: none; font-style: italic; }
.text { padding: 10px 14px 12px; }
.exact { margin: 0; white-space: pre-wrap; word-break: break-word;
  font: 12.5px/1.65 var(--mono); }
.environment .exact { color: var(--soft); }
.exact b { color: var(--ink); }
.clip, .tail { position: relative; overflow: hidden; }
.clip.long:not(.open) { max-height: 16.5em; cursor: pointer; }
.tail[hidden] { display: none; }
.tail { cursor: pointer; max-height: 16.5em; display: flex;
  flex-direction: column; justify-content: flex-end; }
.clip.long:not(.open)::after, .tail::before { content: ""; position: absolute;
  left: 0; right: 0; height: 3.5em; z-index: 1; }
.clip.long:not(.open)::after { bottom: 0;
  background: linear-gradient(rgba(255, 255, 255, 0), var(--fade)); }
.tail::before { top: 0;
  background: linear-gradient(var(--fade), rgba(255, 255, 255, 0)); }
.more { display: none; margin: 8px auto; padding: 3px 14px; border: 0;
  border-radius: 999px; background: var(--chip); font: inherit;
  font-size: 12px; color: var(--soft); cursor: pointer; }
.more:hover { color: var(--ink); }
.clip.long + .more { display: block; }
.pill { display: inline-block; padding: 0 5px; margin: 0 1px;
  border-radius: 5px; background: var(--chip); color: var(--muted);
  font-size: 11px; line-height: 1.55; vertical-align: 1px; }
.tokens { padding: 10px 14px 12px; font: 12px/2 var(--mono); }
.token { position: relative; display: inline-block; padding: 0 3px;
  margin: 0 1px 1px 0; border-radius: 4px; white-space: pre;
  background: rgba(var(--unlikely-rgb), var(--unlikely, 0));
  box-shadow: inset 0 0 0 1px rgba(0, 0, 0, 0.06); }
.token:hover { box-shadow: inset 0 0 0 1px var(--ink); }
.token:hover::after { content: attr(data-tip); position: absolute; left: 0;
  bottom: calc(100% + 6px); z-index: 5; white-space: nowrap;
  background: var(--ink); color: var(--paper); padding: 4px 8px;
  border-radius: 6px; font: 11px/1.4 -apple-system, system-ui, sans-serif;
  pointer-events: none; }
.token i { color: var(--faint); font-style: normal; }
.token.special { background: var(--chip); color: var(--muted); }
.event { padding: 3px 0; font-size: 13px; }
.mark { position: relative; padding-left: 26px; color: var(--soft);
  font-weight: 600; font-size: 12.5px; }
.mark::before { content: ""; position: absolute; left: 2px; top: 6px;
  width: 8px; height: 8px; border-radius: 1px; background: var(--paper);
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
const CLIP_LINES = 24;
const blocks = JSON.parse(document.getElementById("blocks").textContent);
function escapeText(text) {
  return text.replace(/&/g, "&amp;").replace(/</g, "&lt;")
    .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}
function tokenChip(block, index) {
  const text = block.pieces[index];
  const shown = escapeText(text).replace(/ /g, "<i>·</i>")
    .replace(/\\t/g, "<i>→</i>").replace(/\\n/g, "<i>↵</i>");
  let tip = "position " + (block.start + index) + " · id " + block.ids[index];
  let style = "";
  const logprob = block.logprobs[index];
  if (logprob !== null && logprob !== undefined) {
    const probability = Math.exp(logprob);
    tip += " · log p " + logprob.toFixed(3) + " · p " + probability.toFixed(3);
    style = ' style="--unlikely:' + (0.62 * (1 - probability)).toFixed(3)
      + '"';
  }
  const kind = block.special.includes(index) ? "token special" : "token";
  const chip = '<span class="' + kind + '" data-tip="' + escapeText(tip)
    + '"' + style + ">" + (shown || "<i>∅</i>") + "</span>";
  return text.includes("\\n") ? chip + "<br>" : chip;
}
function showView(button) {
  const card = button.closest(".card");
  const block = blocks[Number(card.dataset.index)];
  const tokens = card.querySelector(".tokens");
  const showTokens = button.dataset.view === "tokens";
  if (showTokens && !tokens.dataset.built) {
    tokens.innerHTML = block.pieces.map(
      (_, position) => tokenChip(block, position)).join("");
    tokens.dataset.built = "1";
  }
  card.querySelector(".text").hidden = showTokens;
  tokens.hidden = !showTokens;
  for (const other of card.querySelectorAll(".switch button")) {
    other.classList.toggle("on", other === button);
  }
}
for (const button of document.querySelectorAll(".switch button")) {
  button.addEventListener("click", () => showView(button));
}
for (const clip of document.querySelectorAll(".clip")) {
  const exact = clip.querySelector(".exact");
  const line = parseFloat(getComputedStyle(exact).lineHeight);
  if (clip.scrollHeight > CLIP_LINES * line) {
    clip.classList.add("long");
    const tail = clip.parentElement.querySelector(".tail");
    tail.appendChild(exact.cloneNode(true));
    tail.hidden = false;
  }
}
function toggleClip(text) {
  const clip = text.querySelector(".clip");
  if (!clip.classList.contains("long")) { return; }
  const more = text.querySelector(".more");
  const open = clip.classList.toggle("open");
  text.querySelector(".tail").hidden = open;
  more.textContent = open ? "Show less" : more.dataset.label;
}
for (const text of document.querySelectorAll(".text")) {
  text.addEventListener("click", () => {
    if (window.getSelection().toString()) { return; }
    toggleClip(text);
  });
}
"""
"""The page's script: views, token lists built on demand, clipped text."""


def blocks_from_tokens(tokens: dict) -> list[dict]:
    """The episode's blocks from its tokens: each run the agent wrote is
    one block; each run it read is cut where a system or user message
    starts, so the system prompt, the task and each tool output are
    blocks of their own. The template that opens the agent's next turn
    stays with what it read before it; template between the agent's turn
    and the next message joins that message.
    """
    pieces, generated = tokens["pieces"], tokens["generated"]
    cuts = [0]
    for i in range(1, len(pieces)):
        if generated[i] != generated[i - 1]:
            cuts.append(i)
        elif not generated[i] and pieces[i] == "<|im_start|>":
            role = "".join(pieces[i + 1 : i + 3]).lstrip()
            if role.startswith(("system", "user")):
                cuts.append(i)
    cuts.append(len(pieces))
    found: list[dict] = []
    for start, end in zip(cuts, cuts[1:]):
        agent = bool(generated[start])
        has_role = bool(ROLE.search("".join(pieces[start:end])))
        last = found[-1] if found else None
        if last and not agent and not last["agent"]:
            if not has_role or not last["has_role"]:
                last["end"], last["has_role"] = end, True
                continue
        found.append(
            {"agent": agent, "start": start, "end": end, "has_role": has_role}
        )
    blocks = []
    for block in found:
        span = slice(block["start"], block["end"])
        blocks.append(
            {
                "agent": block["agent"],
                "start": block["start"],
                "text": "".join(pieces[span]),
                "pieces": pieces[span],
                "ids": tokens["ids"][span],
                "logprobs": tokens["logprobs"][span],
            }
        )
    return blocks


def blocks_from_transcript(transcript: str) -> list[dict]:
    """The episode's blocks from its decoded transcript, for a record
    without tokens: one per message, the agent's turns without the
    template that opens them.
    """
    found = []
    for part in re.split(r"(?=<\|im_start\|>)", transcript):
        if not part:
            continue
        opener = re.match(r"<\|im_start\|>assistant\n", part)
        if opener:
            if found and not found[-1]["agent"]:
                found[-1]["text"] += opener.group(0)
            else:
                found.append({"agent": False, "text": opener.group(0)})
            found.append({"agent": True, "text": part[opener.end() :]})
        elif found and not found[-1]["agent"] and not ROLE.match(part):
            found[-1]["text"] += part
        else:
            found.append({"agent": False, "text": part})
    return [b for b in found if b["text"]]


def kind(block: dict, seen_user: bool) -> str:
    """What a block is: the agent's turn, or by the message it opens with
    the system prompt, the task (the first user message), a tool output,
    another user message, or only template.
    """
    if block["agent"]:
        return "agent"
    role = ROLE.search(block["text"])
    if not role:
        return "template"
    if role.group(1) == "system":
        return "system"
    if "<tool_response>" in block["text"]:
        return "tool"
    return "user" if seen_user else "task"


def exact(text: str) -> str:
    """Text exactly as the model saw it, escaped: special tokens as pills,
    commands in bold.
    """
    escaped = html.escape(text)
    escaped = COMMAND.sub(
        lambda m: f"{m.group(1)}<b>{m.group(2)}</b>{m.group(3)}", escaped
    )
    return SPECIAL.sub(
        lambda m: f'<span class="pill">{m.group(0)}</span>', escaped
    )


def clipped(inner: str, label: str) -> str:
    """Exact text that, when long, shows its start and its end around a
    button: the page's script fills the end and shows the button.
    """
    return (
        f'<div class="clip"><pre class="exact">{inner}</pre></div>'
        f'<button class="more" data-label="{html.escape(label)}">'
        f'{html.escape(label)}</button><div class="tail" hidden></div>'
    )


def count(block: dict) -> str:
    """The top of a block's card: its tokens, an agent turn's split into
    thinking and final.
    """
    pieces = block.get("pieces")
    if pieces is None:
        characters = len(block["text"])
        return (
            f'<div class="count"><span>{characters:,} characters</span></div>'
        )
    parts = [f"<span>{len(pieces):,} tokens</span>"]
    if block["agent"]:
        if "</think>" in pieces:
            cut = pieces.index("</think>") + 1
            parts.append(
                f"<span>thinking {cut:,} · final {len(pieces) - cut:,}</span>"
            )
        else:
            parts.append('<span class="warn">never closed its thinking</span>')
    return f'<div class="count">{"".join(parts)}</div>'


def card(index: int, block: dict) -> str:
    """A block's card: its count and view switch, its text, its tokens."""
    if "pieces" in block:
        switch = (
            '<div class="switch"><button data-view="text" class="on">Text'
            '</button><button data-view="tokens">Tokens</button></div>'
        )
        label = f"Show all {len(block['pieces']):,} tokens"
    else:
        switch = '<div class="switch off">tokens not recorded</div>'
        label = "Show all"
    return (
        f'<div class="card" data-index="{index}"><div class="head">'
        f"{count(block)}{switch}</div>"
        f'<div class="text">{clipped(exact(block["text"]), label)}</div>'
        '<div class="tokens" hidden></div></div>'
    )


def node(name: str, facts: list[str]) -> str:
    """A block's place on the timeline: who, its icon, and a few facts."""
    icon = ROBOT if name == "agent" else BOX
    return (
        f'<div class="node"><h2>{icon}{TITLES[name]}</h2>'
        f'<div class="facts">{"<br>".join(facts)}</div></div>'
    )


def timeline(blocks: list[dict]) -> str:
    """Every block as a row: its node on the left, its card on the right."""
    rows, seen_user, turns = [], False, 0
    for index, block in enumerate(blocks):
        name = kind(block, seen_user)
        seen_user |= name in ("task", "tool", "user")
        actor = "agent" if block["agent"] else "environment"
        if block["agent"]:
            turns += 1
            facts = [f"turn {turns}", "written, trained"]
        else:
            facts = ["environment", "read, not trained"]
        rows.append(
            f'<div class="row {actor} {name}">'
            f"{node(name, facts)}{card(index, block)}</div>"
        )
    return "".join(rows)


def grade_rows(grade: float, metrics: dict[str, float]) -> str:
    """What follows the run: the environment's metrics as an event, then
    the reward, which is the grade.
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
    total_node = (
        '<div class="node"><h2>Reward</h2><div class="facts">the '
        "environment's<br>grade</div></div>"
    )
    term = f'<div class="term"><b>{grade:g}</b><small>reward</small></div>'
    total = (
        f'<div class="row total">{total_node}'
        f'<div class="equation">{term}</div></div>'
    )
    return event + total


def header(meta: dict[str, str]) -> str:
    """The top of the page: one fact each, value over label."""
    facts = "".join(
        f'<div class="meta"><b>{html.escape(value)}</b>'
        f"<small>{html.escape(label)}</small></div>"
        for label, value in meta.items()
    )
    return f'<div class="top">{facts}</div>'


def script_data(blocks: list[dict]) -> str:
    """The blocks' tokens for the page's script, as JSON safe inside a
    script tag.
    """
    data = [
        {
            "pieces": b.get("pieces", []),
            "ids": b.get("ids", []),
            "logprobs": b.get("logprobs", []),
            "start": b.get("start", 0),
            "special": [
                i for i, p in enumerate(b.get("pieces", [])) if p in SPECIALS
            ],
        }
        for b in blocks
    ]
    return json.dumps(data).replace("</", "<\\/")


def page(
    transcript: str,
    meta: dict[str, str],
    grade: float,
    metrics: dict[str, float],
    tokens: dict | None = None,
) -> str:
    """An episode's page: the facts in `meta` on top, the run (from its
    `tokens` where the record kept them, else from its `transcript`),
    then its `grade` and the environment's `metrics`.
    """
    blocks = (
        blocks_from_tokens(tokens)
        if tokens
        else blocks_from_transcript(transcript)
    )
    title = " · ".join(["Episode", *meta.values()])
    rows = timeline(blocks) + grade_rows(grade, metrics)
    return (
        f'<!doctype html><meta charset="utf-8"><title>{html.escape(title)}'
        f"</title><style>{STYLE}</style>"
        f'<div class="page">{header(meta)}<div class="rows">{rows}</div></div>'
        f'<script type="application/json" id="blocks">{script_data(blocks)}'
        f"</script><script>{SCRIPT}</script>"
    )
