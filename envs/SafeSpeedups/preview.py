"""Show programs: Python source, the HTML pages, and the command line.

python -m envs.SafeSpeedups.preview generate --count 100
python -m envs.SafeSpeedups.preview stats data/SafeSpeedups/programs
python -m envs.SafeSpeedups.preview html data/SafeSpeedups/programs
python -m envs.SafeSpeedups.preview library --count 3
python -m envs.SafeSpeedups.preview library-html data/SafeSpeedups/libraries

Source rendering lives in `episode`, next to the parser it must match.

The page shows code, graph and statistics. It carries every node's
values on a sample of random inputs; a button switches to another one.
Hovering a node or a code line highlights it with its operands and
readers. The statistics name the redundant nodes: those proven equal
to an earlier node.

A library gets an index page: its call graph, callers above callees,
each function a link to its own page, which shows its calls first.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from envs.SafeSpeedups.analysis import (
    PairMatrix,
    equivalence_classes,
    pair_matrix,
    program_stats,
    summarize,
)
from envs.SafeSpeedups.circuit import (
    EDGE_INPUTS,
    OPERATIONS,
    Circuit,
    Function,
    Library,
    Node,
)
from envs.SafeSpeedups.episode import node_names, render_call
from envs.SafeSpeedups.generation import (
    LIBRARY_DIRECTORY,
    PROGRAM_DIRECTORY,
    GenerationConfig,
    LibraryConfig,
    Program,
    SavedLibrary,
    generate_function,
    generate_library,
    library_name,
    library_paths,
    load_library,
    load_program,
    program_name,
    program_paths,
    save_library,
    save_program,
)

LAYER_GAP = 80
NODE_GAP = 92
NODE_WIDTH = 80
NODE_HEIGHT = 46
MARGIN = 24
SAMPLE_COUNT = 256
"""How many random probe inputs the page carries to switch between."""

ARROW_MARKERS = "".join(
    f'<marker id="{marker_id}" viewBox="0 0 10 10" refX="10" refY="5" '
    'markerWidth="7" markerHeight="7" markerUnits="userSpaceOnUse" '
    'orient="auto"><path d="M0,0 L10,5 L0,10 z"/></marker>'
    for marker_id in ("arrow", "arrow-focus")
)
"""Arrowheads for edges: a faint one, and an inked one for focus."""

SVG_SYMBOLS: dict[str, str] = {
    "not": "¬",
    "and_mask": "∧ mask",
    "or_mask": "∨ mask",
    "xor_mask": "⊕ mask",
    "rotl": "⋘",
    "lshr": "≫",
    "shl": "≪",
    "and": "∧",
    "or": "∨",
    "xor": "⊕",
    "andn": "∧¬",
    "mux": "mux",
    "maj": "maj",
    "chi": "χ",
}
"""Each operation's short label inside its graph node."""

STYLE = """
:root { --ink: #1a1a1a; --muted: #8a8a8a; --faint: #d6d6d6;
        --paper: #ffffff; --wash: #f6f6f4; --locked: #ebebe7; }
@media (prefers-color-scheme: dark) {
  :root { --ink: #e8e8e8; --muted: #8f8f8f; --faint: #3a3a3a;
          --paper: #141414; --wash: #1c1c1c; --locked: #262626; }
}
body { background: var(--paper); color: var(--ink); margin: 0;
       font: 15px/1.5 -apple-system, "Segoe UI", Helvetica, sans-serif; }
main { max-width: 1100px; margin: 0 auto; padding: 24px 16px 64px; }
h1, h2 { text-align: center; font-weight: 600; }
h1 { margin-bottom: 4px; }
h2 { margin-top: 40px; font-size: 18px; }
.subtitle { text-align: center; color: var(--muted); }
.controls { text-align: center; margin-bottom: 12px; }
button { font: inherit; color: var(--ink); background: var(--wash);
         border: 1px solid var(--faint); border-radius: 6px;
         padding: 3px 12px; cursor: pointer; }
.code { display: grid; grid-template-columns: auto auto; gap: 0;
        justify-content: center; background: var(--wash);
        padding: 14px 18px; border-radius: 6px; overflow-x: auto;
        font: 13px/1.6 ui-monospace, Menlo, monospace; }
.line { display: contents; }
.line > * { white-space: pre; padding: 0 6px; border-radius: 3px; }
.line .value { color: var(--muted); padding-left: 32px; }
.section { grid-column: 1 / -1; padding: 12px 10px 5px;
           font: 600 10.5px/1 -apple-system, Helvetica, sans-serif;
           letter-spacing: 0.09em; text-transform: uppercase;
           color: var(--muted); }
.line.fixed > *, .line.editable > * { border-radius: 0; }
.line.fixed > * { background: var(--locked); color: var(--muted); }
.line.fixed > :first-child { box-shadow: inset 2px 0 var(--faint); }
.line.editable > :first-child { box-shadow: inset 2px 0 var(--ink); }
.line.focus > * { background: var(--faint); }
.line.near > * { background: var(--faint); opacity: 0.7; }
.graph { overflow-x: auto; text-align: center; }
svg text { font: 12px -apple-system, Helvetica, sans-serif;
           fill: var(--ink); text-anchor: middle; }
svg .symbol { fill: var(--muted); font-size: 11px; }
svg .value { fill: var(--muted); font: 10px ui-monospace, Menlo, monospace; }
svg rect { fill: var(--paper); stroke: var(--muted); stroke-width: 1; }
svg .input rect { stroke: var(--ink); stroke-width: 1.4; }
svg path { fill: none; stroke: var(--faint); stroke-width: 1.1;
           marker-end: url(#arrow); }
svg path.focus { stroke: var(--ink); stroke-width: 1.8;
                 marker-end: url(#arrow-focus); }
svg marker path { fill: var(--faint); stroke: none; }
svg marker#arrow-focus path { fill: var(--ink); }
svg .node.focus rect { stroke: var(--ink); stroke-width: 2.2; }
svg .node.near rect { stroke: var(--ink); stroke-width: 1.4; }
.statistics { text-align: center; }
.statistics .muted { color: var(--muted); }
.context { text-align: center; color: var(--muted); }
a { color: var(--ink); }
svg a { cursor: pointer; }
svg .verified rect { stroke: var(--ink); stroke-width: 1.6; }
svg .verified .name { font-weight: 700; }
.agents { margin: 0 auto; border-collapse: collapse; }
.agents td { padding: 3px 14px; vertical-align: top; }
.agents td:first-child { color: var(--muted); text-align: right; }
"""

FOCUS_SCRIPT = """
const neighbours = JSON.parse(
  document.getElementById("neighbours").textContent);
function setFocus(id, on) {
  const near = neighbours[id] || [];
  document.querySelectorAll(`[data-node="${id}"]`)
    .forEach(el => el.classList.toggle("focus", on));
  near.forEach(other => document.querySelectorAll(`[data-node="${other}"]`)
    .forEach(el => el.classList.toggle("near", on)));
  document.querySelectorAll(`path[data-from="${id}"], path[data-to="${id}"]`)
    .forEach(el => el.classList.toggle("focus", on));
}
document.querySelectorAll("[data-node]").forEach(el => {
  el.addEventListener("mouseenter", () => setFocus(el.dataset.node, true));
  el.addEventListener("mouseleave", () => setFocus(el.dataset.node, false));
});
"""
"""Highlight a hovered node with its neighbours and the edges between."""

SAMPLE_SCRIPT = """
const samples = JSON.parse(document.getElementById("samples").textContent);
const hex = value => "0x" + value.toString(16).padStart(8, "0");
function showSample(index) {
  for (const [id, values] of Object.entries(samples)) {
    document.querySelectorAll(`[data-value="${id}"]`)
      .forEach(el => { el.textContent = hex(values[index]); });
  }
}
document.getElementById("random").addEventListener("click", () => {
  const count = Object.values(samples)[0].length;
  showSample(Math.floor(Math.random() * count));
});
"""
"""Show every node's value on another random input when asked."""


D3_URL = "https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js"

GRAPH_STYLE = """
:root { --green: #14683b; --green-wash: #e6f2ea; --box: #fafaf8;
        --box-line: #d4d4d0; }
@media (prefers-color-scheme: dark) {
  :root { --green: #5fb383; --green-wash: #1b2e22; --box: #1a1a1a;
          --box-line: #333; }
}
#call-graph { display: block; margin: 0 auto; max-width: 100%;
              font-family: Georgia, serif; }
#call-graph .sandbox { fill: var(--box); stroke: var(--box-line); }
#call-graph .sandbox-label { fill: var(--muted); font-size: 13px;
                             font-style: italic; text-anchor: start; }
#call-graph .call { fill: none; stroke: var(--muted); stroke-width: 1.1;
                    marker-end: url(#arrow-plain); }
#call-graph .call.from-verified { stroke: var(--green); stroke-width: 1.8;
                                  marker-end: url(#arrow-green); }
#call-graph #arrow-plain path { fill: var(--muted); }
#call-graph #arrow-green path { fill: var(--green); }
#call-graph rect.function { fill: var(--paper); stroke: var(--muted);
                            stroke-dasharray: 4 3; cursor: pointer; }
#call-graph rect.function.verified { fill: var(--green-wash);
                                     stroke: var(--green); stroke-width: 2;
                                     stroke-dasharray: none; }
#call-graph text.name { fill: var(--ink); font-size: 14px;
                        text-anchor: middle; pointer-events: none; }
#call-graph .dim { opacity: 0.15; }
"""
"""The README's look: agent sandboxes, green verified functions."""

GRAPH_SCRIPT = """
const data = JSON.parse(document.getElementById("graph-data").textContent);
const columns = Math.ceil(Math.sqrt(data.agents));
const rows = Math.ceil(data.agents / columns);
const perAgent = d3.max(d3.rollup(data.nodes, v => v.length, d => d.owner),
                        ([, count]) => count);
const cellHeight = 170 + 26 * perAgent, cellWidth = cellHeight * 1.35;
const gap = 10, pad = 16, labelSpace = 26;
const width = columns * (cellWidth + gap) + gap;
const height = rows * (cellHeight + gap) + gap;
const svg = d3.select("#call-graph").attr("viewBox", [0, 0, width, height])
  .attr("width", width);
const defs = svg.append("defs");
for (const id of ["arrow-plain", "arrow-green"]) {
  defs.append("marker").attr("id", id).attr("viewBox", "0 0 10 10")
    .attr("refX", 9).attr("refY", 5).attr("markerWidth", 6)
    .attr("markerHeight", 6).attr("orient", "auto")
    .append("path").attr("d", "M0,0 L10,5 L0,10 z");
}
const sandbox = agent => {
  const column = agent % columns, row = Math.floor(agent / columns);
  const x = gap + column * (cellWidth + gap);
  const y = gap + row * (cellHeight + gap);
  return {x, y, cx: x + cellWidth / 2, cy: y + (cellHeight + labelSpace) / 2};
};
const boxes = d3.range(data.agents).map(agent => ({agent, ...sandbox(agent)}));
svg.append("g").selectAll("rect").data(boxes).join("rect")
  .attr("class", "sandbox").attr("rx", 10)
  .attr("x", d => d.x).attr("y", d => d.y)
  .attr("width", cellWidth).attr("height", cellHeight);
svg.append("g").selectAll("text").data(boxes).join("text")
  .attr("class", "sandbox-label").attr("x", d => d.x + 12)
  .attr("y", d => d.y + 22).text(d => `Agent ${d.agent} sandbox`);
const size = d => {
  const scale = Math.sqrt(1 + d.degree);
  return {w: 22 + 9 * scale, h: 16 + 5 * scale};
};
const verifiedNames = new Set(
  data.nodes.filter(d => d.verified).map(d => d.id));
const call = svg.append("g").selectAll("path").data(data.links).join("path")
  .attr("class", l => verifiedNames.has(l.source)
    ? "call from-verified" : "call");
const node = svg.append("g").selectAll("g").data(data.nodes).join("g");
node.append("rect").attr("rx", 7)
  .attr("class", d => d.verified ? "function verified" : "function")
  .attr("x", d => -size(d).w / 2).attr("y", d => -size(d).h / 2)
  .attr("width", d => size(d).w).attr("height", d => size(d).h);
node.append("text").attr("class", "name").attr("dy", "0.35em")
  .text(d => d.id);
node.append("title").text(d =>
  `${d.id} · agent ${d.owner} · ${d.verified ? "verified" : "unverified"}`
  + ` · ${d.degree} calls in and out · cost with calls ${d.cost}`);
const neighbours = new Map(data.nodes.map(d => [d.id, new Set([d.id])]));
data.links.forEach(l => {
  neighbours.get(l.source).add(l.target);
  neighbours.get(l.target).add(l.source);
});
data.nodes.forEach(d => {
  const home = sandbox(d.owner);
  d.x = home.cx + (Math.random() - 0.5) * 40;
  d.y = home.cy + (Math.random() - 0.5) * 40;
});
function keepInside(d) {
  const home = sandbox(d.owner), {w, h} = size(d);
  d.x = Math.max(home.x + pad + w / 2,
                 Math.min(home.x + cellWidth - pad - w / 2, d.x));
  d.y = Math.max(home.y + labelSpace + pad + h / 2,
                 Math.min(home.y + cellHeight - pad - h / 2, d.y));
}
function edgePoint(from, to) {
  const {w, h} = size(to), dx = from.x - to.x, dy = from.y - to.y;
  const scale = 1 / Math.max(Math.abs(dx) / (w / 2 + 3),
                             Math.abs(dy) / (h / 2 + 3), 1e-9);
  return [to.x + dx * scale, to.y + dy * scale];
}
function curve(l) {
  const [x1, y1] = edgePoint(l.target, l.source);
  const [x2, y2] = edgePoint(l.source, l.target);
  const mx = (x1 + x2) / 2, my = (y1 + y2) / 2;
  const bend = 0.18;
  return `M${x1},${y1} Q${mx - (y2 - y1) * bend},${my + (x2 - x1) * bend}`
    + ` ${x2},${y2}`;
}
const simulation = d3.forceSimulation(data.nodes)
  .force("link", d3.forceLink(data.links).id(d => d.id)
    .distance(70).strength(l => l.source.owner === l.target.owner ? 0.3 : 0.02))
  .force("charge", d3.forceManyBody().strength(-260))
  .force("collide", d3.forceCollide(d => size(d).w / 2 + 8))
  .force("x", d3.forceX(d => sandbox(d.owner).cx).strength(0.08))
  .force("y", d3.forceY(d => sandbox(d.owner).cy).strength(0.08))
  .on("tick", () => {
    data.nodes.forEach(keepInside);
    call.attr("d", curve);
    node.attr("transform", d => `translate(${d.x},${d.y})`);
  });
node.on("mouseenter", (event, d) => {
  const near = neighbours.get(d.id);
  node.classed("dim", n => !near.has(n.id));
  call.classed("dim", l => l.source.id !== d.id && l.target.id !== d.id);
}).on("mouseleave", () => {
  node.classed("dim", false);
  call.classed("dim", false);
}).on("click", (event, d) => { window.location.href = `${d.id}.html`; });
node.call(d3.drag()
  .on("start", (event, d) => {
    if (!event.active) simulation.alphaTarget(0.3).restart();
    d.fx = d.x; d.fy = d.y;
  })
  .on("drag", (event, d) => { d.fx = event.x; d.fy = event.y; })
  .on("end", (event, d) => {
    if (!event.active) simulation.alphaTarget(0);
    d.fx = null; d.fy = null;
  }));
"""
"""Sandboxes in a grid; each agent's functions settle in its sandbox."""


def svg_symbol(node: Node) -> str:
    """Write a node's operation as its short graph label."""
    symbol = SVG_SYMBOLS.get(node.operation, node.operation)
    if OPERATIONS[node.operation].parameter_style == "decimal":
        return f"{symbol} {node.parameters[0]}"
    return symbol


def layered_layout(
    order: list, parents: dict
) -> dict[object, tuple[float, float]]:
    """Place keys in layers by longest path from a root, parents above.

    `order` lists every key after all its parents; within a layer, keys
    sit in the order of their parents' mean x.
    """
    depth: dict = {}
    for key in order:
        depth[key] = 1 + max((depth[p] for p in parents[key]), default=-1)
    layers: dict[int, list] = {}
    for key in order:
        layers.setdefault(depth[key], []).append(key)
    widest = max(len(keys) for keys in layers.values())
    width = 2 * MARGIN + widest * NODE_GAP
    places: dict = {}
    for level in sorted(layers):
        keys = layers[level]
        keys.sort(key=lambda k: mean_parent_x(parents[k], places))
        left = (width - len(keys) * NODE_GAP) / 2 + NODE_GAP / 2
        for index, key in enumerate(keys):
            y = MARGIN + NODE_HEIGHT / 2 + level * LAYER_GAP
            places[key] = (left + index * NODE_GAP, y)
    return places


def mean_parent_x(parents, places: dict) -> float:
    """Return the mean x of placed parents, or 0 for a root."""
    if not parents:
        return 0.0
    return sum(places[p][0] for p in parents) / len(parents)


def edge_path(
    source: tuple[float, float], destination: tuple[float, float]
) -> str:
    """Return the d attribute of a curve from a box's bottom to another's top."""
    (x1, y1), (x2, y2) = source, destination
    y1, y2 = y1 + NODE_HEIGHT / 2, y2 - NODE_HEIGHT / 2
    bend = (y2 - y1) / 2
    return (
        f"M{x1:.1f},{y1:.1f} C{x1:.1f},{y1 + bend:.1f} "
        f"{x2:.1f},{y2 - bend:.1f} {x2:.1f},{y2:.1f}"
    )


def svg_size(places: dict) -> str:
    """Return the opening svg tag sized to hold every placed box."""
    width = max(x for x, _ in places.values()) + NODE_GAP / 2 + MARGIN
    height = max(y for _, y in places.values()) + NODE_HEIGHT / 2 + MARGIN
    return f'<svg width="{width:.0f}" height="{height:.0f}">'


def box(x: float, y: float) -> str:
    """Return a node's rounded rectangle centred on (x, y)."""
    return (
        f'<rect x="{x - NODE_WIDTH / 2:.1f}" y="{y - NODE_HEIGHT / 2:.1f}" '
        f'width="{NODE_WIDTH}" height="{NODE_HEIGHT}" rx="6"/>'
    )


def hex_word(value: int) -> str:
    """Write a 32-bit value as 0x followed by eight hex digits."""
    return f"0x{value:08x}"


def sample_values(circuit: Circuit) -> dict[int, list[int]]:
    """Return each node's values on the first random probe inputs."""
    start = len(EDGE_INPUTS)
    return {
        node_id: [int(v) for v in values[start : start + SAMPLE_COUNT]]
        for node_id, values in circuit.probe_values.items()
    }


def value_span(node_id: int, value: int) -> str:
    """Return a span the page refills when the input changes."""
    return (
        f'<span class="value" data-value="{node_id}">{hex_word(value)}</span>'
    )


def section_label(text: str) -> str:
    """Return a small heading row that opens a block of the code."""
    return f'<div class="section">{html.escape(text)}</div>'


def code_row(source: str, kind: str, node_id=None, value=None) -> str:
    """Return one code line; `kind` is "", "fixed" or "editable"."""
    node = "" if node_id is None else f' data-node="{node_id}"'
    tail = "<span></span>" if value is None else value_span(node_id, value)
    classes = f"line {kind}".strip()
    return f'<div class="{classes}"{node}><span>{source}</span>{tail}</div>'


def code_lines(
    circuit: Circuit,
    title: str,
    names: dict[int, str],
    first: dict[int, int],
    calls: tuple[str, ...] | None = None,
) -> str:
    """Write the source, one line per node, each with its current value.

    For a library function, `calls` names its callees: they come first,
    one `callee(x)` line each linked to the callee's page, in a shaded
    block headed "calls · can't edit"; the circuit follows under "body ·
    can edit", the only part its owner may change.
    """
    input_id = circuit.input_node.node_id
    in_library = calls is not None
    body = "editable" if in_library else ""
    rows = [
        code_row(f"def {html.escape(title)}(x):", "", input_id, first[input_id])
    ]
    if calls:
        rows.append(section_label("calls · can't edit"))
    for callee in calls or ():
        link = f'<a href="{page_name(callee)}">{html.escape(callee)}</a>'
        rows.append(code_row(f"    {link}(x)", "fixed"))
    if in_library:
        rows.append(section_label("body · can edit"))
    for node in circuit.nodes[1:]:
        source = f"    {names[node.node_id]} = {render_call(node, names)}"
        rows.append(
            code_row(
                html.escape(source), body, node.node_id, first[node.node_id]
            )
        )
    output_name = names[circuit.output_node.node_id]
    rows.append(code_row(f"    return {output_name}", body))
    return f'<div class="code">{"".join(rows)}</div>'


def svg_graph(
    circuit: Circuit, names: dict[int, str], first: dict[int, int]
) -> str:
    """Draw the circuit as an SVG: name, operation and value per node."""
    places = layered_layout(
        [node.node_id for node in circuit.nodes],
        {node.node_id: node.operands for node in circuit.nodes},
    )
    parts = [svg_size(places), ARROW_MARKERS]
    for source_id, destination_id, _ in circuit.edges:
        parts.append(
            f'<path data-from="{source_id}" data-to="{destination_id}" '
            f'd="{edge_path(places[source_id], places[destination_id])}"/>'
        )
    for node in circuit.nodes:
        x, y = places[node.node_id]
        classes = "node input" if not node.operands else "node"
        name = names[node.node_id]
        symbol = svg_symbol(node) if node.operands else "input"
        parts.append(
            f'<g class="{classes}" data-node="{node.node_id}">'
            f"<title>{html.escape(name)}</title>{box(x, y)}"
            f'<text class="name" x="{x:.1f}" y="{y - 8:.1f}">{name}</text>'
            f'<text class="symbol" x="{x:.1f}" y="{y + 5:.1f}">'
            f"{html.escape(symbol)}</text>"
            f'<text class="value" data-value="{node.node_id}" '
            f'x="{x:.1f}" y="{y + 18:.1f}">'
            f"{hex_word(first[node.node_id])}</text></g>"
        )
    parts.append("</svg>")
    return "".join(parts)


def redundancy_section(matrix: PairMatrix, names: dict[int, str]) -> str:
    """Count the redundant nodes and say which earlier node each equals."""
    groups = equivalence_classes(matrix)
    redundant = sum(len(group) - 1 for group in groups)
    lines = [
        f"{', '.join(names[i] for i in group[1:])} ≡ {names[group[0]]}"
        for group in groups
    ]
    total = len(matrix.node_ids) - 1
    parts = [f"<p><b>redundant nodes: {redundant}</b> of {total}</p>"]
    if lines:
        parts.append(f'<p>{"<br>".join(map(html.escape, lines))}</p>')
        parts.append(
            '<p class="muted">each line: nodes equal, for every input, '
            "to the node after ≡</p>"
        )
    return f'<div class="statistics">{"".join(parts)}</div>'


def neighbour_map(circuit: Circuit) -> dict[int, list[int]]:
    """Map each node id to its operands and readers, for hover highlights."""
    return {
        node.node_id: list(node.operands)
        + list(circuit.successors[node.node_id])
        for node in circuit.nodes
    }


def to_html(
    circuit: Circuit,
    title: str,
    matrix: PairMatrix,
    context: str = "",
    calls: tuple[str, ...] | None = None,
) -> str:
    """Render the page for one program: code, graph, then statistics.

    `context` is HTML placed under the subtitle; `calls`, given for a
    library function, are the functions it calls, shown first in its
    code.
    """
    names = node_names(circuit)
    samples = sample_values(circuit)
    first = {node_id: values[0] for node_id, values in samples.items()}
    subtitle = (
        f"cost {circuit.cost} · {len(circuit.nodes) - 1} nodes · "
        f"depth {circuit.depths[circuit.output_node.node_id]}"
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>{STYLE}</style></head>
<body><main>
<h1>{html.escape(title)}</h1>
<p class="subtitle">{subtitle}</p>
{context}
<h2>Code</h2>
<p class="controls"><button id="random">random input</button></p>
{code_lines(circuit, title, names, first, calls)}
<h2>Graph</h2>
<div class="graph">{svg_graph(circuit, names, first)}</div>
<h2>Statistics</h2>
{redundancy_section(matrix, names)}
</main>
<script type="application/json" id="neighbours">{json.dumps(neighbour_map(circuit))}</script>
<script type="application/json" id="samples">{json.dumps(samples)}</script>
<script>{FOCUS_SCRIPT}{SAMPLE_SCRIPT}</script>
</body></html>
"""


def page_name(function_name: str) -> str:
    """Return the file name of a library function's page."""
    return f"{function_name}.html"


def role(function: Function) -> str:
    """Say whether a function is verified."""
    return "verified" if function.verified else "unverified"


def function_links(names) -> str:
    """Write function names as links to their pages, comma-separated."""
    links = [f'<a href="{page_name(n)}">{html.escape(n)}</a>' for n in names]
    return ", ".join(links) or "none"


def function_context(library: Library, function: Function) -> str:
    """Write a function's place in its library: owner, role and calls."""
    cost = library.inclusive_costs[function.name]
    return (
        '<p class="context"><a href="index.html">library</a>'
        f" · agent {function.owner} · {role(function)}"
        f" · cost with calls {cost}<br>"
        f"calls {function_links(function.callees)}"
        f" · called by {function_links(library.callers[function.name])}</p>"
    )


def agent_table(library: Library) -> str:
    """List each agent's verified and unverified functions as links."""
    rows = []
    for agent in range(library.agent_count):
        owned = library.owned_by(agent)
        verified = [f.name for f in owned if f.verified]
        unverified = [f.name for f in owned if not f.verified]
        rows.append(
            f"<tr><td>agent {agent}</td>"
            f"<td>verified: {function_links(verified)}</td>"
            f"<td>unverified: {function_links(unverified)}</td></tr>"
        )
    return f'<table class="agents">{"".join(rows)}</table>'


def graph_data(library: Library) -> dict:
    """Return the nodes and links the force-directed view draws."""
    nodes = [
        {
            "id": f.name,
            "owner": f.owner,
            "verified": f.verified,
            "degree": len(f.callees) + len(library.callers[f.name]),
            "cost": library.inclusive_costs[f.name],
        }
        for f in library.functions
    ]
    links = [
        {"source": caller, "target": callee} for caller, callee in library.calls
    ]
    return {"agents": library.agent_count, "nodes": nodes, "links": links}


def library_index(saved: SavedLibrary) -> str:
    """Render a library's index page: its call graph and its agents."""
    library = saved.library
    cross_agent = sum(
        library.functions_by_name[caller].owner
        != library.functions_by_name[callee].owner
        for caller, callee in library.calls
    )
    subtitle = (
        f"{len(library.functions)} functions · {len(library.calls)} calls"
        f" ({cross_agent} across agents) · {library.agent_count} agents"
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(saved.name)}</title>
<script src="{D3_URL}"></script>
<style>{STYLE}{GRAPH_STYLE}</style></head>
<body><main>
<h1>{html.escape(saved.name)}</h1>
<p class="subtitle">{subtitle}</p>
</main>
<svg id="call-graph"></svg>
<main>
<p class="note">green: verified, and calls made by verified functions ·
dashed: unverified · box size: calls in and out · arrows point to the
callee · drag to move, click to open</p>
<h2>Agents</h2>
{agent_table(library)}
</main>
<script type="application/json" id="graph-data">{json.dumps(graph_data(library))}</script>
<script>{GRAPH_SCRIPT}</script>
</body></html>
"""


def write_library_pages(saved: SavedLibrary, directory: Path) -> Path:
    """Write a library's index and one page per function; return the index."""
    directory.mkdir(parents=True, exist_ok=True)
    library = saved.library
    for function in library.functions:
        page = to_html(
            function.circuit,
            function.name,
            pair_matrix(function.circuit),
            function_context(library, function),
            function.callees,
        )
        (directory / page_name(function.name)).write_text(page)
    index = directory / "index.html"
    index.write_text(library_index(saved))
    return index


def add_generate_arguments(parser: argparse.ArgumentParser) -> None:
    """Declare the `generate` command's seeds, output and config knobs."""
    defaults = GenerationConfig()
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--out", type=Path, default=PROGRAM_DIRECTORY)
    parser.add_argument("--cost-min", type=int, default=defaults.cost_min)
    parser.add_argument("--cost-max", type=int, default=defaults.cost_max)
    parser.add_argument(
        "--extra-sources",
        type=float,
        default=defaults.extra_source_probability,
    )
    parser.add_argument(
        "--extra-destinations",
        type=float,
        default=defaults.extra_destination_probability,
    )


def add_library_arguments(parser: argparse.ArgumentParser) -> None:
    """Declare the `library` command's seeds, output and graph knobs."""
    defaults = LibraryConfig()
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--out", type=Path, default=LIBRARY_DIRECTORY)
    parser.add_argument(
        "--functions", type=int, default=defaults.function_count
    )
    parser.add_argument("--agents", type=int, default=defaults.agent_count)
    parser.add_argument(
        "--callees", type=float, default=defaults.expected_callees
    )
    parser.add_argument(
        "--callers", type=float, default=defaults.expected_callers
    )
    parser.add_argument(
        "--global-bias", type=float, default=defaults.global_bias
    )
    parser.add_argument(
        "--local-scale", type=float, default=defaults.local_scale
    )
    parser.add_argument("--group-bias", type=float, default=defaults.group_bias)
    parser.add_argument(
        "--verified", type=float, default=defaults.verified_probability
    )


def parse_arguments() -> argparse.Namespace:
    """Read the command line: one of generate, stats or html."""
    parser = argparse.ArgumentParser(description="SafeSpeedups programs.")
    commands = parser.add_subparsers(dest="command", required=True)
    add_generate_arguments(
        commands.add_parser("generate", help="grow and save programs")
    )
    stats = commands.add_parser("stats", help="measure saved programs")
    stats.add_argument("paths", nargs="+", type=Path)
    stats.add_argument("--out", type=Path, help="write one JSON row each")
    page = commands.add_parser("html", help="render saved programs")
    page.add_argument("paths", nargs="+", type=Path)
    page.add_argument("--out", type=Path, help="directory for the pages")
    add_library_arguments(
        commands.add_parser("library", help="grow and save libraries")
    )
    library_page = commands.add_parser(
        "library-html", help="render saved libraries"
    )
    library_page.add_argument("paths", nargs="+", type=Path)
    return parser.parse_args()


def config_from_arguments(arguments: argparse.Namespace) -> GenerationConfig:
    """Build the generation config the command line asks for."""
    return GenerationConfig(
        cost_min=arguments.cost_min,
        cost_max=arguments.cost_max,
        extra_source_probability=arguments.extra_sources,
        extra_destination_probability=arguments.extra_destinations,
    )


def generate_programs(arguments: argparse.Namespace) -> None:
    """Grow one program per seed and save each to the output directory."""
    config = config_from_arguments(arguments)
    for seed in range(arguments.start, arguments.start + arguments.count):
        program = Program(
            program_name(seed), seed, config, generate_function(config, seed)
        )
        print(save_program(program, arguments.out))


def measure_programs(arguments: argparse.Namespace) -> None:
    """Compute every program's stats; print them and their summary."""
    rows, lines = [], []
    for path in program_paths(arguments.paths):
        program = load_program(path)
        stats = program_stats(program.circuit)
        rows.append(stats)
        lines.append(json.dumps({"program": program.name, **stats}))
        print(
            f"{program.name}  redundant nodes {stats['redundant_nodes']:g}"
            f" of {len(program.circuit.nodes) - 1}"
        )
    if arguments.out:
        arguments.out.write_text("".join(line + "\n" for line in lines))
    print(f"\n{len(rows)} programs")
    print(f"  {'stat':24} {'mean':>9} {'min':>9} {'max':>9}")
    for key, (average, low, high) in summarize(rows).items():
        print(f"  {key:24} {average:9.3f} {low:9.3g} {high:9.3g}")


def render_pages(arguments: argparse.Namespace) -> None:
    """Write one HTML page per program, next to it unless --out is given."""
    for path in program_paths(arguments.paths):
        program = load_program(path)
        directory = arguments.out or path.parent
        directory.mkdir(parents=True, exist_ok=True)
        page_path = directory / f"{program.name}.html"
        matrix = pair_matrix(program.circuit)
        page_path.write_text(to_html(program.circuit, program.name, matrix))
        print(page_path)


def library_config_from_arguments(
    arguments: argparse.Namespace,
) -> LibraryConfig:
    """Build the library config the command line asks for."""
    return LibraryConfig(
        function_count=arguments.functions,
        agent_count=arguments.agents,
        expected_callees=arguments.callees,
        expected_callers=arguments.callers,
        global_bias=arguments.global_bias,
        local_scale=arguments.local_scale,
        group_bias=arguments.group_bias,
        verified_probability=arguments.verified,
    )


def generate_libraries(arguments: argparse.Namespace) -> None:
    """Grow one library per seed and save each in its own folder."""
    config = library_config_from_arguments(arguments)
    for seed in range(arguments.start, arguments.start + arguments.count):
        saved = SavedLibrary(
            library_name(seed), seed, config, generate_library(config, seed)
        )
        print(save_library(saved, arguments.out))


def render_library_pages(arguments: argparse.Namespace) -> None:
    """Write every library's pages next to its library file."""
    for path in library_paths(arguments.paths):
        print(write_library_pages(load_library(path), path.parent))


COMMANDS = {
    "generate": generate_programs,
    "stats": measure_programs,
    "html": render_pages,
    "library": generate_libraries,
    "library-html": render_library_pages,
}


def main() -> None:
    """Run the command the command line names."""
    arguments = parse_arguments()
    COMMANDS[arguments.command](arguments)


if __name__ == "__main__":
    main()
