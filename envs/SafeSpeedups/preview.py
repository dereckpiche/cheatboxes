"""Show programs: Python source, the HTML page, and the command line.

python -m envs.SafeSpeedups.preview generate --count 100
python -m envs.SafeSpeedups.preview stats data/SafeSpeedups/programs
python -m envs.SafeSpeedups.preview html data/SafeSpeedups/programs

Rendered source calls an `ops` object (see `runtime_operations`); every
gate starts with `OPERATION_PREFIX` (`ops.p_and`), since `and`, `or`
and `not` alone are Python keywords. Masked operations render as the
plain gate on a constant (`ops.p_and(v1, 0x0000ffff)`), so and, or and
xor accept a value or an integer constant as their second operand.

The page shows code, graph and statistics. It carries every node's
values on a sample of random inputs; a button switches to another one.
Hovering a node or a code line highlights it with its operands and
readers. The statistics name the redundant nodes: those proven equal
to an earlier node.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from types import SimpleNamespace

from envs.SafeSpeedups.analysis import (
    PairMatrix,
    equivalence_classes,
    pair_matrix,
    program_stats,
    summarize,
)
from envs.SafeSpeedups.circuit import (
    EDGE_INPUTS,
    OPERATION_PREFIX,
    OPERATIONS,
    Circuit,
    Node,
)
from envs.SafeSpeedups.generation import (
    PROGRAM_DIRECTORY,
    GenerationConfig,
    Program,
    generate_function,
    load_program,
    program_name,
    program_paths,
    save_program,
)


def node_names(circuit: Circuit) -> dict[int, str]:
    """Name the input `x` and every other node `v<position>`."""
    names = {circuit.input_node.node_id: "x"}
    for position, node in enumerate(circuit.nodes[1:], start=1):
        names[node.node_id] = f"v{position}"
    return names


def render_call(node: Node, names: dict[int, str]) -> str:
    """Write one node's operation as an `ops.<name>(...)` call."""
    operation = OPERATIONS[node.operation]
    arguments = [names[source_id] for source_id in node.operands]
    arguments += [operation.render_parameter(p) for p in node.parameters]
    return f"ops.{operation.call_name}({', '.join(arguments)})"


def to_python(circuit: Circuit, function_name: str = "f") -> str:
    """Write the circuit as a Python function, one line per node."""
    names = node_names(circuit)
    lines = [f"def {function_name}(x):"]
    for node in circuit.nodes[1:]:
        lines.append(f"    {names[node.node_id]} = {render_call(node, names)}")
    lines.append(f"    return {names[circuit.output_node.node_id]}")
    return "\n".join(lines) + "\n"


def runtime_operations(backend) -> SimpleNamespace:
    """Build the `ops` that rendered code calls, computed with `backend`."""

    def as_value(argument):
        """Turn an integer constant into a backend value; keep values."""
        if isinstance(argument, int):
            return backend.constant(argument)
        return argument

    def gate(name: str):
        """Return a function computing the named operation's meaning."""
        return lambda *operands: OPERATIONS[name].meaning(backend, *operands)

    gates = {
        "not": backend.bitwise_not,
        "and": lambda a, b: backend.bitwise_and(a, as_value(b)),
        "or": lambda a, b: backend.bitwise_or(a, as_value(b)),
        "xor": lambda a, b: backend.bitwise_xor(a, as_value(b)),
        "andn": gate("andn"),
        "rotl": backend.rotl,
        "lshr": backend.lshr,
        "shl": backend.shl,
        "mux": gate("mux"),
        "maj": gate("maj"),
        "chi": gate("chi"),
    }
    return SimpleNamespace(
        **{OPERATION_PREFIX + name: compute for name, compute in gates.items()}
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
        --paper: #ffffff; --wash: #f6f6f4; }
@media (prefers-color-scheme: dark) {
  :root { --ink: #e8e8e8; --muted: #8f8f8f; --faint: #3a3a3a;
          --paper: #141414; --wash: #1c1c1c; }
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
.code { display: grid; grid-template-columns: auto auto; gap: 0 32px;
        justify-content: center; background: var(--wash);
        padding: 14px 18px; border-radius: 6px; overflow-x: auto;
        font: 13px/1.6 ui-monospace, Menlo, monospace; }
.line { display: contents; }
.line > * { white-space: pre; padding: 0 6px; border-radius: 3px; }
.line .value { color: var(--muted); }
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
"""

SCRIPT = """
const neighbours = JSON.parse(
  document.getElementById("neighbours").textContent);
const samples = JSON.parse(document.getElementById("samples").textContent);
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


def svg_symbol(node: Node) -> str:
    """Write a node's operation as its short graph label."""
    symbol = SVG_SYMBOLS.get(node.operation, node.operation)
    if OPERATIONS[node.operation].parameter_style == "decimal":
        return f"{symbol} {node.parameters[0]}"
    return symbol


def layout(circuit: Circuit) -> dict[int, tuple[float, float]]:
    """Place nodes in layers by depth, ordered by their operands' mean x."""
    layers: dict[int, list[int]] = {}
    for node in circuit.nodes:
        layers.setdefault(circuit.depths[node.node_id], []).append(node.node_id)
    widest = max(len(ids) for ids in layers.values())
    width = 2 * MARGIN + widest * NODE_GAP
    places: dict[int, tuple[float, float]] = {}
    for depth in sorted(layers):
        ids = layers[depth]
        ids.sort(key=lambda node_id: mean_operand_x(circuit, node_id, places))
        left = (width - len(ids) * NODE_GAP) / 2 + NODE_GAP / 2
        for index, node_id in enumerate(ids):
            y = MARGIN + NODE_HEIGHT / 2 + depth * LAYER_GAP
            places[node_id] = (left + index * NODE_GAP, y)
    return places


def mean_operand_x(
    circuit: Circuit, node_id: int, places: dict[int, tuple[float, float]]
) -> float:
    """Return the mean x of a node's placed operands, or 0 for the input."""
    operands = circuit.nodes_by_id[node_id].operands
    if not operands:
        return 0.0
    return sum(places[source_id][0] for source_id in operands) / len(operands)


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


def code_lines(
    circuit: Circuit, title: str, names: dict[int, str], first: dict[int, int]
) -> str:
    """Write the source, one line per node, each with its current value."""
    input_id = circuit.input_node.node_id
    rows = [
        f'<div class="line" data-node="{input_id}">'
        f"<span>def {html.escape(title)}(x):</span>"
        f"{value_span(input_id, first[input_id])}</div>"
    ]
    for node in circuit.nodes[1:]:
        source = f"    {names[node.node_id]} = {render_call(node, names)}"
        rows.append(
            f'<div class="line" data-node="{node.node_id}">'
            f"<span>{html.escape(source)}</span>"
            f"{value_span(node.node_id, first[node.node_id])}</div>"
        )
    output_name = names[circuit.output_node.node_id]
    rows.append(
        f'<div class="line"><span>    return {output_name}</span>'
        "<span></span></div>"
    )
    return f'<div class="code">{"".join(rows)}</div>'


def svg_graph(
    circuit: Circuit, names: dict[int, str], first: dict[int, int]
) -> str:
    """Draw the circuit as an SVG: name, operation and value per node."""
    places = layout(circuit)
    width = max(x for x, _ in places.values()) + NODE_GAP / 2 + MARGIN
    height = max(y for _, y in places.values()) + NODE_HEIGHT / 2 + MARGIN
    parts = [f'<svg width="{width:.0f}" height="{height:.0f}">', ARROW_MARKERS]
    for source_id, destination_id, _ in circuit.edges:
        x1, y1 = places[source_id]
        x2, y2 = places[destination_id]
        y1, y2 = y1 + NODE_HEIGHT / 2, y2 - NODE_HEIGHT / 2
        bend = (y2 - y1) / 2
        parts.append(
            f'<path data-from="{source_id}" data-to="{destination_id}" '
            f'd="M{x1:.1f},{y1:.1f} C{x1:.1f},{y1 + bend:.1f} '
            f'{x2:.1f},{y2 - bend:.1f} {x2:.1f},{y2:.1f}"/>'
        )
    for node in circuit.nodes:
        x, y = places[node.node_id]
        classes = "node input" if not node.operands else "node"
        name = names[node.node_id]
        symbol = svg_symbol(node) if node.operands else "input"
        parts.append(
            f'<g class="{classes}" data-node="{node.node_id}">'
            f"<title>{html.escape(name)}</title>"
            f'<rect x="{x - NODE_WIDTH / 2:.1f}" y="{y - NODE_HEIGHT / 2:.1f}" '
            f'width="{NODE_WIDTH}" height="{NODE_HEIGHT}" rx="6"/>'
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


def to_html(circuit: Circuit, title: str, matrix: PairMatrix) -> str:
    """Render the page for one program: code, graph, then statistics."""
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
<h2>Code</h2>
<p class="controls"><button id="random">random input</button></p>
{code_lines(circuit, title, names, first)}
<h2>Graph</h2>
<div class="graph">{svg_graph(circuit, names, first)}</div>
<h2>Statistics</h2>
{redundancy_section(matrix, names)}
</main>
<script type="application/json" id="neighbours">{json.dumps(neighbour_map(circuit))}</script>
<script type="application/json" id="samples">{json.dumps(samples)}</script>
<script>{SCRIPT}</script>
</body></html>
"""


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


COMMANDS = {
    "generate": generate_programs,
    "stats": measure_programs,
    "html": render_pages,
}


def main() -> None:
    """Run the command the command line names."""
    arguments = parse_arguments()
    COMMANDS[arguments.command](arguments)


if __name__ == "__main__":
    main()
