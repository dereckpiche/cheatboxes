"""Grow random circuits and libraries; save and load them as files.

A function starts as `x -> z` and grows one node at a time (see
`grow`) until its cost reaches the configured range. One file per
program holds its circuit with the seed and config that grew it, so
statistics can be computed later without regenerating.

A library draws each function's agent, grows its call graph one
function at a time (see `connect_new_function`), favouring calls within
an agent, then flips a verified coin for each function. Owners, graph
and coins draw from separate random streams.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Iterator

from envs.SafeSpeedups.circuit import (
    INPUT,
    Circuit,
    Function,
    Library,
    Node,
    operations_with_arity,
)


@dataclass(frozen=True)
class GenerationConfig:
    """Settings for growing one function; every field has a working default.

    Growth adds nodes until the cost reaches `cost_min`; an attempt that
    overshoots `cost_max` or has fewer than `min_distinct_outputs` distinct
    outputs on the probes is rejected. `operation_weights` maps operation
    names to sampling weights (missing names weigh 1). Each extra source or
    extra destination of a new node is added with its probability, again
    and again until a draw fails.
    """

    cost_min: int = 30
    cost_max: int = 40
    operation_weights: dict[str, float] = field(default_factory=dict)
    extra_source_probability: float = 0.4
    extra_destination_probability: float = 0.2
    min_distinct_outputs: int = 1024
    max_attempts: int = 50


MAX_ARITY = 3


def choose_operation(
    arity: int, rng: random.Random, config: GenerationConfig
) -> tuple[str, tuple[int, ...]]:
    """Draw an operation with `arity` operands, and its parameters."""
    candidates = operations_with_arity(arity)
    weights = [config.operation_weights.get(op.name, 1.0) for op in candidates]
    operation = rng.choices(candidates, weights)[0]
    return operation.name, operation.sample_parameters(rng)


def start_circuit(rng: random.Random, config: GenerationConfig) -> Circuit:
    """Build the starting circuit `x -> z`, with `z` a one-operand node."""
    operation, parameters = choose_operation(1, rng, config)
    return Circuit(
        (
            Node(0, INPUT),
            Node(1, operation, operands=(0,), parameters=parameters),
        )
    )


def pick_edge(circuit: Circuit, rng: random.Random) -> tuple[int, int, int]:
    """Draw an edge uniformly, as (source id, destination id, slot)."""
    return rng.choice(circuit.edges)


def draw_while(probability: float, limit: int, rng: random.Random) -> int:
    """Count successive successful draws of `probability`, up to `limit`."""
    count = 0
    while count < limit and rng.random() < probability:
        count += 1
    return count


def pick_extra_sources(
    circuit: Circuit,
    insert_index: int,
    first_source_id: int,
    rng: random.Random,
    config: GenerationConfig,
) -> list[int]:
    """Draw extra operands for the new node among the nodes before it."""
    candidates = [
        node.node_id
        for node in circuit.nodes[:insert_index]
        if node.node_id != first_source_id
    ]
    count = draw_while(
        config.extra_source_probability,
        min(MAX_ARITY - 1, len(candidates)),
        rng,
    )
    return rng.sample(candidates, count)


def pick_extra_destinations(
    circuit: Circuit,
    insert_index: int,
    first_destination_id: int,
    rng: random.Random,
    config: GenerationConfig,
) -> list[int]:
    """Draw later nodes that will also read the new node."""
    candidates = [
        node.node_id
        for node in circuit.nodes[insert_index:]
        if node.node_id != first_destination_id
        and len(node.operands) < MAX_ARITY
    ]
    count = draw_while(
        config.extra_destination_probability, len(candidates), rng
    )
    return rng.sample(candidates, count)


def add_operand(
    node: Node, operand_id: int, rng: random.Random, config: GenerationConfig
) -> Node:
    """Give a node one more operand and redraw its operation to match."""
    operands = list(node.operands)
    operands.insert(rng.randrange(len(operands) + 1), operand_id)
    operation, parameters = choose_operation(len(operands), rng, config)
    return replace(
        node,
        operation=operation,
        operands=tuple(operands),
        parameters=parameters,
    )


def replace_operand(node: Node, slot: int, operand_id: int) -> Node:
    """Make a node read `operand_id` in place of its operand at `slot`."""
    operands = list(node.operands)
    operands[slot] = operand_id
    return replace(node, operands=tuple(operands))


def grow(
    circuit: Circuit, rng: random.Random, config: GenerationConfig
) -> Circuit:
    """Add one node on a random edge and return the larger circuit."""
    source_id, destination_id, slot = pick_edge(circuit, rng)
    insert_index = rng.randrange(
        circuit.positions[source_id] + 1, circuit.positions[destination_id] + 1
    )
    operands = [source_id] + pick_extra_sources(
        circuit, insert_index, source_id, rng, config
    )
    rng.shuffle(operands)
    extra_destination_ids = set(
        pick_extra_destinations(
            circuit, insert_index, destination_id, rng, config
        )
    )
    new_id = circuit.next_node_id
    operation, parameters = choose_operation(len(operands), rng, config)
    nodes = []
    for node in circuit.nodes:
        if node.node_id == destination_id:
            node = replace_operand(node, slot, new_id)
        elif node.node_id in extra_destination_ids:
            node = add_operand(node, new_id, rng, config)
        nodes.append(node)
    nodes.insert(
        insert_index, Node(new_id, operation, tuple(operands), parameters)
    )
    return Circuit(tuple(nodes))


def grow_to_cost(config: GenerationConfig, rng: random.Random) -> Circuit:
    """Grow a circuit from `x -> z` until its cost reaches `cost_min`."""
    circuit = start_circuit(rng, config)
    while circuit.cost < config.cost_min:
        circuit = grow(circuit, rng, config)
    return circuit


def rejection_reasons(circuit: Circuit, config: GenerationConfig) -> list[str]:
    """List why a grown circuit is unacceptable; empty if it is fine."""
    reasons = []
    if circuit.cost > config.cost_max:
        reasons.append("over_cost")
    if circuit.distinct_output_count < config.min_distinct_outputs:
        reasons.append("few_distinct_outputs")
    return reasons


def attempts(
    config: GenerationConfig, seed: int | str
) -> Iterator[tuple[Circuit, list[str]]]:
    """Yield each attempt's circuit and its rejection reasons, in order."""
    for attempt in range(config.max_attempts):
        rng = random.Random(f"{seed}/attempt/{attempt}")
        circuit = grow_to_cost(config, rng)
        yield circuit, rejection_reasons(circuit, config)


def generate_function(config: GenerationConfig, seed: int | str) -> Circuit:
    """Return the first acceptable circuit grown from `seed`."""
    for circuit, reasons in attempts(config, seed):
        if not reasons:
            return circuit
    raise RuntimeError(
        f"no acceptable function for seed {seed} "
        f"in {config.max_attempts} attempts"
    )


@dataclass(frozen=True)
class LibraryConfig:
    """Settings for growing a library; every field has a working default.

    Every function belongs to an agent from birth: blocks of
    function_count // agent_count, the last agent also taking the
    remainder, in random order. Each new function is inserted at a random
    place in the call order and first connects to an anchor drawn with
    weight (1 + degree)^global_bias, times group_bias when the anchor's
    agent is its own. Every other function then gets its own coin:
    earlier ones may become callees, later ones callers. On each side, a
    function v is linked with probability min(1, expected · w(v) / Σ w),
    where w(v) = (1 + degree v)^global_bias · exp(-distance(anchor, v) /
    local_scale) · (group_bias if same agent else 1), and distance counts
    calls in either direction. So `expected_callees` and
    `expected_callers` are the expected links beyond the anchor; the
    biases only decide who gets them. Each function is verified with
    `verified_probability`.
    """

    function_count: int = 24
    agent_count: int = 4
    expected_callees: float = 0.2
    expected_callers: float = 0.1
    global_bias: float = 1.5
    local_scale: float = 0.5
    group_bias: float = 40.0
    verified_probability: float = 0.5
    function: GenerationConfig = field(default_factory=GenerationConfig)


def undirected_neighbours(
    callees: dict[int, set[int]],
) -> dict[int, set[int]]:
    """Map each function to the functions it calls or is called by."""
    neighbours = {function: set() for function in callees}
    for caller, called in callees.items():
        for callee in called:
            neighbours[caller].add(callee)
            neighbours[callee].add(caller)
    return neighbours


def graph_distances(
    source: int, neighbours: dict[int, set[int]]
) -> dict[int, int]:
    """Count the calls on a shortest path from `source` to each function."""
    distances, frontier = {source: 0}, [source]
    while frontier:
        next_frontier = []
        for function in frontier:
            for neighbour in neighbours[function]:
                if neighbour not in distances:
                    distances[neighbour] = distances[function] + 1
                    next_frontier.append(neighbour)
        frontier = next_frontier
    return distances


def global_weight(degree: int, config: LibraryConfig) -> float:
    """Weigh a function by its degree: well-connected ones attract more."""
    return (1 + degree) ** config.global_bias


def group_weight(same_agent: bool, config: LibraryConfig) -> float:
    """Weigh a function by whether the new function's agent owns it."""
    return config.group_bias if same_agent else 1.0


def local_weight(distance: int | None, config: LibraryConfig) -> float:
    """Weigh a function by its distance to the anchor; unreachable is 0."""
    if distance is None:
        return 0.0
    return math.exp(-distance / config.local_scale)


def flip_link_coins(
    weights: dict[int, float], expected: float, rng: random.Random
) -> list[int]:
    """Flip one coin per function, min(1, expected · w / Σ w); keep wins."""
    total = sum(weights.values())
    if total == 0:
        return []
    return [
        function
        for function, weight in weights.items()
        if rng.random() < min(1.0, expected * weight / total)
    ]


def connect_new_function(
    order: list[int],
    callees: dict[int, set[int]],
    owners: dict[int, int],
    new: int,
    position: int,
    rng: random.Random,
    config: LibraryConfig,
) -> tuple[list[int], list[int]]:
    """Draw the callees and callers of `new`, inserted at `position`."""
    neighbours = undirected_neighbours(callees)
    base = {
        f: global_weight(len(neighbours[f]), config)
        * group_weight(owners[f] == owners[new], config)
        for f in order
    }
    anchor = rng.choices(order, [base[f] for f in order])[0]
    distance = graph_distances(anchor, neighbours)
    weight = {
        f: base[f] * local_weight(distance.get(f), config)
        for f in order
        if f != anchor
    }
    earlier = set(order[:position])
    new_callees = flip_link_coins(
        {f: w for f, w in weight.items() if f in earlier},
        config.expected_callees,
        rng,
    )
    new_callers = flip_link_coins(
        {f: w for f, w in weight.items() if f not in earlier},
        config.expected_callers,
        rng,
    )
    if anchor in earlier:
        new_callees.append(anchor)
    else:
        new_callers.append(anchor)
    return new_callees, new_callers


def grow_call_graph(
    config: LibraryConfig, owners: dict[int, int], rng: random.Random
) -> tuple[list[int], dict[int, set[int]]]:
    """Grow the call graph; return the call order and each one's callees."""
    order: list[int] = []
    callees: dict[int, set[int]] = {}
    for new in range(config.function_count):
        position = rng.randrange(len(order) + 1)
        callees[new] = set()
        if order:
            new_callees, new_callers = connect_new_function(
                order, callees, owners, new, position, rng, config
            )
            callees[new].update(new_callees)
            for caller in new_callers:
                callees[caller].add(new)
        order.insert(position, new)
    return order, callees


def block_sizes(config: LibraryConfig) -> list[int]:
    """Size each agent's block; the last agent also takes the remainder."""
    block = config.function_count // config.agent_count
    sizes = [block] * config.agent_count
    sizes[-1] += config.function_count - block * config.agent_count
    return sizes


def birth_owners(config: LibraryConfig, rng: random.Random) -> list[int]:
    """Draw the agent of each function, in creation order, sizes exact."""
    owners = [
        agent
        for agent, size in enumerate(block_sizes(config))
        for _ in range(size)
    ]
    return rng.sample(owners, len(owners))


def draw_verified(
    functions: list[int], config: LibraryConfig, rng: random.Random
) -> dict[int, bool]:
    """Flip each function's verified coin."""
    return {
        function: rng.random() < config.verified_probability
        for function in functions
    }


def generate_library(config: LibraryConfig, seed: int | str) -> Library:
    """Grow a call graph, split it between agents, and fill each function."""
    if config.function_count < config.agent_count:
        raise ValueError("need at least one function per agent")
    owners = dict(
        enumerate(birth_owners(config, random.Random(f"{seed}/owners")))
    )
    order, callees = grow_call_graph(
        config, owners, random.Random(f"{seed}/calls")
    )
    verified = draw_verified(order, config, random.Random(f"{seed}/verified"))
    position = {function: index for index, function in enumerate(order)}
    names = {function: f"f{position[function] + 1}" for function in order}
    functions = tuple(
        Function(
            name=names[function],
            circuit=generate_function(
                config.function, f"{seed}/function/{names[function]}"
            ),
            callees=tuple(
                names[c] for c in sorted(callees[function], key=position.get)
            ),
            owner=owners[function],
            verified=verified[function],
        )
        for function in order
    )
    return Library(functions, config.agent_count)


PROGRAM_DIRECTORY = Path("data/SafeSpeedups/programs")
PROGRAM_SUFFIX = ".json"


@dataclass(frozen=True)
class Program:
    """A generated circuit with its name and where it came from."""

    name: str
    seed: int
    config: GenerationConfig
    circuit: Circuit


def program_name(seed: int) -> str:
    """Name the program grown from `seed`, sortable by seed."""
    return f"function_{seed:05d}"


def circuit_to_dict(circuit: Circuit) -> dict:
    """Write a circuit as plain JSON-ready data."""
    return {
        "nodes": [
            {
                "id": node.node_id,
                "operation": node.operation,
                "operands": list(node.operands),
                "parameters": list(node.parameters),
            }
            for node in circuit.nodes
        ]
    }


def circuit_from_dict(data: dict) -> Circuit:
    """Read a circuit back from `circuit_to_dict` data."""
    return Circuit(
        tuple(
            Node(
                node["id"],
                node["operation"],
                tuple(node["operands"]),
                tuple(node["parameters"]),
            )
            for node in data["nodes"]
        )
    )


def save_program(program: Program, directory: Path) -> Path:
    """Write a program to `<directory>/<name>.json`; return the path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{program.name}{PROGRAM_SUFFIX}"
    data = {
        "name": program.name,
        "seed": program.seed,
        "config": asdict(program.config),
        "circuit": circuit_to_dict(program.circuit),
    }
    path.write_text(json.dumps(data, indent=1) + "\n")
    return path


def load_program(path: Path) -> Program:
    """Read a program written by `save_program`."""
    data = json.loads(Path(path).read_text())
    return Program(
        name=data["name"],
        seed=data["seed"],
        config=GenerationConfig(**data["config"]),
        circuit=circuit_from_dict(data["circuit"]),
    )


def program_paths(paths: list[Path]) -> list[Path]:
    """Expand directories into their program files, sorted by name."""
    found = []
    for path in map(Path, paths):
        if path.is_dir():
            found.extend(sorted(path.glob(f"*{PROGRAM_SUFFIX}")))
        else:
            found.append(path)
    return found


LIBRARY_DIRECTORY = Path("data/SafeSpeedups/libraries")
LIBRARY_FILE = "library.json"


@dataclass(frozen=True)
class SavedLibrary:
    """A generated library with its name and where it came from."""

    name: str
    seed: int
    config: LibraryConfig
    library: Library


def library_name(seed: int) -> str:
    """Name the library grown from `seed`, sortable by seed."""
    return f"library_{seed:05d}"


def library_to_dict(library: Library) -> dict:
    """Write a library as plain JSON-ready data."""
    return {
        "agent_count": library.agent_count,
        "functions": [
            {
                "name": function.name,
                "owner": function.owner,
                "verified": function.verified,
                "callees": list(function.callees),
                "circuit": circuit_to_dict(function.circuit),
            }
            for function in library.functions
        ],
    }


def library_from_dict(data: dict) -> Library:
    """Read a library back from `library_to_dict` data."""
    functions = tuple(
        Function(
            name=entry["name"],
            circuit=circuit_from_dict(entry["circuit"]),
            callees=tuple(entry["callees"]),
            owner=entry["owner"],
            verified=entry["verified"],
        )
        for entry in data["functions"]
    )
    return Library(functions, data["agent_count"])


def save_library(saved: SavedLibrary, directory: Path) -> Path:
    """Write a library to `<directory>/<name>/library.json`; return it."""
    folder = directory / saved.name
    folder.mkdir(parents=True, exist_ok=True)
    data = {
        "name": saved.name,
        "seed": saved.seed,
        "config": asdict(saved.config),
        **library_to_dict(saved.library),
    }
    path = folder / LIBRARY_FILE
    path.write_text(json.dumps(data, indent=1) + "\n")
    return path


def load_library(path: Path) -> SavedLibrary:
    """Read a library written by `save_library`, from its file or folder."""
    path = Path(path)
    if path.is_dir():
        path = path / LIBRARY_FILE
    data = json.loads(path.read_text())
    config = dict(data["config"])
    config["function"] = GenerationConfig(**config["function"])
    return SavedLibrary(
        name=data["name"],
        seed=data["seed"],
        config=LibraryConfig(**config),
        library=library_from_dict(data),
    )


def library_paths(paths: list[Path]) -> list[Path]:
    """Expand folders into the library files they hold, sorted by name."""
    found = []
    for path in map(Path, paths):
        if path.is_dir() and (path / LIBRARY_FILE).exists():
            found.append(path / LIBRARY_FILE)
        elif path.is_dir():
            found.extend(sorted(path.glob(f"*/{LIBRARY_FILE}")))
        else:
            found.append(path)
    return found
