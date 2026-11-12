"""Grow random circuits, and save and load them as program files.

A function starts as `x -> z` and grows one node at a time (see
`grow`) until its cost reaches the configured range. One file per
program holds its circuit with the seed and config that grew it, so
statistics can be computed later without regenerating.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Iterator

from envs.SafeSpeedups.circuit import (
    INPUT,
    Circuit,
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
    config: GenerationConfig, seed: int
) -> Iterator[tuple[Circuit, list[str]]]:
    """Yield each attempt's circuit and its rejection reasons, in order."""
    for attempt in range(config.max_attempts):
        rng = random.Random(f"{seed}/attempt/{attempt}")
        circuit = grow_to_cost(config, rng)
        yield circuit, rejection_reasons(circuit, config)


def generate_function(config: GenerationConfig, seed: int) -> Circuit:
    """Return the first acceptable circuit grown from `seed`."""
    for circuit, reasons in attempts(config, seed):
        if not reasons:
            return circuit
    raise RuntimeError(
        f"no acceptable function for seed {seed} "
        f"in {config.max_attempts} attempts"
    )


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
