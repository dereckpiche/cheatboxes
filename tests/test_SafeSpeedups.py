"""SafeSpeedups' function generator: structure, determinism, backends,
rendering, storage, equivalence and redundancy. No container needed."""

import random
from dataclasses import replace

import numpy as np
import pytest
import z3

from envs.SafeSpeedups.analysis import (
    PairStatus,
    earliest_equivalents,
    equivalence_classes,
    find_counterexample,
    pair_matrix,
    program_stats,
    prove_equivalent,
    redundancy_stats,
)
from envs.SafeSpeedups.circuit import (
    INPUT,
    NUMPY,
    OPERATIONS,
    PROBE_INPUTS,
    WIDTH,
    Z3,
    Circuit,
    Node,
)
from envs.SafeSpeedups.generation import (
    GenerationConfig,
    Program,
    generate_function,
    load_program,
    save_program,
)
from envs.SafeSpeedups.preview import runtime_operations, to_html, to_python

CONFIG = GenerationConfig()
SEEDS = range(8)


def reaches_output(circuit: Circuit, node_id: int) -> bool:
    """Tell whether a path leads from a node to the output."""
    frontier, seen = [node_id], set()
    while frontier:
        current = frontier.pop()
        if current == circuit.output_node.node_id:
            return True
        if current not in seen:
            seen.add(current)
            frontier.extend(circuit.successors[current])
    return False


def test_the_same_seed_gives_the_same_circuit():
    assert generate_function(CONFIG, 3) == generate_function(CONFIG, 3)
    assert generate_function(CONFIG, 3) != generate_function(CONFIG, 4)


@pytest.mark.parametrize("seed", SEEDS)
def test_a_circuit_is_topological_and_has_no_dead_node(seed):
    circuit = generate_function(CONFIG, seed)
    assert circuit.input_node.operands == ()
    for node in circuit.nodes[1:]:
        operation = OPERATIONS[node.operation]
        assert len(node.operands) == operation.arity
        assert 1 <= operation.arity <= 3
        assert len(set(node.operands)) == len(node.operands)
        assert all(
            circuit.positions[source_id] < circuit.positions[node.node_id]
            for source_id in node.operands
        )
    for node in circuit.nodes:
        assert reaches_output(circuit, node.node_id)


@pytest.mark.parametrize("seed", SEEDS)
def test_cost_lands_in_range_and_output_is_spread(seed):
    circuit = generate_function(CONFIG, seed)
    assert CONFIG.cost_min <= circuit.cost <= CONFIG.cost_max
    assert circuit.distinct_output_count >= CONFIG.min_distinct_outputs


@pytest.mark.parametrize("name", sorted(OPERATIONS))
def test_numpy_and_z3_agree_on_every_operation(name):
    operation = OPERATIONS[name]
    rng = random.Random(name)
    for _ in range(50):
        operands = [rng.getrandbits(WIDTH) for _ in range(operation.arity)]
        parameters = operation.sample_parameters(rng)
        numeric = operation.apply(
            NUMPY, [np.uint32(v) for v in operands], parameters
        )
        symbolic = operation.apply(
            Z3, [z3.BitVecVal(v, WIDTH) for v in operands], parameters
        )
        assert int(numeric) == z3.simplify(symbolic).as_long()


@pytest.mark.parametrize("seed", SEEDS)
def test_rendered_python_computes_the_probe_values(seed):
    circuit = generate_function(CONFIG, seed)
    namespace = {"ops": runtime_operations(NUMPY)}
    exec(to_python(circuit, "rendered"), namespace)
    outputs = namespace["rendered"](PROBE_INPUTS)
    assert np.array_equal(outputs, circuit.output_probe_values)


def flip_a_mask_bit_that_matters(circuit: Circuit) -> Circuit:
    """Flip the first mask bit whose flip changes the probe outputs."""
    for index, node in enumerate(circuit.nodes):
        if not node.operation.endswith("_mask"):
            continue
        for bit in range(WIDTH):
            flipped = replace(node, parameters=(node.parameters[0] ^ 1 << bit,))
            nodes = (
                circuit.nodes[:index] + (flipped,) + circuit.nodes[index + 1 :]
            )
            changed = Circuit(nodes)
            if not np.array_equal(
                changed.output_probe_values, circuit.output_probe_values
            ):
                return changed
    pytest.skip("no mask bit changes this circuit's probe outputs")


def test_prove_equivalent_accepts_a_copy_and_rejects_a_flipped_mask():
    circuit = generate_function(CONFIG, 0)
    assert prove_equivalent(circuit, Circuit(circuit.nodes)) is True
    assert (
        prove_equivalent(circuit, flip_a_mask_bit_that_matters(circuit))
        is False
    )


def test_brute_force_finds_where_a_flipped_mask_differs():
    circuit = generate_function(CONFIG, 0)
    flipped = flip_a_mask_bit_that_matters(circuit)
    differs = flipped.output_probe_values != circuit.output_probe_values
    witness = int(PROBE_INPUTS[np.flatnonzero(differs)[0]])
    found = find_counterexample(circuit, flipped, witness, witness + 4096)
    assert found is not None
    assert int(circuit.output_on(np.uint32(found), NUMPY)) != int(
        flipped.output_on(np.uint32(found), NUMPY)
    )
    copy = Circuit(circuit.nodes)
    assert find_counterexample(circuit, copy, 0, 1 << 20) is None


def planted_circuit() -> Circuit:
    """Build x -> v1 = x ^ 0, v2 = not v1, v3 = not v2 (v1 = v3 = x)."""
    return Circuit(
        (
            Node(0, INPUT),
            Node(1, "xor_mask", (0,), (0,)),
            Node(2, "not", (1,)),
            Node(3, "not", (2,)),
        )
    )


def test_pair_matrix_proves_planted_equivalences():
    matrix = pair_matrix(planted_circuit())
    assert matrix.pair_count() == 6
    assert matrix.count(PairStatus.EQUIVALENT) == 3
    assert matrix.count(PairStatus.DIFFER_ON_PROBES) == 3
    assert (matrix.status == matrix.status.T).all()
    assert earliest_equivalents(matrix) == {1: 0, 3: 0}
    assert equivalence_classes(matrix) == [(0, 1, 3)]


@pytest.mark.parametrize("seed", SEEDS)
def test_every_pair_sharing_probes_goes_to_z3(seed):
    circuit = generate_function(CONFIG, seed)
    stats = redundancy_stats(pair_matrix(circuit))
    assert program_stats(circuit) == {
        "redundant_nodes": stats["redundant_nodes"]
    }
    assert (
        stats["pairs_killed_by_probes"] + stats["pairs_checked_by_z3"]
        == stats["pairs"]
    )
    assert stats["pairs_checked_by_z3"] == (
        stats["equivalent_pairs"]
        + stats["refuted_pairs"]
        + stats["unknown_pairs"]
    )
    assert 0 <= stats["redundant_fraction"] < 1


def test_a_saved_program_loads_back_unchanged(tmp_path):
    config = GenerationConfig(cost_min=20, cost_max=30)
    program = Program("function_00007", 7, config, generate_function(config, 7))
    assert load_program(save_program(program, tmp_path)) == program


def test_html_page_draws_code_graph_and_statistics():
    circuit = generate_function(CONFIG, 0)
    matrix = pair_matrix(circuit)
    page = to_html(circuit, "function_0", matrix)
    assert page.index("<h2>Code") < page.index("<h2>Graph")
    assert page.index("<h2>Graph") < page.index("<h2>Statistics")
    assert page.count('<g class="node') == len(circuit.nodes)
    assert page.count('<div class="line" data-node') == len(circuit.nodes)
    assert page.count("<path data-from") == len(circuit.edges)
    redundant = len(earliest_equivalents(matrix))
    assert f"redundant nodes: {redundant}</b>" in page
