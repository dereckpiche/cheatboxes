"""Decide equivalences and measure programs.

Equivalence: two complete checks of whether two circuits (or two terms)
compute the same function, a z3 proof and numpy brute force over all
2**32 inputs.

Redundancy: a circuit with n nodes (the input included) has
n (n - 1) / 2 unordered pairs. Probes kill every pair whose values
differ on some probe input; z3 then proves or refutes each surviving
pair. The result is a symmetric n x n matrix of `PairStatus` values,
rows in topological order.

Statistics: one flat dict of numbers per program, computed from the
circuit alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from itertools import combinations
from statistics import mean

import numpy as np
import z3

from envs.SafeSpeedups.circuit import NUMPY, WIDTH, Z3, Circuit

DEFAULT_TIMEOUT_MS = 10_000
INPUT_COUNT = 1 << WIDTH
CHUNK_SIZE = 1 << 24


def prove_terms_equal(
    left: z3.BitVecRef,
    right: z3.BitVecRef,
    timeout_ms: int = DEFAULT_TIMEOUT_MS,
) -> bool | None:
    """Prove two terms equal for every input: True, False, None on timeout."""
    solver = z3.Solver()
    solver.set("timeout", timeout_ms)
    solver.add(left != right)
    verdict = solver.check()
    if verdict == z3.unsat:
        return True
    if verdict == z3.sat:
        return False
    return None


def prove_equivalent(
    left: Circuit, right: Circuit, timeout_ms: int = DEFAULT_TIMEOUT_MS
) -> bool | None:
    """Prove equal outputs on all inputs: True, False, or None on timeout."""
    x = z3.BitVec("x", WIDTH)
    return prove_terms_equal(
        left.output_on(x, Z3), right.output_on(x, Z3), timeout_ms
    )


def find_counterexample(
    left: Circuit, right: Circuit, start: int = 0, stop: int = INPUT_COUNT
) -> int | None:
    """Return the first input in [start, stop) where the outputs differ."""
    for chunk_start in range(start, stop, CHUNK_SIZE):
        chunk_stop = min(chunk_start + CHUNK_SIZE, stop)
        inputs = np.arange(chunk_start, chunk_stop, dtype=np.uint64)
        inputs = inputs.astype(np.uint32)
        differ = left.output_on(inputs, NUMPY) != right.output_on(inputs, NUMPY)
        positions = np.flatnonzero(differ)
        if positions.size:
            return int(inputs[positions[0]])
    return None


def brute_force_equivalent(left: Circuit, right: Circuit) -> bool:
    """Check equal outputs by evaluating both circuits on every input."""
    return find_counterexample(left, right) is None


class PairStatus(IntEnum):
    """What is known about one pair of nodes."""

    SELF = 0
    DIFFER_ON_PROBES = 1
    EQUIVALENT = 2
    REFUTED = 3
    UNKNOWN = 4


@dataclass(frozen=True)
class PairMatrix:
    """Pair statuses of a circuit's nodes, rows in topological order."""

    node_ids: tuple[int, ...]
    status: np.ndarray

    def pair_count(self) -> int:
        """Return the number of unordered node pairs, n (n - 1) / 2."""
        size = len(self.node_ids)
        return size * (size - 1) // 2

    def count(self, status: PairStatus) -> int:
        """Return how many unordered pairs have `status`."""
        return int(np.triu(self.status == status, k=1).sum())


def probe_signatures(circuit: Circuit) -> list[bytes]:
    """Return each node's values on the probes, as bytes, in node order."""
    return [
        circuit.probe_values[node.node_id].tobytes() for node in circuit.nodes
    ]


def surviving_pairs(circuit: Circuit) -> list[tuple[int, int]]:
    """Return the index pairs (i < j) whose nodes agree on every probe."""
    groups: dict[bytes, list[int]] = {}
    for index, signature in enumerate(probe_signatures(circuit)):
        groups.setdefault(signature, []).append(index)
    pairs = []
    for indices in groups.values():
        pairs.extend(combinations(indices, 2))
    return sorted(pairs)


def proof_status(verdict: bool | None) -> PairStatus:
    """Turn a z3 verdict into a pair status."""
    if verdict is True:
        return PairStatus.EQUIVALENT
    if verdict is False:
        return PairStatus.REFUTED
    return PairStatus.UNKNOWN


def pair_matrix(
    circuit: Circuit, timeout_ms: int = DEFAULT_TIMEOUT_MS
) -> PairMatrix:
    """Kill pairs that differ on probes; prove or refute the rest with z3."""
    size = len(circuit.nodes)
    status = np.full((size, size), PairStatus.DIFFER_ON_PROBES, dtype=np.int8)
    np.fill_diagonal(status, PairStatus.SELF)
    pairs = surviving_pairs(circuit)
    if pairs:
        terms = circuit.evaluate(z3.BitVec("x", WIDTH), Z3)
        node_ids = [node.node_id for node in circuit.nodes]
        for i, j in pairs:
            verdict = prove_terms_equal(
                terms[node_ids[i]], terms[node_ids[j]], timeout_ms
            )
            status[i, j] = status[j, i] = proof_status(verdict)
    return PairMatrix(tuple(node.node_id for node in circuit.nodes), status)


def earliest_equivalents(matrix: PairMatrix) -> dict[int, int]:
    """Map each redundant node to the earliest node proven equal to it."""
    earliest = {}
    for j, node_id in enumerate(matrix.node_ids):
        for i in range(j):
            if matrix.status[i, j] == PairStatus.EQUIVALENT:
                earliest[node_id] = matrix.node_ids[i]
                break
    return earliest


def equivalence_classes(matrix: PairMatrix) -> list[tuple[int, ...]]:
    """Group nodes proven equal, keeping groups of two or more."""
    groups: dict[int, list[int]] = {}
    earliest = earliest_equivalents(matrix)
    for node_id in matrix.node_ids:
        root = node_id
        while root in earliest:
            root = earliest[root]
        groups.setdefault(root, []).append(node_id)
    return [tuple(group) for group in groups.values() if len(group) > 1]


def redundancy_stats(matrix: PairMatrix) -> dict[str, float]:
    """Count pairs by outcome and the nodes equal to an earlier node."""
    redundant = len(earliest_equivalents(matrix))
    non_input_nodes = len(matrix.node_ids) - 1
    return {
        "pairs": matrix.pair_count(),
        "pairs_killed_by_probes": matrix.count(PairStatus.DIFFER_ON_PROBES),
        "pairs_checked_by_z3": matrix.pair_count()
        - matrix.count(PairStatus.DIFFER_ON_PROBES),
        "equivalent_pairs": matrix.count(PairStatus.EQUIVALENT),
        "refuted_pairs": matrix.count(PairStatus.REFUTED),
        "unknown_pairs": matrix.count(PairStatus.UNKNOWN),
        "redundant_nodes": redundant,
        "redundant_fraction": redundant / non_input_nodes,
    }


def program_stats(circuit: Circuit) -> dict[str, float]:
    """Measure one circuit; for now, its redundant nodes."""
    return {"redundant_nodes": len(earliest_equivalents(pair_matrix(circuit)))}


def summarize(rows: list[dict[str, float]]) -> dict[str, tuple[float, ...]]:
    """Return (mean, min, max) of every stat over the rows that have it."""
    keys = sorted({key for row in rows for key in row})
    summary = {}
    for key in keys:
        values = [row[key] for row in rows if key in row]
        summary[key] = (mean(values), min(values), max(values))
    return summary
