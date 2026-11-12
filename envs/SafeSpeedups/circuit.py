"""A generated function: a one-input, one-output bitwise circuit.

Operations: each bitwise operation's meaning is written once, against a
backend, so one definition both runs on numpy probes and builds z3
terms. Adding an operation is one entry in `OPERATIONS`.

Probes: the fixed inputs every circuit is evaluated on. They catch most
differences instantly but miss rare ones, so they filter and
fingerprint; only a proof decides.

Circuit: nodes stored in topological order, the input `x` first and the
output last. A circuit never changes once built; everything derived
from it is computed on first use and cached.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from functools import cached_property
from typing import Callable

import numpy as np
import z3

WIDTH = 32
ALL_ONES = (1 << WIDTH) - 1
OPERATION_PREFIX = "p_"
"""Starts every name rendered code calls (`ops.p_and`, p for primitive);
`and`, `or` and `not` alone are Python keywords."""


def sample_no_parameters(rng: random.Random) -> tuple[int, ...]:
    """Return no parameters, for operations without constants."""
    return ()


def sample_mask(rng: random.Random) -> tuple[int, ...]:
    """Draw a mask whose density of 1-bits is itself drawn at random."""
    density = rng.random()
    mask = 0
    for bit in range(WIDTH):
        if rng.random() < density:
            mask |= 1 << bit
    return (mask,)


def sample_shift(rng: random.Random) -> tuple[int, ...]:
    """Draw a shift or rotation amount in 1 .. WIDTH - 1."""
    return (rng.randrange(1, WIDTH),)


@dataclass(frozen=True)
class Operation:
    """One operation: its operand count, cost, constants and meaning.

    `meaning(backend, *operands, *parameters)` computes the result;
    `gate` is the name rendered code calls after the prefix (`ops.p_and`),
    so the masked forms of and, or and xor read as the plain gate on a
    constant.
    """

    name: str
    arity: int
    cost: int
    meaning: Callable
    sample_parameters: Callable[[random.Random], tuple[int, ...]]
    gate: str
    parameter_style: str = "hex"

    def apply(self, backend, operands, parameters):
        """Compute this operation on operand values with a backend."""
        return self.meaning(backend, *operands, *parameters)

    @property
    def call_name(self) -> str:
        """Return the name rendered code calls, prefix included."""
        return OPERATION_PREFIX + self.gate

    def render_parameter(self, value: int) -> str:
        """Write one parameter as it appears in rendered code."""
        if self.parameter_style == "hex":
            return f"0x{value:08x}"
        return str(value)


def mux(b, a, c, select):
    """Take each bit from `a` where `select` is 1, from `c` where it is 0."""
    return b.bitwise_or(
        b.bitwise_and(a, select), b.bitwise_and(c, b.bitwise_not(select))
    )


def majority(b, a, c, d):
    """Set each bit to the value held by at least two of the three inputs."""
    return b.bitwise_or(
        b.bitwise_or(b.bitwise_and(a, c), b.bitwise_and(a, d)),
        b.bitwise_and(c, d),
    )


def chi(b, a, c, d):
    """Flip the bits of `a` where `c` is 0 and `d` is 1, as in Keccak."""
    return b.bitwise_xor(a, b.bitwise_and(b.bitwise_not(c), d))


OPERATIONS = {
    operation.name: operation
    for operation in [
        Operation(
            "not",
            1,
            1,
            lambda b, a: b.bitwise_not(a),
            sample_no_parameters,
            "not",
        ),
        Operation(
            "and_mask",
            1,
            1,
            lambda b, a, mask: b.bitwise_and(a, b.constant(mask)),
            sample_mask,
            "and",
        ),
        Operation(
            "or_mask",
            1,
            1,
            lambda b, a, mask: b.bitwise_or(a, b.constant(mask)),
            sample_mask,
            "or",
        ),
        Operation(
            "xor_mask",
            1,
            1,
            lambda b, a, mask: b.bitwise_xor(a, b.constant(mask)),
            sample_mask,
            "xor",
        ),
        Operation(
            "rotl",
            1,
            1,
            lambda b, a, amount: b.rotl(a, amount),
            sample_shift,
            "rotl",
            "decimal",
        ),
        Operation(
            "lshr",
            1,
            1,
            lambda b, a, amount: b.lshr(a, amount),
            sample_shift,
            "lshr",
            "decimal",
        ),
        Operation(
            "shl",
            1,
            1,
            lambda b, a, amount: b.shl(a, amount),
            sample_shift,
            "shl",
            "decimal",
        ),
        Operation(
            "and",
            2,
            1,
            lambda b, a, c: b.bitwise_and(a, c),
            sample_no_parameters,
            "and",
        ),
        Operation(
            "or",
            2,
            1,
            lambda b, a, c: b.bitwise_or(a, c),
            sample_no_parameters,
            "or",
        ),
        Operation(
            "xor",
            2,
            1,
            lambda b, a, c: b.bitwise_xor(a, c),
            sample_no_parameters,
            "xor",
        ),
        Operation(
            "andn",
            2,
            1,
            lambda b, a, c: b.bitwise_and(a, b.bitwise_not(c)),
            sample_no_parameters,
            "andn",
        ),
        Operation("mux", 3, 2, mux, sample_no_parameters, "mux"),
        Operation("maj", 3, 2, majority, sample_no_parameters, "maj"),
        Operation("chi", 3, 2, chi, sample_no_parameters, "chi"),
    ]
}


def operations_with_arity(arity: int) -> list[Operation]:
    """List the operations that take exactly `arity` operands."""
    return [op for op in OPERATIONS.values() if op.arity == arity]


class NumpyBackend:
    """Compute primitives on uint32 values or arrays."""

    def constant(self, value: int):
        """Return a constant as a uint32."""
        return np.uint32(value)

    def bitwise_and(self, a, b):
        """Return the bitwise and of two values."""
        return a & b

    def bitwise_or(self, a, b):
        """Return the bitwise or of two values."""
        return a | b

    def bitwise_xor(self, a, b):
        """Return the bitwise exclusive or of two values."""
        return a ^ b

    def bitwise_not(self, a):
        """Return the bitwise complement of a value."""
        return ~a

    def rotl(self, a, amount: int):
        """Rotate a value left by `amount` bits."""
        return (a << np.uint32(amount)) | (a >> np.uint32(WIDTH - amount))

    def lshr(self, a, amount: int):
        """Shift a value right by `amount` bits, filling with zeros."""
        return a >> np.uint32(amount)

    def shl(self, a, amount: int):
        """Shift a value left by `amount` bits, dropping the top bits."""
        return a << np.uint32(amount)


class Z3Backend:
    """Build z3 bitvector terms for the primitives."""

    def constant(self, value: int):
        """Return a constant as a 32-bit bitvector."""
        return z3.BitVecVal(value, WIDTH)

    def bitwise_and(self, a, b):
        """Return the bitwise and of two terms."""
        return a & b

    def bitwise_or(self, a, b):
        """Return the bitwise or of two terms."""
        return a | b

    def bitwise_xor(self, a, b):
        """Return the bitwise exclusive or of two terms."""
        return a ^ b

    def bitwise_not(self, a):
        """Return the bitwise complement of a term."""
        return ~a

    def rotl(self, a, amount: int):
        """Rotate a term left by `amount` bits."""
        return z3.RotateLeft(a, amount)

    def lshr(self, a, amount: int):
        """Shift a term right by `amount` bits, filling with zeros."""
        return z3.LShR(a, amount)

    def shl(self, a, amount: int):
        """Shift a term left by `amount` bits, dropping the top bits."""
        return a << amount


NUMPY = NumpyBackend()
Z3 = Z3Backend()


EDGE_INPUTS = (
    0x00000000,
    0x00000001,
    0x00000002,
    0x55555555,
    0x7FFFFFFF,
    0x80000000,
    0xAAAAAAAA,
    0xFFFFFFFE,
    0xFFFFFFFF,
)
RANDOM_INPUT_COUNT = 4096
PROBE_SEED = 0


def make_probe_inputs() -> np.ndarray:
    """Build the probe inputs: the edge cases, then seeded random words."""
    rng = np.random.default_rng(PROBE_SEED)
    random_inputs = rng.integers(0, 2**32, RANDOM_INPUT_COUNT, dtype=np.uint64)
    edges = np.array(EDGE_INPUTS, dtype=np.uint64)
    return np.concatenate([edges, random_inputs]).astype(np.uint32)


PROBE_INPUTS = make_probe_inputs()


INPUT = "input"


@dataclass(frozen=True)
class Node:
    """One value of a circuit: an operation applied to earlier nodes."""

    node_id: int
    operation: str
    operands: tuple[int, ...] = ()
    parameters: tuple[int, ...] = ()


@dataclass(frozen=True)
class Circuit:
    """A bitwise function as a DAG of nodes in topological order."""

    nodes: tuple[Node, ...]

    @property
    def input_node(self) -> Node:
        """Return the input node `x`."""
        return self.nodes[0]

    @property
    def output_node(self) -> Node:
        """Return the node whose value the function returns."""
        return self.nodes[-1]

    @property
    def next_node_id(self) -> int:
        """Return the id the next added node will get."""
        return len(self.nodes)

    @cached_property
    def positions(self) -> dict[int, int]:
        """Map each node id to its index in the topological order."""
        return {node.node_id: index for index, node in enumerate(self.nodes)}

    @cached_property
    def nodes_by_id(self) -> dict[int, Node]:
        """Map each node id to its node."""
        return {node.node_id: node for node in self.nodes}

    @cached_property
    def edges(self) -> tuple[tuple[int, int, int], ...]:
        """List every edge as (source id, destination id, operand slot)."""
        return tuple(
            (source_id, node.node_id, slot)
            for node in self.nodes
            for slot, source_id in enumerate(node.operands)
        )

    @cached_property
    def successors(self) -> dict[int, tuple[int, ...]]:
        """Map each node id to the ids of the nodes that read it."""
        readers = {node.node_id: [] for node in self.nodes}
        for source_id, destination_id, _ in self.edges:
            readers[source_id].append(destination_id)
        return {node_id: tuple(ids) for node_id, ids in readers.items()}

    @cached_property
    def cost(self) -> int:
        """Return the summed cost of every operation in the circuit."""
        return sum(OPERATIONS[node.operation].cost for node in self.nodes[1:])

    @cached_property
    def depths(self) -> dict[int, int]:
        """Map each node id to its longest path length from the input."""
        depth_of = {}
        for node in self.nodes:
            depth_of[node.node_id] = 1 + max(
                (depth_of[source_id] for source_id in node.operands),
                default=-1,
            )
        return depth_of

    @cached_property
    def operation_counts(self) -> Counter:
        """Count how many nodes use each operation."""
        return Counter(node.operation for node in self.nodes[1:])

    @cached_property
    def probe_values(self) -> dict[int, np.ndarray]:
        """Map each node id to its values on the probe inputs."""
        return self.evaluate(PROBE_INPUTS, NUMPY)

    @property
    def output_probe_values(self) -> np.ndarray:
        """Return the function's outputs on the probe inputs."""
        return self.probe_values[self.output_node.node_id]

    @cached_property
    def distinct_output_count(self) -> int:
        """Count the distinct outputs over the probe inputs."""
        return len(np.unique(self.output_probe_values))

    def evaluate(self, input_value, backend) -> dict:
        """Compute every node's value for one input with a backend."""
        values = {self.input_node.node_id: input_value}
        for node in self.nodes[1:]:
            operation = OPERATIONS[node.operation]
            operand_values = [values[source_id] for source_id in node.operands]
            values[node.node_id] = operation.apply(
                backend, operand_values, node.parameters
            )
        return values

    def output_on(self, input_value, backend):
        """Compute the function's output for one input with a backend."""
        return self.evaluate(input_value, backend)[self.output_node.node_id]
