"""Demonstration episodes for a supervised warm start.

Each demonstration plays a task the way a competent agent would: read
`library.py`, keep a copy, rewrite the verified functions with
simplifications z3 proves, test every function against the copy on random
inputs (the output shown is what the test really prints), check the
format, submit. A simplification replaces a node by a
constant or by an earlier node whenever the function's output provably
stays the same, then drops the nodes nothing reads any more. Unverified
functions are left alone, so every demonstration is honest. Each is graded
like a real episode and kept only if it earns a reward. They are made
on demand, like the RL instances: `training.sft_dataset` asks for
`demonstrate(i)`.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np

from envs.commons.tool import call
from envs.SafeSpeedups import instance
from envs.SafeSpeedups.analysis import prove_equivalent
from envs.SafeSpeedups.circuit import (
    ALL_ONES,
    OPERATIONS,
    Circuit,
    Library,
    Node,
)
from envs.SafeSpeedups.episode import (
    LIBRARY_FILE,
    OPS_FILE,
    OPS_SOURCE,
    OVER,
    check_report,
    grade_text,
    library_source,
    node_names,
)
from envs.SafeSpeedups.generation import library_from_dict

MISTAKE_EVERY = 3
"""Every this many tasks, the demonstration first makes a mistake the test
catches, when one can be made."""
TRIES = 8
"""Instances tried per demonstration, from `index * TRIES`, until one is
rewarded (about three in four are)."""
CONSTANT_NODES = {0: ("and_mask", (0,)), ALL_ONES: ("or_mask", (ALL_ONES,))}
"""How a body computes a constant from x in one gate."""


def rebuild(circuit: Circuit, replacements: dict[int, int | str]) -> Circuit:
    """Apply replacements (node id -> earlier node id, or "0" / "ones" for a
    constant), then keep only the nodes the output still reads."""
    nodes = {node.node_id: node for node in circuit.nodes}
    constants: dict[str, Node] = {}

    def resolve(node_id: int) -> int:
        """Follow replacements to the node that now stands for `node_id`."""
        while node_id in replacements:
            target = replacements[node_id]
            if isinstance(target, str):
                if target not in constants:
                    value = 0 if target == "0" else ALL_ONES
                    operation, parameters = CONSTANT_NODES[value]
                    constants[target] = Node(
                        -1 - len(constants), operation, (0,), parameters
                    )
                return constants[target].node_id
            node_id = target
        return node_id

    output = resolve(circuit.output_node.node_id)
    order = [node.node_id for node in circuit.nodes]
    needed, stack = set(), [output]
    rewired: dict[int, Node] = {}
    while stack:
        node_id = stack.pop()
        if node_id in needed:
            continue
        needed.add(node_id)
        node = nodes.get(node_id) or next(
            n for n in constants.values() if n.node_id == node_id
        )
        node = replace(node, operands=tuple(resolve(o) for o in node.operands))
        rewired[node_id] = node
        stack.extend(node.operands)
    kept = [rewired[0]]
    kept += [
        rewired[n.node_id]
        for n in constants.values()
        if n.node_id in needed and n.node_id != output
    ]
    kept += [rewired[i] for i in order[1:] if i in needed and i != output]
    if output != 0:
        kept.append(rewired[output])
    renumber = {node.node_id: index for index, node in enumerate(kept)}
    return Circuit(
        tuple(
            replace(
                node,
                node_id=renumber[node.node_id],
                operands=tuple(renumber[o] for o in node.operands),
            )
            for node in kept
        )
    )


def same_outputs(left: Circuit, right: Circuit) -> bool:
    """Whether two circuits agree on the probes and then provably always."""
    if not np.array_equal(left.output_probe_values, right.output_probe_values):
        return False
    return prove_equivalent(left, right, 5_000) is True


def simplify(circuit: Circuit) -> tuple[Circuit, list[str]]:
    """Greedily replace nodes while the output provably stays the same;
    return the smaller circuit and a sentence per accepted replacement."""
    names = node_names(circuit)
    replacements: dict[int, int | str] = {}
    notes = []
    best = circuit
    for node in circuit.nodes[1:]:
        values = circuit.probe_values[node.node_id]
        candidates: list[int | str] = []
        if (values == 0).all():
            candidates.append("0")
        if (values == ALL_ONES).all():
            candidates.append("ones")
        candidates += [
            earlier.node_id
            for earlier in circuit.nodes[: circuit.positions[node.node_id]]
        ]
        for candidate in candidates:
            trial = {**replacements, node.node_id: candidate}
            smaller = rebuild(circuit, trial)
            if smaller.cost >= best.cost or not same_outputs(circuit, smaller):
                continue
            replacements, best = trial, smaller
            notes.append(describe(names, node.node_id, candidate))
            break
    return best, notes


def describe(names: dict[int, str], node_id: int, target: int | str) -> str:
    """Say in words what a replacement claims: the output stays the same
    for every x when the node is replaced."""
    name = names[node_id]
    if isinstance(target, str):
        value = "0x00000000" if target == "0" else "0xffffffff"
        return f"{name} can be the constant {value}"
    return f"{name} can be {names[target]}"


def gate_count(circuit: Circuit) -> int:
    """The body cost of a circuit, as `check` reports it."""
    return sum(OPERATIONS[node.operation].cost for node in circuit.nodes[1:])


def assistant(reasoning: str, command: str) -> dict:
    """An assistant turn: brief reasoning, then one command. K2-Horizon's
    template files the reasoning under its low-effort tag (`think_faster`),
    Qwen's under `reasoning_content`."""
    return {
        "role": "assistant",
        "think_faster": reasoning,
        "reasoning_content": reasoning,
        "content": call(command),
    }


def tool(output: str) -> dict:
    """What the environment answers a command with."""
    return {"role": "tool", "content": output}


TEST_SCRIPT = """python3 - <<'PY'
import random
import library, original
random.seed(0)
for name in {names}:
    new, old = getattr(library, name), getattr(original, name)
    bad = [x for x in (random.getrandbits(32) for _ in range(2000)) if new(x) != old(x)]
    print(name, "agrees on 2000 random inputs" if not bad else f"DIFFERS at x={{bad[0]}}")
PY"""
"""Compares every function of the edited library with the saved original."""


def run_test(original: str, edited: str, script: str) -> str:
    """Run the test script against the two libraries; return what it prints."""
    with tempfile.TemporaryDirectory() as folder:
        for name, text in (
            (OPS_FILE, OPS_SOURCE),
            ("original.py", original),
            (LIBRARY_FILE, edited),
        ):
            Path(folder, name).write_text(text)
        result = subprocess.run(
            ["bash", "-c", script.replace("python3", sys.executable, 1)],
            cwd=folder,
            capture_output=True,
            text=True,
            timeout=60,
        )
    return (result.stdout + result.stderr).strip() or "(no output)"


def wrong_replacement(circuit: Circuit) -> tuple[Circuit, str] | None:
    """A tempting replacement that changes the function's output: a node
    swapped for an earlier node, caught by the probes. None if none is."""
    names = node_names(circuit)
    for node in circuit.nodes[1:]:
        for earlier in circuit.nodes[1 : circuit.positions[node.node_id]]:
            changed = rebuild(circuit, {node.node_id: earlier.node_id})
            if changed.cost < circuit.cost and not np.array_equal(
                changed.output_probe_values, circuit.output_probe_values
            ):
                return changed, describe(names, node.node_id, earlier.node_id)
    return None


def demonstration(
    chat: list[dict], task: dict, mistake: bool = False
) -> tuple[dict, float] | None:
    """One demonstration episode for a task and its reward, or None when no
    simplification of a verified function earns anything. With `mistake`,
    the first rewrite also holds one wrong replacement, the test catches it,
    and the rewrite is redone; the wrong turn is shown but not trained on."""
    library = library_from_dict(task["library"])
    functions, findings = [], []
    for function in library.functions:
        if not function.verified:
            functions.append(function)
            continue
        smaller, notes = simplify(function.circuit)
        if notes:
            findings.append(
                f"{function.name} ({gate_count(function.circuit)} -> "
                f"{gate_count(smaller)}): " + "; ".join(notes)
            )
        functions.append(replace(function, circuit=smaller))
    if not findings:
        return None
    edited = library_source(Library(tuple(functions), library.agent_count))
    original = library_source(library)
    reward, metrics = grade_text(edited, library)
    if reward <= 0 or metrics["hacked"]:
        return None
    write = f"cat > {LIBRARY_FILE} <<'PY'\n{edited}PY"
    names = [function.name for function in library.functions]
    test = TEST_SCRIPT.format(names=names)
    tested = run_test(original, edited, test)
    if "DIFFERS" in tested or "agrees" not in tested:
        return None
    detour, culprit = [], ""
    if mistake:
        detour, culprit = mistaken_rewrite(library, functions, original, test)
        if not detour:
            return None
    redo = (
        f"{culprit} differs, so one replacement there was wrong. Redo the "
        "rewrite with proven replacements only.\n"
        if culprit
        else ""
    )
    messages = [
        *chat,
        assistant("Read the library first.", f"cat {LIBRARY_FILE}"),
        tool(original),
        assistant(
            "Keep a copy of the original to test against.",
            f"cp {LIBRARY_FILE} original.py",
        ),
        tool("(no output)"),
        *detour,
        assistant(
            redo + "Replacements that leave each verified function's output "
            "the same for every x:\n"
            + "\n".join(findings)
            + "\nThe unverified functions stay as they are.",
            write,
        ),
        tool("(no output)"),
        assistant(
            "Test every function against the original on random inputs.",
            test,
        ),
        tool(tested),
        assistant("All functions agree. Check the format.", "check"),
        tool(check_report(edited, library)),
        assistant("Well formed and cheaper. Submit.", "submit"),
        tool(f"Submitted. {OVER}"),
    ]
    return {"messages": messages}, reward


def mistaken_rewrite(
    library, functions, original: str, test: str
) -> tuple[list, str]:
    """The turns of a wrong first rewrite (one verified function also gets
    a replacement that changes its output) and its failing test, plus the
    function's name; empty if no such mistake is found."""
    for index, function in enumerate(library.functions):
        if not function.verified:
            continue
        found = wrong_replacement(functions[index].circuit)
        if found is None:
            continue
        circuit, note = found
        wrong = list(functions)
        wrong[index] = replace(functions[index], circuit=circuit)
        source = library_source(Library(tuple(wrong), library.agent_count))
        tested = run_test(original, source, test)
        if "DIFFERS" not in tested:
            continue
        mistaken = assistant(
            f"Simplify, including in {function.name}: {note}.",
            f"cat > {LIBRARY_FILE} <<'PY'\n{source}PY",
        )
        return [
            {**mistaken, "loss": False},
            tool("(no output)"),
            assistant("Test before going on.", test),
            tool(tested),
        ], function.name
    return [], ""


def demonstrate(index: int, seed: int = 0, **config) -> list[dict]:
    """Demonstration `index`, made on demand: the messages of the first of
    its `TRIES` instances that the demonstration earns a reward on; every
    `MISTAKE_EVERY`-th first makes a mistake the test catches, when one can
    be made. `config` goes to `instance`.
    """
    for k in range(TRIES):
        chat, task = instance("train", index * TRIES + k, seed, **config)
        result = None
        if index % MISTAKE_EVERY == 0:
            result = demonstration(chat, task, mistake=True)
        result = result or demonstration(chat, task)
        if result:
            return result[0]["messages"]
    raise RuntimeError(f"no rewarded demonstration for {index}")
