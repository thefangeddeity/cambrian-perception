from __future__ import annotations

"""
The safe operation vocabulary and its evaluator -- the one place in
this repo that ever "runs" a brain-authored program. See the README's
"fishbowl boundary" section for the full contract; the short version:
this is a closed, fixed set of pure numeric operations, no loops, no
recursion, no I/O. A tree built from these is finite and is walked
exactly once to evaluate it, so evaluation always terminates -- that
is a property of the language, not a timeout wrapped around it.

Deliberately modeled on orbital-organism's modules/scratch_blocks.py
(same project, same author) -- that module already proved this exact
pattern (a closed vocabulary of safe blocks interpreted by a fixed
evaluator, not literal code generation) works for a real, live,
unattended evolutionary loop. This is that same idea, generalized and
given its own repo to explore further, with an explicit "what can this
recursively touch" question scratch_blocks.py was never asked.
"""

from dataclasses import dataclass, field
from typing import Union

import numpy as np

# Protects div/pow the same way orbital-organism's scratch_blocks.py
# learned to the hard way (a real, confirmed divide-by-zero in its own
# pow gradient was the root cause of a recurring "non-finite loss"
# bug there) -- floored inputs, not a try/except around the operation.
_EPS = 1e-6


def _protected_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    safe_b = np.where(np.abs(b) < _EPS, _EPS, b)
    return a / safe_b


def _protected_pow(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    base = np.clip(np.abs(a), _EPS, 1e6)
    exponent = np.clip(b, -4.0, 4.0)
    return np.sign(a) * base**exponent


def _select(cond: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.where(cond > 0.0, a, b)


# name -> (arity, function). Every function here is pure, total (never
# raises for any finite float input, by construction -- protected_div/
# protected_pow above), and vectorized over numpy arrays. Nothing here
# reads a file, opens a socket, imports anything dynamically, or can
# run for unbounded time -- each is O(1) per element.
OPS = {
    "add": (2, lambda a, b: a + b),
    "sub": (2, lambda a, b: a - b),
    "mul": (2, lambda a, b: a * b),
    "div": (2, _protected_div),
    "pow": (2, _protected_pow),
    "neg": (1, lambda a: -a),
    "abs": (1, np.abs),
    "min": (2, np.minimum),
    "max": (2, np.maximum),
    "select": (3, _select),  # select(cond, a, b): a if cond > 0 else b
}

MAX_CONST = 5.0


@dataclass
class Node:
    """
    A leaf is either a variable reference (kind="var", index=which
    input column) or a constant (kind="const", value=a float already
    clamped to [-MAX_CONST, MAX_CONST] at construction time -- not at
    eval time, so a mutation can never smuggle in an unbounded value).
    An op node (kind="op") holds one of OPS's names and that many
    child Nodes.
    """
    kind: str  # "var" | "const" | "op"
    index: int = 0
    value: float = 0.0
    op: str = ""
    children: list["Node"] = field(default_factory=list)

    def node_count(self) -> int:
        return 1 + sum(child.node_count() for child in self.children)

    def depth(self) -> int:
        if not self.children:
            return 1
        return 1 + max(child.depth() for child in self.children)

    def evaluate(self, inputs: np.ndarray) -> np.ndarray:
        """
        inputs: (n_samples, n_vars) array. Returns (n_samples,).
        Pure recursion over a FINITE tree -- no data-driven looping,
        so this always terminates in time proportional to node_count().
        """
        if self.kind == "var":
            return inputs[:, self.index]
        if self.kind == "const":
            return np.full(inputs.shape[0], self.value)
        arity, fn = OPS[self.op]
        args = [child.evaluate(inputs) for child in self.children[:arity]]
        result = fn(*args)
        # Final, unconditional floor -- defense in depth on top of
        # each op already being individually protected. A tree is
        # only ever as safe as its weakest composition; this makes
        # the WHOLE tree's output safe regardless of how deeply
        # nested the unsafe combination is.
        return np.nan_to_num(result, nan=0.0, posinf=MAX_CONST * 1e3, neginf=-MAX_CONST * 1e3)

    def to_dict(self) -> dict:
        return {
            "kind": self.kind, "index": self.index, "value": self.value,
            "op": self.op, "children": [c.to_dict() for c in self.children],
        }

    @staticmethod
    def from_dict(data: dict) -> "Node":
        return Node(
            kind=data["kind"], index=data.get("index", 0),
            value=data.get("value", 0.0), op=data.get("op", ""),
            children=[Node.from_dict(c) for c in data.get("children", [])],
        )
