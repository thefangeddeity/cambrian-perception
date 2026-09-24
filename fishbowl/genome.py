from __future__ import annotations

"""
Two-level genome: a set of named output trees (see blocks.py), and the
weights that decide how those trees get mutated. Mutating the weights
themselves -- changing HOW it changes, not just WHAT it's currently
trying -- is the recursive part this repo exists to watch. See
README's "fishbowl boundary" for why this is deliberately bounded to
two levels plus one scalar meta-mutation rate, not unbounded meta-
regress.

Multiple named trees (not one) as of the foveated-vision design: the
organism needs more than one output per frame -- a response magnitude
(scored against the reflex/conspec signals) AND where to move its
fovea next (pan, tilt) -- see fovea.py. the user's own framing: "we take a
small frame of Tanzania's feed, a fovea, and replicate pan/tilt
digitally, so the reflexes can train." All channels share the exact
same safe block-tree language and mutation machinery; there's no
separate, less-safe code path for "the action-producing tree" versus
"the perception tree."
"""

import copy
import random

import numpy as np

from . import blocks

TASK_OPS = ("mutate_const", "mutate_op", "grow", "shrink", "reroll_subtree")

DEFAULT_CHANNELS = ("response", "pan", "tilt")

# Arity groups -- mutate_op only ever swaps an op for one of the SAME
# arity, so a mutation can never change how many children a node needs
# (which would otherwise require also inventing/discarding children,
# a much larger and riskier mutation than "try a different operator
# here").
_OPS_BY_ARITY: dict[int, list[str]] = {}
for _name, (_arity, _fn) in blocks.OPS.items():
    _OPS_BY_ARITY.setdefault(_arity, []).append(_name)


def _random_leaf(rng: random.Random, n_vars: int) -> blocks.Node:
    if rng.random() < 0.5:
        return blocks.Node(kind="var", index=rng.randrange(0, n_vars))
    return blocks.Node(kind="const", value=rng.uniform(-blocks.MAX_CONST, blocks.MAX_CONST))


def _random_small_tree(rng: random.Random, n_vars: int, max_depth: int = 2) -> blocks.Node:
    if max_depth <= 1 or rng.random() < 0.4:
        return _random_leaf(rng, n_vars)
    op = rng.choice(list(blocks.OPS.keys()))
    arity, _ = blocks.OPS[op]
    children = [_random_small_tree(rng, n_vars, max_depth - 1) for _ in range(arity)]
    return blocks.Node(kind="op", op=op, children=children)


def random_genome(rng: random.Random, n_vars: int = 3, channels: tuple[str, ...] = DEFAULT_CHANNELS) -> "Genome":
    # n_vars: how many named input slots a leaf can reference -- 3 for
    # the original synthetic task, one per retina/fovea cell for the
    # vision task. Stored on the genome itself so mutation later knows
    # the valid range without needing it passed around separately.
    trees = {name: _random_small_tree(rng, n_vars, max_depth=3) for name in channels}
    weights = {name: 1.0 / len(TASK_OPS) for name in TASK_OPS}
    return Genome(trees=trees, mutation_weights=weights, meta_mutation_rate=0.15, n_vars=n_vars)


class Genome:
    def __init__(self, trees: dict[str, blocks.Node], mutation_weights: dict[str, float], meta_mutation_rate: float, n_vars: int = 3):
        self.trees = trees
        self.mutation_weights = mutation_weights
        self.meta_mutation_rate = meta_mutation_rate
        self.n_vars = n_vars

    @property
    def channels(self) -> tuple[str, ...]:
        return tuple(self.trees.keys())

    def clone(self) -> "Genome":
        return Genome(
            {name: copy.deepcopy(tree) for name, tree in self.trees.items()},
            dict(self.mutation_weights),
            self.meta_mutation_rate,
            self.n_vars,
        )

    def evaluate(self, name: str, inputs: np.ndarray) -> np.ndarray:
        return self.trees[name].evaluate(inputs)

    def _all_nodes(self, channel: str) -> list[blocks.Node]:
        out = []

        def walk(node):
            out.append(node)
            for child in node.children:
                walk(child)

        walk(self.trees[channel])
        return out

    def _op_nodes(self, channel: str) -> list[blocks.Node]:
        return [n for n in self._all_nodes(channel) if n.kind == "op"]

    # -- per-channel tree-mutation operators ---------------------------

    def _mutate_const(self, rng: random.Random, channel: str) -> bool:
        consts = [n for n in self._all_nodes(channel) if n.kind == "const"]
        if not consts:
            return False
        node = rng.choice(consts)
        node.value = float(np.clip(node.value + rng.gauss(0.0, 0.5), -blocks.MAX_CONST, blocks.MAX_CONST))
        return True

    def _mutate_op(self, rng: random.Random, channel: str) -> bool:
        ops = self._op_nodes(channel)
        if not ops:
            return False
        node = rng.choice(ops)
        arity, _ = blocks.OPS[node.op]
        candidates = [o for o in _OPS_BY_ARITY[arity] if o != node.op]
        if not candidates:
            return False
        node.op = rng.choice(candidates)
        return True

    def _grow(self, rng: random.Random, channel: str, max_nodes: int, max_depth: int) -> bool:
        tree = self.trees[channel]
        if tree.node_count() >= max_nodes or tree.depth() >= max_depth:
            return False
        leaves = [n for n in self._all_nodes(channel) if n.kind != "op"]
        if not leaves:
            return False
        target = rng.choice(leaves)
        replacement = _random_small_tree(rng, self.n_vars, max_depth=2)
        target.kind, target.op, target.children = replacement.kind, replacement.op, replacement.children
        target.index, target.value = replacement.index, replacement.value
        return True

    def _shrink(self, rng: random.Random, channel: str) -> bool:
        ops = self._op_nodes(channel)
        tree = self.trees[channel]
        if not ops or tree.kind != "op":
            return False
        # Only ever shrinks a node whose parent can safely take its
        # place -- picking the whole tree's root out of ops (if it's
        # an op node) and replacing it with one of its own children is
        # always structurally valid, so that's the one case handled
        # directly rather than needing parent pointers at all.
        node = rng.choice(ops)
        if not node.children:
            return False
        replacement = rng.choice(node.children)
        if node is tree:
            self.trees[channel] = copy.deepcopy(replacement)
        else:
            node.kind, node.op = replacement.kind, replacement.op
            node.index, node.value = replacement.index, replacement.value
            node.children = replacement.children
        return True

    def _reroll_subtree(self, rng: random.Random, channel: str) -> bool:
        nodes = self._all_nodes(channel)
        target = rng.choice(nodes)
        replacement = _random_small_tree(rng, self.n_vars, max_depth=2)
        if target is self.trees[channel]:
            self.trees[channel] = replacement
        else:
            target.kind, target.op = replacement.kind, replacement.op
            target.index, target.value = replacement.index, replacement.value
            target.children = replacement.children
        return True

    def mutate_task(self, rng: random.Random, max_nodes: int, max_depth: int, channel: str | None = None) -> tuple[str, str]:
        """
        Picks a channel (random, if not given) and ONE task-mutation
        operator for it, weighted by self.mutation_weights -- the same
        weights govern every channel, not one set per channel, since
        the interesting thing to watch is whether the genome converges
        on a mutation STYLE at all, not per-channel bookkeeping.
        Returns (channel, operator_applied) -- operator is "noop" if
        the chosen one couldn't apply (e.g. grow at the size ceiling).
        """
        if channel is None:
            channel = rng.choice(self.channels)

        names = list(self.mutation_weights.keys())
        probs = np.array([self.mutation_weights[n] for n in names], dtype=float)
        probs = probs / probs.sum()
        choice = rng.choices(names, weights=probs, k=1)[0]

        applied = {
            "mutate_const": lambda: self._mutate_const(rng, channel),
            "mutate_op": lambda: self._mutate_op(rng, channel),
            "grow": lambda: self._grow(rng, channel, max_nodes, max_depth),
            "shrink": lambda: self._shrink(rng, channel),
            "reroll_subtree": lambda: self._reroll_subtree(rng, channel),
        }[choice]()

        return channel, (choice if applied else "noop")

    def mutate_weights(self, rng: random.Random) -> None:
        """
        The recursive step: nudges ONE mutation weight by an amount
        scaled by meta_mutation_rate, renormalizes, and lets meta_
        mutation_rate itself drift slightly -- this is what makes it
        "how it changes how it changes," not just "how it changes."
        Clipped to a sane range so meta_mutation_rate can't drift to
        zero (frozen forever) or explode (pure noise every step).
        """
        name = rng.choice(list(self.mutation_weights.keys()))
        delta = rng.gauss(0.0, self.meta_mutation_rate)
        self.mutation_weights[name] = max(0.01, self.mutation_weights[name] + delta)
        total = sum(self.mutation_weights.values())
        for key in self.mutation_weights:
            self.mutation_weights[key] /= total

        self.meta_mutation_rate = float(
            np.clip(self.meta_mutation_rate * rng.uniform(0.9, 1.1), 0.01, 1.0)
        )

    def to_dict(self) -> dict:
        return {
            "trees": {name: tree.to_dict() for name, tree in self.trees.items()},
            "mutation_weights": self.mutation_weights,
            "meta_mutation_rate": self.meta_mutation_rate,
            "n_vars": self.n_vars,
        }

    @staticmethod
    def from_dict(data: dict) -> "Genome":
        return Genome(
            trees={name: blocks.Node.from_dict(t) for name, t in data["trees"].items()},
            mutation_weights=dict(data["mutation_weights"]),
            meta_mutation_rate=float(data["meta_mutation_rate"]),
            n_vars=int(data.get("n_vars", 3)),
        )
