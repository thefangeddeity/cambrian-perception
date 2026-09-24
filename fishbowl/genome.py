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

# How fast the per-operator success estimate tracks new evidence, and
# how fast meta_mutation_rate itself settles toward its floor once
# real evidence starts arriving. See update_mutation_weights()'s own
# docstring for why these replace the old blind random-walk approach.
# META_DECAY validated against fishbowl/task.py's synthetic task
# (15000-generation, 12-seed sweep): 0.9995 beat 0.999/0.998 on mean
# final fitness AND, more importantly, never produced the total-
# stagnation failure mode 0.995 did on one in twelve seeds (locking
# onto a bad operator mix before enough evidence had accumulated to
# correct it).
OP_SUCCESS_EMA_ALPHA = 0.05
META_DECAY = 0.9995

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
    op_success = {name: 0.5 for name in TASK_OPS}
    return Genome(trees=trees, mutation_weights=weights, meta_mutation_rate=0.15, n_vars=n_vars, op_success=op_success)


class Genome:
    def __init__(
        self,
        trees: dict[str, blocks.Node],
        mutation_weights: dict[str, float],
        meta_mutation_rate: float,
        n_vars: int = 3,
        op_success: dict[str, float] | None = None,
    ):
        self.trees = trees
        self.mutation_weights = mutation_weights
        self.meta_mutation_rate = meta_mutation_rate
        self.n_vars = n_vars
        # Per-operator EMA of how often ITS attempts get accepted --
        # the real evidence update_mutation_weights() nudges
        # mutation_weights toward. Defaults to a neutral 0.5 prior for
        # any operator with no track record yet (e.g. resuming an old
        # checkpoint that predates this field).
        self.op_success = dict(op_success) if op_success is not None else {name: 0.5 for name in TASK_OPS}

    @property
    def channels(self) -> tuple[str, ...]:
        return tuple(self.trees.keys())

    def clone(self) -> "Genome":
        return Genome(
            {name: copy.deepcopy(tree) for name, tree in self.trees.items()},
            dict(self.mutation_weights),
            self.meta_mutation_rate,
            self.n_vars,
            dict(self.op_success),
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

    def _grow(self, rng: random.Random, channel: str, max_nodes: int, max_depth: int) -> tuple[bool, bool]:
        """
        Returns (applied, hit_ceiling) -- hit_ceiling is True ONLY when
        this failed because the tree is genuinely at the real size/
        depth ceiling. Kept separate from the plain bool because
        mutate_task (below) needs to tell a REAL ceiling-hit apart from
        an ordinary structural noop, for sandbox.note_ceiling -- see
        its own comment for why conflating the two produced a real,
        confusing false signal.
        """
        tree = self.trees[channel]
        if tree.node_count() >= max_nodes or tree.depth() >= max_depth:
            return False, True
        leaves = [n for n in self._all_nodes(channel) if n.kind != "op"]
        if not leaves:
            return False, False
        target = rng.choice(leaves)
        replacement = _random_small_tree(rng, self.n_vars, max_depth=2)
        target.kind, target.op, target.children = replacement.kind, replacement.op, replacement.children
        target.index, target.value = replacement.index, replacement.value
        return True, False

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

        Returns (channel, operator_applied) -- operator_applied is the
        real chosen op name on success. On failure it's one of two
        DIFFERENT strings, not one generic "noop":
        - "noop_ceiling": grow genuinely couldn't apply because the
          tree is already at the real max_nodes/max_depth ceiling.
        - "noop_inapplicable": the chosen operator just doesn't apply
          to this tree's current SHAPE, regardless of ceilings -- e.g.
          mutate_const picked on a tree with no const leaves, or
          shrink/mutate_op picked on a single-leaf tree with no op
          nodes at all. This is a normal, frequent, and harmless
          outcome for a small tree (most of a tiny tree's 5 possible
          operators simply don't apply to it yet) -- real bug, caught
          live: the caller used to log EVERY noop as "hit
          tree_size_or_depth," which made a perfectly healthy small
          tree look like it was slamming into a 300-node ceiling when
          it was nowhere close -- see run_vision.py's own use of this.
        """
        if channel is None:
            channel = rng.choice(self.channels)

        names = list(self.mutation_weights.keys())
        probs = np.array([self.mutation_weights[n] for n in names], dtype=float)
        probs = probs / probs.sum()
        choice = rng.choices(names, weights=probs, k=1)[0]

        if choice == "grow":
            applied, hit_ceiling = self._grow(rng, channel, max_nodes, max_depth)
        else:
            applied = {
                "mutate_const": lambda: self._mutate_const(rng, channel),
                "mutate_op": lambda: self._mutate_op(rng, channel),
                "shrink": lambda: self._shrink(rng, channel),
                "reroll_subtree": lambda: self._reroll_subtree(rng, channel),
            }[choice]()
            hit_ceiling = False

        if applied:
            return channel, choice
        return channel, ("noop_ceiling" if hit_ceiling else "noop_inapplicable")

    def update_mutation_weights(self, op_applied: str, accepted: bool) -> None:
        """
        The recursive step, done right. The original version of this
        method proposed a fully random, undirected nudge to ONE
        mutation_weights entry, as its own candidate competing in the
        SAME "does fitness improve" accept/reject gate the run loop
        uses for tree mutations (run_vision.py picked this branch on
        15% of generations, INSTEAD of a tree mutation). That's
        structurally broken: mutation_weights never feed
        Genome.evaluate() at all, so a weights-only candidate's
        fitness was always bit-identical to its parent's -- "strictly
        greater than parent + margin" can never pass for a value that
        is, by construction, never greater. Confirmed against
        production's real evolution_log.json: mutation_weights and
        meta_mutation_rate had not moved one bit off their uniform
        defaults across 8500+ real logged generations, and every one
        of those weight-mutation generations was ALSO a fully wasted
        generation for the trees (no tree mutation was even attempted
        on it).

        Real fix: there is no fitness to gate this on -- these weights
        don't produce behavior, they bias which operator gets TRIED
        next. So track it as what it actually is: a slow, per-operator
        exponential moving average of how often that operator's
        attempts get accepted (self.op_success), and continuously nudge
        mutation_weights toward whatever's currently working. Called
        every generation from the run loop, after its accept/reject
        decision, on whichever genome persists (the newly-accepted
        candidate, or the unchanged parent on a reject) -- using the
        real evidence from THIS generation's real attempt, not a
        separate hypothetical draw.

        meta_mutation_rate still governs the step size (how hard to
        chase the current evidence) and still drifts -- geometrically
        toward its floor as attempts accumulate, same clip range as
        before (0.01 floor, so it settles rather than freezes: even at
        the floor, mutation_weights keep tracking new evidence every
        generation, just slowly, which is what actually lets this
        genome settle into a mutation STYLE instead of wandering
        forever). Validated empirically (fishbowl/task.py's synthetic
        task, 15000 generations x 12 seeds) against the old behavior
        and against a merely-faster meta_mutation_rate decay -- see
        META_DECAY's own comment for the sweep that picked 0.9995.
        """
        # Covers both "noop_ceiling" and "noop_inapplicable" -- neither
        # is a key in op_success, so this same check already excludes
        # both without needing to name either explicitly.
        if op_applied not in self.op_success:
            return

        self.op_success[op_applied] = (
            (1.0 - OP_SUCCESS_EMA_ALPHA) * self.op_success[op_applied]
            + OP_SUCCESS_EMA_ALPHA * (1.0 if accepted else 0.0)
        )

        total_success = sum(self.op_success.values())
        target = {name: v / total_success for name, v in self.op_success.items()}
        for name in self.mutation_weights:
            self.mutation_weights[name] += self.meta_mutation_rate * (target[name] - self.mutation_weights[name])
            self.mutation_weights[name] = max(0.01, self.mutation_weights[name])
        total_weight = sum(self.mutation_weights.values())
        for name in self.mutation_weights:
            self.mutation_weights[name] /= total_weight

        self.meta_mutation_rate = float(np.clip(self.meta_mutation_rate * META_DECAY, 0.01, 1.0))

    def to_dict(self) -> dict:
        return {
            "trees": {name: tree.to_dict() for name, tree in self.trees.items()},
            "mutation_weights": self.mutation_weights,
            "meta_mutation_rate": self.meta_mutation_rate,
            "n_vars": self.n_vars,
            "op_success": self.op_success,
        }

    @staticmethod
    def from_dict(data: dict) -> "Genome":
        return Genome(
            trees={name: blocks.Node.from_dict(t) for name, t in data["trees"].items()},
            mutation_weights=dict(data["mutation_weights"]),
            meta_mutation_rate=float(data["meta_mutation_rate"]),
            n_vars=int(data.get("n_vars", 3)),
            # Old checkpoints (pre-dating this field) fall back to the
            # same neutral 0.5-for-everyone prior random_genome() uses
            # for a fresh genome.
            op_success=data.get("op_success") or {name: 0.5 for name in TASK_OPS},
        )
