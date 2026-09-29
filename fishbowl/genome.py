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
AND where to move its
fovea next (pan, tilt) -- see fovea.py. All channels share the exact
same safe block-tree language and mutation machinery; there's no
separate, less-safe code path for "the action-producing tree" versus
"the perception tree."
"""

import copy
import math
import random

import numpy as np

from . import blocks, fovea
from .controller import MosquitoBrain
from .mushroom import MAX_KC
from .prey import PREY_CLASSES
from .state import METABOLISM_GUARD, MIN_METABOLISM, MOBILIZE_BELOW, PUMP_REF, STORE_ABOVE

# mutate_fovea grows or shrinks its eye by one ring of receptors
# (genome.receptors, fovea.py) instead of touching a tree -- same fitness
# gate, same operator-weight learning, so how often resizing gets TRIED is
# itself learned from evidence.
# mutate_brain perturbs the recurrent motor brain (controller.py).
# mutate_pace changes its pace of life (how often it looks; see run_vision.py).
# mutate_colour adds or removes a colour-opponent channel in the gaze (see retina.py).
# mutate_cones grows or shrinks the gaze's central patch of cones (colour receptors) by a ring (fovea.py).
# grow_unit / shrink_unit add (a duplicated, unconnected copy) or remove a hidden unit.
# duplicate_layer / remove_layer stack a silent copy of the brain's top layer or remove one (controller.py).
# grow_kc / shrink_kc add or remove a cohort of Kenyon cells, mutate_learning changes the learning
# rate of its mushroom body (fishbowl/mushroom.py: lifetime reward learning).
# grow_channel / add_prediction / shrink_channel add or remove a brain
# channel -- an output wired back in as an input (controller.py): a latch
# duplicated from an existing output, or a predictor of one of its inputs.
# mutate_stabilizer changes its image-stabilization reflex gain (run_vision.py).
# mutate_zoom changes the gain of its zoom lens (fovea.magnification: 0 = off).
# mutate_metabolism moves its metabolic strategy between ectotherm and endotherm (state.py).
# mutate_host retunes its prey sense across the living classes (host preference; people never below even).
# mutate_replay changes how much it replays experience: awake in quiet moments, asleep, and the REM share.
# mutate_vigilance changes how big a change in the field it takes to wake it (the sentry's threshold).
TASK_OPS = ("mutate_const", "mutate_op", "grow", "shrink", "reroll_subtree", "mutate_fovea", "mutate_brain", "mutate_pace",
            "mutate_colour", "grow_channel", "add_prediction", "shrink_channel", "mutate_stabilizer",
            "mutate_prey_sense", "grow_unit", "shrink_unit", "duplicate_layer", "remove_layer", "mutate_cones",
            "grow_kc", "shrink_kc", "mutate_learning", "mutate_zoom", "mutate_metabolism", "mutate_host",
            "mutate_replay", "mutate_vigilance", "mutate_pump", "mutate_aversive", "mutate_receptor_speed", "mutate_plant_sense",
            "mutate_imagery", "mutate_recall", "mutate_scenes", "mutate_pool", "mutate_sleep_set",
            "mutate_setpoints", "mutate_bore", "mutate_archetypes")
STABILIZER_SIGMA = 0.1
MAX_SLEEP_SET = 1024  # a safety bound on its sleep test set (looks kept); its price is the edits tested on it
MAX_SCENES = 64  # a safety bound only: its price (matching each look) is what limits it
# The feeding pump has no biological bounds: its upkeep and its intake set
# its limits, and log-normal steps can shrink it toward a nip without ever
# reaching zero. These only keep the float sane (numerical guards).
PUMP_GUARD = (1e-6, 1e6)
SLOWNESS_GUARD = 1e6  # photoreceptor slowness: a numerical guard only (its cost and its blur set its limits)
ZOOM_SIGMA = STABILIZER_SIGMA  # a reflex gain, 0..1, mutates as the stabilizer's does
# The traits from the 2026-09-27 panels mutate at the same scale (assumption):
# metabolism (0.1 ectotherm .. 1 endotherm, born 1: today's body), host
# preference (a weight per living class, born even), replay (counts per look,
# born 0 -- off) and vigilance (born 1: today's wake thresholds).
TRAIT_SIGMA = STABILIZER_SIGMA
MAX_REPLAYS = 16                 # replays per look: a bound only -- the price and the look's deadline limit it
MIN_VIGILANCE, MAX_VIGILANCE = 0.25, 4.0   # bounds only: a quarter to four times as easily woken
# Prey sense (run_vision.py): 0 = eyes only, 1 = scent (prey somewhere in
# view), 2 = + a coarse direction to it. mutate_prey_sense steps it by one.
MAX_PREY_SENSE = 3  # 3 = + the velocity of the host it follows (2026-09-28 audit: pursuit)
# Structural additions that change nothing at birth AND cost nothing until
# used (run_vision.py keeps them on a tie, so they can drift until they are
# useful). Not grow_kc or grow_unit (2026-09-28, a panel): Kenyon cells and
# hidden units are computed -- and priced -- every look from birth, but the
# price was below the tie window, so a tie always kept growth while a shrink
# was kept one time in ten: a one-way ratchet that drove the mushroom body to
# its 4096 ceiling (tina 3712, tanzania's old lineage 4096) without selection
# ever weighing it. Priced growth now faces the same odds as shrinking.
NEUTRAL_GROWTH_OPS = ("grow_channel", "add_prediction", "mutate_prey_sense", "duplicate_layer", "mutate_learning")
# The mushroom body (fishbowl/mushroom.py): Kenyon cells come and go a cohort
# at a time; a first learning rate is drawn log-uniformly from 0.001 up to 1
# (above 1 a single update overshoots its own prediction error), later ones
# step multiplicatively. Both chosen, documented in the constants audit.
KC_STEP = 64
LEARNING_MIN, LEARNING_MAX, LEARNING_SIGMA = 1e-3, 1.0, 0.5
MAX_COLOUR_CHANNELS = 2  # 0 = light only, 1 = + red-green, 2 = + blue-yellow
MIN_PACE, MAX_PACE = 1, 6  # resting gaze interval: every 1st .. 6th frame
BRAIN_FLOOR = 0.3  # share of mutations always given to the brain
# The perception tree's receptor planes (blocks.Node kind "cell"): the gaze
# now, the previous gaze, red-green, blue-yellow (zero without colour vision).
RETINA_PLANES = 4
# The tree's inputs before receptors were read by position (a flat vector:
# 12x12 gaze now, 12x12 previous, 2 movement, the brain's units, 2 x 12x12
# colour) -- only for migrating older checkpoints.
_OLD_SIDE, _OLD_CELLS = 12, 144

# Motor control (pan/tilt/zoom) moved to the recurrent brain; the tree
# genome keeps the perception readout only.
DEFAULT_CHANNELS = ("response",)

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


def _random_leaf(rng: random.Random, n_vars: int, n_side: int = 0) -> blocks.Node:
    """A constant half the time; otherwise any input equally likely -- one of
    the n_vars plain inputs or one of the receptors its eye has now (n_side x
    n_side per plane; none for a task without an eye)."""
    if rng.random() < 0.5:
        k = rng.randrange(0, n_vars + RETINA_PLANES * n_side * n_side)
        if k < n_vars:
            return blocks.Node(kind="var", index=k)
        plane, pos = divmod(k - n_vars, n_side * n_side)
        iy, ix = divmod(pos, n_side)
        return blocks.Node(kind="cell", index=plane, kx=ix - n_side // 2, ky=iy - n_side // 2)
    return blocks.Node(kind="const", value=rng.uniform(-blocks.MAX_CONST, blocks.MAX_CONST))


def _random_small_tree(rng: random.Random, n_vars: int, max_depth: int = 2, n_side: int = 0) -> blocks.Node:
    if max_depth <= 1 or rng.random() < 0.4:
        return _random_leaf(rng, n_vars, n_side)
    op = rng.choice(list(blocks.OPS.keys()))
    arity, _ = blocks.OPS[op]
    children = [_random_small_tree(rng, n_vars, max_depth - 1, n_side) for _ in range(arity)]
    return blocks.Node(kind="op", op=op, children=children)


def edit_tree(tree: blocks.Node, rng: random.Random, n_vars: int, receptors: int,
              max_nodes: int, max_depth: int) -> blocks.Node | None:
    """One small edit to a copy of a tree -- the operators evolution uses
    (sleep programming: the organism trying edits to its own tree). None if
    the edit didn't apply."""
    holder = object.__new__(Genome)
    holder.trees, holder.n_vars, holder.receptors = {"t": copy.deepcopy(tree)}, n_vars, receptors
    kind = rng.choice(("mutate_const", "mutate_op", "grow", "shrink", "reroll_subtree", "mutate_pool"))
    if kind == "grow":
        ok = holder._grow(rng, "t", max_nodes, max_depth)[0]
    else:
        ok = {"mutate_const": holder._mutate_const, "mutate_op": holder._mutate_op, "shrink": holder._shrink,
              "reroll_subtree": holder._reroll_subtree, "mutate_pool": holder._mutate_pool}[kind](rng, "t")
    return holder.trees["t"] if ok else None


def _become(target: blocks.Node, other: blocks.Node) -> None:
    """Turns target into other in place (its parent keeps pointing at it)."""
    target.kind, target.op, target.children = other.kind, other.op, other.children
    target.index, target.value, target.kx, target.ky, target.angle = other.index, other.value, other.kx, other.ky, other.angle


def random_genome(rng: random.Random, n_vars: int = 3, channels: tuple[str, ...] = DEFAULT_CHANNELS,
                  receptors: int = 0) -> "Genome":
    # n_vars: how many plain input slots a leaf can reference -- 3 for the
    # original synthetic task; for vision, its own last movement and the
    # brain's units, with the eye's receptors read by position besides
    # (receptors per side; 0 = no eye). Stored on the genome itself so
    # mutation later knows the valid range without needing it passed around.
    trees = {name: _random_small_tree(rng, n_vars, max_depth=3, n_side=receptors) for name in channels}
    weights = {name: 1.0 / len(TASK_OPS) for name in TASK_OPS}
    op_success = {name: 0.5 for name in TASK_OPS}
    return Genome(trees=trees, mutation_weights=weights, meta_mutation_rate=0.15, n_vars=n_vars,
                  op_success=op_success, brain=MosquitoBrain.random(rng), receptors=receptors)


class Genome:
    def __init__(
        self,
        trees: dict[str, blocks.Node],
        mutation_weights: dict[str, float],
        meta_mutation_rate: float,
        n_vars: int = 3,
        op_success: dict[str, float] | None = None,
        receptors: int = 0,
        brain: MosquitoBrain | None = None,
        pace: int = 1,
        colour_channels: int = 0,
        stabilizer: float = 0.0,
        prey_sense: int = 0,
        cones: int | None = None,
        kc: int = 0,
        kc_seed: int | None = None,
        learning_rate: float = 0.0,
        zoom: float = 0.0,
        metabolism: float = 1.0,
        host_pref: dict | None = None,
        awake_replay: int = 0,
        sleep_replay: int = 0,
        rem_share: float = 0.0,
        replay_backup: float = 0.0,
        dream_steps: int = 0,
        vigilance: float = 1.0,
        pump: float | None = None,
        aversive_rate: float = 0.0,
        receptor_slowness: float = 0.0,
        plant_sense: int = 0,
        imagery: int = 0,
        recall: int = 0,
        scenes: int = 1,
        sleep_set: int = 0,
        bore: float = 1.0,
        archetypes: int = 0,
        archetype_classes: list | None = None,
        mobilize: float = MOBILIZE_BELOW,
        store: float = STORE_ABOVE,
    ):
        self.trees = trees
        self.mutation_weights = mutation_weights
        self.meta_mutation_rate = meta_mutation_rate
        self.n_vars = n_vars
        # Its eye: receptors per side (fovea.py), 0 for a task without one.
        self.receptors = fovea.even_receptors(receptors) if receptors else 0
        # Its cones: the central cones x cones of its receptors (the rest are
        # rods); by default all of them, like the eye before rods and cones.
        self.cones = self.receptors if cones is None else int(np.clip(cones, 0, self.receptors))
        # Its mushroom body: Kenyon cells, their wiring, how fast it learns.
        self.kc = int(np.clip(kc, 0, MAX_KC))
        self.kc_seed = int(kc_seed) if kc_seed is not None else random.randrange(2 ** 31)
        self.learning_rate = float(np.clip(learning_rate, 0.0, LEARNING_MAX))
        # Its aversive compartment's learning rate (fishbowl/mushroom.py): born 0, off.
        self.aversive_rate = float(np.clip(aversive_rate, 0.0, LEARNING_MAX))
        # Its photoreceptors' speed (organism.py): extra integration time, in
        # reference frames (1/15 s). Born 0: as fast as the camera's frames.
        self.receptor_slowness = float(np.clip(receptor_slowness, 0.0, SLOWNESS_GUARD))
        # Its plant sense (0 none, 1 scent, 2 + a coarse direction), like its
        # prey sense; born 0. Without it, it can still learn where plants fed it.
        self.plant_sense = int(np.clip(plant_sense, 0, 2))
        # Imagery (mushroom.py prototypes): born off; costs its multiply-adds.
        self.imagery = int(np.clip(imagery, 0, 1))
        # Recall (pattern completion over its own episodes): born off; costs
        # a multiply-add per stored cell it compares.
        self.recall = int(np.clip(recall, 0, 1))
        # Scenes it can keep (a scene library; born 1: one scene, as before).
        self.scenes = int(np.clip(scenes, 1, MAX_SCENES))
        # Sleep programming (born 0: off): how many of its waking looks it keeps
        # to test its own tree edits against while it sleeps.
        self.sleep_set = int(np.clip(sleep_set, 0, MAX_SLEEP_SET))
        # Its proboscis's bore (state.py: flow ~ bore^4, upkeep ~ bore^2); born 1.
        self.bore = float(np.clip(bore, PUMP_GUARD[0], PUMP_GUARD[1]))
        # Archetype heads (mushroom.py; born 0) and which detector class each is taught by.
        self.archetypes = int(np.clip(archetypes, 0, 4))
        self.archetype_classes = ([int(c) for c in (archetype_classes or [])] + [0, 0, 0, 0])[:4]
        # Its fuel set points (state.py), inherited, born at the rulebook's
        # values: blood sugar below `mobilize` counts as fasted (glucagon),
        # above `store` a surplus goes to fat (insulin). 0 < mobilize < store < 1.
        self.mobilize, self.store = float(mobilize), float(store)
        if not 0.0 < self.mobilize < self.store < 1.0:
            self.mobilize, self.store = MOBILIZE_BELOW, STORE_ABOVE
        self.brain = brain if brain is not None else MosquitoBrain.random(random.Random(0))
        self.pace = int(pace)
        self.colour_channels = int(colour_channels)
        self.stabilizer = float(np.clip(stabilizer, 0.0, 1.0))
        # Its zoom lens's gain (fovea.magnification): 0 = its zoom output does nothing.
        self.zoom = float(np.clip(zoom, 0.0, 1.0))
        # Its metabolic strategy (state.py): 1 = endotherm (today's body), down to an ectotherm's 0.1.
        self.metabolism = float(np.clip(metabolism, MIN_METABOLISM, METABOLISM_GUARD))
        # Its host preference: how strongly its prey sense answers each living class
        # (an odorant receptor's tuning, McBride et al. 2014); mean 1, people never below 1.
        self.host_pref = _host_pref(host_pref)
        # Replay (fishbowl/organism.py): per quiet waking look, per sleeping look, and
        # the share of sleep replays that are REM (recombined) rather than NREM.
        self.awake_replay = int(np.clip(awake_replay, 0, MAX_REPLAYS))
        self.sleep_replay = int(np.clip(sleep_replay, 0, MAX_REPLAYS))
        self.rem_share = float(np.clip(rem_share, 0.0, 1.0))
        # Sequence replay (organism._replay): how much of the next moment's
        # value flows back along a replayed path. Born 0: replay as before.
        self.replay_backup = float(np.clip(replay_backup, 0.0, 1.0))
        # Closed-loop dreaming (organism._dream): brain steps per settled
        # sleeping look run on its own maps. Born 0: no dreams.
        self.dream_steps = int(np.clip(dream_steps, 0, MAX_REPLAYS))
        # Vigilance: how easily a change in the field wakes it (state.big_change).
        self.vigilance = float(np.clip(vigilance, MIN_VIGILANCE, MAX_VIGILANCE))
        # Its feeding pump (state.py): how fast blood flows in while a host is
        # at its mouth. Born where the old per-look meal fed it at its own
        # resting tempo, so no lineage jumps.
        self.pump = float(np.clip(pump if pump is not None else PUMP_REF / max(1, int(pace)),
                                  PUMP_GUARD[0] * PUMP_REF, PUMP_GUARD[1] * PUMP_REF))
        self.prey_sense = int(np.clip(prey_sense, 0, MAX_PREY_SENSE))
        # Per-operator EMA of how often ITS attempts get accepted --
        # the real evidence update_mutation_weights() nudges
        # mutation_weights toward. Defaults to a neutral 0.5 prior for
        # any operator with no track record yet (e.g. resuming an old
        # checkpoint that predates this field).
        self.op_success = dict(op_success) if op_success is not None else {name: 0.5 for name in TASK_OPS}

    @property
    def fovea_fraction(self) -> float:
        """Its gaze's side as a fraction of the frame's height (for the record)."""
        return fovea.extent(self.receptors)

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
            self.receptors,
            self.brain.clone(),
            self.pace,
            self.colour_channels,
            self.stabilizer,
            self.prey_sense,
            self.cones,
            self.kc,
            self.kc_seed,
            self.learning_rate,
            self.zoom,
            self.metabolism,
            dict(self.host_pref),
            self.awake_replay,
            self.sleep_replay,
            self.rem_share,
            self.replay_backup,
            self.dream_steps,
            self.vigilance,
            self.pump,
            self.aversive_rate,
            self.receptor_slowness,
            self.plant_sense,
            self.imagery,
            self.recall,
            self.scenes,
            self.sleep_set,
            self.bore,
            self.archetypes,
            list(self.archetype_classes),
            self.mobilize,
            self.store,
        )

    def evaluate(self, name: str, inputs: np.ndarray, retina: np.ndarray | None = None) -> np.ndarray:
        return self.trees[name].evaluate(inputs, retina)

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

    def _mutate_pool(self, rng: random.Random, channel: str) -> bool:
        """A receptor leaf becomes a pooled 3x3 patch around it; a pool widens
        or narrows by a ring (narrowed to nothing, it is one receptor again),
        or becomes oriented (an edge detector at a random angle); an oriented
        pool turns (by a constant's mutation step, 0.5 rad), widens or narrows,
        or loses its orientation."""
        leaves = [n for n in self._all_nodes(channel) if n.kind in ("cell", "pool", "edge")]
        if not leaves:
            return False
        node = rng.choice(leaves)
        ring = lambda: int(node.value) + rng.choice((-1, 1))  # noqa: E731
        cap = max(1, self.receptors // 2)
        if node.kind == "cell":
            node.kind, node.value = "pool", 1.0
        elif node.kind == "pool":
            if rng.random() < 0.5:
                node.kind, node.angle = "edge", rng.uniform(0.0, math.pi)
                node.value = float(max(1, int(node.value)))
            else:
                r = ring()
                if r <= 0:
                    node.kind, node.value = "cell", 0.0
                else:
                    node.value = float(min(r, cap))
        else:
            what = rng.choice(("turn", "ring", "unorient"))
            if what == "turn":
                node.angle = float((node.angle + rng.gauss(0.0, 0.5)) % math.pi)
            elif what == "ring":
                node.value = float(min(max(1, ring()), cap))
            else:
                node.kind = "pool"
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
        _become(target, _random_small_tree(rng, self.n_vars, max_depth=2, n_side=self.receptors))
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
            _become(node, replacement)
        return True

    def _reroll_subtree(self, rng: random.Random, channel: str) -> bool:
        """
        Real bug fix (external audit, 2026-09-24): candidates used to
        include the tree's own ROOT, so this could replace an ENTIRE
        tree -- however large it had grown -- with a fresh depth<=2
        stub in one single mutation. Verified empirically (20,000-
        iteration neutral-drift test, uniform operator weights): mean
        accepted tree size settled around ~11 nodes regardless of the
        size ceiling (60, 300, or 1000 -- raising it never helped,
        because it was never the constraint). This was the actual
        reason: whatever `_grow` accumulated a few nodes at a time,
        this could erase in one step, just as often. Excluding the
        root keeps this operator doing what its name says -- rerolling
        A subtree, not potentially the whole tree -- while still
        falling back to the root on a single-node tree (there's
        nothing else to target then).
        """
        nodes = self._all_nodes(channel)
        root = self.trees[channel]
        candidates = [n for n in nodes if n is not root] or nodes
        target = rng.choice(candidates)
        replacement = _random_small_tree(rng, self.n_vars, max_depth=2, n_side=self.receptors)
        if target is self.trees[channel]:
            self.trees[channel] = replacement
        else:
            _become(target, replacement)
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

        # The brain is guaranteed a real share of the search (audit: learned
        # operator weights had starved it to ~1% of mutations).
        if rng.random() < BRAIN_FLOOR:
            choice = "mutate_brain"
        else:
            names = list(self.mutation_weights.keys())
            probs = np.array([self.mutation_weights[n] for n in names], dtype=float)
            probs = probs / probs.sum()
            choice = rng.choices(names, weights=probs, k=1)[0]

        if choice == "mutate_fovea":
            old = self.receptors
            if old:
                self.receptors = fovea.even_receptors(old + rng.choice((-2, 2)))  # one ring
                self.cones = min(self.cones, self.receptors)
            return "fovea", (choice if self.receptors != old else "noop_inapplicable")
        if choice == "grow_kc":
            old = self.kc
            self.kc = min(MAX_KC, old + KC_STEP)
            return "brain", (choice if self.kc != old else "noop_inapplicable")
        if choice == "shrink_kc":
            old = self.kc
            self.kc = max(0, old - KC_STEP)
            return "brain", (choice if self.kc != old else "noop_inapplicable")
        if choice == "mutate_learning":
            old = self.learning_rate
            if old <= 0.0:
                self.learning_rate = float(LEARNING_MIN * (LEARNING_MAX / LEARNING_MIN) ** rng.random())
            else:
                self.learning_rate = float(np.clip(old * np.exp(rng.gauss(0.0, LEARNING_SIGMA)), LEARNING_MIN, LEARNING_MAX))
            return "brain", (choice if self.learning_rate != old else "noop_inapplicable")
        if choice == "mutate_imagery":
            self.imagery = 1 - self.imagery
            return "brain", choice
        if choice == "mutate_recall":
            self.recall = 1 - self.recall
            return "brain", choice
        if choice == "mutate_sleep_set":
            old = self.sleep_set  # doubling or halving: 0 <-> 16 <-> 32 ...
            self.sleep_set = (16 if rng.random() < 0.5 else 0) if old == 0 else \
                int(np.clip(old * 2 if rng.random() < 0.5 else (old // 2 if old > 16 else 0), 0, MAX_SLEEP_SET))
            return "brain", (choice if self.sleep_set != old else "noop_inapplicable")
        if choice == "mutate_scenes":
            old = self.scenes
            self.scenes = int(np.clip(old + rng.choice((-1, 1)), 1, MAX_SCENES))
            return "brain", (choice if self.scenes != old else "noop_inapplicable")
        if choice == "mutate_plant_sense":
            old = self.plant_sense
            self.plant_sense = int(np.clip(old + rng.choice((-1, 1)), 0, 2))
            return "prey", (choice if self.plant_sense != old else "noop_inapplicable")
        if choice == "mutate_archetypes":  # one more or fewer head, or a head taught by another class
            from .prey import PLANT_CLASS, PREY_CLASSES
            classes = sorted(PREY_CLASSES) + [PLANT_CLASS]
            if self.archetypes and rng.random() < 0.5:
                k = rng.randrange(self.archetypes)
                self.archetype_classes[k] = rng.choice(classes)
            else:
                old = self.archetypes
                self.archetypes = int(np.clip(old + rng.choice((-1, 1)), 0, 4))
                if self.archetypes > old:
                    self.archetype_classes[old] = rng.choice(classes)
                if self.archetypes == old:
                    return "brain", "noop_inapplicable"
            return "brain", choice
        if choice == "mutate_bore":  # log-normal steps, like the pump's
            self.bore = float(np.clip(self.bore * math.exp(rng.gauss(0.0, TRAIT_SIGMA)), PUMP_GUARD[0], PUMP_GUARD[1]))
            return "pump", choice
        if choice == "mutate_setpoints":
            # One operator steps both lines (Wagner: many small traits dilute the
            # search); a step that would cross them (or leave 0..1) doesn't apply.
            m = self.mobilize + rng.gauss(0.0, TRAIT_SIGMA)
            s = self.store + rng.gauss(0.0, TRAIT_SIGMA)
            if not 0.0 < m < s < 1.0:
                return "body", "noop_inapplicable"
            self.mobilize, self.store = float(m), float(s)
            return "body", choice
        if choice == "mutate_receptor_speed":
            # Steps scale with its size, so it can creep off 0 and still move once slow.
            old = self.receptor_slowness
            self.receptor_slowness = float(np.clip(old + rng.gauss(0.0, TRAIT_SIGMA * (1.0 + old)), 0.0, SLOWNESS_GUARD))
            return "fovea", (choice if self.receptor_slowness != old else "noop_inapplicable")
        if choice == "mutate_aversive":  # drawn and stepped as the reward learning rate is
            old = self.aversive_rate
            if old <= 0.0:
                self.aversive_rate = float(LEARNING_MIN * (LEARNING_MAX / LEARNING_MIN) ** rng.random())
            else:
                self.aversive_rate = float(np.clip(old * np.exp(rng.gauss(0.0, LEARNING_SIGMA)), LEARNING_MIN, LEARNING_MAX))
            return "brain", (choice if self.aversive_rate != old else "noop_inapplicable")
        if choice == "mutate_cones":
            old = self.cones
            self.cones = int(np.clip(old + rng.choice((-2, 2)), 0, self.receptors))  # one ring of cones
            return "fovea", (choice if self.cones != old else "noop_inapplicable")
        if choice == "mutate_pace":
            old = self.pace
            self.pace = int(np.clip(old + rng.choice((-1, 1)), MIN_PACE, MAX_PACE))
            return "pace", (choice if self.pace != old else "noop_inapplicable")
        if choice == "mutate_colour":
            old = self.colour_channels
            self.colour_channels = int(np.clip(old + rng.choice((-1, 1)), 0, MAX_COLOUR_CHANNELS))
            return "colour", (choice if self.colour_channels != old else "noop_inapplicable")
        if choice == "mutate_prey_sense":
            old = self.prey_sense
            self.prey_sense = int(np.clip(old + rng.choice((-1, 1)), 0, MAX_PREY_SENSE))
            return "prey_sense", (choice if self.prey_sense != old else "noop_inapplicable")
        if choice == "mutate_stabilizer":
            old = self.stabilizer
            self.stabilizer = float(np.clip(old + rng.gauss(0.0, STABILIZER_SIGMA), 0.0, 1.0))
            return "stabilizer", (choice if self.stabilizer != old else "noop_inapplicable")
        if choice == "mutate_metabolism":
            old = self.metabolism
            self.metabolism = float(np.clip(old + rng.gauss(0.0, TRAIT_SIGMA), MIN_METABOLISM, METABOLISM_GUARD))
            return "metabolism", (choice if self.metabolism != old else "noop_inapplicable")
        if choice == "mutate_host":
            old = dict(self.host_pref)
            w = dict(self.host_pref)
            c = rng.choice(sorted(w))
            w[c] *= math.exp(rng.gauss(0.0, TRAIT_SIGMA))
            self.host_pref = _host_pref(w)
            return "host", (choice if self.host_pref != old else "noop_inapplicable")
        if choice == "mutate_replay":
            old = (self.awake_replay, self.sleep_replay, self.rem_share, self.replay_backup, self.dream_steps)
            which = rng.randrange(5)
            if which == 0:
                self.awake_replay = int(np.clip(self.awake_replay + rng.choice((-1, 1)), 0, MAX_REPLAYS))
            elif which == 1:
                self.sleep_replay = int(np.clip(self.sleep_replay + rng.choice((-1, 1)), 0, MAX_REPLAYS))
            elif which == 2:
                self.rem_share = float(np.clip(self.rem_share + rng.gauss(0.0, TRAIT_SIGMA), 0.0, 1.0))
            elif which == 3:
                self.replay_backup = float(np.clip(self.replay_backup + rng.gauss(0.0, TRAIT_SIGMA), 0.0, 1.0))
            else:
                self.dream_steps = int(np.clip(self.dream_steps + rng.choice((-1, 1)), 0, MAX_REPLAYS))
            new = (self.awake_replay, self.sleep_replay, self.rem_share, self.replay_backup, self.dream_steps)
            return "replay", (choice if new != old else "noop_inapplicable")
        if choice == "mutate_pump":
            old = self.pump
            self.pump = float(np.clip(old * math.exp(rng.gauss(0.0, TRAIT_SIGMA)), PUMP_GUARD[0] * PUMP_REF, PUMP_GUARD[1] * PUMP_REF))
            return "pump", (choice if self.pump != old else "noop_inapplicable")
        if choice == "mutate_vigilance":
            old = self.vigilance
            self.vigilance = float(np.clip(old * math.exp(rng.gauss(0.0, TRAIT_SIGMA)), MIN_VIGILANCE, MAX_VIGILANCE))
            return "vigilance", (choice if self.vigilance != old else "noop_inapplicable")
        if choice == "mutate_zoom":
            old = self.zoom
            self.zoom = float(np.clip(old + rng.gauss(0.0, ZOOM_SIGMA), 0.0, 1.0))
            return "zoom", (choice if self.zoom != old else "noop_inapplicable")
        if choice == "mutate_brain":
            return "brain", (choice if self.brain.mutate(rng) > 0 else "noop_inapplicable")
        if choice == "grow_channel":
            return "brain", (choice if self.brain.grow_channel(rng, "latch") else "noop_inapplicable")
        if choice == "add_prediction":
            return "brain", (choice if self.brain.grow_channel(rng, "predict") else "noop_inapplicable")
        if choice == "grow_unit":
            return "brain", (choice if self.brain.grow_unit(rng) else "noop_inapplicable")
        if choice == "shrink_unit":
            return "brain", (choice if self.brain.shrink_unit(rng) else "noop_inapplicable")
        if choice == "duplicate_layer":
            return "brain", (choice if self.brain.duplicate_layer(rng) else "noop_inapplicable")
        if choice == "remove_layer":
            return "brain", (choice if self.brain.remove_layer(rng) else "noop_inapplicable")
        if choice == "shrink_channel":
            return "brain", (choice if self.brain.shrink_channel(rng) else "noop_inapplicable")
        if choice == "grow":
            applied, hit_ceiling = self._grow(rng, channel, max_nodes, max_depth)
        else:
            applied = {
                "mutate_const": lambda: self._mutate_const(rng, channel),
                "mutate_op": lambda: self._mutate_op(rng, channel),
                "shrink": lambda: self._shrink(rng, channel),
                "reroll_subtree": lambda: self._reroll_subtree(rng, channel),
                "mutate_pool": lambda: self._mutate_pool(rng, channel),
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
            "receptors": self.receptors,
            "cones": self.cones,
            "kc": self.kc,
            "kc_seed": self.kc_seed,
            "learning_rate": self.learning_rate,
            "brain": self.brain.to_dict(),
            "pace": self.pace,
            "colour_channels": self.colour_channels,
            "stabilizer": self.stabilizer,
            "prey_sense": self.prey_sense,
            "zoom": self.zoom,
            "metabolism": self.metabolism,
            "host_pref": {str(k): round(v, 4) for k, v in self.host_pref.items()},
            "awake_replay": self.awake_replay,
            "sleep_replay": self.sleep_replay,
            "rem_share": self.rem_share,
            "replay_backup": self.replay_backup,
            "dream_steps": self.dream_steps,
            "vigilance": self.vigilance,
            "pump": self.pump,
            "aversive_rate": self.aversive_rate,
            "receptor_slowness": self.receptor_slowness,
            "plant_sense": self.plant_sense,
            "imagery": self.imagery,
            "recall": self.recall,
            "scenes": self.scenes,
            "sleep_set": self.sleep_set,
            "bore": self.bore,
            "archetypes": self.archetypes,
            "archetype_classes": list(self.archetype_classes),
            "mobilize": self.mobilize,
            "store": self.store,
        }

    @staticmethod
    def from_dict(data: dict) -> "Genome":
        # Old checkpoints predate newer operators (e.g. mutate_fovea):
        # give any missing one a fair uniform share and the neutral 0.5
        # success prior, then renormalize -- evolved trees and learned
        # weights for existing operators are kept as-is.
        weights = dict(data["mutation_weights"])
        op_success = dict(data.get("op_success") or {})
        for name in TASK_OPS:
            weights.setdefault(name, 1.0 / len(TASK_OPS))
            op_success.setdefault(name, 0.5)
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}
        # Pre-brain checkpoints carried pan/tilt trees; motor control now
        # lives in the brain, so only the perception channels are kept.
        trees = {name: blocks.Node.from_dict(t) for name, t in data["trees"].items() if name in DEFAULT_CHANNELS}
        brain = MosquitoBrain.from_dict(data["brain"]) if data.get("brain") else MosquitoBrain.random(random.Random(0))
        n_vars = int(data.get("n_vars", 3))
        if "receptors" in data:
            receptors = int(data["receptors"])
        else:
            # From the zoom eye (a 12x12 grid stretched over a gaze of
            # fovea_fraction of the frame; tree inputs a flat vector): the eye
            # keeps its extent (so its upkeep is unchanged), and each receptor
            # a tree read becomes a read of the same spot relative to the
            # gaze's centre, rescaled to the new receptor size.
            frac = float(data.get("fovea_fraction", 0.35))
            receptors = fovea.even_receptors(frac / fovea.RECEPTOR_PITCH)
            if n_vars > 2 * _OLD_CELLS + 2:
                scale = (frac / _OLD_SIDE) / fovea.RECEPTOR_PITCH
                for tree in trees.values():
                    _migrate_flat_inputs(tree, n_vars, scale)
                n_vars -= 4 * _OLD_CELLS
        return Genome(
            trees=trees,
            mutation_weights=weights,
            meta_mutation_rate=float(data["meta_mutation_rate"]),
            n_vars=n_vars,
            op_success=op_success,
            receptors=receptors,
            brain=brain,
            pace=int(np.clip(data.get("pace", 1), MIN_PACE, MAX_PACE)),
            colour_channels=int(np.clip(data.get("colour_channels", 0), 0, MAX_COLOUR_CHANNELS)),
            stabilizer=float(np.clip(data.get("stabilizer", 0.0), 0.0, 1.0)),
            prey_sense=int(np.clip(data.get("prey_sense", 0), 0, MAX_PREY_SENSE)),
            cones=data.get("cones"),
            kc=int(data.get("kc", 0)),
            kc_seed=data.get("kc_seed"),
            learning_rate=float(data.get("learning_rate", 0.0)),
            zoom=float(np.clip(data.get("zoom", 0.0), 0.0, 1.0)),
            metabolism=float(data.get("metabolism", 1.0)),
            host_pref={int(k): float(v) for k, v in (data.get("host_pref") or {}).items()},
            awake_replay=int(data.get("awake_replay", 0)),
            sleep_replay=int(data.get("sleep_replay", 0)),
            replay_backup=float(data.get("replay_backup", 0.0)),
            dream_steps=int(data.get("dream_steps", 0)),
            rem_share=float(data.get("rem_share", 0.0)),
            vigilance=float(data.get("vigilance", 1.0)),
            pump=data.get("pump"),
            aversive_rate=float(data.get("aversive_rate", 0.0)),
            receptor_slowness=float(data.get("receptor_slowness", 0.0)),
            plant_sense=int(data.get("plant_sense", 0)),
            imagery=int(data.get("imagery", 0)),
            recall=int(data.get("recall", 0)),
            scenes=int(data.get("scenes", 1)),
            sleep_set=int(data.get("sleep_set", 0)),
            bore=float(data.get("bore", 1.0)),
            archetypes=int(data.get("archetypes", 0)),
            archetype_classes=data.get("archetype_classes"),
            mobilize=float(data.get("mobilize", MOBILIZE_BELOW)),
            store=float(data.get("store", STORE_ABOVE)),
        )


PERSON = 0  # COCO's person class: the prey every lineage must be able to track


def _host_pref(w: dict | None) -> dict:
    """A host preference over the living classes (prey.PREY_CLASSES): mean 1
    (tuning redistributes attention, it doesn't add any), people never below
    the even weight 1, so every lineage can track people (the clade's rule).
    Missing classes start even."""
    classes = sorted(PREY_CLASSES)
    w = {c: max(1e-3, float((w or {}).get(c, 1.0))) for c in classes}
    mean = sum(w.values()) / len(w)
    w = {c: v / mean for c, v in w.items()}
    if w[PERSON] < 1.0:
        others = [c for c in classes if c != PERSON]
        rest = sum(w[c] for c in others)
        scale = (len(classes) - 1.0) / rest if rest > 0 else 1.0
        w = {c: (1.0 if c == PERSON else w[c] * scale) for c in classes}
    return {c: round(v, 6) for c, v in w.items()}


def _migrate_flat_inputs(node: blocks.Node, old_n_vars: int, scale: float) -> None:
    """Old flat tree inputs -> plain inputs and receptors read by position
    (see Genome.from_dict)."""
    if node.kind == "var":
        i, colour_start = node.index, old_n_vars - 2 * _OLD_CELLS
        plane = None
        if i < 2 * _OLD_CELLS:
            plane, cell = divmod(i, _OLD_CELLS)
        elif i >= colour_start:
            plane, cell = divmod(i - colour_start, _OLD_CELLS)
            plane += 2
        if plane is None:
            node.index = i - 2 * _OLD_CELLS  # movement and the brain's units, now first
        else:
            row, col = divmod(cell, _OLD_SIDE)
            node.kind, node.index = "cell", plane
            node.kx = int(np.floor((col - _OLD_SIDE / 2 + 0.5) * scale))
            node.ky = int(np.floor((row - _OLD_SIDE / 2 + 0.5) * scale))
    for child in node.children:
        _migrate_flat_inputs(child, old_n_vars, scale)
