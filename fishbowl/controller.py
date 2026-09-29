from __future__ import annotations

"""
Visual controller -- a small recurrent network that moves the gaze.

Modeled on insect neurobiology (e.g. Diptera / mosquito):
  - Optic Lobe Inputs:
      the gaze's luminance, motion and optic flow (dx, dy); loom and where
      motion is (dx, dy) from the whole field; current gaze (cx, cy, eye size) and
      eye velocity; interoception (blood sugar, arousal, threat, search,
      hunger, curiosity, gut, reserve, sleep pressure, asleep); the whole
      field's light and its trend (day and night); and the perception tree's
      output.
  - Central Complex (Recurrent Ring):
      A 16-unit recurrent neural circuit (RNN) with temporal hidden memory.
      Maintains smooth gaze stabilization, pursuit, and active visual casting.
  - Outputs: pan, tilt, zoom (in only, through its inherited lens gain -- fovea.py), alarm, tempo, sleep. There is no hard-wired
    escape reflex: a fast reaction to looming is rewarded instead
    (run_vision.py), so the flinch has to evolve. Sleep is its own choice
    (the body adds only physiological overrides -- state.py).

Growable channels (after a design panel on how new outputs and feedback
loops evolve -- e.g. vocal circuits thought to arise from duplicated motor
circuits, and birdsong learning's comparison of what it hears with what it
meant to sing). A channel is an extra output wired back as an extra input
on the next step:
  - latch: the output itself comes back -- a loop it can use as working
    memory, or a signal to itself.
  - predict: the output predicts one of its own inputs; what comes back is
    the error (actual - predicted), a comparator.
New channels are born by duplication (a copy of an existing output's
weights) or as a blank predictor, always with zero weights on the way back
in, so a newborn channel changes nothing: it can drift until it is useful.
Its energy price grows with those feedback weights (run_vision.py): a loop
that does nothing is free, one that matters has to pay for itself.

Growable hidden layer (Stage B's neural budget, after a design panel with
Sterling and Laughlin's principles of neural design in mind: a neuron costs
its wiring and activity, not its existence). A mutation duplicates a hidden
unit, the way genes duplicate: the copy listens like the original but is
born unconnected (nothing reads it), so it changes nothing and is kept on a
tie, free to diverge; another mutation removes one. Thinking is priced by
the brain's real arithmetic (multiply-adds per step, relative to the
original 16-unit brain) x CPU scarcity, so a host with cores to spare can
afford a big brain and a starved one keeps a lean one. The perception tree
reads the first TREE_HIDDEN units.

Stacked layers by duplication (after a deep-learning panel -- Net2Net's
function-preserving "deeper net", Chen, Goodfellow & Shlens 2016, in its
residual form; and the vertebrate genome's two rounds of whole-genome
duplication, Ohno 1970, most duplicates later lost, the survivors taking on
new or divided jobs, Force et al. 1999). A mutation stacks a copy of the top
layer on top of the brain:

    h_k = h_(k-1) + gate_k * tanh(b_k + W_k h_(k-1) + U_k h_k(previous step))

born with gate 0 -- so it changes nothing, and is kept on a tie -- and
silent layers are not computed and cost nothing (a duplicated gene that is
not expressed); a mutation that opens the gate switches it on, from then on
priced by its arithmetic like the rest of the brain. The motor outputs read
the top layer; the perception tree reads the first layer. Another mutation
removes a stacked layer. All layers have the same units (the skip needs
it), so growing or removing a unit does it in every layer. No depth is
chosen by hand: how deep it gets is what pays.

Runs on numpy (a matrix step, 10-20x faster than the old Python loops).
"""

import math
import random
from typing import Any, NamedTuple

import numpy as np

from .state import MosquitoState

BASE_INPUTS = 68  # ... + terrain at its gaze + how near what it looks at is + how near it feels it is + how soon it will reach it + turning, tilting, heading sin/cos; 19 + gut, reserve, sleep pressure, asleep, field light, light trend + prey scent, prey dir x/y + food value + place dx/dy/value + intruder + danger + plant scent, dir x/y + mismatch, dir x/y + recalled value, dir x/y + protein + host velocity x/y + own pace, missed + uncertainty + ground near, horizon + parallax, camera moving + 4 archetypes + collicular dx, dy, strength
PREY_INPUTS = (25, 26, 27)  # scent, direction x, direction y (run_vision.py's prey sense)
# Its place map (fishbowl/organism.py): the direction from its gaze to the
# spot that has fed it best, and how good that spot was; and the intruder
# sense -- a person where its own experience says people don't appear.
PLACE_INPUTS = (29, 30, 31)
INTRUDER_INPUT = 32
DANGER_INPUT = 33  # its aversive compartment's learned danger of what it sees (fishbowl/mushroom.py)
PLANT_INPUTS = (34, 35, 36)  # its plant sense (genome.plant_sense): scent, then a coarse direction
SPEED_INPUTS = (46, 47)       # its own pace (seconds per look) and whether it just missed a look (interoception)
UNCERTAINTY_INPUT = 48        # how wrong its mushroom body's predictions have been lately (Friston's precision)
GROUND_INPUTS = (49, 50)      # how near the ground at its gaze is, and where the horizon is (a learned ground plane)
PARALLAX_INPUTS = (51, 52)    # parallax at its gaze and how much the camera itself moves (moving streams)
ARCHETYPE_INPUTS = (53, 54, 55, 56)
COLLICULUS_INPUTS = (57, 58, 59)  # where its collicular priority map's winner is, and how strong (organism.py)  # its archetype heads: mushroom-body readouts taught by the detector's classes
TURN_INPUT, TILT_INPUT = 64, 65  # its vestibular sense: yaw and roll per look, in its half field of view (organism.py)
HEADING_INPUTS = (66, 67)  # its compass (head direction): sin, cos of its heading
CONTACT_INPUT = 63  # how soon what it looks at will reach it (Lee's tau, in its own looks; organism.py)
FELT_NEARNESS_INPUT = 62  # its terrain head's own estimate of nearness, from its eye (organism.py): a sense that needs no teacher
NEARNESS_INPUT = 61  # how near what it looks at is (relative: from its feet on its ground; organism.py)
TERRAIN_INPUT = 60  # the ground's rise or fall at its gaze, from its terrain map (organism.py), in camera heights
PROTEIN_INPUT = 43  # its protein store (for eggs; only blood fills it)
VELOCITY_INPUTS = (44, 45)  # the followed host's velocity (prey sense level 3)
RECALL_INPUTS = (40, 41, 42)  # what the memory its view recalls held, and where it happened (pattern completion)
MISMATCH_INPUTS = (37, 38, 39)  # how much of its field differs from its slow model of the room, and where (orienting)
INPUTS = BASE_INPUTS  # kept for older callers: the base inputs
HIDDEN = 16         # a newborn brain's hidden layer
TREE_HIDDEN = 16    # how many hidden units the perception tree reads (its inputs keep fixed positions)
MIN_HIDDEN, MAX_HIDDEN = 4, 256  # MAX is a safety bound only; the price is what limits growth
MAX_LAYERS = 16  # stacked layers: a safety bound only, like MAX_HIDDEN
BASE_OUTPUTS = 6  # [pan, tilt, zoom, alarm, tempo, sleep]
OUTPUTS = BASE_OUTPUTS
MAX_CHANNELS = 4
# Mutation steps are heavy-tailed, like real mutations' effects (mostly tiny,
# occasionally large -- the distribution of fitness effects): 1 nudge in 10 is
# Cauchy-sized. Gaussian steps alone never jump, so a brain stuck behind a
# threshold -- e.g. an eye pinned at an edge by a saturated push, where every
# small nudge changes nothing -- could not get off the plateau.
HEAVY_TAIL_P = 0.1
HEAVY_TAIL_SCALE = 0.1
HEAVY_TAIL_MAX = 2.0
INPUT_NAMES = ("light", "motion", "flow x", "flow y", "loom", "gaze x", "gaze y", "eye size", "blood sugar",
               "arousal", "threat", "search", "motion dx", "motion dy", "eye vx", "eye vy", "hunger",
               "curiosity", "tree", "gut", "reserve", "sleep pressure", "asleep", "field light", "light trend",
               "prey scent", "prey dir x", "prey dir y", "food value", "place dx", "place dy", "place value",
               "intruder", "danger", "plant scent", "plant dir x", "plant dir y", "mismatch", "mismatch dx", "mismatch dy",
               "recalled value", "recalled dx", "recalled dy", "protein", "host vx", "host vy",
               "own pace", "missed", "uncertainty", "ground near", "horizon", "parallax", "camera moving",
               "archetype 1", "archetype 2", "archetype 3", "archetype 4",
               "collicular dx", "collicular dy", "collicular strength")
OUTPUT_NAMES = ("pan", "tilt", "zoom", "alarm", "tempo", "sleep")


class Motor(NamedTuple):
    pan: float
    tilt: float
    zoom: float
    alarm: float
    tempo: float
    sleep: float


def _macs(n_hidden: int, n_in: int, n_out: int) -> int:
    """Multiply-adds in one brain step."""
    return n_hidden * (n_in + n_hidden) + n_out * n_hidden


REFERENCE_MACS = _macs(HIDDEN, BASE_INPUTS, BASE_OUTPUTS)  # the original brain: thinking costs its THINK_COST


class MosquitoBrain:
    def __init__(self, weights_ih, weights_hh, weights_ho, bias_h, bias_o, channels: list[dict] | None = None,
                 layers: list[dict] | None = None, apical: float = 0.0):
        self.weights_ih = np.array(weights_ih, dtype=float)  # (hidden, inputs)
        self.weights_hh = np.array(weights_hh, dtype=float)  # (hidden, hidden)
        self.weights_ho = np.array(weights_ho, dtype=float)  # (outputs, hidden)
        self.bias_h = np.array(bias_h, dtype=float)
        self.bias_o = np.array(bias_o, dtype=float)
        self.channels = [dict(c) for c in (channels or [])]
        # Stacked layers (see the module docstring): W from the layer below,
        # U its own recurrence, b, and its gate (a 1-element array, so a
        # mutation can nudge it like any weight).
        self.layers = [{"W": np.array(l["W"], dtype=float), "U": np.array(l["U"], dtype=float),
                        "b": np.array(l["b"], dtype=float), "gate": np.array(l["gate"], dtype=float).reshape(1)}
                       for l in (layers or [])]
        # Pyramidal units (a 2026-09-29 panel; Larkum 2013): each hidden unit
        # has a basal compartment (what the senses say, W_ih x) and an apical
        # one (context: its own recurrent state, W_hh h). With apical gain a,
        # h = tanh(b + basal + context + a * basal * tanh(context)): a unit
        # fires far more when evidence and context agree (coincidence
        # detection). Born 0 -- the old point neuron -- and evolvable.
        self.apical = float(apical)
        self.last_top = None     # what fed its outputs last step, and what they were (sleep distillation)
        self.last_outputs = None
        # Hidden units still alive in a wasting body (fishbowl/organism.py); None = all.
        self.live_units: int | None = None
        self.reset_hidden()

    @classmethod
    def random(cls, rng: random.Random) -> MosquitoBrain:
        def matrix(rows: int, cols: int, scale: float = 0.4) -> list[list[float]]:
            return [[rng.uniform(-scale, scale) for _ in range(cols)] for _ in range(rows)]

        return cls(
            weights_ih=matrix(HIDDEN, BASE_INPUTS, scale=0.4),
            weights_hh=matrix(HIDDEN, HIDDEN, scale=0.3),
            weights_ho=matrix(BASE_OUTPUTS, HIDDEN, scale=0.4),
            bias_h=[rng.uniform(-0.05, 0.05) for _ in range(HIDDEN)],
            bias_o=[rng.uniform(-0.05, 0.05) for _ in range(BASE_OUTPUTS)],
        )

    @property
    def n_hidden(self) -> int:
        return len(self.bias_h)

    def reset_hidden(self) -> None:
        self.hidden = np.zeros(self.n_hidden)
        self.layer_hidden = [np.zeros(self.n_hidden) for _ in self.layers]
        self.loop_in = np.zeros(len(self.channels))  # what each channel feeds back this step
        self._pred = np.zeros(len(self.channels))    # each predictor's last prediction

    def tree_view(self) -> np.ndarray:
        """The hidden units the perception tree reads (the first TREE_HIDDEN, zero-padded)."""
        h = self.hidden[:TREE_HIDDEN]
        return h if len(h) == TREE_HIDDEN else np.concatenate([h, np.zeros(TREE_HIDDEN - len(h))])

    def step(
        self,
        luminance: float,
        motion: float,
        flow_x: float,
        flow_y: float,
        loom: float,
        gaze_cx: float,
        gaze_cy: float,
        gaze_zoom: float,
        state: MosquitoState,
        periph_dx: float = 0.0,
        periph_dy: float = 0.0,
        eye_vx: float = 0.0,
        eye_vy: float = 0.0,
        tree_out: float = 0.0,
        field_light: float = 0.5,
        prey_scent: float = 0.0,
        prey_dx: float = 0.0,
        prey_dy: float = 0.0,
        food_value: float = 0.0,
        place_dx: float = 0.0,
        place_dy: float = 0.0,
        place_value: float = 0.0,
        intruder: float = 0.0,
        danger: float = 0.0,
        plant_scent: float = 0.0,
        plant_dx: float = 0.0,
        plant_dy: float = 0.0,
        mismatch: float = 0.0,
        mismatch_dx: float = 0.0,
        mismatch_dy: float = 0.0,
        recalled: float = 0.0,
        recalled_dx: float = 0.0,
        recalled_dy: float = 0.0,
        host_vx: float = 0.0,
        host_vy: float = 0.0,
        own_pace: float = 0.0,
        missed: float = 0.0,
        uncertainty: float = 0.0,
        ground_near: float = 0.0,
        horizon: float = 0.0,
        parallax: float = 0.0,
        camera_moving: float = 0.0,
        archetypes=(0.0, 0.0, 0.0, 0.0),
        colliculus=(0.0, 0.0, 0.0),
        terrain=0.0,
        nearness=0.0,
        felt_nearness=0.0,
        contact=0.0,
        turning=0.0,
        tilting=0.0,
        heading=(0.0, 0.0),
    ) -> Motor:
        """Runs one tick of the brain. Returns its motor outputs (Motor)."""
        base = np.array([
            luminance, motion, flow_x, flow_y, loom,
            gaze_cx - 0.5, gaze_cy - 0.5, gaze_zoom,
            state.energy, state.arousal, state.threat, state.search,
            periph_dx, periph_dy,
            eye_vx * 2.5, eye_vy * 2.5,  # terminal eye speed 0.4 -> ~1
            state.hunger, state.curiosity, tree_out,
            state.gut, state.stores, state.sleep_pressure, state.asleep,  # "reserve": glycogen + fat, on the old 6 h scale
            field_light, state.light_trend,
            prey_scent, prey_dx, prey_dy,
            food_value,  # its mushroom body's learned value of what it sees (fishbowl/mushroom.py)
            place_dx, place_dy, place_value,  # its place map: where food has been (fishbowl/organism.py)
            intruder,  # a person where its own experience says people don't appear
            danger,  # what its aversive compartment has learned comes before a swat
            plant_scent, plant_dx, plant_dy,  # its plant sense (nectar)
            mismatch, mismatch_dx, mismatch_dy,  # its field vs its slow model of the room, and where (orienting)
            recalled, recalled_dx, recalled_dy,  # the episode its view recalls: what it held, where it was
            state.protein, host_vx, host_vy,  # protein for eggs; the followed host's velocity (prey sense 3)
            own_pace, missed,  # its own speed (2026-09-29: "it needs the neurons to")
            uncertainty,  # how unsure its memory's predictions are
            ground_near, horizon,  # its learned ground plane
            parallax, camera_moving,  # depth from a moving camera
            *archetypes,  # its archetype heads
            *colliculus,  # its collicular priority map's winner
            terrain,  # its terrain map at its gaze
            nearness,  # how near what it looks at is
            felt_nearness,  # how near it feels it is (its own terrain head)
            contact,  # how soon what it looks at will reach it
            turning, tilting,  # its vestibular sense
            *heading,  # its compass
        ], dtype=float)
        # Predictors: what comes back is how wrong last step's prediction was.
        for k, ch in enumerate(self.channels):
            if ch["kind"] == "predict":
                self.loop_in[k] = max(-1.0, min(1.0, base[ch["target"]] - self._pred[k]))
        x = np.concatenate([base, self.loop_in]) if self.channels else base
        # Recurrent hidden update h = tanh(W_ih x + W_hh h_prev + b_h); motor readout o = tanh(W_ho h + b_o).
        basal, context = self.weights_ih @ x, self.weights_hh @ self.hidden
        if self.apical:
            self.hidden = np.tanh(self.bias_h + basal + context + self.apical * basal * np.tanh(context))
        else:
            self.hidden = np.tanh(self.bias_h + basal + context)
        live = self.live_units
        if live is not None and live < self.n_hidden:  # units lost to wasting are silent
            self.hidden[live:] = 0.0
        top = self.hidden
        for k, layer in enumerate(self.layers):
            g = float(layer["gate"][0])
            if g != 0.0:  # a silent layer is not computed (and costs nothing)
                self.layer_hidden[k] = np.tanh(layer["b"] + layer["W"] @ top + layer["U"] @ self.layer_hidden[k])
                if live is not None and live < self.n_hidden:
                    self.layer_hidden[k][live:] = 0.0
                top = top + g * self.layer_hidden[k]
        outputs = np.tanh(self.bias_o + self.weights_ho @ top)
        self.last_top, self.last_outputs = top, outputs
        for k, ch in enumerate(self.channels):
            o = outputs[BASE_OUTPUTS + k]
            if ch["kind"] == "predict":
                self._pred[k] = o
            else:
                self.loop_in[k] = o
        return Motor(*(float(v) for v in outputs[:BASE_OUTPUTS]))

    # ---- what it costs to think -------------------------------------------------
    def think_factor(self) -> float:
        """This brain's arithmetic per step relative to the original 16-unit
        brain (a stacked layer counts only while its gate is open; units lost
        to wasting cost nothing)."""
        h = self.n_hidden if self.live_units is None else max(0, min(self.n_hidden, self.live_units))
        stacked = sum(2 * h * h + h for l in self.layers if float(l["gate"][0]) != 0.0)
        apical = 2 * h if self.apical else 0  # the coincidence term: two more multiplies a unit
        return (_macs(h, self.weights_ih.shape[1], self.weights_ho.shape[0]) + stacked + apical) / REFERENCE_MACS

    def macs(self) -> int:
        """Multiply-adds in one step of this brain (open stacked layers included)."""
        return int(round(self.think_factor() * REFERENCE_MACS))

    def active_layers(self) -> int:
        return sum(1 for l in self.layers if float(l["gate"][0]) != 0.0)

    def loop_synapses(self) -> float:
        """Total weight on the channels' way back in (their energy price)."""
        return float(np.abs(self.weights_ih[:, BASE_INPUTS:]).sum())

    def sense_synapses(self, inputs: tuple) -> float:
        """Weight on a sense's inputs (its synaptic price, like the prey sense's)."""
        return float(np.abs(self.weights_ih[:, list(inputs)]).sum())

    def prey_synapses(self, level: int) -> float:
        """Weight on the prey-sense inputs it has (level 1: scent; 2: + direction)."""
        live = list(PREY_INPUTS[:1] if level == 1 else PREY_INPUTS if level >= 2 else ()) + (list(VELOCITY_INPUTS) if level >= 3 else [])
        return float(np.abs(self.weights_ih[:, live]).sum()) if live else 0.0

    # ---- growable hidden layer ----------------------------------------------------
    def grow_unit(self, rng: random.Random) -> bool:
        """A duplicated hidden unit, born unconnected (nothing reads it yet)."""
        h = self.n_hidden
        if h >= MAX_HIDDEN:
            return False
        src = rng.randrange(h)
        self.weights_ih = np.vstack([self.weights_ih, self.weights_ih[src]])
        listens = np.append(self.weights_hh[src], self.weights_hh[src, src])  # as the original, incl. to itself
        self.weights_hh = np.vstack([np.hstack([self.weights_hh, np.zeros((h, 1))]), listens])
        self.weights_ho = np.hstack([self.weights_ho, np.zeros((self.weights_ho.shape[0], 1))])
        self.bias_h = np.append(self.bias_h, self.bias_h[src])
        for l in self.layers:  # the same unit in every stacked layer: listens like its original, unread
            for key in ("W", "U"):
                m = l[key]
                l[key] = np.vstack([np.hstack([m, np.zeros((h, 1))]), np.append(m[src], m[src, src] if key == "U" else 0.0)])
            l["b"] = np.append(l["b"], l["b"][src])
        self.reset_hidden()
        return True

    def shrink_unit(self, rng: random.Random) -> bool:
        h = self.n_hidden
        if h <= MIN_HIDDEN:
            return False
        k = rng.randrange(h)
        self.weights_ih = np.delete(self.weights_ih, k, axis=0)
        self.weights_hh = np.delete(np.delete(self.weights_hh, k, axis=0), k, axis=1)
        self.weights_ho = np.delete(self.weights_ho, k, axis=1)
        self.bias_h = np.delete(self.bias_h, k)
        for l in self.layers:
            l["W"] = np.delete(np.delete(l["W"], k, axis=0), k, axis=1)
            l["U"] = np.delete(np.delete(l["U"], k, axis=0), k, axis=1)
            l["b"] = np.delete(l["b"], k)
        self.reset_hidden()
        return True

    # ---- stacked layers ------------------------------------------------------------
    def duplicate_layer(self, rng: random.Random) -> bool:
        """Stacks a copy of the top expressed layer on top, silent (gate 0):
        its W and U copy that layer's recurrence, its bias that layer's. A
        silent copy doesn't mutate (only its gate does), so while one waits a
        second would be the identical copy again: refused."""
        if len(self.layers) >= MAX_LAYERS or any(float(l["gate"][0]) == 0.0 for l in self.layers):
            return False
        src = self.layers[-1] if self.layers else {"U": self.weights_hh, "b": self.bias_h}
        self.layers.append({"W": src["U"].copy(), "U": src["U"].copy(), "b": src["b"].copy(), "gate": np.zeros(1)})
        self.reset_hidden()
        return True

    def remove_layer(self, rng: random.Random) -> bool:
        if not self.layers:
            return False
        del self.layers[rng.randrange(len(self.layers))]
        self.reset_hidden()
        return True

    # ---- growable channels ---------------------------------------------------------
    def grow_channel(self, rng: random.Random, kind: str = "latch") -> bool:
        """A new channel: a latch duplicated from an existing output, or a
        blank predictor of one of its inputs. Zero weights on the way back
        in, so nothing changes until evolution wires it up."""
        if len(self.channels) >= MAX_CHANNELS:
            return False
        if kind == "predict":
            row, bias = np.zeros(self.n_hidden), 0.0
            ch = {"kind": "predict", "target": rng.randrange(BASE_INPUTS)}
        else:
            src = rng.randrange(self.weights_ho.shape[0])
            row, bias = self.weights_ho[src].copy(), float(self.bias_o[src])
            ch = {"kind": "latch", "copy_of": src}
        self.weights_ho = np.vstack([self.weights_ho, row])
        self.bias_o = np.append(self.bias_o, bias)
        self.weights_ih = np.hstack([self.weights_ih, np.zeros((self.n_hidden, 1))])
        self.channels.append(ch)
        self.reset_hidden()
        return True

    def shrink_channel(self, rng: random.Random) -> bool:
        if not self.channels:
            return False
        k = rng.randrange(len(self.channels))
        self.weights_ho = np.delete(self.weights_ho, BASE_OUTPUTS + k, axis=0)
        self.bias_o = np.delete(self.bias_o, BASE_OUTPUTS + k)
        self.weights_ih = np.delete(self.weights_ih, BASE_INPUTS + k, axis=1)
        del self.channels[k]
        self.reset_hidden()
        return True

    def clone(self) -> MosquitoBrain:
        return MosquitoBrain(self.weights_ih.copy(), self.weights_hh.copy(), self.weights_ho.copy(),
                             self.bias_h.copy(), self.bias_o.copy(), channels=self.channels,
                             layers=[{k: v.copy() for k, v in l.items()} for l in self.layers], apical=self.apical)

    def mutate(self, rng: random.Random, sigma: float = 0.05) -> int:
        """
        Nudges 1-3 randomly chosen weights/biases (every one equally likely)
        by a small step -- heavy-tailed, see HEAVY_TAIL_P. Small steps give
        selection something it can climb; the rare big ones get it off plateaus.
        A silent stacked layer offers only its gate (its weights wait unexpressed,
        and don't dilute the search -- Hinton's caveat on growing networks).
        """
        arrays = (self.weights_ih, self.weights_hh, self.weights_ho, self.bias_h, self.bias_o,
                  *(l[key] for l in self.layers
                    for key in (("W", "U", "b", "gate") if float(l["gate"][0]) != 0.0 else ("gate",))))
        sizes = np.cumsum([a.size for a in arrays])
        k = rng.randint(1, 3)
        for flat in rng.sample(range(int(sizes[-1])), k):
            a = int(np.searchsorted(sizes, flat, side="right"))
            i = flat - (int(sizes[a - 1]) if a else 0)
            if rng.random() < HEAVY_TAIL_P:
                step = max(-HEAVY_TAIL_MAX, min(HEAVY_TAIL_MAX, HEAVY_TAIL_SCALE * math.tan(math.pi * (rng.random() - 0.5))))
            else:
                step = rng.gauss(0.0, sigma)
            arrays[a].flat[i] += step
        return k

    def to_dict(self) -> dict[str, Any]:
        return {
            "weights_ih": self.weights_ih.tolist(),
            "weights_hh": self.weights_hh.tolist(),
            "weights_ho": self.weights_ho.tolist(),
            "bias_h": self.bias_h.tolist(),
            "bias_o": self.bias_o.tolist(),
            "channels": self.channels,
            "layers": [{k: v.tolist() for k, v in l.items()} for l in self.layers],
            "base_inputs": BASE_INPUTS,
            "base_outputs": BASE_OUTPUTS,
            "apical": self.apical,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MosquitoBrain:
        # Brains saved before inputs or outputs were added get zero weights
        # for them (inserted before any channels) -- identical behavior at
        # the switch; evolution can start using them from there.
        channels = [dict(c) for c in data.get("channels", [])]
        n_ch = len(channels)
        n_hid = len(data["bias_h"])
        n_in_saved = data.get("base_inputs", len(data["weights_ih"][0]) - n_ch)
        n_out_saved = data.get("base_outputs", len(data["weights_ho"]) - n_ch)
        weights_ih = [list(r[:n_in_saved]) + [0.0] * (BASE_INPUTS - n_in_saved) + list(r[n_in_saved:])
                      for r in data["weights_ih"]]
        ho, bo = [list(r) for r in data["weights_ho"]], list(data["bias_o"])
        weights_ho = ho[:n_out_saved] + [[0.0] * n_hid for _ in range(BASE_OUTPUTS - n_out_saved)] + ho[n_out_saved:]
        bias_o = bo[:n_out_saved] + [0.0] * (BASE_OUTPUTS - n_out_saved) + bo[n_out_saved:]
        return cls(weights_ih, data["weights_hh"], weights_ho, data["bias_h"], bias_o, channels=channels,
                   layers=data.get("layers"), apical=float(data.get("apical", 0.0)))
