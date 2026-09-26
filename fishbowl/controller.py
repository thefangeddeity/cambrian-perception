from __future__ import annotations

"""
Visual controller -- a small recurrent network that moves the gaze.

Modeled on insect neurobiology (e.g. Diptera / mosquito):
  - Optic Lobe Inputs:
      the gaze's luminance, motion and optic flow (dx, dy); loom and where
      motion is (dx, dy) from the whole field; current gaze (cx, cy, zoom) and
      eye velocity; interoception (blood sugar, arousal, threat, search,
      hunger, curiosity, gut, reserve, sleep pressure, asleep); the whole
      field's light and its trend (day and night); and the perception tree's
      output.
  - Central Complex (Recurrent Ring):
      A 16-unit recurrent neural circuit (RNN) with temporal hidden memory.
      Maintains smooth gaze stabilization, pursuit, and active visual casting.
  - Outputs: pan, tilt, zoom, alarm, tempo, sleep. There is no hard-wired
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
"""

import math
import random
from typing import Any, NamedTuple

from .state import MosquitoState

BASE_INPUTS = 25  # 19 + gut, reserve, sleep pressure, asleep, field light, light trend
INPUTS = BASE_INPUTS  # kept for older callers: the base inputs
HIDDEN = 16
BASE_OUTPUTS = 6  # [pan, tilt, zoom, alarm, tempo, sleep]
OUTPUTS = BASE_OUTPUTS
MAX_CHANNELS = 4
INPUT_NAMES = ("light", "motion", "flow x", "flow y", "loom", "gaze x", "gaze y", "zoom", "blood sugar",
               "arousal", "threat", "search", "motion dx", "motion dy", "eye vx", "eye vy", "hunger",
               "curiosity", "tree", "gut", "reserve", "sleep pressure", "asleep", "field light", "light trend")
OUTPUT_NAMES = ("pan", "tilt", "zoom", "alarm", "tempo", "sleep")


class Motor(NamedTuple):
    pan: float
    tilt: float
    zoom: float
    alarm: float
    tempo: float
    sleep: float


def _tanh(x: float) -> float:
    return math.tanh(x)


class MosquitoBrain:
    def __init__(
        self,
        weights_ih: list[list[float]],
        weights_hh: list[list[float]],
        weights_ho: list[list[float]],
        bias_h: list[float],
        bias_o: list[float],
        channels: list[dict] | None = None,
    ):
        self.weights_ih = weights_ih
        self.weights_hh = weights_hh
        self.weights_ho = weights_ho
        self.bias_h = bias_h
        self.bias_o = bias_o
        self.channels = [dict(c) for c in (channels or [])]
        self.reset_hidden()

    @classmethod
    def random(cls, rng: random.Random) -> MosquitoBrain:
        def matrix(rows: int, cols: int, scale: float = 0.4) -> list[list[float]]:
            return [
                [rng.uniform(-scale, scale) for _ in range(cols)]
                for _ in range(rows)
            ]

        return cls(
            weights_ih=matrix(HIDDEN, BASE_INPUTS, scale=0.4),
            weights_hh=matrix(HIDDEN, HIDDEN, scale=0.3),
            weights_ho=matrix(BASE_OUTPUTS, HIDDEN, scale=0.4),
            bias_h=[rng.uniform(-0.05, 0.05) for _ in range(HIDDEN)],
            bias_o=[rng.uniform(-0.05, 0.05) for _ in range(BASE_OUTPUTS)],
        )

    def reset_hidden(self) -> None:
        self.hidden = [0.0] * HIDDEN
        self.loop_in = [0.0] * len(self.channels)  # what each channel feeds back this step
        self._pred = [0.0] * len(self.channels)    # each predictor's last prediction

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
    ) -> Motor:
        """Runs one tick of the brain. Returns its motor outputs (Motor)."""
        base = [
            luminance,
            motion,
            flow_x,
            flow_y,
            loom,
            gaze_cx - 0.5,
            gaze_cy - 0.5,
            gaze_zoom,
            state.energy,
            state.arousal,
            state.threat,
            state.search,
            periph_dx,
            periph_dy,
            eye_vx * 2.5,  # terminal eye speed 0.4 -> ~1
            eye_vy * 2.5,
            state.hunger,
            state.curiosity,
            tree_out,
            state.gut,
            state.reserve,
            state.sleep_pressure,
            state.asleep,
            field_light,
            state.light_trend,
        ]
        # Predictors: what comes back is how wrong last step's prediction was.
        for k, ch in enumerate(self.channels):
            if ch["kind"] == "predict":
                self.loop_in[k] = max(-1.0, min(1.0, base[ch["target"]] - self._pred[k]))
        inputs = base + self.loop_in
        n_in = len(inputs)

        # Recurrent hidden update: h_t = tanh(W_ih * x + W_hh * h_{t-1} + b_h)
        new_hidden = []
        for i in range(HIDDEN):
            val = self.bias_h[i]
            row = self.weights_ih[i]
            for j in range(n_in):
                val += row[j] * inputs[j]
            for j in range(HIDDEN):
                val += self.weights_hh[i][j] * self.hidden[j]
            new_hidden.append(_tanh(val))
        self.hidden = new_hidden

        # Motor readout: o_t = tanh(W_ho * h_t + b_o)
        outputs = []
        for i in range(len(self.weights_ho)):
            val = self.bias_o[i]
            for j in range(HIDDEN):
                val += self.weights_ho[i][j] * self.hidden[j]
            outputs.append(_tanh(val))

        for k, ch in enumerate(self.channels):
            o = outputs[BASE_OUTPUTS + k]
            if ch["kind"] == "predict":
                self._pred[k] = o
            else:
                self.loop_in[k] = o
        return Motor(*outputs[:BASE_OUTPUTS])

    # ---- growable channels ------------------------------------------------
    def loop_synapses(self) -> float:
        """Total weight on the channels' way back in (their energy price)."""
        return sum(abs(row[BASE_INPUTS + k]) for row in self.weights_ih for k in range(len(self.channels)))

    def grow_channel(self, rng: random.Random, kind: str = "latch") -> bool:
        """A new channel: a latch duplicated from an existing output, or a
        blank predictor of one of its inputs. Zero weights on the way back
        in, so nothing changes until evolution wires it up."""
        if len(self.channels) >= MAX_CHANNELS:
            return False
        if kind == "predict":
            self.weights_ho.append([0.0] * HIDDEN)
            self.bias_o.append(0.0)
            ch = {"kind": "predict", "target": rng.randrange(BASE_INPUTS)}
        else:
            src = rng.randrange(len(self.weights_ho))
            self.weights_ho.append(list(self.weights_ho[src]))
            self.bias_o.append(self.bias_o[src])
            ch = {"kind": "latch", "copy_of": src}
        for row in self.weights_ih:
            row.append(0.0)
        self.channels.append(ch)
        self.reset_hidden()
        return True

    def shrink_channel(self, rng: random.Random) -> bool:
        if not self.channels:
            return False
        k = rng.randrange(len(self.channels))
        del self.weights_ho[BASE_OUTPUTS + k]
        del self.bias_o[BASE_OUTPUTS + k]
        for row in self.weights_ih:
            del row[BASE_INPUTS + k]
        del self.channels[k]
        self.reset_hidden()
        return True

    def clone(self) -> MosquitoBrain:
        return MosquitoBrain(
            weights_ih=[row[:] for row in self.weights_ih],
            weights_hh=[row[:] for row in self.weights_hh],
            weights_ho=[row[:] for row in self.weights_ho],
            bias_h=self.bias_h[:],
            bias_o=self.bias_o[:],
            channels=self.channels,
        )

    def mutate(self, rng: random.Random, sigma: float = 0.05) -> int:
        """
        Nudges 1-3 randomly chosen weights/biases by a small Gaussian step.
        Audit: the old version changed ~32 of ~650 weights at sigma 0.12 per
        mutation -- almost never neutral, usually worse, so the brain was
        effectively never improved. Small steps give selection something
        it can actually climb.
        """
        slots = [(m, r, j) for m in (self.weights_ih, self.weights_hh, self.weights_ho)
                 for r in range(len(m)) for j in range(len(m[r]))]
        slots += [(b, None, i) for b in (self.bias_h, self.bias_o) for i in range(len(b))]
        k = rng.randint(1, 3)
        for container, r, j in rng.sample(slots, k):
            if r is None:
                container[j] += rng.gauss(0.0, sigma)
            else:
                container[r][j] += rng.gauss(0.0, sigma)
        return k

    def to_dict(self) -> dict[str, Any]:
        return {
            "weights_ih": self.weights_ih,
            "weights_hh": self.weights_hh,
            "weights_ho": self.weights_ho,
            "bias_h": self.bias_h,
            "bias_o": self.bias_o,
            "channels": self.channels,
            "base_inputs": BASE_INPUTS,
            "base_outputs": BASE_OUTPUTS,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MosquitoBrain:
        # Brains saved before inputs or outputs were added get zero weights
        # for them (inserted before any channels) -- identical behavior at
        # the switch; evolution can start using them from there.
        channels = [dict(c) for c in data.get("channels", [])]
        n_ch = len(channels)
        n_in_saved = data.get("base_inputs", len(data["weights_ih"][0]) - n_ch)
        n_out_saved = data.get("base_outputs", len(data["weights_ho"]) - n_ch)
        weights_ih = [list(r[:n_in_saved]) + [0.0] * (BASE_INPUTS - n_in_saved) + list(r[n_in_saved:])
                      for r in data["weights_ih"]]
        ho, bo = [list(r) for r in data["weights_ho"]], list(data["bias_o"])
        weights_ho = ho[:n_out_saved] + [[0.0] * HIDDEN for _ in range(BASE_OUTPUTS - n_out_saved)] + ho[n_out_saved:]
        bias_o = bo[:n_out_saved] + [0.0] * (BASE_OUTPUTS - n_out_saved) + bo[n_out_saved:]
        return cls(
            weights_ih=weights_ih,
            weights_hh=data["weights_hh"],
            weights_ho=weights_ho,
            bias_h=data["bias_h"],
            bias_o=bias_o,
            channels=channels,
        )
