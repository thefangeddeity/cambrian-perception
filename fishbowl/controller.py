from __future__ import annotations

"""
Mosquito Visual Controller -- Recurrent Central Complex with Giant Fiber Reflex.

Modeled on insect neurobiology (e.g. Diptera / mosquito):
  - Optic Lobe Inputs:
      12 ommatidia/optic channels: luminance, motion, optic flow (dx, dy), loom,
      current gaze (cx, cy, zoom), and homeostatic interoception (energy, arousal,
      threat, search).
  - Central Complex (Recurrent Ring):
      A 16-unit recurrent neural circuit (RNN) with temporal hidden memory.
      Maintains smooth gaze stabilization, pursuit, and active visual casting.
  - Giant Fiber Evasion Reflex (Subcortical Override):
      High looming stimulus triggers an immediate emergency escape saccade
      and fovea expansion, bypassing deliberative processing.
"""

import math
import random
from typing import Any

from .state import MosquitoState

INPUTS = 18  # Gemini's 12 + where the whole field saw motion relative to the look (dx, dy) + own eye velocity (vx, vy) + hunger, curiosity
HIDDEN = 16
OUTPUTS = 5  # [pan, tilt, zoom, alarm, tempo] -- tempo: speed up / slow down its gazing (see run_vision.py)


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
    ):
        self.weights_ih = weights_ih
        self.weights_hh = weights_hh
        self.weights_ho = weights_ho
        self.bias_h = bias_h
        self.bias_o = bias_o
        self.hidden = [0.0] * HIDDEN

    @classmethod
    def random(cls, rng: random.Random) -> MosquitoBrain:
        def matrix(rows: int, cols: int, scale: float = 0.4) -> list[list[float]]:
            return [
                [rng.uniform(-scale, scale) for _ in range(cols)]
                for _ in range(rows)
            ]

        return cls(
            weights_ih=matrix(HIDDEN, INPUTS, scale=0.4),
            weights_hh=matrix(HIDDEN, HIDDEN, scale=0.3),
            weights_ho=matrix(OUTPUTS, HIDDEN, scale=0.4),
            bias_h=[rng.uniform(-0.05, 0.05) for _ in range(HIDDEN)],
            bias_o=[rng.uniform(-0.05, 0.05) for _ in range(OUTPUTS)],
        )

    def reset_hidden(self) -> None:
        self.hidden = [0.0] * HIDDEN

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
    ) -> tuple[float, float, float, float, bool]:
        """
        Runs one tick of the mosquito brain.
        Returns: (pan_dx, tilt_dy, d_zoom, alarm_response, tempo, is_reflex)
        """
        # No hard-wired escape reflex (User: "No hard-wired flinch. A fast
        # reaction to looming gets rewarded so it can evolve."). Gemini's
        # giant-fiber override lived here; run_vision.py now rewards a
        # fast reaction to real approach instead, and this network has to
        # evolve the flinch itself from its loom/threat inputs.

        # -------------------------------------------------------------
        # 2. Central Complex: Recurrent Processing
        # -------------------------------------------------------------
        inputs = [
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
        ]

        # Recurrent hidden update: h_t = tanh(W_ih * x + W_hh * h_{t-1} + b_h)
        new_hidden = []
        for i in range(HIDDEN):
            val = self.bias_h[i]
            for j in range(INPUTS):
                val += self.weights_ih[i][j] * inputs[j]
            for j in range(HIDDEN):
                val += self.weights_hh[i][j] * self.hidden[j]
            new_hidden.append(_tanh(val))

        self.hidden = new_hidden

        # Motor readout: o_t = tanh(W_ho * h_t + b_o)
        outputs = []
        for i in range(OUTPUTS):
            val = self.bias_o[i]
            for j in range(HIDDEN):
                val += self.weights_ho[i][j] * self.hidden[j]
            outputs.append(_tanh(val))

        pan_dx = outputs[0]
        tilt_dy = outputs[1]
        d_zoom = outputs[2]
        alarm = outputs[3]
        tempo = outputs[4]

        return pan_dx, tilt_dy, d_zoom, alarm, tempo, False

    def clone(self) -> MosquitoBrain:
        return MosquitoBrain(
            weights_ih=[row[:] for row in self.weights_ih],
            weights_hh=[row[:] for row in self.weights_hh],
            weights_ho=[row[:] for row in self.weights_ho],
            bias_h=self.bias_h[:],
            bias_o=self.bias_o[:],
        )

    def mutate(self, rng: random.Random, rate: float = 0.05, sigma: float = 0.12) -> int:
        """Applies Gaussian mutation to a subset of synaptic weights and biases."""
        mutated = 0

        for matrix in (self.weights_ih, self.weights_hh, self.weights_ho):
            for row in matrix:
                for j in range(len(row)):
                    if rng.random() < rate:
                        row[j] += rng.gauss(0.0, sigma)
                        mutated += 1

        for bias_list in (self.bias_h, self.bias_o):
            for i in range(len(bias_list)):
                if rng.random() < rate:
                    bias_list[i] += rng.gauss(0.0, sigma)
                    mutated += 1

        return mutated

    def to_dict(self) -> dict[str, Any]:
        return {
            "weights_ih": self.weights_ih,
            "weights_hh": self.weights_hh,
            "weights_ho": self.weights_ho,
            "bias_h": self.bias_h,
            "bias_o": self.bias_o,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MosquitoBrain:
        # Brains saved before inputs were added get zero weights for the
        # new inputs -- identical behavior at the switch, evolution can
        # start using them from there.
        weights_ih = [list(row) + [0.0] * (INPUTS - len(row)) for row in data["weights_ih"]]
        return cls(
            weights_ih=weights_ih,
            weights_hh=data["weights_hh"],
            # Brains saved before an output was added get a silent (zero)
            # row for it: identical behavior until evolution uses it.
            weights_ho=[list(r) for r in data["weights_ho"]] + [[0.0] * HIDDEN for _ in range(OUTPUTS - len(data["weights_ho"]))],
            bias_h=data["bias_h"],
            bias_o=list(data["bias_o"]) + [0.0] * (OUTPUTS - len(data["bias_o"])),
        )
