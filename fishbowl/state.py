from __future__ import annotations

"""
Mosquito Brain Homeostasis -- Interoceptive & Metabolic State.

A real insect (e.g. mosquito, fruit fly) does not optimize an abstract
score; it maintains physiological equilibrium (homeostasis) under real
metabolic constraints:
  - Basal metabolic expenditure: every tick of existence costs energy.
  - Motor effort: large, rapid saccades cost energy and accumulate fatigue.
  - Arousal: visual motion and luminance flicker excite the optic lobe.
  - Threat: looming dark objects excite giant fibers, triggering evasive arousal.
  - Search drive: when starved or stagnant, appetitive search pressure rises.
  - Homeostatic viability: distance from physiological collapse.
"""

from dataclasses import dataclass


def _clamp(v: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, v))


@dataclass
class MosquitoState:
    energy: float = 1.0        # metabolic reserve [0, 1]
    arousal: float = 0.1       # sensory alertness / motor readiness [0, 1]
    threat: float = 0.0        # looming danger signal [0, 1]
    search: float = 0.0        # appetitive / host-seeking drive [0, 1]
    fatigue: float = 0.0       # motor wear from high-velocity saccades [0, 1]
    # Gemini's plan lists "Curiosity & Hunger: appetite for novel visual
    # structure" under interoception; its first state.py left both out.
    hunger: float = 0.0        # builds while energy is low (visual-organism's form)
    curiosity: float = 0.0     # appetite for novelty: grows while nothing new comes in, drops when fed
    previous_drive: float = 0.0

    def drive(self) -> float:
        """
        Homeostatic distance from optimal equilibrium.
        Minimizing drive means maintaining high energy, low threat, and low fatigue.
        """
        energy_deficit = 1.0 - self.energy
        return (
            energy_deficit * energy_deficit
            + self.threat * self.threat
            + self.fatigue * self.fatigue
        )

    def update(
        self,
        motion: float,
        loom: float,
        motor_effort: float,
        aperture_cost: float = 0.0,
    ) -> None:
        self.previous_drive = self.drive()

        # Arousal: fast excitation from motion, steady decay
        self.arousal = _clamp(0.92 * self.arousal + 0.08 * motion)

        # Threat: sensitive to loom (approaching shadow/swatter)
        self.threat = _clamp(0.85 * self.threat + 0.15 * loom)

        # Basal metabolism: existing costs energy; high arousal burns faster
        basal_cost = 0.003 + 0.003 * self.arousal
        effort_cost = 0.010 * motor_effort

        # A wider look processes more pixels: priced per tick by the
        # caller (area x real CPU scarcity -- see run_vision.py).
        self.energy = _clamp(self.energy - (basal_cost + effort_cost + aperture_cost))

        # Fatigue: accumulates with violent saccades, recovers when still
        self.fatigue = _clamp(0.95 * self.fatigue + 0.08 * motor_effort)

        # Hunger tracks energy deficit; search pressure follows hunger.
        self.hunger = _clamp(0.97 * self.hunger + 0.03 * (1.0 - self.energy))
        self.search = _clamp(0.96 * self.search + 0.04 * self.hunger)

    def feed_visual_sustenance(self, tracking_quality: float) -> None:
        """
        Tracking salient visual structure provides cognitive sustenance,
        offsetting some basal decay (information foraging / active engagement).
        """
        # 0.012, not 0.004: at 0.004 the best possible meal was below
        # basal burn alone, so every organism starved whatever it did
        # (measured on real camera frames: 12/12 random brains and the
        # live genome ended at ~0 energy). Now eating well while moving
        # economically can run a surplus; frantic or badly-placed can't.
        gain = 0.012 * _clamp(tracking_quality)
        self.energy = _clamp(self.energy + gain)
        # Curiosity: slowly rises every frame, satisfied by real novelty.
        self.curiosity = _clamp(self.curiosity + 0.01 - 0.05 * _clamp(tracking_quality))

    def drive_reduction(self) -> float:
        """
        Homeostatic reward = D(t-1) - D(t).
        Positive when moving toward health/safety, negative when depleting/panicking.
        """
        return self.previous_drive - self.drive()

    def to_dict(self) -> dict[str, float]:
        return {
            "energy": round(self.energy, 4),
            "arousal": round(self.arousal, 4),
            "threat": round(self.threat, 4),
            "search": round(self.search, 4),
            "fatigue": round(self.fatigue, 4),
            "hunger": round(self.hunger, 4),
            "curiosity": round(self.curiosity, 4),
        }
