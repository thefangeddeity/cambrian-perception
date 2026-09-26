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


# Energy on a real clock. Its body persists across generations and
# restarts (run_vision.py), so these are sized for a life of hours, not
# the ~1-minute lives Gemini's per-tick numbers implied (measured: with
# surprise as food every brain starved within one 80 s window).
BASAL_PER_SECOND = 1.0 / 1200.0  # full energy lasts ~20 min at hummingbird pace with no food, ~2 h at reptile pace
ACCLIMATIZE_HALF_LIFE_S = 80.0  # ~2 min time constant for metabolic rate to follow tempo
EFFORT_COST = 1e-4               # per push, x force^2 (a full saccade ~ 0.25 s of basal)
FOOD_PER_LOOK = 2e-4             # SNACK: x surprise (0..1) -- alone it can't sustain a fast tempo
PREY_FOOD_PER_LOOK = 1.2e-3      # MEAL: x prey held in the gaze center (0..1) -- see fishbowl/prey.py


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
    # Metabolic rate, acclimatizing: drifts toward the rate its current
    # tempo implies over ~2 minutes -- animals acclimatize to different
    # metabolic conditions (a bear is sluggish in winter, active in
    # spring). 1.0 = gazing every frame; slower tempo -> lower.
    metabolic_rate: float = 1.0

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
        dt: int = 1,
        pace: int = 1,
        dt_seconds: float | None = None,
    ) -> None:
        """
        One gaze. dt = frames since the last gaze (for the fast-decaying
        signals); dt_seconds = the same interval in real seconds; pace =
        the gaze interval it is using right now.
        Everything that happens in real time (basal burn, and the decay
        and build-up of arousal, threat, fatigue, hunger, search) is
        scaled by dt, so looking less often doesn't slow its metabolism
        down. Things that happen per look (the motor push, the cost of
        processing a look of this size) are charged once per look.

        Basal metabolic rate scales with pace (a hummingbird must eat
        almost constantly; many reptiles eat rarely): a fast-paced
        organism idles hot, a slow one idles cheap -- otherwise a slow
        pace could never eat enough to survive at all.
        """

        def leak(old: float, decay: float, target: float) -> float:
            k = decay ** dt
            return _clamp(k * old + (1.0 - k) * target)

        self.arousal = leak(self.arousal, 0.92, motion)
        self.threat = leak(self.threat, 0.85, loom)

        # Basal rate follows its ACCLIMATIZED metabolic rate, which drifts
        # toward 1/pace (the tempo it is gazing at right now) with a ~2 min
        # time constant -- slowing down only pays off once it has been slow
        # for a while, so it can't flip between hummingbird and reptile
        # for free every frame.
        seconds = dt_seconds if dt_seconds is not None else dt / 15.0
        k = 0.5 ** (seconds / ACCLIMATIZE_HALF_LIFE_S)
        self.metabolic_rate = k * self.metabolic_rate + (1.0 - k) * (1.0 / max(1, pace))
        bmr = self.metabolic_rate
        basal_cost = BASAL_PER_SECOND * (1.0 + self.arousal) * bmr * seconds
        effort_cost = EFFORT_COST * motor_effort
        # A wider look processes more pixels: priced per look by the
        # caller (area x real CPU scarcity -- see run_vision.py).
        self.energy = _clamp(self.energy - (basal_cost + effort_cost + aperture_cost))

        # Fatigue: each push adds wear; wear recovers in real time.
        self.fatigue = _clamp(0.95 ** dt * self.fatigue + 0.08 * motor_effort)

        # Hunger tracks energy deficit; search pressure follows hunger.
        self.hunger = leak(self.hunger, 0.97, 1.0 - self.energy)
        self.search = leak(self.search, 0.96, self.hunger)
        self._dt = dt

    def feed_visual_sustenance(self, tracking_quality: float) -> None:
        """
        Tracking salient visual structure provides cognitive sustenance,
        offsetting some basal decay (information foraging / active engagement).
        """
        # A small snack (FOOD_PER_LOOK); real meals are prey (feed_prey).
        gain = FOOD_PER_LOOK * _clamp(tracking_quality)
        self.energy = _clamp(self.energy + gain)
        # Curiosity: slowly rises every frame, satisfied by real novelty.
        # Rises with real time, falls with what it actually took in (audit:
        # the fall was 0.05 per gaze vs a rise of 0.01 per FRAME, so at slow
        # tempos curiosity could never come down and sat pinned at 1.0).
        self.curiosity = _clamp(self.curiosity + 0.01 * getattr(self, "_dt", 1) - 0.25 * _clamp(tracking_quality))

    def idle(self, seconds: float) -> None:
        """Time passing with no food and nothing seen (e.g. while the
        process was down): basal burn at its acclimatized rate, and hunger
        and search following the energy deficit."""
        self.energy = _clamp(self.energy - BASAL_PER_SECOND * self.metabolic_rate * seconds)
        self.hunger = _clamp(1.0 - self.energy) if seconds > 60 else self.hunger
        self.search = self.hunger if seconds > 60 else self.search
        self.arousal = 0.0 if seconds > 60 else self.arousal
        self.threat = 0.0 if seconds > 60 else self.threat

    def feed_prey(self, amount: float) -> None:
        """A real meal: prey (a person or animal, per YOLO) held in the
        center of the gaze. The scarce, earned food source."""
        self.energy = _clamp(self.energy + PREY_FOOD_PER_LOOK * _clamp(amount))

    @classmethod
    def from_dict(cls, data: dict) -> "MosquitoState":
        fields = ("energy", "arousal", "threat", "search", "fatigue", "hunger", "curiosity", "metabolic_rate")
        return cls(**{k: float(data[k]) for k in fields if k in data})

    def to_dict(self) -> dict[str, float]:
        return {
            "energy": round(self.energy, 4),
            "arousal": round(self.arousal, 4),
            "threat": round(self.threat, 4),
            "search": round(self.search, 4),
            "fatigue": round(self.fatigue, 4),
            "hunger": round(self.hunger, 4),
            "curiosity": round(self.curiosity, 4),
            "metabolic_rate": round(self.metabolic_rate, 4),
        }
