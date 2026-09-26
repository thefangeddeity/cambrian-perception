from __future__ import annotations

"""
The organism's body -- homeostatic, on a real clock, persisting across
generations and restarts (run_vision.py). Fixed, human-written physics;
only the brain's behaviour evolves.

Energy (after a nutrition/sleep design panel -- Sterling's allostasis,
Borbely's two-process sleep model, Keramati & Gutkin's homeostatic RL):
  - gut: meals land here (a full gut can't take more -> satiety) and
    digest into blood sugar over a few minutes.
  - blood sugar (`energy`, 0..1): pays for everything.
  - reserve (0..1, ~6 h of waking burn): surplus blood sugar is stored at
    75% efficiency; the reserve can top blood sugar back up only at a
    capped rate that covers SLEEPING metabolism but not waking. So in an
    empty room, awake means blood sugar falls and hunger rises; asleep
    holds steady -- which lets sleep evolve without any rule saying
    "sleep at night".
  - no death: an empty body burns less and degrades instead (the caller
    switches off colour/zoom and slows gazing), so the homeostatic signal
    never goes flat at zero.
Sleep:
  - sleep pressure (Process S) builds with time awake and brain load and
    clears during sleep.
  - sleep is the brain's own choice (a `sleep` output), with a settling
    period on falling asleep (no clearing yet) and grogginess on waking
    (can't eat yet); two physiological overrides, not behaviour rules:
    collapse when pressure maxes out, and hunger that wakes it.
Sleep pays for itself twice: it saves energy when there is nothing to eat,
and it restores what tiredness takes (a tired hunter gets less of what it
catches; sleep also consolidates its habituation memory -- run_vision.py).
Also: arousal and threat (from the whole visual field), a sense of day and
night from the field's light, muscle fatigue, search and curiosity, and a
metabolic rate that acclimatizes to its tempo.
"""

import math
from dataclasses import dataclass

# ---- units -----------------------------------------------------------
# B = waking basal burn per second at metabolic rate 1. Stores are sized
# in B-seconds. Per-gaze costs and meals are passed in the older
# "energy unit" (1 unit = LEGACY_UNIT B-seconds) and converted here.
BASAL_PER_SECOND = 1.0 / 1200.0   # kept for callers that price in legacy units
LEGACY_UNIT = 1200.0
G_CAP = 600.0          # blood sugar: ~10 min of waking basal burn
GUT_CAP = 1200.0       # gut: ~20 min
R_CAP = 21600.0        # reserve: ~6 h
DIGEST_TAU_S = 240.0   # gut -> blood sugar time constant
STORE_ABOVE = 0.8      # blood sugar above this is stored...
STORE_RATE = 1.0       # ...at up to this many B per second
STORE_EFFICIENCY = 0.75
MOBILIZE_BELOW = 0.5   # reserve tops blood sugar up below this level...
MOBILIZE_RATE = 0.5    # ...at up to 0.5 B/s: enough for sleep, not for waking
SLEEP_METABOLISM = 0.3  # B/s while asleep
# Awake: a fixed cost of being awake at all, plus a share that follows its
# tempo (a fast gaze is expensive, a slow one cheap). The floor sits above
# what the reserve can supply (MOBILIZE_RATE), so however slowly it gazes,
# staying awake with nothing to eat still drains it -- sleep is the only
# way to hold steady in an empty room.
WAKE_FLOOR = 0.6
TEMPO_SHARE = 0.4
# Its sense of day and night: the whole field's light, averaged over about a
# minute and about 20 minutes; their difference says dawn (rising) or dusk.
LIGHT_FAST_S = 60.0
LIGHT_SLOW_S = 1200.0
# A tired hunter misses: sleep pressure cuts how much of what it catches it
# gets (up to half at full pressure).
TIRED_EFFICIENCY = 0.5
ACCLIMATIZE_HALF_LIFE_S = 80.0  # metabolic rate follows tempo over ~2 min
EFFORT_COST = 1e-4               # legacy units per push, x force^2
FOOD_PER_LOOK = 2e-4             # SNACK (legacy units) x surprise (0..1)
PREY_FOOD_PER_LOOK = 1.2e-3      # MEAL (legacy units) x prey held in the gaze center (0..1)
# Sleep pressure (Process S)
S_RISE_S = 14 * 3600.0
S_FALL_S = 3 * 3600.0
SLEEP_SETTLE_S = 10.0   # falling asleep: no clearing yet
SLEEP_INERTIA_S = 7.0   # waking up: groggy, can't eat yet
COLLAPSE_S = 0.95       # sleep pressure that forces sleep...
COLLAPSE_RELEASE_S = 0.8  # ...and holds it until pressure is back below this
HUNGER_WAKE_G = 0.15    # blood sugar that forces waking -- with an empty gut AND an empty
HUNGER_WAKE_R = 0.05    # reserve: while the reserve can still carry sleep, a hungry animal may sleep
EMPTY_G = 0.1           # below this the body degrades


def _clamp(v: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, v))


@dataclass
class MosquitoState:
    energy: float = 1.0        # blood sugar [0, 1] (was: the single energy store)
    gut: float = 0.0           # undigested food [0, 1]
    reserve: float = 0.5       # slow reserve [0, 1]
    sleep_pressure: float = 0.0
    asleep: float = 0.0        # 1.0 while asleep
    sleep_clock: float = 0.0   # seconds since falling asleep / waking (settle, inertia)
    arousal: float = 0.1
    threat: float = 0.0
    search: float = 0.0
    fatigue: float = 0.0
    hunger: float = 0.0
    curiosity: float = 0.0
    metabolic_rate: float = 1.0
    light_fast: float = 0.5    # whole-field light, ~1 min average
    light_slow: float = 0.5    # whole-field light, ~20 min average
    cleared: float = 0.0       # sleep pressure cleared since last asked (for memory consolidation)

    # ---- what the organism "feels" ------------------------------------
    def drive(self) -> float:
        """Distance from a viable state. Hunger counts the gut (a full
        stomach cuts hunger before absorption, like ghrelin) and a small
        reserve term, so building reserves shows up within a window."""
        fed = min(1.0, self.energy + 0.5 * self.gut)
        return ((1.0 - fed) ** 2 + 0.3 * (1.0 - self.reserve) ** 2
                + self.threat ** 2 + self.fatigue ** 2 + 0.3 * self.sleep_pressure ** 2)

    @property
    def degraded(self) -> bool:
        return self.energy < EMPTY_G

    @property
    def light_trend(self) -> float:
        return max(-1.0, min(1.0, 4.0 * (self.light_fast - self.light_slow)))

    @property
    def efficiency(self) -> float:
        return 1.0 - TIRED_EFFICIENCY * self.sleep_pressure

    @property
    def can_eat(self) -> bool:
        return self.asleep < 0.5 and self.sleep_clock >= SLEEP_INERTIA_S

    # ---- sleep --------------------------------------------------------
    def set_sleep(self, wants_sleep: bool, loom: float = 0.0, field_motion: float = 0.0) -> None:
        """The brain's choice, with the body's two overrides and a raised
        arousal threshold while asleep (only a big change wakes it)."""
        asleep = self.asleep >= 0.5
        want = wants_sleep
        starving = self.energy < HUNGER_WAKE_G and self.gut < 0.02 and self.reserve < HUNGER_WAKE_R
        # Exhaustion: collapse, and no waking by choice until it has recovered
        # -- unless it is starving, which keeps even an exhausted animal up.
        if not starving and ((not asleep and self.sleep_pressure > COLLAPSE_S)
                             or (asleep and self.sleep_pressure > COLLAPSE_RELEASE_S)):
            want = True
        # What wakes even an exhausted animal (and keeps it up): starving, or a big change.
        if starving:
            want = False
        if asleep and (loom > 0.18 or field_motion > 0.6):
            want = False
        if want != asleep:
            self.asleep = 1.0 if want else 0.0
            self.sleep_clock = 0.0

    # ---- one gaze -----------------------------------------------------
    def update(
        self,
        motion: float,
        loom: float,
        motor_effort: float,
        aperture_cost: float = 0.0,
        dt: int = 1,
        pace: int = 1,
        dt_seconds: float | None = None,
        field_light: float | None = None,
    ) -> None:
        """
        One gaze. dt = frames since the last gaze (fast signals);
        dt_seconds = the same interval in real seconds; pace = the gaze
        interval in use now. motor_effort and aperture_cost (per-gaze
        costs: thinking, gaze size, colour) are in legacy units.
        field_light = the whole field's mean light (0..1), its day/night sense.
        """
        seconds = dt_seconds if dt_seconds is not None else dt / 15.0
        asleep = self.asleep >= 0.5
        self.sleep_clock += seconds

        def leak(old: float, decay: float, target: float) -> float:
            k = decay ** dt
            return _clamp(k * old + (1.0 - k) * target)

        self.arousal = leak(self.arousal, 0.92, 0.0 if asleep else motion)
        if field_light is not None:
            self.light_fast += (1.0 - math.exp(-seconds / LIGHT_FAST_S)) * (field_light - self.light_fast)
            self.light_slow += (1.0 - math.exp(-seconds / LIGHT_SLOW_S)) * (field_light - self.light_slow)
        self.threat = leak(self.threat, 0.85, loom)

        # metabolic rate acclimatizes toward the current tempo
        k = 0.5 ** (seconds / ACCLIMATIZE_HALF_LIFE_S)
        self.metabolic_rate = k * self.metabolic_rate + (1.0 - k) * (1.0 / max(1, pace))

        # --- spending (B-seconds) ---
        if asleep:
            burn = SLEEP_METABOLISM * seconds
        else:
            burn = (WAKE_FLOOR + TEMPO_SHARE * (1.0 + self.arousal) * self.metabolic_rate) * seconds
        burn += (EFFORT_COST * motor_effort + aperture_cost) * LEGACY_UNIT
        if self.degraded:  # soft floor: an empty body runs on less
            burn *= 0.5 + 0.5 * self.energy / EMPTY_G
        g_bs = self.energy * G_CAP - burn

        # --- digestion: gut -> blood sugar ---
        gut_bs = self.gut * GUT_CAP
        moved = gut_bs * (1.0 - math.exp(-seconds / DIGEST_TAU_S))
        gut_bs -= moved
        g_bs += moved

        # --- storage and mobilization ---
        r_bs = self.reserve * R_CAP
        if g_bs > STORE_ABOVE * G_CAP:
            store = min(g_bs - STORE_ABOVE * G_CAP, STORE_RATE * seconds)
            g_bs -= store
            r_bs += STORE_EFFICIENCY * store
        elif g_bs < MOBILIZE_BELOW * G_CAP:
            draw = min(MOBILIZE_BELOW * G_CAP - g_bs, MOBILIZE_RATE * seconds, r_bs)
            r_bs -= draw
            g_bs += draw

        self.energy = _clamp(g_bs / G_CAP)
        self.gut = _clamp(gut_bs / GUT_CAP)
        self.reserve = _clamp(r_bs / R_CAP)

        # --- sleep pressure (Process S) ---
        if asleep:
            if self.sleep_clock > SLEEP_SETTLE_S:
                before = self.sleep_pressure
                self.sleep_pressure = self.sleep_pressure * math.exp(-seconds / S_FALL_S)
                self.cleared += before - self.sleep_pressure
        else:
            load = 0.5 + 0.5 * _clamp(self.metabolic_rate)
            self.sleep_pressure = 1.0 - (1.0 - self.sleep_pressure) * math.exp(-seconds * load / S_RISE_S)

        self.fatigue = _clamp(0.95 ** dt * self.fatigue + 0.08 * motor_effort)
        self.hunger = _clamp(1.0 - min(1.0, self.energy + 0.5 * self.gut))
        self.search = leak(self.search, 0.96, self.hunger)
        self._dt = dt

    # ---- eating -------------------------------------------------------
    def _swallow(self, legacy_amount: float) -> None:
        room = (1.0 - self.gut) * GUT_CAP
        self.gut = _clamp(self.gut + min(room, legacy_amount * LEGACY_UNIT) / GUT_CAP)

    def feed_visual_sustenance(self, tracking_quality: float) -> None:
        """A small snack: genuinely new structure in the gaze center."""
        if self.can_eat:
            self._swallow(FOOD_PER_LOOK * _clamp(tracking_quality) * self.efficiency)
        # Curiosity rises with real time, falls with what it took in.
        self.curiosity = _clamp(self.curiosity + 0.01 * getattr(self, "_dt", 1) - 0.25 * _clamp(tracking_quality))

    def feed_prey(self, amount: float) -> None:
        """A real meal: prey (a person or animal, per YOLO) held in the
        center of the gaze. Only while awake and past waking grogginess."""
        if self.can_eat:
            self._swallow(PREY_FOOD_PER_LOOK * _clamp(amount) * self.efficiency)

    def idle(self, seconds: float) -> None:
        """Time passing while the process was down: no food, nothing seen,
        treated as sleep (the body stays viable on its reserve)."""
        self.asleep, self.sleep_clock = 1.0, SLEEP_SETTLE_S + 1.0
        step = 30.0
        while seconds > 0:
            s = min(step, seconds)
            self.update(0.0, 0.0, 0.0, 0.0, dt=1, pace=12, dt_seconds=s)
            seconds -= s
        self.arousal = self.threat = 0.0

    def take_cleared(self) -> float:
        c, self.cleared = self.cleared, 0.0
        return c

    # ---- persistence --------------------------------------------------
    FIELDS = ("energy", "gut", "reserve", "sleep_pressure", "asleep", "sleep_clock", "arousal", "threat",
              "search", "fatigue", "hunger", "curiosity", "metabolic_rate", "light_fast", "light_slow")

    @classmethod
    def from_dict(cls, data: dict) -> "MosquitoState":
        return cls(**{k: float(data[k]) for k in cls.FIELDS if k in data})

    def to_dict(self) -> dict[str, float]:
        return {k: round(float(getattr(self, k)), 4) for k in self.FIELDS}
