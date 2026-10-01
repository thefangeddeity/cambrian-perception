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
    switches off colour and slows gazing), so the homeostatic signal
    never goes flat at zero.
Sleep:
  - sleep pressure (Process S) builds with time awake and brain load and
    clears during sleep.
  - sleep is the brain's own choice (a `sleep` output), with a settling
    period on falling asleep (no clearing yet) and grogginess on waking
    (can't eat yet); one physiological override, not a behaviour rule:
    collapse when pressure maxes out. (Starvation used to force it awake
    too; retired 2026-09-27 -- it made the body flap between sleep and
    waking as the gut crossed its threshold, losing a grogginess each time.)
Metabolic strategy (after a design panel on poikilothermy -- Bennett &
Ruben 1979, Secor, Huey & Pianka; genome.metabolism, inherited): from an
endotherm (1, today's body: a high resting burn, any tempo sustainable) to an
ectotherm (0.1: a tenth of the resting and sleeping burn -- cheap waiting,
sensors still on -- but sustained activity capped at AEROBIC_SCOPE x its
resting rate; faster than that is an anaerobic burst whose debt it feels as
fatigue and repays over hours, like a crocodile after a struggle). Sensors
and thinking keep their prices whatever the strategy (Niven & Laughlin).
Digestion costs energy for everyone (specific dynamic action, Secor): part
of every meal is spent digesting it.
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
DIGEST_TAU_S = 240.0   # gut -> blood sugar time constant
# ---- fuel stores and their hormonal controls (docs/physiology.md) ----
# Insulin (fed): blood sugar above STORE_ABOVE is stored -- glycogen first,
# then fat. Glucagon (fasted): below MOBILIZE_BELOW, glycogen tops blood sugar
# up and fat is burned directly. Fat can't become glucose (only its glycerol,
# ~5%), so it fuels aerobic work, never a burst or the brain -- the brain's
# backup is ketones, made from fat after glycogen runs out.
STORE_ABOVE = 0.8      # blood sugar above this is stored (insulin)...
STORE_RATE = 1.0       # ...into fat at up to this many B per second
STORE_EFFICIENCY = 0.75  # making fat from sugar (~75-80%)
MOBILIZE_BELOW = 0.5   # below this, glycogen is released and fat burned (glucagon)
GLYCOGEN_CAP = 6 * 3600.0     # ~6 h of waking basal (human liver glycogen ~100 g, ~1/4 of a day's basal burn)
GLYCOGEN_FILL_S = 4 * 3600.0  # empty -> full after meals in ~4 h (liver refills within hours; Jentjens & Jeukendrup 2003)
GLYCOGEN_EFFICIENCY = 0.97    # glycogen synthesis costs ~1 of ~32 ATP per glucose
FAT_CAP = 72 * 3600.0         # ~3 days of waking basal: between a mouse (~1/3 of its fat gone in a 14 h fast) and Aedes
                              # aegypti on water alone (median 3-4 days, carbohydrate first, then fat; Briegel et al.)
FAT_MAX_SHARE = 0.5           # fat burns at most ~half the aerobic ceiling ("Fatmax", Achten & Jeukendrup 2003)
KETONE_TAU_S = 12 * 3600.0    # ketosis ramps over ~12 h after glycogen runs out (mice: 12-24 h of fasting)
KETONE_MAX_SHARE = 2.0 / 3.0  # ketones cover at most ~2/3 of a brain's energy (Owen et al. 1967)
# The rest of a brain's need, with no sugar left, is protein: tissue broken
# down, felt as wasting -- the costly structure (Kenyon cells, hidden units,
# receptors) goes first (organism.py). Rebuilt when fed again.
PROTEIN_CAP = 24 * 3600.0     # assumption: a day of waking basal's worth of tissue before it is fully wasted
WASTING_REBUILD_S = 72 * 3600.0  # assumption: lost tissue regrows over ~3 days of being fed
OLD_R_CAP = 21600.0           # the old single "reserve" (6 h), for migrating saved bodies into fat
SLEEP_METABOLISM = 0.3  # B/s while asleep
# Awake: a fixed cost of being awake at all, plus a share that follows its
# tempo (a fast gaze is expensive, a slow one cheap). The floor sits above
# what the reserve can supply (MOBILIZE_RATE), so however slowly it gazes,
# staying awake with nothing to eat still drains it -- sleep is the only
# way to hold steady in an empty room.
WAKE_FLOOR = 0.6
TEMPO_SHARE = 0.4
# Metabolic strategy (genome.metabolism: the resting and sleeping burn's
# multiplier, 0.1 = ectotherm .. 1 = endotherm):
# No floor on metabolic strategy (2026-09-28 audit: the old 0.1 floor was
# binding on two lineages -- design, not physics): a cold, slow animal's brain
# runs slower (neural processing speed scales with metabolic rate; Seebacher),
# so it misses looks (organism.py), and evolution finds its own floor. Only a
# numerical guard remains.
MIN_METABOLISM = 1e-3
# No upper limit but a numerical guard: a body can run hotter than a newborn
# (hummingbirds, shrews), paying a higher resting burn for a higher aerobic
# ceiling -- Bennett & Ruben's aerobic-capacity trade, which is what this
# trait models (endothermy as stamina; warmth came along).
METABOLISM_GUARD = 1e3
AEROBIC_SCOPE = 10.0   # sustained activity up to ~10x resting (vertebrate factorial aerobic scope)
# The three energy systems for activity above the aerobic ceiling: the
# phosphagen first (seconds), then glycolysis (minutes, running a debt).
PHOSPHAGEN_S = 10.0    # ~10 s of maximal effort (twice the ceiling) from the phosphagen (PCr; insects: arginine phosphate)
PHOSPHAGEN_HALF_S = 30.0  # it refills with a ~30 s half-time (Harris et al. 1976), paid aerobically
BURST_S = 120.0        # glycolytic capacity: exhausted in ~2 min (Gastin 2001: crossover ~75 s; lizards, Bennett 1978)
DEBT_TAU_S = 7200.0    # the debt and its lactate clear over hours (crocodiles recover from a struggle in hours; assumption: 2 h)
ANAEROBIC_FUEL_RATIO = 16.0  # glycolysis yields 2 of glucose's ~32 ATP: a burst draws 16x the fuel...
LACTATE_RETURN = 0.93  # ...but the lactate keeps the rest, returned as the debt clears: in recovery ~half burned directly
                       # (~all its energy kept) and ~half rebuilt to glucose via the Cori cycle (6 of ~32 ATP spent, ~87% kept)
                       # (Brooks' lactate shuttle: ~50% oxidized at rest, 75-80% in exercise). Net: a burst costs ~2x aerobic fuel.
SDA_FRACTION = 0.2     # share of each meal spent digesting it (specific dynamic action, in Secor's range; assumption)
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
PREY_FOOD_PER_LOOK = 1.2e-3      # MEAL (legacy units) x prey under the gaze centre, its mouth (0..1: the catch's confidence)
# Feeding is a flow, not a per-look event (Holling's handling time; a
# 2026-09-28 panel): while a host is under its mouth, blood flows into the
# gut at its pump's rate (genome.pump, inherited) x the catch's confidence,
# until the gut is full -- so gazing faster doesn't eat faster, and staying
# on a host pays. A faster pump is more muscle to keep (Sterling & Laughlin):
PUMP_REF = PREY_FOOD_PER_LOOK * LEGACY_UNIT * 15.0  # B/s: the old per-look meal at a look every frame (15/s) -- a newborn's
# Its proboscis is a narrow tube: however strong the pump, flow through a tube
# is bounded by its bore (Poiseuille: flow ~ bore^4). At the inherited bore 1
# a full gut takes ENGORGE_S -- a mosquito engorges in about 1.5 minutes
# (Clements 1992; Chadee & Beier 1995: 1-2.5 min). The bore is inherited
# (genome.bore, born 1); a wider tube is more tissue to keep (upkeep ~ its
# cross-section, bore^2, at the pump's upkeep share). 2026-09-28 audit.
# Protein (2026-09-28 audit; Cahill, Vosshall, Dawkins): blood carries what
# nectar can't -- the protein that makes eggs. A protein store (0..1, one
# gut-full of blood = a full clutch's worth) fills only from blood and is
# spent over a gonotrophic cycle (blood meal to eggs: ~3 days, Clements 1992;
# 2-4 days). Keeping it up is part of its drive, weighted like its fat stores;
# it feels it (a brain input). Born full (teneral reserves).
GONOTROPHIC_S = 3 * 24 * 3600.0
ENGORGE_S = 90.0
TUBE_FLOW = GUT_CAP / ENGORGE_S  # B/s through the reference bore
PUMP_UPKEEP_SHARE = 0.05  # assumption: a pump at PUMP_REF costs 5% of the resting burn (the brain's measured share, Mink 1981)
# Sleep pressure (Process S): the two-process model's fitted time constants
# (Daan, Beersma & Borbely 1984) -- rising with time awake, x brain load here.
S_RISE_S = 18.2 * 3600.0
S_FALL_S = 4.2 * 3600.0
SLEEP_SETTLE_S = 10.0   # falling asleep: no clearing yet
SLEEP_INERTIA_S = 7.0   # waking up: groggy, can't eat yet
COLLAPSE_S = 0.95       # sleep pressure that forces sleep...
COLLAPSE_RELEASE_S = 0.8  # ...and holds it until pressure is back below this
HUNGER_WAKE_R = 0.05    # of the old 6 h reserve: fat below this (~18 min) counts as empty (with blood sugar under EMPTY_G and no glycogen: degraded)
# What wakes it from sleep: a big change in the field, at these levels x its
# inherited vigilance's inverse (genome.vigilance: 1 = today's thresholds).
WAKE_LOOM, WAKE_MOTION = 0.18, 0.6
EMPTY_G = 0.1           # below this the body degrades
# Sleep needs sleep pressure (Borbely's two thresholds, both lowered by the
# dark -- the circadian part, from its sense of the field's light): it can
# fall asleep only above the upper one and wakes by itself below the lower
# one. A rested animal can't sleep through a busy room, and sleep ends once
# it has done its work.
SLEEP_ONSET_DARK, SLEEP_ONSET_DAY = 0.15, 0.45
SLEEP_END_DARK, SLEEP_END_DAY = 0.01, 0.04
NIGHT_LIGHT, DAY_LIGHT = 0.15, 0.5  # field light (~20 min average) that counts as night / day
# Its clock (Process C; a 2026-09-29 panel -- Borbely, Siegel, Nesse): an
# internal phase that runs a day long (Earth's: physics) whatever it sees, and
# that light sets, as a zeitgeber, by at most about an hour a day (the size of
# real phase shifts: human phase-response curves, Czeisler). The sleep
# thresholds follow the clock, not the moment's light. And in its night, once
# sleep pressure reaches the night threshold in a quiet moment, it falls asleep
# whatever its brain prefers -- sleep pays over hours, and a choice judged over
# minutes never picks it (the Tanzania lineage never slept in 48 h): a gate, like
# collapse, is the body plan's answer. Aedes, its model, sleeps at night.
# Its life history (a 2026-09-29 panel -- Kooijman's Dynamic Energy Budget
# theory, Pearl's rate of living, Kirkwood, Stearns, Charnov, Gelman):
# - the kappa rule: of what it assimilates (what digestion moves into its blood,
#   less the cost of digesting it), a share kappa (genome.kappa) goes to its body
#   as before and 1 - kappa into a reproduction buffer (repro), from which it
#   lays eggs; stores mobilised while starving never feed it
# - an egg costs what a newborn is made of: a fresh body's stores
#   (newborn_energy()), made from sugar at STORE_EFFICIENCY
# - ageing: a share of all the energy it burns damages it (DAMAGE_FRACTION,
#   the share of the respiratory chain's electrons that leak to superoxide:
#   0.15%, St-Pierre, Buckingham, Roebuck & Brand 2002, J Biol Chem 277:44784);
#   it dies of age when that damage equals its own tissue (PROTEIN_CAP, the
#   same tissue wasting breaks down) -- a faster metabolism ages it faster
# - it dies of starvation when wasting reaches its ceiling (1: all that tissue)
DAMAGE_FRACTION = 0.0015
# Torpor (a 2026-09-30 panel -- Heller, Geiser, Nesse, Gelman, Sterling &
# Laughlin): when its eyes get no world at all (a dead camera: black, blank or
# frozen frames), it hibernates -- burning TORPOR_SHARE of what it would
# (hibernators cut metabolism to below 5% of basal: Geiser 2004), ageing as
# slowly (its damage is a share of what it burns), and never wasting: torpor
# can't starve it to death. A real, food-poor world still can.
TORPOR_SHARE = 0.05
# A developed body's cyst (a 2026-09-30 panel -- Storey, Clegg, Hand, Boothby,
# Jonsson, Gelman): dormancy is a prepared capacity with a cost, not a reward
# for being accomplished. To encyst it makes a protective sugar of
# PROTECTANT_SHARE of its body (brine shrimp cysts are ~15% trehalose by dry
# weight: Clegg 1962), from its sugar and glycogen only (as Artemia makes it
# from glycogen), at STORE_EFFICIENCY; encysted, nothing runs; at revival the
# sugar returns to its glycogen, as a cyst's trehalose is burned on waking.
PROTECTANT_SHARE = 0.15
CIRC_PERIOD_S = 24 * 3600.0
CIRC_SHIFT_RATE = (2 * math.pi / 24.0) / (24 * 3600.0)  # rad/s of phase at most: one hour a day


def _clamp(v: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, v))


@dataclass
class MosquitoState:
    energy: float = 1.0        # blood sugar [0, 1] (was: the single energy store)
    gut: float = 0.0           # undigested food [0, 1]
    reserve: float = 0.04      # fat [0, 1] of FAT_CAP (triglycerides: dense, slow, aerobic only)
    glycogen: float = 0.0      # glycogen [0, 1] of GLYCOGEN_CAP (fast in, fast out)
    phosphagen: float = 1.0    # the phosphagen store [0, 1] (seconds of burst)
    lactate: float = 0.0       # B-s of lactate from bursts, returned as the debt clears
    ketone: float = 0.0        # ketosis [0, 1]: how much of the brain's need fat can cover (x KETONE_MAX_SHARE)
    wasting: float = 0.0       # tissue broken down for a brain with no sugar [0, 1]
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
    debt: float = 0.0          # anaerobic debt (0..1 = exhausted), felt as fatigue, repaid over hours
    metabolism: float = 1.0    # its inherited metabolic strategy (genome.metabolism; set by the organism, not saved)
    pump: float = PUMP_REF     # its feeding pump's rate, B/s (genome.pump; set by the organism, not saved)
    bore: float = 1.0          # its proboscis's bore, relative (genome.bore; set by the organism, not saved)
    protein: float = 1.0       # protein from blood, for eggs (0..1 of a clutch); born full, spent over a gonotrophic cycle
    mobilize: float = MOBILIZE_BELOW  # its fasted line (glucagon), inherited (genome.mobilize; set by the organism, not saved)
    store: float = STORE_ABOVE        # its storage line (insulin -> fat), inherited (genome.store; set by the organism, not saved)
    bite_blood: float = 0.0    # blood (legacy units) taken in the bite going on now: what a swat takes back
    sleep_mismatch: float = 0.0  # the field's mismatch when it fell asleep (or lowest since): the room it fell asleep in
    woke_by: str = ""          # what ended its last sleep (hourly metrics): mismatch, loom, motion, rested, choice
    circ_phase: float = -1.0   # its clock's phase (radians, 0 = its midday); -1 until it first sees light
    repro: float = 0.0         # its reproduction buffer, B (the kappa rule's 1 - kappa of what it assimilates)
    damage: float = 0.0        # the damage its burning has done, B (DAMAGE_FRACTION of all it burned); age = damage / PROTEIN_CAP
    kappa: float = 1.0         # its share of assimilation kept for its body (genome.kappa; set by the organism, not saved)
    torpid: bool = False       # hibernating: its eyes get no world (set by its live body, not saved)
    encysted: float = 0.0      # 1 while in its cyst (a developed body's dormancy; saved: a restart keeps it)
    protectant: float = 0.0    # B of protective sugar its cyst holds (returned at revival)
    burned: float = 0.0        # B burned so far (a running count, for its energy balance; not saved)
    assimilated: float = 0.0   # B assimilated so far (the same)

    # ---- its life history ----------------------------------------------
    @property
    def age(self) -> float:
        """How much of its tissue its burning has damaged (0..1; 1 = dead of age)."""
        return self.damage / PROTEIN_CAP

    @property
    def sugar(self) -> float:
        """What it can make its cyst's sugar from: blood sugar and glycogen, B."""
        return self.energy * G_CAP + self.glycogen * GLYCOGEN_CAP

    def encyst(self, cost: float) -> None:
        """Into its cyst: cost B of protective sugar made from its sugar (glycogen first)."""
        need = cost / STORE_EFFICIENCY
        gly = min(self.glycogen * GLYCOGEN_CAP, need)
        self.glycogen = _clamp(self.glycogen - gly / GLYCOGEN_CAP)
        self.energy = _clamp(self.energy - (need - gly) / G_CAP)
        self.protectant, self.encysted = cost, 1.0

    def revive(self) -> None:
        """Out of its cyst: its protective sugar back into its glycogen (what overflows, into its blood)."""
        gly_bs = self.glycogen * GLYCOGEN_CAP + self.protectant
        self.glycogen = _clamp(gly_bs / GLYCOGEN_CAP)
        self.energy = _clamp(self.energy + max(0.0, gly_bs - GLYCOGEN_CAP) / G_CAP)
        self.protectant, self.encysted = 0.0, 0.0

    def death(self) -> str | None:
        """What it died of, or None: starvation (wasting at its ceiling) or age."""
        if self.wasting >= 1.0:
            return "starvation"
        if self.damage >= PROTEIN_CAP:
            return "age"
        return None

    # ---- what the organism "feels" ------------------------------------
    @property
    def tiredness(self) -> float:
        """Muscle fatigue plus anaerobic debt, as it feels them."""
        return _clamp(self.fatigue + self.debt)

    @property
    def stores(self) -> float:
        """Glycogen and fat together, on the old reserve's 6 h scale (so the
        drive and the brain's "reserve" sense keep their meaning)."""
        return _clamp((self.glycogen * GLYCOGEN_CAP + self.reserve * FAT_CAP) / OLD_R_CAP)

    def drive(self) -> float:
        """Distance from a viable state. Hunger counts the gut (a full
        stomach cuts hunger before absorption, like ghrelin) and a small
        stores term, so building stores shows up within a window; wasting
        (tissue burned for a brain with no sugar) is felt like fatigue."""
        fed = min(1.0, self.energy + 0.5 * self.gut)
        return ((1.0 - fed) ** 2 + 0.3 * (1.0 - self.stores) ** 2 + 0.3 * (1.0 - self.protein) ** 2
                + self.threat ** 2 + self.tiredness ** 2 + 0.3 * self.sleep_pressure ** 2 + self.wasting ** 2)

    @property
    def degraded(self) -> bool:
        # Truly empty: blood sugar, glycogen AND fat. A hungry body with food
        # in store forages at full strength (degrading it made hunger a trap:
        # a narrow, slow eye catches less).
        return (self.energy < EMPTY_G and self.glycogen < 0.01
                and self.reserve * FAT_CAP < HUNGER_WAKE_R * OLD_R_CAP)

    @property
    def light_trend(self) -> float:
        return max(-1.0, min(1.0, 4.0 * (self.light_fast - self.light_slow)))

    @property
    def daylight(self) -> float:
        return _clamp((self.light_slow - NIGHT_LIGHT) / (DAY_LIGHT - NIGHT_LIGHT))

    @property
    def circ_day(self) -> float:
        """Its clock's day (1 midday, 0 midnight); its light's until the clock is set."""
        return self.daylight if self.circ_phase < 0 else 0.5 * (1.0 + math.cos(self.circ_phase))

    @property
    def efficiency(self) -> float:
        return 1.0 - TIRED_EFFICIENCY * self.sleep_pressure

    @property
    def can_eat(self) -> bool:
        return self.asleep < 0.5 and self.sleep_clock >= SLEEP_INERTIA_S

    # ---- sleep --------------------------------------------------------
    def big_change(self, loom: float, field_motion: float, vigilance: float = 1.0) -> bool:
        """A change in the field big enough to wake it (or cut short a
        replay): its inherited vigilance lowers or raises the bar."""
        v = max(1e-6, vigilance)
        return loom > WAKE_LOOM / v or field_motion > WAKE_MOTION / v

    def set_sleep(self, wants_sleep: bool, loom: float = 0.0, field_motion: float = 0.0,
                  vigilance: float = 1.0, mismatch: float = 0.0, host: bool = False, host_arousal: float = 0.0) -> None:
        """The brain's choice, with the body's override (collapse) and a raised
        arousal threshold while asleep (only a big change wakes it -- or the
        room no longer being the room it fell asleep in)."""
        asleep = self.asleep >= 0.5
        want = wants_sleep
        cause = "choice"
        day = self.circ_day  # the clock's day, not the moment's light (Process C)
        onset = SLEEP_ONSET_DARK + (SLEEP_ONSET_DAY - SLEEP_ONSET_DARK) * day
        # a host in view (jing) holds sleep off as far as its inherited host
        # arousal says: the onset rises toward the pressure that collapses it
        # (0: no change; 1: not until it nearly collapses) -- and wakes it below that
        if host and host_arousal > 0.0:
            onset = onset + host_arousal * (COLLAPSE_S - onset)
            if asleep and self.sleep_pressure < onset and self.sleep_pressure <= COLLAPSE_RELEASE_S:
                want, cause = False, "host"
        if want and not asleep and self.sleep_pressure < onset:
            want = False  # not tired enough to fall asleep
        v = max(1e-6, vigilance)
        if not asleep and day < 0.5 and self.sleep_pressure >= onset and loom <= WAKE_LOOM / v and field_motion <= WAKE_MOTION / v:
            want = True  # its night, tired enough, a quiet moment: the gate
        if asleep and day < 0.5 and cause != "host":
            want = True  # and it holds: in its night it wakes rested or disturbed, not by choice
        if asleep and self.sleep_pressure < SLEEP_END_DARK + (SLEEP_END_DAY - SLEEP_END_DARK) * day:
            want, cause = False, "rested"  # slept enough: wakes by itself
        # Exhaustion: collapse, and no waking by choice until it has recovered.
        if (not asleep and self.sleep_pressure > COLLAPSE_S) or (asleep and self.sleep_pressure > COLLAPSE_RELEASE_S):
            want = True
        # What wakes even an exhausted animal: a big change -- the sentry's
        # sensors stay on while it sleeps.
        if asleep:
            # The orienting reflex (Sokolov): its field's mismatch with its
            # slow model of the room, against what it was when it fell asleep
            # (following it down as it habituates) -- someone who came in
            # quietly, and is still there, wakes it; someone who was already
            # there when it dropped off doesn't. At the looming line.
            self.sleep_mismatch = min(self.sleep_mismatch, mismatch)
            if loom > WAKE_LOOM / v:
                want, cause = False, "loom"
            elif field_motion > WAKE_MOTION / v:
                want, cause = False, "motion"
            elif mismatch - self.sleep_mismatch > WAKE_LOOM / v:
                want, cause = False, "mismatch"
        if want != asleep:
            self.asleep = 1.0 if want else 0.0
            self.sleep_clock = 0.0
            if want:
                self.sleep_mismatch = mismatch
            else:
                self.woke_by = cause

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
        self.protein = max(0.0, self.protein - seconds / GONOTROPHIC_S)  # eggs made from it
        self.sleep_clock += seconds

        def leak(old: float, decay: float, target: float) -> float:
            k = decay ** dt
            return _clamp(k * old + (1.0 - k) * target)

        self.arousal = leak(self.arousal, 0.92, 0.0 if asleep else motion)
        if field_light is not None:
            self.light_fast += (1.0 - math.exp(-seconds / LIGHT_FAST_S)) * (field_light - self.light_fast)
            self.light_slow += (1.0 - math.exp(-seconds / LIGHT_SLOW_S)) * (field_light - self.light_slow)
            if self.circ_phase < 0:  # a founder's clock, set from the light in view now (its slow average starts at a guess)
                now = _clamp((field_light - NIGHT_LIGHT) / (DAY_LIGHT - NIGHT_LIGHT))
                self.circ_phase = math.acos(2.0 * now - 1.0)
            # light entrains it: the phase moves to bring its day toward the
            # light's (gradient of the squared difference), at most the rate above
            err = self.daylight - self.circ_day
            self.circ_phase -= CIRC_SHIFT_RATE * seconds * err * math.sin(self.circ_phase)
        if self.circ_phase >= 0:
            self.circ_phase = (self.circ_phase + 2 * math.pi * seconds / CIRC_PERIOD_S) % (2 * math.pi)
        self.threat = leak(self.threat, 0.85, loom)

        # metabolic rate acclimatizes toward the current tempo
        k = 0.5 ** (seconds / ACCLIMATIZE_HALF_LIFE_S)
        self.metabolic_rate = k * self.metabolic_rate + (1.0 - k) * (1.0 / max(1, pace))

        # --- spending (B-seconds): the three energy systems -------------------
        # Resting and sleeping burn follow its metabolic strategy; activity
        # (tempo, muscle) costs the same per unit for all (assumption: equal
        # cost of moving) -- what differs is how much it can SUSTAIN. Up to
        # AEROBIC_SCOPE x its resting rate it is aerobic; beyond that the
        # phosphagen pays first (seconds), then glycolysis: 16x the fuel,
        # most of it parked as lactate and returned as the debt clears, and
        # a debt felt as fatigue. Sensing and thinking (aperture_cost) are
        # the brain's: sugar or, after glycogen runs out, ketones -- never fat.
        m = min(METABOLISM_GUARD, max(MIN_METABOLISM, self.metabolism))
        ceiling = AEROBIC_SCOPE * WAKE_FLOOR * m  # B/s it can sustain aerobically
        muscle = EFFORT_COST * motor_effort * LEGACY_UNIT  # B-seconds this gaze
        fade = 1.0 - math.exp(-seconds / DEBT_TAU_S)
        self.debt *= 1.0 - fade
        back = self.lactate * fade
        self.lactate -= back
        if asleep:
            rest, activity = SLEEP_METABOLISM * m * seconds, muscle
        else:
            rest = WAKE_FLOOR * m * seconds
            activity = TEMPO_SHARE * (1.0 + self.arousal) * self.metabolic_rate * seconds + muscle
        aerobic_act = min(activity, ceiling * seconds)
        excess = activity - aerobic_act
        phos_cap = PHOSPHAGEN_S * ceiling
        phos = self.phosphagen * phos_cap
        from_phos = min(excess, phos)
        phos -= from_phos
        excess -= from_phos
        # the phosphagen refills from spare aerobic capacity (~30 s half-time)
        refill = min((phos_cap - phos) * (1.0 - 0.5 ** (seconds / PHOSPHAGEN_HALF_S)),
                     max(0.0, ceiling * seconds - aerobic_act))
        phos += refill
        glyco_fuel = excess * ANAEROBIC_FUEL_RATIO
        self.lactate += excess * (ANAEROBIC_FUEL_RATIO - 1.0)
        self.debt = _clamp(self.debt + excess / (ceiling * BURST_S))
        # its feeding pump's upkeep (muscle: scales with its metabolic strategy)
        aerobic_need = (rest + aerobic_act + refill
                        + PUMP_UPKEEP_SHARE * WAKE_FLOOR * m * (self.pump / PUMP_REF + self.bore ** 2) * seconds)
        brain = aperture_cost * LEGACY_UNIT
        if self.degraded:  # soft floor: an empty body runs on less
            f = 0.5 + 0.5 * self.energy / EMPTY_G
            aerobic_need, brain, glyco_fuel = aerobic_need * f, brain * f, glyco_fuel * f

        if self.torpid:  # hibernating: a sliver of its burn
            aerobic_need, brain, glyco_fuel = aerobic_need * TORPOR_SHARE, brain * TORPOR_SHARE, glyco_fuel * TORPOR_SHARE
        # the damage of burning (its ageing): a share of everything it burns this step
        self.damage += DAMAGE_FRACTION * (aerobic_need + brain + glyco_fuel)
        self.burned += aerobic_need + brain + glyco_fuel
        # --- digestion: gut -> blood sugar (costs part of the meal: SDA) ---
        gut_bs = self.gut * GUT_CAP
        moved = gut_bs * (1.0 - math.exp(-seconds / DIGEST_TAU_S))
        gut_bs -= moved
        assimilated = moved * (1.0 - SDA_FRACTION)
        self.assimilated += assimilated
        self.repro += (1.0 - self.kappa) * assimilated  # the kappa rule: only what it assimilates feeds its eggs
        g_bs = self.energy * G_CAP + self.kappa * assimilated + back * LACTATE_RETURN
        gly_bs, fat_bs = self.glycogen * GLYCOGEN_CAP, self.reserve * FAT_CAP

        # --- fasted (glucagon): fat pays aerobic work directly, up to its limit.
        # At the line counts as fasted: glycogen holds blood sugar there, and
        # fat must do the work meanwhile (strictly below, fat never burned while
        # any glycogen was left -- glycogen paid the whole body) ---
        if g_bs <= self.mobilize * G_CAP + 1e-9:
            from_fat = min(fat_bs, FAT_MAX_SHARE * ceiling * seconds, aerobic_need)
            fat_bs -= from_fat
            aerobic_need -= from_fat
        # ketones (made from fat once glycogen is gone) cover part of the brain
        from_ket = min(fat_bs, brain * self.ketone * KETONE_MAX_SHARE)
        fat_bs -= from_ket
        brain -= from_ket
        g_bs -= aerobic_need + brain + glyco_fuel
        # glucagon / adrenaline: glycogen tops blood sugar back up -- fast
        # enough for waking at the aerobic ceiling, plus a burst's fuel
        if g_bs < self.mobilize * G_CAP:
            release = min(gly_bs, self.mobilize * G_CAP - g_bs, ceiling * seconds + glyco_fuel)
            gly_bs -= release
            g_bs += release
        # what sugar still can't pay is paid by the body's own tissue
        if g_bs < 0.0:
            if not self.torpid:  # torpor never breaks down its tissue: what its stores can't pay, it goes without
                self.wasting = _clamp(self.wasting - g_bs / PROTEIN_CAP)
            g_bs = 0.0
        elif g_bs > self.mobilize * G_CAP:
            self.wasting *= math.exp(-seconds / WASTING_REBUILD_S)  # fed: tissue regrows
        # --- fed (insulin): glycogen fills as soon as it is fed (above the
        # fasted line), so a body climbing out of a deficit banks its first
        # meals there; fat is made from sugar only from a real surplus ---
        if g_bs > self.mobilize * G_CAP:
            to_gly = min(g_bs - self.mobilize * G_CAP, GLYCOGEN_CAP / GLYCOGEN_FILL_S * seconds,
                         (GLYCOGEN_CAP - gly_bs) / GLYCOGEN_EFFICIENCY)
            g_bs -= to_gly
            gly_bs += to_gly * GLYCOGEN_EFFICIENCY
        if g_bs > self.store * G_CAP:
            to_fat = min(g_bs - self.store * G_CAP, STORE_RATE * seconds)
            g_bs -= to_fat
            fat_bs += to_fat * STORE_EFFICIENCY
        # ketosis ramps while glycogen is empty, fades once it is back
        target = 1.0 if gly_bs < 0.01 * GLYCOGEN_CAP else 0.0
        self.ketone += (1.0 - math.exp(-seconds / KETONE_TAU_S)) * (target - self.ketone)

        self.energy = _clamp(g_bs / G_CAP)
        self.gut = _clamp(gut_bs / GUT_CAP)
        self.glycogen = _clamp(gly_bs / GLYCOGEN_CAP)
        self.reserve = _clamp(fat_bs / FAT_CAP)
        self.phosphagen = _clamp(phos / phos_cap) if phos_cap > 0 else 1.0

        # --- sleep pressure (Process S) ---
        if asleep:
            if self.sleep_clock > SLEEP_SETTLE_S:
                before = self.sleep_pressure
                self.sleep_pressure = self.sleep_pressure * math.exp(-seconds / S_FALL_S)
                self.cleared += before - self.sleep_pressure
        else:
            load = 0.5 + 0.5 * _clamp(self.metabolic_rate)
            self.sleep_pressure = 1.0 - (1.0 - self.sleep_pressure) * math.exp(-seconds * load / S_RISE_S)

        # Muscle fatigue recovers with rest, twice as fast asleep.
        self.fatigue = _clamp((0.9 if asleep else 0.95) ** dt * self.fatigue + 0.08 * motor_effort)
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
            self._swallow(FOOD_PER_LOOK * max(0.0, tracking_quality) * self.efficiency)  # its gut bounds it (ram feeding can pass one look's worth)
        # Curiosity rises with real time, falls with what it took in.
        self.curiosity = _clamp(self.curiosity + 0.01 * getattr(self, "_dt", 1) - 0.25 * _clamp(tracking_quality))

    @property
    def flow(self) -> float:
        """What reaches its gut, B/s: its pump, but never more than its tube carries."""
        return min(self.pump, TUBE_FLOW * self.bore ** 4)

    def feed_host(self, confidence: float, seconds: float) -> None:
        """A bite as a flow: a host under its mouth for `seconds`, blood at
        its pump's rate x the catch's confidence (awake, past grogginess)."""
        if self.can_eat:
            before = self.gut
            self._swallow(self.flow * _clamp(confidence) * max(0.0, seconds) * self.efficiency / LEGACY_UNIT)
            self.bite_blood += (self.gut - before) * GUT_CAP / LEGACY_UNIT
            self.protein = _clamp(self.protein + (self.gut - before))  # a gut-full of blood is a clutch's protein

    def swat(self) -> float:
        """Host defense (a 2026-09-28 panel): the host it is biting came at
        it, and this bite's blood is lost -- a blood-full mosquito flies slower
        and is hit more (Roitberg et al. 2003), so staying longer risks more
        (Lima & Dill 1990). Returns what was lost (legacy units)."""
        lost = min(self.bite_blood, self.gut * GUT_CAP / LEGACY_UNIT)
        self.gut = _clamp(self.gut - lost * LEGACY_UNIT / GUT_CAP)
        self.bite_blood = 0.0
        return lost

    def feed_nectar(self, legacy_amount: float) -> float:
        """A sip of nectar (sugar) into the gut; returns what it took (legacy
        units) -- a full gut takes less, and the plant loses only that."""
        if not self.can_eat:
            return 0.0
        before = self.gut
        self._swallow(legacy_amount * self.efficiency)
        return (self.gut - before) * GUT_CAP / LEGACY_UNIT

    def bite_over(self) -> None:
        """The host left its mouth (or it left the host): the bite is over."""
        self.bite_blood = 0.0

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
              "search", "fatigue", "hunger", "curiosity", "metabolic_rate", "light_fast", "light_slow", "debt",
              "glycogen", "phosphagen", "lactate", "ketone", "wasting", "sleep_mismatch", "protein", "circ_phase",
              "repro", "damage", "encysted", "protectant")

    @classmethod
    def from_dict(cls, data: dict) -> "MosquitoState":
        s = cls(**{k: float(data[k]) for k in cls.FIELDS if k in data})
        if "glycogen" not in data and "reserve" in data:
            # A body saved before the fuel stores (2026-09-28): its single 6 h
            # "reserve" becomes fat, keeping its energy; glycogen starts empty.
            s.reserve = _clamp(float(data["reserve"]) * OLD_R_CAP / FAT_CAP)
        return s

    def to_dict(self) -> dict[str, float]:
        return {k: round(float(getattr(self, k)), 6) for k in self.FIELDS}


def newborn_energy() -> float:
    """What a newborn is made of: a fresh body's stores, B (blood sugar, glycogen, fat)."""
    b = MosquitoState()
    return b.energy * G_CAP + b.glycogen * GLYCOGEN_CAP + b.reserve * FAT_CAP


EGG_COST = newborn_energy() / STORE_EFFICIENCY  # an egg: a newborn's stores, made from sugar

CYST_COST = PROTECTANT_SHARE * PROTEIN_CAP  # B of protective sugar in its cyst: 15% of its body (its tissue)
