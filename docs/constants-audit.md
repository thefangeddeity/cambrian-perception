# Constants audit (2026-09-26)

Every number the organism, its world and its evolution run on, sorted by
where it comes from (after a design panel; Dennett's rule). A constant
should be one of:

- **P, physics:** a property of its simulated world chosen on purpose and
  documented, the way a game needs gravity.
- **S, standard:** an external standard, such as a published measurement,
  a model's shipped defaults or a protocol's convention.
- **M, measured:** measured from the organism's own data or its host.
- **B, bound:** a safety bound, a structural size or an engineering budget.
  It doesn't shape behaviour, or only caps what a price already limits.

Anything else is a **G, guess**. Guesses are flagged here and replaced as
yardsticks are found.

## Changed in this audit

| Constant | Was | Now | Basis |
|---|---|---|---|
| `RECEPTOR_COST` (organism) | 21% of resting burn for a newborn eye | 8% | S: blowfly photoreceptors' share of resting metabolic rate (Laughlin, de Ruyter van Steveninck & Anderson 1998) |
| `THINK_COST` (organism) | 36% of resting burn for a newborn brain | 5% | S: the CNS takes 2-8% of body metabolism across vertebrates, ~20% in humans (Mink, Blumenschine & Adams 1981); midpoint |
| `CONE_COST` (organism) | old colour price / 144 | = `RECEPTOR_COST` per channel | P: a colour channel is one more signal per cone, priced like a receptor's signal |
| brain deadline (organism) | none | a brain slower than its next look misses it | M: multiply-adds x this host's measured time per multiply-add (`hostspeed.py`, the best of its run) vs the look's interval. Speed is time, not price: the energy price stays the granted CPU share (a panel vote to price by speed, 6-2-1, was reversed after it starved the slow laptop -- a snail's neurons aren't dearer than a fly's, its world is slower; Healy et al. 2013) |
| drive reduction (run_vision) | x 1200 s / window length | the window's own drive change | Removed an extrapolation that blew seconds of noise up into the largest fitness term |
| `S_RISE_S`, `S_FALL_S` (state) | 14 h, 3 h | 18.2 h, 4.2 h | S: two-process model fit (Daan, Beersma & Borbely 1984) |
| meal / snack grouping (viewer) | 1 s gap, snack > 0.05 | bout criterion from its own gaps | M: `bouts.py` (Sibly, Nott & Fletcher 1990), item D |
| receptor size, counts (retina, fovea) | 12x12 stretched over any gaze | fixed-size square receptors, evolvable count | P, documented, item C |

## Everything, by module

### fishbowl/organism.py

| Constant | Value | Kind | Notes |
|---|---|---|---|
| `REFERENCE_QUOTA_PCT` | 150 | P | the anchor quota (the original service cap) |
| `REFERENCE_GAZES_PER_S` | 15 | P | the reference pace the shares are stated at |
| `EYE_SHARE`, `BRAIN_SHARE` | 0.08, 0.05 | S | see above |
| `CHANNEL_COST`, `PREY_SENSE_COST` | = `THINK_COST` per unit weight | **G** | relative to thinking; a per-synapse yardstick is still missing |
| `STABILIZER_COST` | 0.5 x `THINK_COST` | **G** | same |
| `CONSOLIDATE_RATE` | 3.0 | **G** | proposed to become measured with the sleep redefinition (awaiting a go) |
| `TEMPO_RANGE`, `MAX_INTERVAL` | 3, 12 | **G** | how far tempo can swing |
| `MOTION_GAIN`, `FLOW_GAIN` | 10, 20 | **G** | calibrated on the 12x12 eye; the finer receptors changed their scale. Candidate: divisive normalisation (Carandini & Heeger 2012), measured from its own signal |
| `PERIPH_MOTION_GAIN`, `EXPANSION_GAIN` | 300, 10 | M (dated) | calibrated on a real camera and synthetic looming; recalibrate per host (same candidate) |
| `MEM_H`, `MEM_W` | 24, 32 | B | surprise-memory resolution |
| `UNSEEN_NOVELTY`, `FOOD_GAIN` | 0.25, 400 | **G** | tuned on synthetic scenes |
| `SURPRISE_SIGMAS` | 4 | P | a 4-sigma surprise threshold |
| `NOISE_FLOOR` | 0.02 | **G** | should be measured: the camera's own temporal noise per receptor |
| `MEAN_RATE`, `VAR_RATE` | 0.1, 0.05 | **G** | habituation speeds |
| `SHIFT_WIDTH`, `SHIFT_MIN_RESPONSE`, `SHIFT_MAX` | 160, 0.2, 0.08 | B / **G** | phase-correlation settings for the stabilizer |

### fishbowl/state.py (the body)

| Constant | Value | Kind | Notes |
|---|---|---|---|
| `G_CAP`, `GUT_CAP`, `R_CAP` | 10 min, 20 min, 6 h of burn | P | the body's stores |
| `DIGEST_TAU_S` | 240 s | P | |
| `STORE_ABOVE`, `STORE_RATE`, `MOBILIZE_BELOW`, `MOBILIZE_RATE` | 0.8, 1.0, 0.5, 0.5 | P | chosen so the reserve can carry sleep but not waking |
| `STORE_EFFICIENCY` | 0.75 | P | order of real fat-storage efficiency |
| `SLEEP_METABOLISM`, `WAKE_FLOOR`, `TEMPO_SHARE` | 0.3, 0.6, 0.4 B/s | P | |
| `LIGHT_FAST_S`, `LIGHT_SLOW_S` | 60 s, 1200 s | P | its day/night sense |
| `TIRED_EFFICIENCY` | 0.5 | **G** | the invented cost of skipping sleep; the proposed sleep redefinition would retire it (awaiting a go) |
| `ACCLIMATIZE_HALF_LIFE_S` | 80 s | **G** | |
| `EFFORT_COST`, `FOOD_PER_LOOK`, `PREY_FOOD_PER_LOOK` | 1e-4, 2e-4, 1.2e-3 | P | prices and food values of its world |
| `S_RISE_S`, `S_FALL_S` | 18.2 h, 4.2 h | S | Daan, Beersma & Borbely 1984 |
| `SLEEP_SETTLE_S`, `SLEEP_INERTIA_S` | 10 s, 7 s | **G** | |
| `COLLAPSE_S`, `COLLAPSE_RELEASE_S` | 0.95, 0.8 | **G** | safety net |
| `HUNGER_WAKE_G`, `HUNGER_WAKE_R`, `EMPTY_G` | 0.15, 0.05, 0.1 | P | |
| `SLEEP_ONSET_*`, `SLEEP_END_*`, `NIGHT_LIGHT`, `DAY_LIGHT` | 0.15/0.45, 0.01/0.04, 0.15/0.5 | **G** | Borbely's two thresholds; published threshold values are for humans in normalised units, not yet mapped |

### fishbowl/fovea.py, retina.py (the eye)

| Constant | Value | Kind | Notes |
|---|---|---|---|
| `RECEPTOR_PITCH` | 1/64 of frame height | P | documented in fovea.py |
| `DEFAULT_RECEPTORS`, `MIN_RECEPTORS`, `MAX_RECEPTORS` | 22, 4, 38 | P / B / M (dated) | MAX from the zoom eye's measured collapse past 0.6 |
| `FIELD_RECEPTORS` | 144 | P | the old whole-field count, now square |
| `DAMPING`, `FORCE_GAIN`, `SPRING` | 0.5, 0.2, 0.08 | P | overdamped oculomotor plant (Robinson); saccade speed and hold cost documented |

### fishbowl/controller.py, genome.py, blocks.py (brain and search)

| Constant | Value | Kind | Notes |
|---|---|---|---|
| `HIDDEN`, `TREE_HIDDEN` | 16, 16 | P | a newborn's size; the tree's view |
| `MIN_HIDDEN`, `MAX_HIDDEN`, `MAX_LAYERS` | 4, 256, 16 | B | price limits growth |
| `MAX_CHANNELS` | 4 | **G** | a hand cap on loops; could become a bound only, as for units |
| `HEAVY_TAIL_P`, `HEAVY_TAIL_SCALE`, `HEAVY_TAIL_MAX` | 0.1, 0.1, 2.0 | **G** | heavy-tailed mutations are grounded (the distribution of fitness effects is leptokurtic, Eyre-Walker & Keightley 2007); the mix is not |
| `STABILIZER_SIGMA` | 0.1 | **G** | |
| `MIN_PACE`, `MAX_PACE` | 1, 6 | **G** | |
| `BRAIN_FLOOR` | 0.3 | **G** | an audit fix against operator starvation |
| `OP_SUCCESS_EMA_ALPHA` | 0.05 | **G** | |
| `META_DECAY` | 0.9995 | M | validated on the synthetic task (15000 generations x 12 seeds) |
| `MAX_CONST` | 5 | B | the tree language's bound |

### fishbowl/mushroom.py (lifetime reward learning)

| Constant | Value | Kind | Notes |
|---|---|---|---|
| `KC_INPUTS` | 7 | S | inputs per Kenyon cell (Caron et al. 2013) |
| `KC_ACTIVE` | 5% | S | share of Kenyon cells responding (Turner et al. 2008; Honegger et al. 2011) |
| `MAX_KC` | 4096 | B | the price limits it |
| `KC_STEP` (genome) | 64 | P | cells added or removed per mutation |
| `LEARNING_MIN`, `LEARNING_MAX`, `LEARNING_SIGMA` (genome) | 0.001, 1, 0.5 | P / M | the maximum is principled: above 1, one update overshoots its own prediction error |
| reward scale | 1 = a full look at prey | P | the energy it ate, in meals |

### fishbowl/prey.py, bouts.py, video_source.py

| Constant | Value | Kind | Notes |
|---|---|---|---|
| `MIN_CONFIDENCE`, `NMS_IOU`, `LETTERBOX_GREY`, `INPUT_SIZE` | 0.25, 0.7, 114, 640 | S | Ultralytics defaults / the export |
| `MAX_GAPS`, `EM_ITERATIONS`, `EM_TOLERANCE` | 5000, 200, 1e-9 | B | numerics |
| `DEFAULT_MAX_DIM` | 320 | B | processing budget; it also caps acuity (~2.8 px per receptor) |
| `FRAME_RING`, `SLOW_SOURCE_FPS`, `DETECT_INTERVAL_S` | 1200, 12, 0.3 | B | the detector's duty cycle is measured (>= 3x its own time) |

### run_vision.py (fitness and the loop)

| Constant | Value | Kind | Notes |
|---|---|---|---|
| `HOMEOSTASIS_WEIGHT`, `DRIVE_REDUCTION_WEIGHT` | 3, 1 | **G** | the body is the one non-arbitrary part of fitness (viability); its weights are not |
| `TEACHER_WEIGHT` | 3 | **G** | |
| `CURIOSITY_WEIGHT`, `CURIOSITY_GRID` | 18, 5 | **G** | the largest weight in the file |
| `CORNER_PENALTY_WEIGHT`, `EDGE_PENALTY_WEIGHT` | 1, 0.5 | **G** | Tina still hugs the frame's edge (edge penalty ~0.95 on a test clip) |
| `DEAD_FIELD_*` | 0.01, 10, 1 | **G** | |
| `FLINCH_WEIGHT`, `FLINCH_THRESHOLD`, `FLINCH_WINDOW` | 1, 0.18, 3 | **G** / S | the window is ~200 ms, an escape-latency order of magnitude |
| `NEUTRAL_EPSILON`, `NEUTRAL_ACCEPT_PROB` | 0.001, 0.1 | **G** | |
| `WORLD_REFRESH_GENERATIONS`, `WORLD_REFRESH_MAX_S` | 5, 10 | B | |
| `STALL_CADENCES` | 3 | S | the HLS convention of a few missed reloads |

### tools/resource_handler.py (handling; may differ per host)

| Constant | Value | Kind | Notes |
|---|---|---|---|
| `MAX_QUOTA_PCT` | (cores - 1) x 100 | M | |
| `MIN_QUOTA_PCT`, `IDLE_STEP_PCT`, `STEP_PCT` | 50, 100, 25 | B | |
| `LOAD_STRAIN_PER_CORE`, `FREE_MEM_STRAIN_MB`, `HUNGER_DECAY` | 1.3, 800, 0.85 | **G** | |

## The biggest open item

The fitness function's weights (run_vision.py) are the largest concentration of
guesses left. The panel's direction is that viability, meaning the body's own
drive, is the one fitness that isn't arbitrary, and the other terms are
scaffolding: a curriculum. Retiring or deriving them is a design question of its
own, not a constants fix.
