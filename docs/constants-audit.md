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
| `G_CAP`, `GUT_CAP` | 10 min, 20 min of burn | P | blood sugar and gut. `R_CAP` (6 h) retired 2026-09-28: the old reserve became fat (docs/physiology.md) |
| `GLYCOGEN_CAP`, `GLYCOGEN_FILL_S`, `GLYCOGEN_EFFICIENCY` | 6 h, 4 h, 0.97 | S | human liver glycogen ~1/4 of a day's basal; refills within hours (Jentjens & Jeukendrup 2003); synthesis ~1 of ~32 ATP |
| `FAT_CAP`, `FAT_MAX_SHARE` | 72 h, 0.5 of the aerobic ceiling | S (range) | mouse: 1/3 of its fat in a 14 h fast; Aedes aegypti on water: median 3-4 days (Briegel); Fatmax (Achten & Jeukendrup 2003) |
| `KETONE_TAU_S`, `KETONE_MAX_SHARE` | 12 h, 2/3 | S | mice reach ketosis in 12-24 h; ketones cover ~2/3 of a fasting brain (Owen et al. 1967) |
| `PROTEIN_CAP`, `WASTING_REBUILD_S` | 24 h, 72 h | **G** | how much tissue is there to burn, and how fast it regrows: assumptions |
| `PHOSPHAGEN_S`, `PHOSPHAGEN_HALF_S` | 10 s, 30 s | S | phosphocreatine: ~10 s of maximal effort; resynthesis half-time ~30 s (Harris et al. 1976) |
| `ANAEROBIC_FUEL_RATIO`, `LACTATE_RETURN` | 16, 0.93 | S | 2 vs ~32 ATP per glucose; lactate ~half oxidized, ~half rebuilt (Cori, ~87% kept) in recovery (Brooks): net ~2x aerobic |
| `DIGEST_TAU_S` | 240 s | P | |
| `STORE_ABOVE`, `STORE_RATE`, `MOBILIZE_BELOW` | 0.8, 1.0, 0.5 | P | the hormonal switches: fed (insulin) above 0.5 fills glycogen, above 0.8 makes fat; fasted (glucagon) below 0.5 releases glycogen and burns fat. `MOBILIZE_RATE` (fat -> sugar) retired: fat can't become glucose |
| `STORE_EFFICIENCY` | 0.75 | S | making fat from sugar (de novo lipogenesis ~75-80%) |
| `SLEEP_METABOLISM`, `WAKE_FLOOR`, `TEMPO_SHARE` | 0.3, 0.6, 0.4 B/s | P | the endotherm end; resting and sleeping burn scale with the inherited metabolic strategy (2026-09-27) |
| `MIN_METABOLISM` | 0.1 | S | an ectotherm rests at ~1/10 of an endotherm (Bennett & Ruben 1979: 5-10x, the upper end) |
| `AEROBIC_SCOPE` | 10 x resting | S | vertebrate factorial aerobic scope (Bennett & Ruben); insects' is far higher -- not modelled |
| `BURST_S`, `DEBT_TAU_S` | 120 s, 2 h | S / **G** | glycolytic capacity (Gastin 2001: crossover ~75 s; lizards exhaust within minutes, Bennett 1978); recovery over hours (crocodiles) -- 2 h assumed |
| `SDA_FRACTION` | 0.2 | S / **G** | digestion's cost (specific dynamic action; within Secor's range for ectotherms, the exact value chosen) |
| activity's per-unit cost | same for every strategy | **G** | assumption: equal cost of moving; only what is sustainable differs |
| `LIGHT_FAST_S`, `LIGHT_SLOW_S` | 60 s, 1200 s | P | its day/night sense |
| `TIRED_EFFICIENCY` | 0.5 | **G** | the invented cost of skipping sleep; the proposed sleep redefinition would retire it (awaiting a go) |
| `ACCLIMATIZE_HALF_LIFE_S` | 80 s | **G** | |
| `EFFORT_COST`, `FOOD_PER_LOOK`, `PREY_FOOD_PER_LOOK` | 1e-4, 2e-4, 1.2e-3 | P | prices and food values of its world. A meal is now one bite times the chance the catch is real (the detector's confidence), whatever the eye's size or the prey's distance (2026-09-27; it used to be the share of the gaze centre prey covered, which made the eye a mouth). Blood meals are no longer per look (2026-09-28): see `PUMP_REF`. `PREY_FOOD_PER_LOOK` now only sets where the pump is born |
| `PUMP_REF` | 21.6 B/s | S / **I** | a bite is a flow while a host is at its mouth (Holling's handling time; panel 8-1). The reference is the old per-look meal at a look every frame (15 fps). Each lineage is born at `PUMP_REF / pace`, what it ate at its own resting tempo, so none jumps. The rate is inherited (`genome.pump`), mutates at `TRAIT_SIGMA` (log-normal). No biological bounds: upkeep and intake set its limits, and log-normal steps approach a nip without reaching zero; `PUMP_GUARD` (1e-6-1e6x) only keeps the float sane |
| swat rule | `WAKE_LOOM`, box growth > 0 | P / **S** | a swat = the host at its mouth came nearer (same host, box grew) while the whole field looms past `WAKE_LOOM`, the line that already counts as a big change (reused, not a new constant). World physics: the host side of host defense (panel 2026-09-28) |
| swat cost | this bite's blood | P | a swat takes back the blood taken in the bite going on (Roitberg et al. 2003: engorged mosquitoes are slower and hit more; Lima & Dill 1990). No tissue damage: nothing measures its size (panel 9-2) |
| `PLANT_CLASS`, `NECTAR_CROP` | 58, one gut-full | S / **P** | COCO's "potted plant" as nectar; a full plant holds about one gut-full, a flower's standing crop being about one mosquito sugar meal |
| `NECTAR_REFILL_S` | 3 h | **G** | a drunk plant refills its nectar with this time constant; nectar secretion refills in hours, the exact value a guess |
| `plant_sense` | born 0; 0/1/2 | I | none / scent / + coarse direction to plants; priced per synapse like the prey sense |
| warning cost | alarm² per frame, x `EFFORT_COST` | P | the alarm output, while positive, costs what the eye's muscle costs for the same output (a 2026-09-28 panel: signals cost their sender); reuses the muscle price, no new constant |
| `imagery`, `PROTO_SIDE` | born 0; 16 | I / B | imagery (Kenyon-cell prototypes, and seeing its dreams) is inherited, born off; the prototypes' 16x16 grid relative to its eye is a storage and display bound |
| `mobilize`, `store` | born 0.5, 0.8 | I | the fasted and storage lines (glucagon, insulin), inherited since 2026-09-28 (panel 8-2); one operator steps both by TRAIT_SIGMA; kept 0 < mobilize < store < 1 (their meaning) |
| `METABOLISM_GUARD` | 1e3 | B | metabolic strategy has no upper limit but this numerical guard (panel 8-2): above 1 is hotter than a newborn, paying more rest for more stamina |
| `receptor_slowness` | born 0; steps TRAIT_SIGMA x (1 + slowness) | I / **S** | extra photoreceptor integration time, in reference frames (1/15 s); the gaze sees the frames low-passed with that time constant. Receptor cost x 1 / (1 + slowness): pumping cost follows the membrane conductance G, and speed is G / C (Laughlin & Weckstrom 1993; Niven, Anderson & Laughlin 2007). Never faster than the camera (world physics). Step scaling with size: chosen so it can leave 0 and still move once slow (flagged) |
| `aversive_rate` | born 0; first draw log-uniform 1e-3-1 | I | the aversive compartment's learning rate, drawn and stepped as the reward learning rate is |
| `PUMP_UPKEEP_SHARE` | 0.05 | **G** | a pump at `PUMP_REF` costs 5% of the resting burn, scaled by pump size and metabolic strategy. Borrowed from the brain's share (Mink 1981), not measured for a feeding pump: flagged |
| `MOUTH_SIDE` (fishbowl/prey.py) | 0.172 of frame height | P | a square mouth at the gaze centre, fixed: a given (mouths barely evolve; not modelled). Exactly a newborn eye's old eating zone, so default eyes eat as before; square like the eye's mosaic (panel 7-2: a round one of the same width would cut the catch zone to pi/4) |
| `S_RISE_S`, `S_FALL_S` | 18.2 h, 4.2 h | S | Daan, Beersma & Borbely 1984 |
| `SLEEP_SETTLE_S`, `SLEEP_INERTIA_S` | 10 s, 7 s | **G** | |
| `COLLAPSE_S`, `COLLAPSE_RELEASE_S` | 0.95, 0.8 | **G** | safety net |
| `HUNGER_WAKE_R`, `EMPTY_G` | 0.05, 0.1 | P | an empty body (degraded). `HUNGER_WAKE_G` retired 2026-09-27 with the starvation-forces-awake rule (it made sleep flap as the gut crossed its line) |
| `WAKE_LOOM`, `WAKE_MOTION` | 0.18, 0.6 | **G** | what wakes it from sleep, now divided by the inherited vigilance (born 1: unchanged) |
| `MISMATCH_TAU_S` | = `LIGHT_SLOW_S` (1200 s) | P | the field mismatch's model of the room adapts over its day/night sense's slow timescale (reused, not new); a cell mismatches beyond `SURPRISE_SIGMAS` x its own learned variation, floored at `NOISE_FLOOR` (the snack memory's rule), the area is read with `EXPANSION_GAIN` and wakes at `WAKE_LOOM` (the same kind of reading: an area differing from a background) |
| `NECTAR_OIV7` | Open Images' "Plant" node, 12 classes | S | the detector's own published class hierarchy (bbox_labels_600_hierarchy.json, /m/05s2s), not a pick |
| `FLOWER_EVERY_S` | 30 s | B | plants don't move; the flower detector's CPU budget (under ~1% of a core on Tanzania; 2026-09-28 infrastructure review) |
| scene match | r > 1.96 / sqrt(N_eff), N_eff = N (1 - ra rb) / (1 + ra rb) | S | the 5% test with Bretherton et al. (1999)'s effective sample size for autocorrelated fields |
| `MAX_SCENES` | 64 | B | a safety bound; matching's price limits the library |
| `MAX_SLEEP_SET` | 1024 | B | a safety bound on its sleep test set; the edits tested on it are what it pays |
| `MAX_HEADS` | 4 | B | archetype heads have fixed input slots |
| camera moves | a whole-frame shift of >= 1 pixel of the shift frame | B | the measurement's own resolution; below it is noise |
| parallax scale | residual / (change + `NOISE_FLOOR`) | P | a ratio: no gain; reuses the sensor-noise floor |
| test-set weights | disagreement + running mean disagreement | P | self-scaling (Efraimidis & Spirakis 2006 keys) |
| uncertainty rate | its own learning rate | P | reused |
| `EVENTS_MAX_BYTES` | 5 MB, one rotation | B | disk wear |
| `KEEP_BACKUPS` (fleet.py) | 3 per host | B | fleet-made backups only |
| `ENGORGE_S` | 90 s | S | a mosquito engorges in ~1.5 min (Clements 1992; Chadee & Beier 1995: 1-2.5 min): the tube's flow at bore 1 |
| bore upkeep | bore^2 x `PUMP_UPKEEP_SHARE` | P | tissue to keep scales with the tube's cross-section; reuses the pump's upkeep share |
| `GONOTROPHIC_S` | 3 days | S | blood meal to eggs (Clements 1992: 2-4 days) |
| protein in the drive | 0.3 | P | weighted like the fat stores (reused) |
| episode capacity | one per Kenyon cell | B | a sparse associative memory holds at least as many patterns as cells |
| `MIN_METABOLISM` | 1e-3 (was 0.1) | B | a numerical guard; a slow brain misses looks instead |
| sleep set steps | 0 <-> 16, then doubling / halving | B | a mutation's step size (a set of 16 looks is the smallest worth testing on); the size itself evolves |
| `SLEEP_ONSET_*`, `SLEEP_END_*`, `NIGHT_LIGHT`, `DAY_LIGHT` | 0.15/0.45, 0.01/0.04, 0.15/0.5 | **G** | Borbely's two thresholds; published threshold values are for humans in normalised units, not yet mapped |

### fishbowl/fovea.py, retina.py (the eye)

| Constant | Value | Kind | Notes |
|---|---|---|---|
| `RECEPTOR_PITCH` | 1/64 of frame height | P | documented in fovea.py |
| `DEFAULT_RECEPTORS`, `MIN_RECEPTORS`, `MAX_RECEPTORS` | 22, 4, 38 | P / B / M (dated) | MAX from the zoom eye's measured collapse past 0.6 |
| `FIELD_RECEPTORS` | 144 | P | the old whole-field count, now square |
| `DAMPING`, `FORCE_GAIN`, `SPRING` | 0.5, 0.2, 0.08 | P | overdamped oculomotor plant (Robinson); saccade speed and hold cost documented |
| `max_mag` (zoom lens limit) | `RECEPTOR_PITCH` x frame height (2.81 at 180 rows) | M | measured from each frame: one receptor per pixel, so it never upsamples (no pixel inflation); zoom in only (added 2026-09-27) |

### fishbowl/controller.py, genome.py, blocks.py (brain and search)

| Constant | Value | Kind | Notes |
|---|---|---|---|
| `HIDDEN`, `TREE_HIDDEN` | 16, 16 | P | a newborn's size; the tree's view |
| `MIN_HIDDEN`, `MAX_HIDDEN`, `MAX_LAYERS` | 4, 256, 16 | B | price limits growth |
| `MAX_CHANNELS` | 4 | **G** | a hand cap on loops; could become a bound only, as for units |
| `HEAVY_TAIL_P`, `HEAVY_TAIL_SCALE`, `HEAVY_TAIL_MAX` | 0.1, 0.1, 2.0 | **G** | heavy-tailed mutations are grounded (the distribution of fitness effects is leptokurtic, Eyre-Walker & Keightley 2007); the mix is not |
| `STABILIZER_SIGMA` | 0.1 | **G** | |
| `ZOOM_SIGMA` | = `STABILIZER_SIGMA` | **G** | a 0..1 reflex gain, mutated at the same scale; the zoom gain is born 0 (off) |
| `TRAIT_SIGMA` | = `STABILIZER_SIGMA` | **G** | the 2026-09-27 traits (metabolism, host preference, REM share, vigilance) mutate at the stabilizer's scale |
| `MAX_REPLAYS`, `MIN_VIGILANCE`, `MAX_VIGILANCE` | 16, 0.25, 4 | B | bounds only: replay is limited by its price and the look's deadline |
| replay rules | prioritized by surprise (Mattar & Daw); REM = two experiences' halves at mean reward (Hoel) | P | Crick & Mitchison's REM unlearning not modelled; the place map's NREM downscaling reuses `CONSOLIDATE_RATE` |
| place map learning rate | = `genome.learning_rate` | P | the mushroom body's own rate (one lifetime learner) |
| intruder memory | over `LIGHT_SLOW_S` (~20 min), per light phase | P | its day/night sense's slow average reused as "lately" |
| host preference | mean 1, people >= 1 | P | tuning redistributes attention; people always tracked (the clade's rule) |
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
| `MAX_KC` | 16384 | B | the price limits it; raised from 4096 when a lineage pressed it (Tina at 3,904, 2026-09-28; panel 8-2) |
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
| `MAX_QUOTA_PCT` | (cores - 1) x 100, or lower per host (`state/host_limits.json`) | M | a host someone uses (a laptop) may keep more for itself: Ariana 200% of 4 cores (2026-09-28 infrastructure review) |
| `MIN_QUOTA_PCT`, `IDLE_STEP_PCT`, `STEP_PCT` | 50, 100, 25 | B | |
| space (worker bodies) | cores - 1, -1 per run while memory is short, +1 when not | B | memory strain shrinks the worker pool ("space"), CPU strain cuts the quota ("oxygen"); cutting CPU frees no memory (2026-09-28 infrastructure review) |
| `LOAD_STRAIN_PER_CORE`, `FREE_MEM_STRAIN_MB`, `HUNGER_DECAY` | 1.3, 800, 0.85 | **G** | |

### Ground, terrain, nearness (fishbowl/organism.py, fishbowl/prey.py; 2026-09-29)

| Constant | Value | Kind | Notes |
|---|---|---|---|
| horizon pooling | inverse variance + random effects | S | calibration variance of a line's zero; DerSimonian & Laird 1986 |
| residuals before a class counts | 3 | B | the least a line's scatter can be read from |
| frame-cut margin | one detector pixel, max(1, w/h) / 640 | S | the detector's own resolution, letterboxed |
| terrain clip | +-1 camera height | B | display and sense range; pinhole geometry below it |
| terrain blend | Gaussian, 1 cell | S | the map's own resolution (Grimson's interpolation) |
| terrain shrink | the kernel's centre weight | S | one measurement counts as in its own cell |
| nearness terrain clip | +-0.9 | B | keeps / (1 - e) finite |

### Its visual cortex (fishbowl/cortex.py; 2026-09-29)

| Constant | Value | Kind | Notes |
|---|---|---|---|
| `IOU_MATCH` | 0.3 | S | SORT's association threshold (Bewley et al. 2016) |
| `CHI2_95` | 3.84 ... | S | the chi-square table |
| gait band `GAIT_HZ` | 0.5 - 4 Hz | S | step rates of walking to trotting people and pets |
| white-noise band | 2 / sqrt(N) | S | the autocorrelation's textbook significance |
| colour histogram | 8 hue x 4 saturation (16 grey) | **G** | the panel: fine as a start; measure how often individuals split or merge wrongly |
| `MAX_AGE`, `MIN_HITS` | 1, 3 detections | S | SORT's defaults (was 2 s and 2 s: guesses, replaced after the panel) |
| `GAIT_WINDOW_S` | 2 / 0.5 Hz = 4 s | S | two periods of the slowest gait, from `GAIT_HZ` |
| `LIBRARY_MAX` | 64 | B | memory bound; the least seen, longest ago go first |
| `MET_AGAIN_S` | 60 s | B | the events log's rarity only; no behaviour |
| per-track history | 50 detections | B | memory bound |
| height needs a base below the horizon | one detector pixel | S | the detector's resolution (was 0.02, a guess) |

### Seeding (tools/seed.py, run_vision._default_brain; 2026-09-29)

| Constant | Value | Kind | Notes |
|---|---|---|---|
| seeded input weights | N(0, 0.05) | S | the brain's own mutation step |
| seeded traits | three of their own mutation steps | S | Dennett's compromise: random, from each trait's own walk |
| receptor leaves | each constant leaf with even odds, at least one | **G** | the panel: acceptable (a coin, not a tuned count) |
| founder: archetype heads | 1 - 4 | B | the heads' structural range |
| founder: scenes | three of its mutation steps from 1 | S | was 1 - 8, a guess |
| founder: plasticity | its birth distribution | S | |

### Viewer only (tools/viewer.py)

Display constants (mesh density, blend widths, the phone's 820 px and 600 px
thresholds) shape pictures, not the organism, and are listed in the code.

## The biggest open item

The fitness function's weights (run_vision.py) are the largest concentration of
guesses left. The panel's direction is that viability, meaning the body's own
drive, is the one fitness that isn't arbitrary, and the other terms are
scaffolding: a curriculum. Retiring or deriving them is a design question of its
own, not a constants fix.
