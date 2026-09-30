# The physiology rulebook

These are the inherited rules every organism lives by: fixed physics, the
same on every host, never evolved. Evolution works *within* them, and what
it achieves is only as honest as they are. Each rule names its source.
Where a number is a judgement rather than a measurement, it says so;
[constants-audit.md](constants-audit.md) has the full list.

Tested design choices are recorded as they were made, by design panels
(Sterling & Laughlin, Dennett, Secor, Heller, Seebacher, Vosshall, and
others).

## What kind of creature it is

**A cambrioid (since 2026-09-30).** A panel (Dennett's intentional stance,
Scholl & Tremoulet on perceived animacy, Morton & Johnson) found the mosquito
metaphor forced: what it feeds on is **essence**, what people treat as
animate -- living beings, and the things we lend a life to (a teddy bear,
vehicles). Its hosts are whatever has essence; a catch is still a bite. Where
this document says blood, read essence: the numbers below were measured on
mosquitoes and vampire bats and stay cited as such, as sources, not as what
it is. Next (a design note): essence read from self-propelled motion instead
of a list of classes.

**It began as a chimera, openly.** It has a mosquito's niche: hosts with blood, bites,
YOLO as its CO₂ that switches the hunt on. Its metabolism has a vertebrate's
framing (endotherm to ectotherm, crocodile bursts). Its fuel store makes
ketones like a vertebrate liver, and a shark leans on them too. Real insects
have a **fat body** instead of a liver, and it makes no ketones. We call the
organ "fat body / liver": the organ that stores fuel and makes the brain's
backup fuel.

## Food: essence (it began as "things with blood")

- **What counts as food.** Its hosts are anything with essence that the
  detector names: people, every animal COCO has a word for, the teddy bear,
  and every vehicle (bicycle, car, motorcycle, airplane, bus, train, truck,
  boat) -- `prey.PREY_CLASSES`; a host can change the list (cambrian.json's
  "diet", README). Plants and food items are not food. People are always food and always sensed, so every
  lineage can track people. Which hosts draw it most evolves (host
  preference, a trait).
- **A catch is a bite.** A host under its **mouth** is a bite: a fixed square
  at the gaze centre, 0.172 of the frame height. The bite is worth the same
  whatever the eye's size or the host's distance (Land & Nilsson; Holling
  1959; gape limitation). One bite = one blood meal's worth, times the chance
  the catch is real (the detector's confidence).
- **A bite is a flow, not a gulp** (handling time; Holling 1959). While a host
  is under its mouth, blood flows into the gut at its **pump rate** times the
  detector's confidence, until the gut is full. Looking faster doesn't eat
  faster; staying on the host does. The pump rate is inherited and evolves.
  Each lineage is born at what the old per-look meal fed it at its own resting
  pace (a newborn: 21.6 s of waking life per second on a host). A bigger pump
  is more muscle to keep. At the reference rate its upkeep is 5% of the resting
  burn, scaled with pump size and metabolic strategy. **This 5% is a guess**,
  borrowed from the brain's measured share (Mink 1981); see the audit.
- **Host defense: a swat.** A host it is biting can swat it: the same host,
  nearer than at its last look (its box overlaps the last one and grew),
  while the field looms past the line that already counts as a big change
  (`WAKE_LOOM`). A swat takes back **that bite's blood**. Blood-full
  mosquitoes fly slower and are hit more (Roitberg et al. 2003), so staying
  longer risks more: the foraging-under-risk trade-off (Lima & Dill 1990),
  which lets nipping versus gorging evolve. Nothing forces it to leave.
  Tissue damage was left out: nothing measures its size (panel 9–2).
- **Danger is learned, if evolution wants it.** An aversive compartment of the
  mushroom body (like the fly's punishment compartments, Aso et al. 2014)
  learns, by the same three-factor rule, what came before a swat. It feeds the
  brain a "danger" input, born with zero weights. Its learning rate is its own
  trait, born 0 (off). Once on, its output neuron costs its multiply-adds.
- **Plants are nectar** (a 2026-09-28 panel; Vosshall: mosquitoes of both
  sexes drink nectar for flight's sugar). A potted plant (COCO's "potted
  plant") under its mouth, when no host is, gives sugar at its pump's rate ×
  how full the plant still is. A full plant holds about one gut-full (a
  flower's standing crop is about one mosquito sugar meal); each sip draws it
  down, and it refills over hours (**3 h: a guess**, flagged in the audit),
  so the marginal value theorem applies: it should leave a drunk plant. Blood
  comes first. Its mushroom body learns sips as food, so its place map learns
  where the pot is. A plant sense (scent, then direction) is inherited, born
  off, priced like the prey sense. Plants are never hosts.
  Every plant counts, not only potted ones: COCO has no others, so a second
  YOLOv8n, trained on Open Images V7, finds them, about every 30 s on its own
  thread (plants don't move). Which classes are plants is Open Images' own
  hierarchy (everything under its "Plant" node: tree, flower, houseplant,
  rose, lily, sunflower, lavender, maple, willow, palm tree, Christmas tree),
  not a pick of ours. All hold sugar under the same rule (nectar, and plant
  tissue, which mosquitoes also drink from: Foster 1995); which are worth it
  is for its plant sense and learned values to find.
- **Snacks are surprise** (nectar in name only).

## Signals cost their sender

Its warning (the alarm output, WARN) is made like a movement: while positive,
it costs force squared per frame, as the eye's muscle does (a 2026-09-28
panel; alarm calls cost their callers, Sherman; Zahavi's handicap). Free, it
drifted until most lineages warned most of the time; priced (for one lineage
about 8% of its resting burn at the warning it had evolved), an untrained
warning is pushed toward silence until something -- the owner's feedback --
makes it pay. Surprise in the gaze pays a little. That keeps it
  going but can't sustain it.
- **Digestion costs a fifth of every meal** (specific dynamic action, Secor).

## Fuel stores and their controls

| Store | Holds | In | Out | Source |
|---|---|---|---|---|
| Gut | ~20 min of waking burn | a bite or a sip | into blood sugar over ~4 min | |
| Blood sugar | ~10 min | digestion; lactate returning | pays everything it can | |
| Glycogen | ~6 h | fills first once fed; empty to full in ~4 h (97% efficient) | fast: up to the aerobic ceiling, plus a burst's fuel | human liver glycogen ~¼ of a day's basal burn; Jentjens & Jeukendrup 2003 |
| Fat (triglycerides) | ~3 days | only from a real surplus (blood sugar above 80%), 75% efficient | slow: at most half the aerobic ceiling, and aerobic only | mouse: ⅓ of its fat in a 14 h fast; *Aedes aegypti* on water: median 3–4 days (Briegel); Fatmax (Achten & Jeukendrup 2003) |
| Phosphagen | ~10 s of a burst | refills from spare aerobic capacity, half-time ~30 s | first to pay a burst | phosphocreatine (insects: arginine phosphate); Harris et al. 1976 |
| Lactate | from bursts | a burst parks 15 of every 16 parts of its fuel here | returned as the debt clears, 93% kept | Brooks' lactate shuttle; the Cori cycle |

**The hormonal switches** (their two lines are inherited -- `genome.mobilize`,
born at a half, and `genome.store`, born at 80% -- stepped together by one
mutation; insulin sensitivity and how readily a body stores fat vary between
species and individuals):
- **Fed (insulin):** blood sugar above half. Glycogen fills; fat stays put.
  Above 80%, the surplus becomes fat.
- **Fasted (glucagon):** blood sugar below half. Glycogen tops blood sugar
  back up, and fat pays aerobic work directly.
- **Burst (adrenaline):** glycogen releases a burst's fuel at once.

## The three energy systems

Activity up to its **aerobic ceiling** (10× its resting rate; Bennett &
Ruben) is sustainable. Beyond it:
1. **The phosphagen** pays for the first ~10 s.
2. **Glycolysis** pays after that. It yields 2 of glucose's ~32 ATP, so a
   burst draws 16× the fuel. Most of it waits as lactate and comes back as
   the debt clears. **Net: about 2× the aerobic cost.**
3. **The debt** is felt as fatigue and fades over hours, as a crocodile
   recovers from a struggle. Glycolytic capacity is ~2 min (Gastin 2001;
   Bennett 1978).

Fat can't fuel a burst or a brain.

## The brain's fuel, and what starvation costs

- **Sugar feeds the brain**, and so do the sensors (receptors, thinking). Fat
  can't: fatty acids don't fuel a brain, and fat can't become glucose.
- **How long glycogen lasts is not a constant.** Fat pays the body in a
  fast, so glycogen carries only the glucose-obligate part: the brain and its
  sensors (the retina's analogue). That share is the brain's real cost, set by
  its size (evolved) and its host's climate (CPU scarcity). Measured on
  2026-09-28, it was 15-43% of the waking rest across lineages and newborns.
  At that share, glycogen alone carries the brain for 17-69 h. For comparison,
  a mouse's lasts 12-36 h and a human's ~24 h, where other obligate tissue
  (red cells, renal medulla) draws on it too.
- **Ketones.** Once glycogen is gone, the fat body / liver makes ketones. They
  ramp up over ~12 h and cover up to two-thirds of the brain's need (Owen et
  al. 1967).
- **The rest is tissue: wasting.** It is felt in the drive, and the costliest
  structure goes first, in proportion to the wasting:
  - Kenyon cells, the newest first;
  - hidden units, which go silent;
  - the eye's outer rings, which go blind.

  What is lost stops costing. Tissue regrows over ~3 days once fed again.
- **Why this rule exists.** A starving body used to feel costs as zero,
  because stores stopped at empty and nothing died. That let costly structure
  grow for free. Real biochemistry closes that hole.

## Metabolic strategy (inherited)

What this models is Bennett & Ruben's aerobic-capacity trade: a higher resting
burn buys a higher aerobic ceiling (stamina); warmth came along in real
endotherms, but here every organism thinks at its host's speed whatever its
metabolism, so "-thermy" means rest cost vs stamina, nothing more. Born at 1;
it can fall to an ectotherm's 0.1 (a tenth of the resting and sleeping burn)
or rise above 1, hotter than a newborn (hummingbirds, shrews), with only a
numerical guard on top. Cheap waiting, sensors still on, like a crocodile at the
waterhole. The aerobic ceiling scales with it, so a low-burn body sustains
less and bursts sooner. Sensing and thinking cost the same either way.
(Bennett & Ruben 1979; Secor; Huey & Pianka.)

## Photoreceptor speed (inherited)

Its gaze's receptors can evolve to be slower than the camera's frames
(`genome.receptor_slowness`, in reference frames of 1/15 s, born 0). A slow
receptor integrates light over longer: it blurs motion and sees change
later, and it costs less, 1 / (1 + slowness) of a fast one's price. A
receptor's pumping cost follows the membrane conductance that sets its
speed (Laughlin & Weckström 1993: fast-flying flies have fast, costly
photoreceptors, slow flies slow, cheap ones). It can never be faster than
the camera: the camera's frames are the fastest light its world has. The
whole field's sentinel eyes are unaffected.

## Its world's climate

- **Spare computing time on its host is its temperature.** It sets how fast it
  thinks, not what it eats: i7 creatures live in the tropics, i3 creatures
  near the poles.
- **The camera's frame rate** sets how fast the world can be sampled.
- **Image quality** is water clarity: less to see per receptor, not colder.

## Sleep

Sleep is its own choice (Borbely's thresholds, lowered in the dark). Two
things override it:
- **Collapse** when it is exhausted.
- **A big change** in the field wakes it. How big is inherited: vigilance. Its
  sensors stay on while it sleeps.
- **The room changing** wakes it too (the orienting reflex, Sokolov 1963: a
  violation of its own model of its surroundings). Its whole-field eyes keep a
  slow model of the room's structure (each receptor less the field's mean, so
  the lights dimming isn't news), adapting over ~20 minutes: each spot's
  usual value and how much it usually varies, so leaves that always wave
  become expected. The share of the field that is off by more than its own
  usual variation (the snack memory's rule) is its **mismatch**. Motion is gone in a
  second; mismatch stays while the change stays, so someone who came in
  quietly and stood still still wakes it. It wakes when the mismatch rises
  past the looming line (x vigilance) above what it was when it fell asleep:
  someone already there when it dropped off doesn't wake it. Awake, the brain
  gets the mismatch and where it is as a sense (priced like the others), to
  turn to the change. What woke it (the room, looming, movement, rested, its
  own choice) is logged hourly. Waking on a host's scent alone is its brain's
  choice: scent is an input while it sleeps, so evolution can learn it.

Starvation no longer forces it awake. That rule made sleep flap as the gut
crossed a line.

Sleep replays experience:
- **NREM:** the biggest surprises first, then the place map scaled down.
- **REM:** experiences recombined.

- **Paths (sequence replay):** with an inherited backup (born 0), a replay
  goes on as a path, walking back from the surprise it started at. Each earlier
  moment learns its own reward plus backup × the value of the moment that
  followed. This is reverse replay after reward (Foster & Wilson 2006): it
  hands a meal's value back along the route that led to it. Each step costs
  one replay.

- **Dreams (closed loop):** asleep and settled, its brain can steer an
  imagined gaze through the real eye physics, its senses fed by its own maps
  instead of its eyes. This is its internal representation standing in for the
  world: preplay of paths not yet taken (Pfeiffer & Foster 2013), and dreaming
  as model-running (Hobson & Friston 2012). Along the imagined path it learns a
  value map by the same backup (Sutton's Dyna): a place is worth its food plus
  backup × what the path reached next. Its place sense then points to the
  place worth most. Each dream step costs one brain step. How many steps it
  dreams is inherited, born 0; a dream teaches nothing without a backup above 0.

- **Recall (pattern completion)** (inherited, born off; a 2026-09-28 panel):
  what it sees calls up the episode of this life whose Kenyon-cell code
  overlaps it most, when the overlap beats chance (two random codes of
  those sizes). This is Marr's CA3, which at low memory load retrieves like
  a Hopfield network: the stored pattern nearest the cue. The brain gets what
  that episode held (its reward) and where it happened, as a sense. Asleep
  and dreaming, the dream's own code is the cue, so a dream calls up the
  memories it resembles. It pays a multiply-add per stored cell compared
  (an inverted index from each Kenyon cell to its episodes), so remembering
  more costs more. Recalls are counted hourly.

- **Scenes** (inherited, born 1; a 2026-09-28 panel on hippocampal
  remapping, O'Keefe & Moser): a library of the places it has lived in. Each
  scene has its layout (the field's structure, adapting over ~20 min) and its
  own maps: food, dreamt values, where people are expected, the surprise
  memory, the plants' crops. Each look it correlates the field's structure
  with the scene it is in. While the match is significant it stays: someone
  walking in hardly changes the layout. When it isn't, it goes to the stored
  scene that matches best, if one does, or starts a new one; a full library
  forgets the scene it visited least recently. "Significant" is the standard
  5% test on the field's effective number of cells, since neighbouring cells
  are alike (Bretherton et al. 1999). A blank frame is never placed. Matching
  costs a multiply-add per cell per stored scene per look. Born 1: one scene,
  never swapped, as before. Scene switches are counted hourly.

- **Teaching itself asleep (sleep programming)** (inherited, born off; a
  2026-09-28 panel: Dennett's Popperian creature, Friston's structure
  learning in sleep, the Baldwin effect). It keeps a uniform sample of its
  waking looks (Vitter's Algorithm R): each look's retina and the teacher's
  label. How many it keeps is inherited (`sleep_set`: 0, 16, 32 ... doubling).
  Asleep and settled, it tries one edit to its own perception tree per look,
  with the same operators evolution uses, and keeps the edit if the tree
  then predicts the teacher better on that sample. It pays for both trees
  over the sample. What it teaches itself lasts this life only (a transplant
  that leaves its tree alone keeps it); lineages that can learn are what
  evolution favours. On a synthetic test, 400 sleeping edits cut its error
  against the teacher sixfold.

- **Imagery, and seeing its dreams** (inherited, born off): each Kenyon cell
  learns a prototype, the average of what its eye saw (on a 16×16 grid
  relative to the eye) when it fired: a way back from memory to the eye, like
  feedback connections carrying predictions. Summed over a replayed code, the
  prototypes are its own reconstruction of the memory, and asleep its eye sees
  that reconstruction instead of darkness, so its brain and mushroom body run
  on the dream. It never sees recorded frames. The learning costs its
  multiply-adds; the prototypes are saved with its memory (8-bit).

It can also replay awake, in quiet moments. How much of each is inherited,
starting at 0.

## The perception tree pays for itself

The tree now pays for its arithmetic like the brain does (a 2026-09-28
panel, Sterling & Laughlin): a step per node per look, plus each receptor a
`pool` averages. Before this, a 1000-node tree ran free. `pool` is a block
that averages a square patch of receptors on one plane: a receptive field in
a single block, so evolution (and sleep programming) can find one in a
single step instead of chaining a dozen adds (Wagner, Land).

## After the 2026-09-28 audit

- **Its proboscis is a tube.** However strong its pump, flow through a tube
  is bounded by its bore (Poiseuille: flow ~ bore^4). At the inherited bore 1
  a full gut takes about 90 s, as a mosquito's engorgement does (Clements
  1992; Chadee & Beier 1995). A wider bore is more tissue to keep (upkeep ~
  bore^2, at the pump's upkeep share). Nectar comes through the same tube.
- **Protein: what blood has that nectar hasn't.** A protein store (one
  gut-full of blood = a clutch's worth) fills only from blood and is spent
  over a gonotrophic cycle, about 3 days (Clements 1992). Keeping it up is
  part of its drive, weighted like its fat stores; it feels it. Sugar keeps
  it alive; blood is what it's for.
- **No metabolism floor.** A slow animal's brain runs slower (processing
  speed scales with metabolic rate): its thinking takes longer against each
  look's deadline and it misses looks. Only a numerical guard remains.
- **Memories survive a restart.** Its episodes (about one per Kenyon cell;
  when full, the least surprising gives way: Mattar & Daw 2018) and its
  sleep test set are saved every 10 minutes and when it stops.
- **It teaches itself awake too**, in the quiet moments it replays in.
- **Pursuit:** prey sense level 3 adds the velocity of the host it follows,
  matched between detections by overlapping boxes.

An **oriented pool** (2026-09-29) splits the patch by a line through its
centre at an evolvable angle: the mean of one side minus the other, an edge
detector at that orientation (V1's simple cells; insects' oriented cells). A
pool can become oriented in one mutation. Both pay per receptor read, and
every tree cost is multiplied by CPU scarcity, so a host with more resources
can afford more and bigger ones.

## Senses added 2026-09-29 (all born unwired; evolution decides)

- **Its own speed** (interoception, unpriced like the gut): its pace (seconds
  per look, around its reference pace) and whether it just missed a look
  because its brain was still thinking.
- **Uncertainty** (Friston's precision): a running mean of its mushroom body's
  prediction errors, at its own learning rate. Priced as a sense.
- **A ground plane per scene** (Gibson; the self-driving bird's-eye view):
  every host is a measuring stick. Under a flat ground and a pinhole eye a
  thing's height in the frame grows in proportion to how far below the
  horizon its base stands; per class, a running line through (base, height)
  gives the horizon. Senses: how near the ground at its gaze is, and where
  the horizon is. Kept per scene. Plants measure it too (rooted: a plant's base is
  ground), each class's horizon counted by inverse variance -- the textbook
  calibration variance of a line's zero -- so a class whose members fit their
  line badly (flowers up on a bush, pots on a sill) counts for little. A thing
  the frame cuts (its base below the bottom edge, or its top above the top)
  measures nothing: where it stands, or its height, is unseen. The viewer
  draws its ground through feet and roots: flat, bent where a thing stands
  bigger or smaller than its kind's line predicts (elevation = camera height
  x (1 - predicted / seen)).
- **Every object measures the ground; a terrain map in its mind** (a
  2026-09-29 panel): its detector keeps every COCO class. Its cognition still
  sees only hosts and plants; the other things feed only its early vision's
  ground plane (a thing on a table works too: parallel planes share one
  horizon). Per field cell, a terrain map holds how far the ground under the
  things measured there rises or falls (camera heights), relative to its
  kind's line -- the surfaces mammals draw (V2 border ownership, CIP surface
  slant, the occipital place area, boundary vector cells). Kept per scene
  (memory item 14); input 60 is the terrain at its gaze (born unwired; seed
  set "2026-09-29 terrain" wires it at the mutation step).
- **Its visual cortex** (`fishbowl/cortex.py`, live organism only; a
  2026-09-29 panel): tracks (SORT's IoU rule), each tracked thing's true
  height and ground speed from its ground plane, its step rate (the
  autocorrelation of the motion inside its box, Johansson; Cutting &
  Kozlowski), and a library of individuals it has met (who, not just what),
  matched by a chi-square test on true height and colour. How much one
  individual varies is learned from its own tracks -- a thing it keeps its eyes
  on is one thing (Spelke's spatiotemporal continuity) -- so it never has to be
  told. Individuals that turn out alike merge. Not fed to its brain or charged
  for yet; kept in `state/cortex.json`; its meetings go to the events log. Means are
  compared at their own precision (the two-sample test: one detection's
  spread over the detections each rests on, plus how much one individual
  varies between meetings -- random effects, learned from its re-meetings),
  so a crowd's wobbling boxes don't make everyone one person. **In NREM**
  (Tononi & Cirelli's synaptic homeostasis) every individual's memory
  strength is scaled back so the library's total returns to its size after
  the last sleep; whoever falls below one meeting's worth is washed away.
  Meetings add one; a bite on someone, or a swat from them, adds its
  surprisal (-ln of how often such events come per meeting) -- the
  consequential stay. Known weakness: red sits on the hue wheel's seam, so
  red things split more often.
- **How near what it looks at is** (input 61; a 2026-09-29 panel): it knew
  where it looks (its gaze is among its inputs, an efference copy) but not how
  far. Relative, as a monocular eye's must be: the nearest thing whose box holds
  its gaze, read from its feet on its ground plane and terrain (else the ground
  at its gaze): 1 at the frame's bottom edge, 0 at the horizon. Born unwired;
  seed set "2026-09-29 nearness" wires it.
- **A default brain** (2026-09-29): a fresh install's founder is a random
  genome with every seed set applied and, as a young brain overproduces, the
  seeded values drawn at random from each trait's own distribution
  (`tools/seed.py random_draws`); its prices prune what doesn't pay. Seed set
  "2026-09-29 everything" switches on, at random, whatever else can evolve and
  is still off (replay backup, dream steps, colour, any unwired input, and
  receptor leaves where a tree reads only constants -- a tree that never reads
  its eye can never grow a line). A founder is saved the moment it is made: one
  killed before its first save used to be founded afresh at every restart
  (7elwe, 2026-09-29: three founders in ten minutes).
- **Its terrain head, its model card, the lite export** (2026-09-29): a
  mushroom-body readout (gene `felt_terrain`, born off) taught at each look by
  its ground model's nearness at its gaze -- lesson and look share one gaze, so
  locations match by construction -- each lesson weighted by the teacher's
  precision against its running mean (at most 1). Its estimate is input 62,
  there when the teacher is silent. Every head is scored prequentially (before
  it learns from a look; Dawid): mean error +- SE with the AR(1) effective n,
  and correlation. `champion.json` is its model card: genome, rules and code
  version (source fingerprints), frozen learned parts (mushroom body readouts,
  heads, distilled output layer), those scores, and fitness at adoption and on
  later snapshots. `tools/export_lite.py` turns it into the light CV module's
  bundle (refusing one without scores); `fishbowl/live.py` applies the learned
  parts. The perception tree has no held-out score yet, and the card says so.
- **Seeing the present despite its lag** (2026-09-29; Nijhawan's flash-lag,
  Berry & Meister's retinal anticipation, Changizi, Lee's tau): (1) **tau**,
  input 63 -- corners tracked where it looks (attention), last look to this
  one, fitted on their own: their expansion over one look is its look interval
  / time-to-contact, counted when significant and replicated, clipped to
  [0, 1]; (2) **lookahead** (gene, 0/1): its terrain head taught toward the
  next look; (3) **extrapolation** (gene, born 0, seeded near 1): the host it
  follows carried forward by its velocity x its own measured lag x the gene
  (1 = exact compensation; evolution tunes the over/undershoot). All seeded
  (seed set "2026-09-29 looking ahead"), priced, for evolution to prune. Scored
  prequentially: the extrapolated host position against the next detection,
  beside the plain position (3x closer in a synthetic walk). Shown on its HUD
  ("contact ~N looks", a dashed ghost where it expects the host) and in its
  eye's view (a reticle that tightens with tau, taught vs felt nearness, the
  ghost standing on its ground).
- **Feeling its rotation** (2026-09-29; Taube's head-direction cells,
  Jayaraman's fly compass ring, Jeffery and O'Keefe on landmark anchoring): a
  vestibular sense -- yaw from the frame's shift (focal length ~ frame
  height), roll from the ego-motion fit (significant and replicated) -- summed
  over each look as turning and tilting, in its half field of view (inputs
  64-65); a compass (gene, born off, seeded on) integrating the yaw (inputs
  66-67, sin and cos), re-anchored to the heading each scene of its library
  was lived at. Parallax and camera-motion undo the roll too. Tested: a pan
  integrates to 73.7 deg against 76.4 deg of geometry. Next (proposed): the
  entorhinal step, grid-cell path integration of speed x heading.
- **Its entorhinal map and hippocampal places** (2026-09-29; `fishbowl/entorhinal.py`;
  O'Keefe, the Mosers, McNaughton, Burgess, Jeffery, Taube; Kropff, Stensola,
  Solstad; Foster, Morris & Dayan): speed cells (forward speed in eye-heights/s
  = expansion rate x its ground's typical depth, 2 / (1 - horizon); input 68),
  an otolith (its change per look: a start, a stop; input 69), path integration
  (speed x its compass heading), grid cells (4 modules, 1 to 2.8 eye-heights in
  sqrt(2) steps, 16 random-phase cells each, three cosines 60 deg apart), place
  cells (128, each summing 7 grid cells, 5% firing, a fresh random map per scene:
  global remapping) with a value map (what each place has been worth; input 70),
  re-anchored to the position each scene was lived at. Gene `entorhinal`, seeded
  on. A passenger: it knows where on its route it is, not where to go. Tested:
  still 0, moving 0.28 eye-heights/s, +0.33 at a start and -0.29 at a stop, 3.9
  eye-heights straight ahead. Next (proposed): boundary vector cells (Burgess),
  a successor representation (Stachenfeld).
- **Its frames of reference** (2026-09-29; Galileo's ship; Jeffery, Burgess,
  Wolpert): it tells what rides with it (a tram's cab, a bonnet, the glass)
  from the world it moves through, the way a passenger does. It learns this while
  it moves and remembers it when it stops. On each frame where it moves and
  most of what it tracks flowed (vection needs wide-field flow: Brandt,
  Dichgans & Koenig 1973; when most stood still it may be stopped among people
  walking, so nothing votes), each
  corner of its ego-motion fit that the world's motion should have moved at least a
  pixel votes. A corner that stayed put votes local frame; one that moved with
  the world (within RANSAC's pixel) votes world; one that moved some other way
  votes nothing. Votes are counted per field cell, one map a life (its body is
  the same in every place), and forgotten at its room model's rate
  (`MISMATCH_TAU_S`) in seconds of moving only, so a stop forgets nothing. A
  cell is its local frame when its still votes beat half at the standard 5%
  test. Tau ignores corners there (what rides with it can't approach it), its
  colliculus gives them no pull, and the share of its view that rides with it
  is a sense (riding, input 71; seeded). Monocular: a second eye would add one
  more cue, weighed by its reliability. Known limit: on a long straight run,
  far scenery near the focus of expansion barely moves and can look as if it
  rides along, as the moon seems to follow a car. Turns vote it back to the world. The map
  is memory item 15. Tested on a synthetic ride with a still dashboard band:
  its corners vote local (mean height 0.91 of the frame; the band starts at
  0.75), the world's vote world, nothing votes once stopped; the band's cells
  are marked (riding 13%) and still marked 200 frames after the stop. First
  deploy: without the vection gate, a stopped tram with people walking past
  voted its whole still view local (62% on Tina); fixed and the maps cleared.
- **Flow-taught terrain** (2026-09-29; Gibson, Longuet-Higgins & Prazdny
  1980, Friston): its terrain map predicts how its ground should flow, and the
  error teaches the map, so no detector is needed. On a frame where it moves
  straight ahead (a significant expansion, no pixel of shift, no significant
  roll), each corner that moved with the world (its frames of reference), below
  its horizon, is read. A ground point at row y on ground raised e lies (1 - e)
  / (y - horizon) eye-heights away and flows out of the focus by speed / depth
  a frame. Its speed is read off the ground's own flow through its map (the
  median over the frame's corners, at least 8), not from the whole-frame fit:
  on rendered ground that fit reads ~35% fast (it locks onto the near rows);
  its speed cells share that bias, a finding for the panel. The lesson is 1 -
  predicted / measured camera heights, the detector's own formula, weighted
  `flow_teacher` x (1 - 1 px / how far it moved). The detector's lessons now
  count 1 - `flow_teacher`. Corners within half the tracker's window of an
  edge teach nothing (past the edge it reads fast flow short). Its felt-terrain
  head learns the result through its nearness, which reads the map. Gene
  `flow_teacher` (0-1, born 0, seeded near 0.5). The model card scores the map
  against each teacher before each lesson (`terrain_map_vs_detector`,
  `terrain_map_vs_flow`). Tested on rendered ground: flat reads flat (within
  0.05; MAE 0.032), and a plateau raised 0.3 on the right half reads a step
  of 0.38. Flow gives relative terrain; with half the ground raised, the level
  floats, and the detector's lessons anchor it.
- **Its V4** (2026-09-29; `fishbowl/v4.py`; Zeki, Conway, Roe, Pasupathy &
  Connor, Freeman & Simoncelli, Gibson): the ventral stream's middle stage, as
  functions (bees keep colour constant, an insect's lobula builds shape), each
  an evolvable, priced trait.
  - **Colour constancy:** its cones' von Kries gains adapt to what they have
    seen, over `colour_constancy` seconds (gene; 0 = none; gained at Fairchild
    & Reniff 1995's human time course, 20 s, then stepped). Luminance is kept.
    Its wide field stays monochrome, so its surround is what its cones saw, in
    time. Priced as a second pass over its cones. Tested: a grey card under
    warm light goes from red-green 0.60 / blue-yellow 0.35 to 0.51 / 0.49
    (neutral 0.5).
  - **The visual cortex** removes the frame's colour cast before its hue
    histograms (grey world, Buchsbaum 1980). Individuals seen before relearn
    their colour (a new histogram kind).
  - **Texture statistics** per field cell (gene `texture`): contrast, fineness
    (the spectrum's second moment, with the derivative filter's
    sin(2 pi f) response undone exactly: 9.0 / 18.0 / 36.0 read for 9 / 18 /
    36 cycles), and anisotropy (the structure tensor's coherence: 1.0 on
    stripes, 0.17 on isotropic noise). At its gaze they are inputs 72-74
    (contrast x 2, fineness over its measurable limit, anisotropy). Priced 3
    a field cell a look.
  - **The texture gradient** teaches its terrain (gene `texture_teacher`):
    across-fineness grows as depth on ground of one texture, so each ground
    cell is compared with the median of its own row (at least 8 cells). Along
    a row, the depth on flat ground and the camera's resolution limit are the
    same; pooled over rows, texture reaching the pixel scale near the horizon
    read as a rise toward it. It teaches what is raised or sunk beside its
    row, never the whole ground's tilt; its local frame is excluded. Tested
    on rendered ground: flat within +-0.1; a 0.3 plateau reads a 0.27 step.
    Scored `terrain_map_vs_texture`.
  - **Curvature:** an oriented pool can bend (its dividing line becomes an
    arc; Pasupathy & Connor's curved contour fragments). Mutation bends,
    straightens or turns it, at an edge's price. Edges now save their exact
    angle: rounded to 4 decimals, a receptor near a wide edge's line could
    change sides, so a reloaded tree answered differently from the one scored
    (found by a 5000-mutation save/load test).
- **Outside its model** (2026-09-29; Friston, Wolpert, Gelman, Dennett,
  Nesse; a shark turned upside down goes into tonic immobility, and this
  organism must not freeze, nor quietly mislearn). Each look, its detections
  are tested against its ground model: each whole box's residual from its
  class's line, over that class's own residual variance, summed as a
  chi-square (one degree of freedom a box; p by Wilson & Hilferty 1931).
  Failing the standard 5% test, it holds its world model's learning that look
  (ground lines, all three terrain teachers, the felt-terrain head); rewards,
  memories and its frames of reference still learn. The residual variance
  always learns: censoring the data that sets its own test would close the
  gate tighter and tighter, and left open, a long spell of strangeness widens
  what it accepts, as people adapt to inverting goggles (Stratton 1897), at a
  pace set only by how much it has lived. Its strangeness, 1 - p (uniform
  while it sees what it knows), is a sense (input 75, seeded). Tested: in
  model it holds 6.7% of looks (5% expected; heavy-tailed sizes); upside-down
  boxes are held 84% of the first 200 looks with its line moving 0.004, then
  it adapts.
- **Its clock (Process C)** (2026-09-29; Borbely, Siegel, Nesse): an internal
  phase with Earth's 24 h period, set by light as a zeitgeber at most about an
  hour a day (phase-response curves, Czeisler); a founder's clock starts from the
  light in view. The sleep thresholds follow the clock, not the moment's light.
  In its night, once sleep pressure reaches the night threshold in a quiet
  moment, it falls asleep whatever its brain prefers, and it holds: it wakes
  rested or disturbed, not by choice. (Sleep pays over hours; a choice judged
  over minutes never picked it -- the Tanzania lineage never slept in 48 h.
  Heller's partial dissent: a gate from the homeostat and the clock, not a
  schedule.) Aedes sleeps at night. In simulation: asleep ~18:00-03:00 of the
  stream's day, whenever born; a busy stream fragments it (~60% of the night).
- **Maturation** (2026-09-29; Hensch, Benna & Fusi, Kirkpatrick, Nader): what
  it has learned that works locks, and the rest stays plastic. Each output
  synapse its sleep distills into counts the good lessons that shaped it
  (credited in proportion to how far each moved it); its lock is lessons /
  (lessons + maturation x its lessons per night), and its updates shrink by
  that much. Maturation is evolvable, in nights (consolidation needs sleep):
  born 0 (never locking), first drawn between one night (Garcia's one-trial
  learning) and an adult Aedes' life (~30 nights); its lessons per night are
  averaged over its own last `maturation` nights. What it distilled, and its
  locks, are saved with its episodes and come back after a restart when the
  genome's brain is the one they were learned on. Its mushroom
  body -- the fast learner -- and all its maps and libraries never lock. And it
  doesn't go dumb: each night locks loosen by how wrong its predictions have
  been lately (reconsolidation). On a test, a skill's synapses locked to 0.71
  while unused ones stayed at 0; one night at 60% surprise loosened them to 0.60.
- **One long process, its watchdog, streams switched in place** (2026-09-29):
  no hourly bound (a stream's expiring address is re-resolved by the feed);
  checkpoints every 10 min; its own watchdog thread ends it (for its supervisor
  to restart) when its body lives no frame for the feed's stall line (30 s)
  while frames arrive, or its main loop makes no progress for its longest wait
  (600 s) plus a generation -- and under systemd feeds WatchdogSec=30. Another
  stream chosen is opened beside the old one and its body moves onto it; only
  camera <-> stream still restarts (the camera suite's handover).
- **Living in the present, and oxygen** (2026-09-29 panels): its present is
  its own look interval; a frame older than that when it gets to it isn't
  lived, only the newest is (freshness over completeness: it never queues a
  backlog). Like the diving reflex (Scholander; brain sparing, Ramirez), when
  its work per frame outgrows the frame's own time (load > 1, whatever the
  cause -- load, heat, a host short of memory) it sheds, one at a time:
  evolution (the next generation waits, at most as long as the last took),
  then its visual cortex, then its expansion measurement. Each shedding's
  saving is measured once its timings have settled (50 frames, the running
  mean's span), and each comes back when its load plus that saving fits in a
  frame. Its brain, gaze and senses are never shed. Every genome, the parent
  included, is scored in worker processes (a pool of one, if memory is
  short): its body's process only lives.
- **Its body in view** (2026-09-29; Gibson's nose, Neisser's ecological
  self): on a tram's cab view 60% of what it tracked never moved -- the cab --
  so its one fit said "still" and it never saw itself travel. When most of what
  it tracks is still and the rest agree on a motion of their own, the still
  part is its body, travelling with the camera, and the rest is the world. An
  expansion counts when it is beyond twice its standard error and the frame
  before showed one the same way (replication: neighbouring corners share
  tracking windows, so one frame's errors are correlated). Tram cab: moving in
  83% of steps (was 0%); walking: 83%; a still camera with sensor noise: 0 of 80.
- **Walking cameras** (2026-09-29): a camera walking forward shifts the frame
  hardly at all -- the image flows out of its centre (Gibson) -- so it read as
  still, and parallax never switched on (on a 24/7 walking stream: moving in
  0% of steps). Its camera's motion is now its shift and its expansion:
  corners tracked frame to frame (Lucas-Kanade), one similarity transform
  fitted with RANSAC (things moving on their own are outliers); the camera
  moves when either moves the frame's corners a pixel or more, and parallax
  undoes both. On the same stream: moving in about half its steps (the walker
  strolls, at the one-pixel line), parallax with it.
- **Parallax** (moving cameras): with the camera's own motion undone, what is
  left of each cell's change over what changed at all (+ sensor noise): 0 for
  the scene sliding past, towards 1 for what is nearer or moving on its own.
  No gain. The camera "moves" only when the whole frame shifts by at least a
  pixel of the frame it is measured on; while it moves, the scene library
  doesn't switch places.
- **Archetype heads** (inherited count, born 0; Tooby & Cosmides, Menzel):
  up to 4 more mushroom-body readouts, each taught by the three-factor rule to
  predict one detector class in its gaze; which class is inherited and
  evolves. Categories grounded by a teacher.
- **Self-teaching favours its mistakes** (the self-driving "data engine"): its
  test set is a weighted reservoir sample (Efraimidis & Spirakis 2006), each
  look weighted by its tree's disagreement with the teacher plus the running
  mean disagreement (self-scaling; no look is excluded).
- **Rare events are logged** as they happen (`state/events.jsonl`): waking and
  why, falling asleep, swats, new and revisited scenes, kept self-edits.
- Memories formed on a stream of another shape replay without touching maps
  whose cells don't exist on the current one.

## Life history: eggs, death and hatching (2026-09-29)

A panel (6-3; Kooijman, Pearl, Kirkwood, Stearns, Charnov, Wilkinson,
Dawkins, Gelman, Sterling & Laughlin) gave it a life history, because replay
trials are blind to slow costs: a lineage could drift into a costly metabolism
and starve for hours, and a starving body never died.

- **The kappa rule** (Kooijman's Dynamic Energy Budget theory): of what it
  assimilates -- what digestion moves into its blood, less the cost of
  digesting it -- a share kappa goes to its body as before and 1 - kappa into a
  reproduction buffer. Stores mobilised while starving never feed it. Gene
  `kappa`, born 1 (no eggs); seed set "reproduction" sets the Add-my-Pet
  collection's median across ~3000 species, 0.9 (Marques et al. 2018; DEB's
  prior of 0.8 barely moves fitted values), exactly: random seed steps clipped
  at 1 would have seeded lineages that can never lay.
- **Kappa is never mutated in replay trials**: they see its cost (less for the
  body) but never its benefit (eggs), and would push it to 1. It changes only
  when an egg is laid, one step on its log-odds (by `LEARNING_SIGMA`, as its
  rates step on their logs), so it nears 0 or 1 but never reaches them, and is
  selected by which lineages go on (unanimous).
- **An egg** costs what a newborn is made of: a fresh body's stores (blood sugar
  and fat, 10,968 B) made from sugar at `STORE_EFFICIENCY`: 14,624 B, about 4 h
  of waking burn. The live body lays one whenever its buffer holds one; its
  genome (kappa stepped) waits in `state/eggs/`.
- **Death** comes from its own body. Starvation: wasting reaches its ceiling
  (all the tissue `PROTEIN_CAP` measures; from a newborn's stores with no food,
  48 h). Age, by the rate of living (Pearl 1928): a share of all the energy it
  burns damages it -- the share of respiratory electrons that leak to
  superoxide, 0.15% (St-Pierre et al. 2002) -- and it dies when that damage
  equals the same tissue. At metabolism 1 that is ~629 days; at 2, 0.62 of it
  (not all of its burn scales with metabolism). It has its own lifetime, not
  a mosquito's (~8 days in the field) nor a vampire bat's (~30 years).
- **At death**: its state moves aside (`backup-<time>-died`); its newest egg
  hatches -- that genome, a newborn body, no memories; the lineage's founder
  and seed record go on with it, the other eggs stay with their mother. With no
  egg the lineage is extinct and a new random founder starts. Every founding,
  death (cause, age, eggs), hatching and extinction is recorded in
  `state/life_history.json` (the host's record, kept across lineages) and on
  the model card (`life_history`).
- Proposed, not built: eggs that also need blood protein (the protein store
  already exists; real mosquitoes need a blood meal to make eggs), and egg
  resorption when starving (oosorption).

## Migration, and a self-test that guards it (2026-09-30)

**Migration** (Wright's island model; Gelman): when a lineage goes extinct
(it died with no egg), its host's next start holds the fleet's tournament --
every organism it can reach (tools/fleet.py's discovery, pulled, never pushed)
and a new random founder, each scored as it would arrive (its genome in a
newborn body, no memories) on this host's own first snapshot at the reference
prices. The winner arrives as an egg (kappa stepped). No peer reachable, or
any failure: a new random founder, as before. Recorded in life_history.

**Self-test** (tools/selftest.py): the checks it was developed with -- genome,
tree, brain bound, body, organisms at both size extremes, scoring in a
worker, a live body, adoptions, statistics, migration, the viewer's scripts --
offline, in a temporary folder. Every installer runs it on the incoming code
before touching the running organism, and keeps the running code if it
fails. On its first night it caught two real faults: its own gate running from
a folder the service user couldn't enter (Linux), and OpenCV segfaulting in a
forked worker whose parent had started OpenCV's thread pool (macOS) -- workers
now run OpenCV single-threaded.

## Development kept: the cyst and upbringing (2026-09-30)

(See docs/design-ontogeny.md.) Dormancy is a prepared capacity with a cost,
not a reward for being accomplished (Storey's wood frog, Clegg's brine shrimp
cysts, tardigrades that must dry slowly). A body is **developed** when its
paired lead over its own founder -- scored beside it, newborn, on the same
frames at each new snapshot -- passes a one-sided paired t-test at the
standard 5%, its pairs counted as the effective number (successive snapshots
overlap), from 2 pairs (the model card's `development`). Only a developed body may encyst, and only
when its energy over its last hour has gone down (heading to starvation, not
between meals), at the last moment it can still pay: a cyst's protective
sugar is 15% of its body (Clegg 1962), made from its sugar and glycogen. A
starving newborn can't afford it and dies as before. Encysted, nothing runs
-- no burn, no ageing, no evolution -- until a host big enough to bite comes
into view or its stream changes; its sugar then returns to its glycogen.
**Upbringing**: an egg carries its mother's learned parts (its food and
danger readouts, heads, distilled habits), given to the hatchling once, as a
young vampire bat learns from its mother. Tested: encysted 3 s into a losing
fast with 15 B to spare; no damage while encysted; a person in reach revived
it with 0.60 glycogen back; a hatchling took its mother's readouts.

## Ram feeding (2026-09-30: a trial, then how every organism feeds)

On a tram it starved: its snack -- the new structure in its gaze -- was paid
once a look, however fast the world poured past. A ram filter feeder (whale
sharks, mantas; barnacles in a current) takes in density x speed x mouth. So
a snack pays surprise x
(1 + how far the scene flowed past its mouth during the look, from its own
ego-motion, over `MOUTH_SIDE`): still, the same; moving, a mouthful more per
mouth-length. No new number: the snack's own dilution (a sixth of blood's),
its gut bounding it. Tested on a synthetic ride: still 4.8 either way; fast,
15.0 -> 17.4 (+16%). It ran as a per-host trial on Tanzania (Tina the
control) the morning of 2026-09-30 and was made universal the same day: every
host, its live body and its evaluations alike, so evolution scores what its
body lives. The nav display shows what the flow is paying its snacks now
(`RAM x1.8`: nothing shown while still).

## When its eyes get no world (2026-09-30)

A panel (Heller, Geiser, Nesse, Gelman, Sterling & Laughlin) separated the
world having no food -- real scarcity, which may kill -- from its losing its
senses, which must never cost it its life or its evolution (a blind animal
doesn't revert to a sea squirt).
- **Cryptobiosis**: while its process isn't running (the machine off, its
  feed down) it isn't living, so no time passes: no burn, no ageing, no clock.
  An earlier audit charged downtime as idle, foodless time; once it could
  starve to death, an outage it never lived through could have killed it.
- **Torpor**: when frames arrive but carry no world -- a real sensor always
  has noise, so a frame with less spread than `NOISE_FLOOR` (black, blank) or
  identical to the one before (frozen) has none -- for longer than its feed's
  stall line, it hibernates: no looks, no learning, 5% of its burn (Geiser
  2004), ageing as slowly, and no wasting, so torpor can't starve it.
  Evolution waits (nothing is scored on blank frames); the first frame with a
  world wakes it. A dark night is still a world (its sensor noise is there).

## How big a brain, how big an eye (2026-09-30)

No handwritten caps (a panel; Sterling & Laughlin, Nilsson, Gregg, Changeux,
Dennett). The reference brain (16 units) costs 5% of its resting burn (the CNS
takes 2-8% across vertebrates: Mink et al. 1981), and no brain may cost more of
it than the most any brain is measured to: ~60%, the elephantnose fish (Nilsson
1996). So a brain's arithmetic is at most 12x the reference's -- stacked layers
(counted as if open: a gate can open) and channels included; that replaced the
caps of 256 units and 16 layers. A founder draws its hidden units
log-uniformly from 1 (the smallest brain) to 102 (the bound, one layer):
overproduction, then prices prune. Its host's CPU is only a backstop (hundreds
of times looser; a brain slower than a look misses looks). The largest eye is
the frame's height in receptors (64); the old 38 came from the retired zoom
eye's collapse, and its receptors' price is what limits it now.

The constants audit now tells derived numbers (D) from handwritten ones (H,
until today filed as "physics"), and gives each H a fate: cite, measure,
evolve or bound, most influential first. `PROTEIN_CAP` is calibrated: with a
newborn's stores, a starved newborn dies in ~48 h, as starved *Aedes aegypti*
females must feed about every other day in the field.

## Consolidation and pyramidal units (2026-09-29)

- **Sleep distillation** (inherited plasticity, born 0; complementary learning
  systems, McClelland, McNaughton & O'Reilly 1995; skills consolidate in
  sleep, Walker & Stickgold): in NREM, a replayed episode that went better
  than its usual makes what it did then more decisive -- its brain's output
  weights (the plastic slice) are pulled toward the full commitment of that
  action, scaled by plasticity x advantage. It stops by itself as outputs
  saturate. Its own copy of the brain, this life only (a transplant that
  leaves the brain's genes alone keeps it). Priced per weight changed.
- **Pyramidal units** (inherited apical gain, born 0; Larkum 2013): each
  hidden unit has a basal compartment (the senses) and an apical one (its own
  recurrent context); h = tanh(b + basal + context + a x basal x
  tanh(context)) -- far more firing when evidence and context agree. Two more
  multiplies a unit when on.

## A collicular priority map (2026-09-29)

The superior colliculus is a topographic priority map: every point of the
field sums evidence, and the strongest pulls the eyes (Land, Nilsson, Tooby &
Cosmides, Friston). Ours is on the 144 whole-field cells: salience = a
weighted sum of motion, mismatch (novelty), parallax, host presence, host
size and plants; the brain gets the direction to the winning cell and its
strength. The six weights are inherited; the host-size weight is a cat's size
gate made evolvable -- negative prefers small fast things (a cat), positive
big ones (a mosquito). About 900 multiply-adds a look, priced; off while every
weight is 0.

## Seeding, and when not to wait (panel 11-3)

Brains overproduce and prune (Changeux, Edelman). A capacity whose estimated
discovery time on the fleet is over a month is seeded rather than waited for
(six months is the outer bound): switched on at values a few steps along its
own trait's distribution, always priced, always prunable, never hard-wired
(`tools/seed.py`; each seed set applies once per lineage, listed in
`state/seeded.txt`). Dennett's side and Urmson's meet there.
