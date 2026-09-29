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

**A chimera, openly.** It has a mosquito's niche: hosts with blood, bites,
YOLO as its CO₂ that switches the hunt on. Its metabolism has a vertebrate's
framing (endotherm to ectotherm, crocodile bursts). Its fuel store makes
ketones like a vertebrate liver, and a shark leans on them too. Real insects
have a **fat body** instead of a liver, and it makes no ketones. We call the
organ "fat body / liver": the organ that stores fuel and makes the brain's
backup fuel.

## Food: things with blood

- **What counts as food.** Its hosts are anything with blood that the detector
  names: people and pets, and every vertebrate COCO has a word for. Plants and
  food items are not food. People are always food and always sensed, so every
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

**The hormonal switches:**
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

From endotherm (1, as born) to ectotherm (0.1: a tenth of the resting and
sleeping burn). Cheap waiting, sensors still on, like a crocodile at the
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
