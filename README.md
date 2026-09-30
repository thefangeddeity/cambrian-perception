# cambrian-perception

An artificial animal that lives in a camera feed. It has a movable eye, a
whole-field sentinel eye, a recurrent brain, a mushroom body that learns in its
lifetime, a body with metabolism, stores, sleep and a clock, and memories it
replays and dreams. It evolves continuously on real hardware, and it is priced
for everything it does.

- The body and its senses: [docs/physiology.md](docs/physiology.md)
- Every constant and where it comes from: [docs/constants-audit.md](docs/constants-audit.md)
- Living live on a feed: [docs/live-actor.md](docs/live-actor.md)
- Packages and installers: [docs/packaging.md](docs/packaging.md)
- The camera suite (organism and livecam never run together): [docs/suite.md](docs/suite.md)

Licence: GPL-3.0-only.

## Reading the viewer

The viewer (`tools/viewer.py`, port 8090) keeps its captions to what you need
to read each card. What those cards show is explained here.

**Gestures.** The page keeps its own scroll and pinch-zoom. A graphic (brain,
mushroom body, perception tree) takes wheel, drag and pinch only after you
click or tap it (a thin outline shows which); Esc or a click outside releases
it. Clicking a card shows it full screen; clicking the video shows the picture
alone full screen. On a phone the brain and mushroom body open in 2D.

**Video stream.** What it saw, a few seconds behind live, in step with the
visual field (frames stay in RAM, never on disk). The HUD: the reticle is its
gaze; red corners are hosts; SCAN / TRACK / LOCK is what it is doing; "contact ~N looks" means something is approaching its gaze
and will reach it in about N of its looks (Lee's tau); a dashed ghost ahead of
a host is where it expects that host to be by the time it acts (it carries the
host forward by the host's velocity times its own lag). WARN is its warning.
The client view is `/live`. The HUD's text is repeated in Lebanese Arabic,
Ukrainian and Taiwanese Mandarin.

**Navigation.** Its sense of space, seen from its own eye (it drives into this;
nobody steers it). The ground is its terrain map, drawn through feet and roots:
flat where it has no evidence, raised or lowered where things stood bigger or
smaller than their kind usually does at that depth. Pink boxes are hosts, green
plants, grey other things it measures the ground by; dashed ones are cut by
the frame's edge and measure nothing; cyan columns are parallax (nearer, or
moving on their own). The reticle is where it looks; its ring tightens as
something approaches; the two small bars beside it are the nearness its ground
model teaches ("taught") and the nearness its own terrain head feels ("felt").
"felt nearness: error ... (n ...)" is that head scored before each lesson.
"Who it knows" lists the individuals its visual cortex has met, by true height
(in camera heights), step rate and when last seen. Top left: its path, as it
has integrated it (speed x heading), with a tick for where it faces now. Grey
shade is what rides with it (a cab's dashboard): it learns this while moving
(what stays put while the world slides past) and remembers it when stopped;
tau and its gaze ignore it.

**Gaze.** Its eye rebuilt from the frame on screen (a reconstruction): the
receptor grid, the cone patch in the middle (the only colour), where it looks
and how much of the frame its gaze covers.

**Perception tree.** Its evolved program that guesses "a host in my gaze?",
graded by YOLO, drawn as a cone tree growing down onto its retina: leaves plug
in where they read (receptors, pooled patches, oriented edges; a curved line is
an edge that bends, its V4's curvature).

**Visual field.** Its wide-field eyes: where things move, as heat. The box is
its gaze; dashed boxes are hosts, dotted ones plants; a red frame means
something is looming. Layers add its food places, where it expects people,
what still surprises it and its collicular priority. The senses strip lists
its newer senses: pace, uncertainty, camera motion, horizon, what it looks at
and how near, how near it feels it is, contact, turning, tilting, heading,
its speed (eye-heights per second), starting or stopping, what the place it
is at has been worth, how much of its view rides with it, and the texture at
its gaze (contrast, fineness, grain). The navigation card's HUD reads like a
fighter's: heading tape on top, speed (SPD) on the left, nearness (NR: T
taught, F felt) on the right, the flight-path marker where it is going, and
a data block; ODD is how strange the look is to its model, and LEARNING HELD
means it failed its model's test, so it isn't learning from that look.

**Brain.** Inputs, hidden units and outputs with every weight; cyan excites,
orange inhibits.

**Mushroom body.** Its Kenyon cells, each coloured by what it has learned:
green food, red danger, amber food it learned to avoid, grey lost to wasting.
Firing cells glow phosphor green, brighter the more they have learned. In 3D
the firing cells send their axons down the peduncle to the medial lobe (food
value) and the vertical lobe (danger); in 2D the lobes are the two bars.

**Sleep, replay and dreams.** The sleep line: awake or asleep, sleep pressure,
its clock (day or night), replays awake/asleep and the REM share, habits it has
distilled, how much of them has matured (locked), its self-edits kept/tried,
its scene, what woke it and how much the room changed. "Recalled" is the part
of its eye a memory's cells listen to; "mind's eye" is the picture those cells
rebuild (asleep, its eye sees it: a dream); "what it saw" is for you only,
never the organism. The place map shows replay paths: blue NREM, violet REM,
grey awake replay, gold dreamt.

**Body.** Gut, blood sugar and stores:
- blood sugar pays for everything (~10 min of waking burn);
- the gut digests into blood sugar over minutes;
- glycogen lasts ~6 h and is released fast;
- fat lasts ~3 days, made from a real surplus and burned only aerobically;
- protein, for eggs, comes only from blood and is spent over ~3 days;
- phosphagen covers the first ~10 s of a burst;
- once glycogen is gone, fat feeds up to 2/3 of the brain as ketones;
- wasting burns tissue for a brain with no sugar (Kenyon cells, hidden units,
  the outer rings of its eye);
- sleep pressure builds awake and clears asleep; tiredness halves what it
  catches at full pressure;
- hunger, search, curiosity, arousal, threat, fatigue and anaerobic debt are
  its drives and costs.

Its tempo can speed up or slow down 3x either way around its inherited resting
pace. "How it moves" is measured, never rewarded (Land 1969).

**Charts.** Fitness (the current genome re-scored each generation, and the
peak ever), its body per run, its homeostatic drive (lower is healthier), gaze
size, resting pace (every Nth frame), the CPU quota the resource handler
grants (grows with real improvement, shrinks under strain), perception tree
size, blood and nectar per hour, and which kinds of change evolution accepted.

**Genome and host** (the last card). Its fitness and peak ever, what it is
watching, its inherited traits (gaze size, resting pace, colour, stabilizer,
zoom lens, metabolism, feeding pump, vigilance, replay, prey sense), the CPU
quota its host grants, and whether the livecam is off while it runs.

