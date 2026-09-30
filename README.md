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
- A developing organism, and fitness that knows its stage (proposed): [docs/design-ontogeny.md](docs/design-ontogeny.md)
- A second organism per host (parked, documented for later): [docs/second-organism.md](docs/second-organism.md)
- Its self-test, which every installer runs first: `python tools/selftest.py`

Licence: GPL-3.0-only.

## What it is, and what it isn't

It is a program: a population of small numeric programs (perception trees of
fixed arithmetic operations, a recurrent network's weights) selected against
live video under an energy budget, plus the bookkeeping that budget needs. It
is not alive, and nothing here suggests it feels anything.

The biological words throughout -- gut, blood sugar, sleep, eggs, death,
torpor, a cyst, migration, a hive -- are **design vocabulary**: each names a
piece of bookkeeping after the biological mechanism it was modelled on, so the
model can borrow that mechanism's measured numbers and its known trade-offs
(docs/physiology.md cites each). "It died of starvation" means an energy
counter reached its floor and the lineage record moved to the next genome.
Read the words as names for mechanisms, not as claims about experience.

**Its boundary.** What evolves can only compose the fixed operations in
`fishbowl/blocks.py`: nothing it makes can read or write files, open sockets,
start processes or run for unbounded time. Raw camera frames are never
written to disk. Resource use is capped by the operating system (systemd,
Job Objects), not by anything the organism reports.

**The hive (opt-in, off for a new install).** Hosts that join it may, after an
extinction, take a genome from a peer. It is **pull-only** (nothing is pushed
to a host), **data-only** (a genome is numbers; it is vetted against this
host's own limits before use -- `tools/fleet.py vet_genome`) and **removable**
at any time (`--no-hive`). See docs/packaging.md.

**How it is run.** It learns only over long undisturbed stretches. Changes go
in batches, each gated by the self-test on every host before it runs, and then
it is left alone: a deploy restarts every organism, and each restart is time
it doesn't spend living.

**Surviving a power cut.** Its checkpoint, memories and life history are
written durably, each with its previous copy kept. A cut loses at most the
last few minutes; a damaged file is set aside (never silently replaced) and
it resumes from the previous copy, or at worst from its own founder -- it
never crash-loops and never starts a stranger over its lineage. Downtime
isn't charged: it doesn't starve or age while its host is off. After a boot
it waits for the network, and a stream that can't open because the machine
has no network yet is not held against the stream (your choice of video is
kept).

## Running it

**Install** (docs/packaging.md has every platform and package):

| Host | Install or update | Join the hive |
|---|---|---|
| Linux (a checkout in `/srv/cambrian/cambrian-perception`) | `sudo sh deploy/install.sh` | `--hive` (`--no-hive` leaves) |
| macOS | `sh deploy/macos/install.sh` | `--hive` / `--no-hive` |
| Windows (elevated) | `deploy\windows\install.ps1` | `-Hive` / `-NoHive` |

Every installer runs the self-test on the incoming code first; if it fails,
nothing changes and the running organism keeps running. A new install is solo;
an update keeps the host's choice. Installing is starting: the organism takes
the camera (docs/suite.md).

**Control:** `cambrian --start | --stop | --restart | --yield | --status`.
`--stop` saves first; `--yield` also keeps it off at boot until the next
`--start`.

**Its navigation display in a terminal (Linux):** `cambrian --tui` -- the
viewer's nav display alone, drawn in Braille dots with the standard library
(no camera access: it reads only the viewer's `/state` on the same machine).
`q` quits, `c` toggles colour. A small trick for a headless Linux box: put
this in the console user's login profile (`.bash_profile`/`.profile` for bash,
`.zprofile` for zsh) to boot straight into it on the first console:

```sh
[ "$(tty)" = /dev/tty1 ] && exec cambrian --tui
```

**What happens to a lineage over time:**
- it lives on the feed, evolves in the background, and lays eggs when it has
  energy to spare; a body dies of starvation or of age, and its newest egg
  hatches;
- a dead camera or stalled stream puts it in torpor (5% of its usual burn,
  no starvation, evolution waits) until the world comes back;
- a lineage that has developed (it beats its own founder, by a paired
  t-test) and is losing energy encysts instead of dying, and revives when food
  or a new stream appears;
- if a lineage dies out, a new founder starts -- or, in the hive, whichever of
  its peers' genomes and a new founder does best on this host's own frames
  arrives as an egg.

Its life history (`state/life_history.json`) records each of these.

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
gaze; red corners are hosts; SCAN / TRACK / LOCK is what it is doing; a dashed ghost ahead of
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
orange inhibits. New random brain (two clicks) draws its brain afresh as a founder's at its
next generation, keeping everything else; New random founder (two clicks) replaces it with a new
random founder, as a fresh install has: eye, body, memories and traits included. Reset to founder
(two clicks) returns to its founder exactly as it was at birth -- the same
genome, a fresh body, no memories -- so a lineage can be run again from the
same start (lineages born before founders were kept can't). The cards and the
video each have a full-screen button (clicking them works too). Both keep a
backup in its state folder.

**Mushroom body.** Its HUD is its life's tally, each per hour lived (its last
60 minutes) and since its birth. Good: meals (separate bites), sips (nectar),
snacks (looks that fed on something new), eggs laid. Bad: swats, missed looks
(still thinking when it had to act), minutes starving, and deaths on this host
(a body dies of starvation or of age; its newest egg hatches, or its lineage
ends). Bottom right: HIVE on or SOLO (and, for a lineage that arrived by
migration, which host it came from), its age, and the lifespan its own damage
rate so far gives. Neither: approaches (something
began coming at its gaze; the navigation card's ring tightens as it comes). Torpor: minutes it
hibernated because its eyes got no world (a dead camera: black, blank or frozen
frames); torpid, it burns 5% of its usual, can't starve, and evolution waits.
While its process isn't running at all (the machine off, its feed down) no time
passes for it (cryptobiosis). Cyst: minutes a developed body (one that beats its own
founder on the same frames) spent encysted, starving in a world with no food:
nothing runs until a host comes within reach. An egg also carries its mother's
learned parts, given to the hatchling (its upbringing).
Its Kenyon cells, each coloured by what it has learned:
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

**Body.** "Toward an egg" is its reproduction buffer (a share of what it
digests, set by its gene kappa); "age" is the damage its own burning has done
(full: it dies). The first line says whether it is awake and gives its sense of time:
how long one of its looks takes and how many it takes a second (a look is its
moment; people make about 3-4 fixations a second). Gut, blood sugar and stores:
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

