# cambrian-perception

A **cambrioid**: an artificial animal that lives in a camera feed and feeds on
**jīng** 精 (essence): perceived animacy -- whatever is seen as a being, living
or not (people, animals, a teddy bear, cars, boats, every vehicle its detector
names). It has a movable eye, a
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
- A second organism per ecohost (parked, documented for later): [docs/second-organism.md](docs/second-organism.md)
- Next designs (not built): when its ground stops being its ground, binocular vision: [docs/next-designs.md](docs/next-designs.md)
- Its self-test, which every installer runs first: `python tools/selftest.py`

Licence: GPL-3.0-only.

## What it is, and what it isn't

It is a program: a population of small numeric programs (perception trees of
fixed arithmetic operations, a recurrent network's weights) selected against
live video under an energy budget, plus the bookkeeping that budget needs. It
is not alive, and nothing here suggests it feels anything.

The biological words throughout -- gut, sugar, sleep, eggs, death,
torpor, a cyst, migration, a hive -- are **design vocabulary**: each names a
piece of bookkeeping after the biological mechanism it was modelled on, so the
model can borrow that mechanism's measured numbers and its known trade-offs
(docs/physiology.md cites each). "It died of starvation" means an energy
counter reached its floor and the lineage record moved to the next genome.
Read the words as names for mechanisms, not as claims about experience.

**A cambrioid, not a mosquito.** It began as a mosquito-bat chimera, and many
of its numbers still come from mosquitoes and vampire bats, as measured
sources (a bite's size, a gonotrophic cycle, a lifespan's shape). Its own
terms, after the Three Treasures (三寶) of Daoist thought -- borrowed as
names for mechanisms, not as claims about them:

| It eats | The perception of | From | Counted in | The mechanism |
|---|---|---|---|---|
| **jīng** 精 (essence) | **animacy**: this is a being, or is taken for one | **hosts** -- people, animals, a teddy bear, vehicles: the detector's classes | **kǒu** 口, mouthfuls | a bite: a host held under its mouth |
| **qì** 氣 (breath) | **life without agency**: growing, still | plants: the detector's plant classes | **xī** 息, breaths | a sip: nectar, flowing in |
| **shén** 神 (spirit) | **change**: this is new | anything in its gaze: its own receptors | **niàn** 念, thought-moments | a snack: what a look at something new gives; speed pays more (ram feeding) |

Jīng is perceived animacy (Scholl & Tremoulet's term), not motion: a parked
car or a sleeping cat is recognised as a being by its kind, at a glance, as
people recognise one -- and whether a slow car is moving is below what the
cambrioid can resolve anyway. The teddy bear is jīng because it is perceived
as animate. Jīng and qì are perceptions it borrows from its detector (a model
trained on human labels); shén is the one it makes itself. The long-term aim
in the code is for its own perception to learn animacy, with the detector as
its teacher.

The computer it lives on is its **ecohost**: its CPU is its energy economy,
its camera its world, the resource handler its climate; the hive is a network
of ecohosts. In the code some names keep older words (`PREY_CLASSES`,
`bite_blood`, `host_pref`, `snacks`, `nectar`; "host" in the installers means
the machine): they mean the same things.

**Its boundary.** What evolves can only compose the fixed operations in
`fishbowl/blocks.py`: nothing it makes can read or write files, open sockets,
start processes or run for unbounded time. Raw camera frames are never
written to disk. Resource use is capped by the operating system (systemd,
Job Objects), not by anything the organism reports.

**The hive (opt-in, off for a new install).** Ecohosts that join it may, after an
extinction, take a genome from a peer. It is **pull-only** (nothing is pushed
to an ecohost), **data-only** (a genome is numbers; it is vetted against this
ecohost's own limits before use -- `tools/fleet.py vet_genome`) and **removable**
at any time (`--no-hive`). See docs/packaging.md.

**How it is run.** It learns only over long undisturbed stretches. Changes go
in batches, each gated by the self-test on every ecohost before it runs, and then
it is left alone: a deploy restarts every organism, and each restart is time
it doesn't spend living.

**Surviving a power cut.** Its checkpoint, memories and life history are
written durably, each with its previous copy kept. A cut loses at most the
last few minutes; a damaged file is set aside (never silently replaced) and
it resumes from the previous copy, or at worst from its own founder -- it
never crash-loops and never starts a stranger over its lineage. Downtime
isn't charged: it doesn't starve or age while its ecohost is off. After a boot
it waits for the network, and a stream that can't open because the machine
has no network yet is not held against the stream (your choice of video is
kept).

## Install from scratch

You need a camera (or a stream to show it, chosen in its viewer), git, and
the network once. Everything else the installer brings: Python 3.12+ where it
can, the organism's own venv with the pinned libraries (`requirements.lock`),
and the detector models, downloaded from this repo's
[models-v1 release](https://github.com/thefangeddeity/cambrian-perception/releases/tag/models-v1)
and checked against their SHA-256 (`tools/fetch_models.py`; the release notes
say how to make them yourself). Without the models it still runs, with
shén alone (no hosts to bite).

**Linux** (systemd; Debian/Ubuntu, Arch and others):

```sh
git clone https://github.com/thefangeddeity/cambrian-perception.git
sudo sh cambrian-perception/deploy/install.sh            # add --hive to join the hive
```

It makes a system account `cambrian` (in the video group, no login) and its
home, a checkout in `/srv/cambrian/cambrian-perception`, then the venv, the
models, the services under `cambrian.target`, a polkit rule (its viewer may
restart it when you pick a video) and the `cambrian` command. It watches
`/dev/video0`; for another camera see Ecohost settings below. Or install a
package instead (the same layout): `sh deploy/debian/build-deb.sh`, then
`sudo apt install ./dist/cambrian-perception_*.deb`; on Arch,
`cd deploy/arch && makepkg -si`. Don't put a package over a checkout install.

**macOS** (as the user who'll own it; it asks for your password once):

```sh
git clone https://github.com/thefangeddeity/cambrian-perception.git
cd cambrian-perception && sh deploy/macos/install.sh    # add --hive to join the hive
```

It installs into `~/Library/Application Support/cambrian-perception` and
runs from login (macOS lets only a logged-in session use the camera; the
first time, it asks on screen to allow the camera). It uses Homebrew's
Python 3.12+ and ffmpeg if you have them, else installs python.org's Python.

**Windows 11** (an elevated PowerShell, in a clone or unzipped copy):

```powershell
powershell -ExecutionPolicy Bypass -File deploy\windows\install.ps1    # add -Hive to join the hive
```

It installs into `C:\ProgramData\cambrian\cambrian-perception`, runs from
boot as you (no stored password), installs Python 3.13 with winget if needed,
and opens port 8090 on private networks.

macOS and Windows install from these scripts; native packages (a `.pkg`, an
MSI) are still to come. Every installer runs the self-test on the incoming
code first; if it fails, nothing changes and the running organism keeps
running. A new install is solo; an update keeps the ecohost's choice (re-run the
same command to update). Installing is starting: the organism takes the
camera (docs/suite.md). Then open `http://<this machine>:8090/`.

## Ecohost settings

Everything you can set on an ecohost (the computer it lives on), with its default. None of these change the
organism's code; a restart (`cambrian --restart`) picks them up.

| Where | Setting | Default | Notes |
|---|---|---|---|
| `cambrian.json` (next to the code) | `"hive"` | `false` (solo) | the installers' `--hive` / `-Hive` set it |
| `cambrian.json` | `"diet"` | living beings, the teddy bear and vehicles give jīng; plants give qì | see "What it eats" below |
| `cambrian.json` (macOS, Windows) | `"source"`, `"viewer_port"` | `"0"` (the camera), `8090` | the installers' `--source` / `-Source` |
| its viewer | the video it watches | its camera | a stream you pick there (`state/selected_source.json`), with an optional end time |
| systemd drop-in (Linux) | which camera, its priority | `/dev/video0` | `sudo systemctl edit cambrian-perception`: `[Service]` / `ExecStart=` / `ExecStart=/srv/cambrian/cambrian-perception/.venv/bin/python run_vision.py /dev/v4l/by-id/<your camera>` (a by-id path survives reboots), and e.g. `Nice=10` |
| `state/host_limits.json` | `{"max_quota_pct": N}` | the ecohost's cores less one | a ceiling on the CPU the resource handler may grant (e.g. a laptop you use) |
| the unit (Linux) | `CPUQuota`, `MemoryMax` | 150%, 1.5 GB | only until the resource handler's first run (every 15 min): it grants CPU as cores go idle and sizes memory from what the host can spare |
| `state/experiments.json` | per-ecohost trials | none | a trial's name: `true` (e.g. ram feeding was one, before it became how every organism feeds) |
| environment | `CAMBRIAN_PREY_MODEL` | `models/yolov8n.onnx` next to the code | another detector model |
| environment | `CAMBRIAN_WORKERS` | from the CPU granted: a worker per core, less one | a fixed number of evolution workers (`1`: serial) |
| environment | `CAMBRIAN_RUNTIME_DIR` | `/dev/shm/cambrian-perception` | where its live status and frames live (RAM) |
| environment | `CAMBRIAN_CAMERA_PREVIEW` | `1` | `0`: no recent frames kept for the viewer's replay |
| environment | `CAMBRIAN_INSTANCE` | unset | parked: a second organism per ecohost (docs/second-organism.md) |

On the machines this was developed on: all in the hive; Tina and Tanzania
each watch their USB camera through a by-id drop-in (Tina's at `Nice=10`);
streams are chosen per ecohost in the viewer.

## Running it

**Control:** `cambrian --start | --stop | --restart | --yield | --status`.
`--stop` saves first; `--yield` also keeps it off at boot until the next
`--start`.

**Its navigation display in a terminal (Linux):** `cambrian --tui` -- the
viewer's nav display alone, drawn in Braille dots with the standard library
(it never opens the camera: it reads the viewer's `/state` on the same
machine). `q` quits; `c` twice switches the organism back to its camera (the
chosen stream cleared; it restarts); `m` toggles colour. A small trick for a headless Linux box: put
this in the console user's login profile (`.bash_profile`/`.profile` for bash,
`.zprofile` for zsh) to boot straight into it on the first console:

```sh
[ "$(tty)" = /dev/tty1 ] && exec cambrian --tui
```

**What it eats (an ecohost setting).** By default its jīng (bites) comes from
people, every animal the detector names, the teddy bear and every vehicle, and
its qì (sips) from plants. An ecohost can change that in `cambrian.json` next
to the code -- e.g. only a teddy bear and a few animals while testing:

```json
"diet": {"food":   ["person", "cat", "dog", "bird", "teddy bear"],
         "nectar": ["potted plant", "Flower", "Tree"]}
```

`food` (its jīng) names are COCO classes (the list is `COCO_NAMES` in
`fishbowl/prey.py`); `nectar` (its qì) also takes the flower model's Open
Images names. A list you give replaces that default; people always give jīng.
It takes effect at the next restart, and the life history records it: a
lineage fed on teddy bears learns teddy bears. (Shén isn't a class: it is what
any look at something new gives.)

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
  its peers' genomes and a new founder does best on this ecohost's own frames
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
60 minutes) and since its birth. Good: jīng 精 (separate bites of a host, in kǒu, mouthfuls), qì 氣 (sips from
plants, in xī, breaths), shén 神 (looks that fed on something new, in niàn,
thought-moments), eggs laid. Bad: swats, missed looks
(still thinking when it had to act), minutes starving, and deaths on this ecohost
(a body dies of starvation or of age; its newest egg hatches, or its lineage
ends). Bottom right: HIVE on or SOLO (and, for a lineage that arrived by
migration, which ecohost it came from), its age, and the lifespan its own damage
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
moment; people make about 3-4 fixations a second). Gut, sugar and stores:
- sugar pays for everything (~10 min of waking burn);
- the gut digests into sugar over minutes;
- glycogen lasts ~6 h and is released fast;
- fat lasts ~3 days, made from a real surplus and burned only aerobically;
- protein, for eggs, comes only from jīng and is spent over ~3 days;
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
size, jīng and qì per hour, and which kinds of change evolution accepted.

**Genome and ecohost** (the last card). Its fitness and peak ever, what it is
watching, its inherited traits (gaze size, resting pace, colour, stabilizer,
zoom lens, metabolism, feeding pump, vigilance, replay, prey sense), the CPU
quota its ecohost grants, and whether the livecam is off while it runs.

