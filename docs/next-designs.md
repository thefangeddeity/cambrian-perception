# Next designs (notes, 2026-09-30) -- not built

Three designs a panel cleared in direction, written down so whoever builds
them starts from the reasoning, not from scratch. Each still goes to its own
clearance table (every number derived, cited or labelled) before code.

## 1. When its ground stops being its ground

**What exists.** Its ground model (organism.py: per-class fits of where things
stand, the horizon they imply, a terrain map on top) is learned from what
stands on the ground. Since 2026-09-30 it starts afresh when the *source*
changes: a checkpoint remembers the world it was saved in (`world_source`),
and a restart on a different one, or an in-place switch of streams, clears
its ground and terrain. Before, a lineage switched from a tram stream to its
room kept the tram's horizon.

**Why that is only a proxy.** The source is not the world. Three cases:
- A **lift's security camera** never changes source, but when its doors open
  the world outside is another floor. (Inside the closed car, the floor it
  stands on really is the same ground, and its model stays right.)
- A **live stream that cuts** between scenes changes world with no switch.
- A **lifted camera** (a drone, a hand) keeps the scene but not its height:
  every fit that assumes one camera height is wrong by the ratio.

**The general trigger: its own surprise.** It already measures how well its
ground model predicts (`_model_check`: where it said a thing should stand vs
where it stood; LEARNING HELD when it fails). The design: when that error
stays high for longer than its model's own correlation time, the ground is
not its ground any more -- it holds the old fit aside (a scene may come back:
the lift returns to the ground floor; its scene library already stores
places) and learns afresh. No new threshold: the error's own spread while the
model held, as the development test uses its own pairs.

**Height from flow (Srinivasan's bees; Gibson).** For the lifted camera: optic
flow from the ground scales as speed / height, so with its own speed known
(its speed cells, the flow teacher) the flow of the ground below the horizon
gives height relative to the height its model learned at. Rising shows as
ground flow shrinking at the same speed. A brain input "height" (relative,
1 = where it learned), and the ground fits scaled by it, let one model serve
all heights -- which is also what a drone would need.

Panel so far: Gibson, Srinivasan, Gelman. To add before building: someone on
change detection (e.g. Page's CUSUM, 1954) for the "error stays high" test.

## 2. Essence from motion, not a list

**What exists.** Essence is a list of classes (prey.PREY_CLASSES: living
beings, the teddy bear, vehicles), editable per host (cambrian.json "diet").

**The design.** Animacy as people perceive it is mostly motion: things that
start, stop and turn on their own read as alive (Heider & Simmel 1944;
Scholl & Tremoulet 2000; Tremoulet & Feldman). It already tracks individuals
(cortex.py) and measures its own ego-motion, so it can tell a thing's own
motion from the camera's. Essence of a tracked thing = how self-propelled it
has been (changes of speed and heading not explained by the camera), learned
per individual, the class list kept only as a prior for things it hasn't seen
move yet. A parked car then has little essence and a driven one a lot; a
teddy bear has essence only as far as people treat it (its class prior).

Panel so far: Dennett, Scholl & Tremoulet, Morton & Johnson. Open: whether
the prior should fade as its own measure takes over (Gelman: a shrinkage
estimate, with the class as the group).

## 3. Binocular vision (parked)

Ariana and 7elwe watch one room from two places, a natural stereo pair with
a wide baseline. What it needs first: the two machines' frames matched in
time (their clocks agree to ~10 ms over Tailscale only with care), one
organism reading two feeds (today one process reads one), and the two
cameras' relative pose (calibrated once, or learned from the things both see).
Parked until one organism can read two feeds; then depth from disparity is a
sense it can be priced for like any other (Land & Nilsson: most animals'
stereo is for the near field -- a wide baseline is an unusual eye).
