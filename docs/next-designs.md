# Next designs (notes, 2026-09-30) -- not built

Designs a panel cleared in direction (and one it dropped), written down so whoever builds
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

## 2. Essence from motion -- superseded (2026-09-30)

Proposed: read jīng from self-propelled motion instead of a list of classes
(a parked car little, a driven one a lot). Dropped, for two reasons:

- **Jīng is perceived animacy, not motion.** People recognise a parked car or
  a sleeping cat as a being by its kind, at a glance; so does the detector it
  borrows the perception from. The class list is the right mechanism, and it
  already makes the teddy bear no exception: it is perceived as animate.
- **It couldn't see the difference.** Its motion sense is a 16 x 9 field on
  frames 320 px wide; a car creeping a block away moves less than a cell per
  look, and its ego-motion estimate is noisier than that. What it does sense
  -- the flow left after undoing its own motion -- mixes things moving on
  their own with near things that are still (parallax).

Its own motion sense already feeds shén (novelty, paid more with speed); jīng
stays the perception of animacy, which its perception tree learns with the
detector as its teacher.

## 3. Binocular vision (parked)

Ariana and 7elwe watch one room from two places, a natural stereo pair with
a wide baseline. What it needs first: the two machines' frames matched in
time (their clocks agree to ~10 ms over Tailscale only with care), one
organism reading two feeds (today one process reads one), and the two
cameras' relative pose (calibrated once, or learned from the things both see).
Parked until one organism can read two feeds; then depth from disparity is a
sense it can be priced for like any other (Land & Nilsson: most animals'
stereo is for the near field -- a wide baseline is an unusual eye).

## 4. A card collapse toggle (parked for a UX/UI sprint)

Clicking a card used to maximize it; with a full-screen button on every
display the two stepped on each other, and the click was removed
(2026-09-30). What the page lacks instead is a way to fold away the cards a
viewer isn't watching. A collapse toggle of some kind belongs to a UX/UI
sprint -- designed with the page's layout as a whole (the quad, the side
charts, phone widths) and its military-standard HUD vocabulary, not bolted
onto one card.

## 5. The brain card's caption: a "View gates" link (parked, low priority)

The caption lists every stacked layer's gate inline ("gates 0.05, 0.26,
0.31, ..."): with a founder's dozens of layers it runs to several lines of
numbers nobody reads there. Replace the list with a "View gates" link that
shows them on demand (the flatmap already rings each layer's units by its
gate). Not scheduled.
