# Design note: a developing organism, and fitness that knows its stage (proposed)

Status: **proposed** (a 2026-09-30 panel -- West-Eberhard, Waddington, Dunn,
Seeley, Hoelldobler & Wilson, Gelman, Dennett, Nesse). Nothing here is built
except what "Already in place" lists.

## Why

The organism is judged today as if it were always an adult: every genome is
scored on the same fitness, whatever its body has lived. But what a night of
life builds -- maps, memories, habits, the individuals it knows, its terrain
-- is development, and a young body's job is to build it, not yet to use it
well. A tram with no food, or a dead camera, used to cost a night of that.

## The organism as a colony

A Portuguese man o' war (a siphonophore) is one animal made of many zooids,
added and replaced while the colony persists; a hive persists while its
generations turn over. This organism is already built that way: its live body
persists while genomes are transplanted into it (a winning child's genome
adopted, its body and memories kept). The body is the colony; the genomes
are its zooids.

## Already in place

- **Protecting development**: torpor when its eyes get no world, cryptobiosis
  while it isn't running, and a developed body's cyst when its world starves
  it (docs/physiology.md) -- none of them cost it its development.
- **Measuring development**: its paired lead over its own founder on the
  same frames, at each new snapshot (run_vision `card["dev"]`; the model
  card's `development`).
- **Passing development on**: an egg carries its mother's learned parts
  (its upbringing), given to the hatchling once.
- **Its stage**: its maturation, in nights (genome.maturation), already
  sets how its lessons lock in.

## Proposed: fitness that knows its stage

1. **Its stage from its own data**, not an age cut-off: a body is juvenile
   while its development test hasn't yet found it better than its founder,
   and while its prequential scores (how well its heads predict before they
   learn) are still improving; adult once both have levelled off (a trend
   test on its own scores, at the standard 5%).
2. **A juvenile is judged on development**: how fast its prequential errors
   fall, and its lead over its founder growing -- learning, not yet
   performing. An adult is judged on viability (its body's own drive) and
   performance, as now.
3. **Replay trials score each candidate against the stage of the body it
   would be transplanted into**, since the body, not the genome, has the
   stage.
4. **Recorded from day one** (the panel's minority condition, as for life
   history): each body's stage over time, on the model card.

## Open questions for the panel

- Whether juvenile fitness should include how much of its development it
  would keep across an adoption (a transplant that wipes its distilled habits
  costs development).
- Whether upbringing should also pass its maps and individuals library, or
  only what its body learned to do (the panel's split vote: Dennett, Nesse).
- How the fleet's lineages should compete: today each host's lineage only
  replaces itself (and, in the hive, an extinct one may take a migrant).

## Parked (2026-09-30), so the organisms can run undisturbed

- **A second organism per host.** The resource panel (Gregg, Poettering,
  Russinovich, Gelman): only one host of four has the room, and on it two
  bodies would share one USB disk (the contention behind the fleet's worst
  watchdog kills); two organisms on one machine and one stream are weak
  replicates of each other. The plumbing is in and inert
  (`CAMBRIAN_INSTANCE`: its own `state-<i>`, runtime folder, service name;
  fleet discovery probes 8090 and 8091); nothing runs a second one.
- **Sensing each other, and sharing energy** (Wilkinson's reciprocity). It
  needs no second organism per host -- the four hosts' organisms can do it
  across machines, in the hive only, pull-only and data-only -- but it is a
  new mechanism: its clearance table comes first, after a stretch of
  undisturbed running.
