# The live actor: act live, learn offline

Decided by a design panel on 2026-09-28 (Friston, Dennett; the votes are
below). The model is the animal's own: it acts awake, in the present, and
consolidates from replay offline.

## Before

Evolution scored the parent and its children on a snapshot of the
newest ~600 frames. The survivor's run on that snapshot then did three
jobs:
- it moved the lasting body toward its end-of-window body;
- its memory became the lineage's;
- the viewer replayed it, several seconds behind live (3–11 s across hosts).

The organism never lived the present. It only relived recent windows.

## Now

On a live feed (a camera or a stream), one `LiveLife`
(`fishbowl/livelife.py`) runs on its own thread. It lives every frame the
feed keeps, once, in order, as the frame arrives. It is the same
`Organism` a generation's run uses, with the accepted genome.

- **Body of record:** the live body. Each generation's children start from
  a copy of it, and past windows no longer move the body (vote 9–1).
- **Memory of record:** the live memory (the habituation map, the mushroom
  body's food and danger weights, the place and people maps). It is what
  actually lived (vote 8–2).
- **Adoption:** when a child wins, the living body adopts its genome, a brain
  transplant as in the livecam port's `LiveActor.adopt`. Body, memory and
  gaze go on. The moment goes on too, wherever the new genome's shapes still fit: the brain's recurrent state, the eye's last look, and the frame it has reached. Most adoptions are neutral drift, dozens a minute, and a fresh brain state each time would wipe its short-term memory every few seconds.
- **Fitness is unchanged:** children are still scored on past windows,
  starting from the living body and memory.
- **Feeding record and hourly metrics** come from the live frames, each frame
  counted exactly once.
- **The viewer** gets the newest 150 lived frames every second
  (`live_actor.json` beside `live_status.json`). The server merges them over
  the generation's status. The HUD is always drawn for the picture actually
  on screen: when a picture lags, the HUD shows that frame's state, never a
  live HUD over a stale picture.
- **A recorded file** has no live body. There, the survivor's window carries
  the body on, as before.

## Failure mode

If the live thread dies, the run says so and falls back to the old way (the
survivors carry the body) until it ends. The next run starts a new live body
from the saved checkpoint. On stop, the live body and memory are what get
saved.

## Cost

The live actor costs about one organism step per kept frame: its field sense
every frame, and a brain step per look. That runs under the same resource cap
as everything else.
