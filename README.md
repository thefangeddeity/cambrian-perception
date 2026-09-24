# cambrian-perception

A self-improving organism that evolves its own perception from raw
video — no pretrained models, sandboxed, reflex-driven, growing its
own visual machinery from scratch. The point is to watch what it comes
up with, not to deploy it for anything.

Named for the real thing it's meant to echo: watching a Cambrian-era
nervous system slowly discover what real eyes are for, under real
evolutionary pressure, on hardware that can iterate millions of times
faster than 400 million years of actual selection.

**In plain, precise terms** (added 2026-09-24 after an external audit):
the search algorithm is a (1+1) evolutionary strategy — one parent, one
mutated offspring per generation, strict-improvement acceptance (plus
a small amount of neutral drift) — with adaptive operator selection,
not population-based evolution with crossover, and not unbounded
recursive self-modification (see the fishbowl boundary's note on
`update_mutation_weights` below). "Recursive" is accurate only in the
narrow, explicitly bounded sense described there. Small-scale, honest
framing over an impressive-sounding one.

## The fishbowl boundary

The safety design this whole project stands on. Everything the
organism can touch is enumerated here — if it isn't listed, it can't
do it.

**What the organism can do:**
- Compose and mutate small trees of pure, numeric operations (see
  `fishbowl/blocks.py`) — arithmetic, comparison, a protected
  conditional. No loops or recursion exist in this language; a tree is
  finite and is walked exactly once per evaluation, so evaluating it
  always terminates. There's no way to write an infinite loop in it,
  not "there's a timeout in case you do."
- Hold multiple such trees at once, one per output channel
  (`fishbowl/genome.py`'s `DEFAULT_CHANNELS`: `response`, `pan`,
  `tilt`) — a response magnitude scored against real reflex signals,
  and where to move its own attention next (see "Foveated vision"
  below). All channels share the exact same safe tree language; there
  is no separate, less-safe code path for the channels that produce
  actions versus the one that produces a response.
- Have its own accept/reject evidence (which operator got tried, did
  it get accepted) continuously reshape which mutation operator is
  likely to be tried next (`Genome.update_mutation_weights`). Correction
  (2026-09-24): earlier versions of this README called this "mutating
  the weights that decide how it mutates itself" and called it "the
  recursive part." That's no longer accurate and arguably never was in
  the strong sense implied — the update rule is a fixed, human-written
  EMA, not something the organism mutates or that competes for
  acceptance; the organism has no influence over it. What's real here:
  operator selection adapts to evidence instead of staying uniform.
  What's not real: unbounded or organism-driven recursion into its own
  mutation process. Call it adaptive operator selection, not recursive
  self-modification — see `fishbowl/genome.py`'s own docstring for the
  full history of why this changed.
- Log a structured, plain-language request when it hits one of its own
  ceilings repeatedly (`fishbowl/sandbox.py`'s `log_request`). Purely
  observational — it cannot act on a request or change a ceiling. A
  human, or the observation tool being scoped separately, reads
  requests and decides.

**What it cannot do, by construction, not by policy:**
- No `eval`, `exec`, `os.system`, `subprocess`, or dynamic `import`
  **inside `fishbowl/`** — the package the organism's own authored
  programs actually run in. The block-tree interpreter in `blocks.py`
  is the only thing that ever "runs" an organism-authored program, and
  it only ever does arithmetic on numpy arrays. This is NOT a claim
  about the rest of the repo: `run_vision.py` (the harness, not the
  organism) calls `subprocess` to resolve a live stream URL via
  `yt-dlp`, `tools/resource_handler.py` calls `sudo systemctl`, and
  `tools/fetch_curriculum_videos.py` is a human-run utility that shells
  out too. All of that is fixed, human-written infrastructure the
  organism never touches, authored the same day this rule was
  written — corrected here after an external audit caught the
  blanket wording no longer matching the code.
- No network access the organism's own logic can trigger. The ONE
  outbound fetch per run — its configured video source — is a fixed,
  human-curated list (`run_vision.py`'s `LIVE_SOURCES`, or a local
  device path), never anything the organism chooses. Separately,
  `tools/viewer.py` runs a small, **unauthenticated**, LAN-facing HTTP
  server so a human can watch derived state — real network exposure,
  just not organism-controlled or write-capable. Treat the viewer as
  trusted-LAN-only until it has auth; that's a known gap, not
  something to read past.
- No filesystem access beyond `state/` **from `fishbowl/` or
  `run_vision.py`**, and only through one whitelisted writer
  (`sandbox.py`'s `_write_json_atomic`). `tools/` scripts are outside
  that boundary by design (see above) and have their own, separate
  file writes (`resource_handler.py` has its own atomic-write helper;
  `fetch_curriculum_videos.py` writes to `media/`).
- **Raw video frames are never written to disk.** Frames are streamed
  from `video_source.py`, held in memory for one run, and discarded.
  Only derived numeric signals (retina vectors reduced to a 12x12
  grid, reflex scores, fitness, mutation history) ever reach `state/`.
  This matters more than the other rules here: one of the real camera
  sources this runs against faces the user's own bedroom. Worth being
  precise, not just reassuring: the 12x12 grid IS a (very coarse,
  144-value) derived image, written to `state/live_status.json` every
  generation and served by the viewer above — nowhere near
  reconstructable into anything recognizable, but not literally
  "nothing," either. The hard rule is narrower and still holds: no
  FULL-resolution frame, not even transiently, ever reaches disk.
- No unbounded loops, recursion, or wall-clock time — every run is
  hard-capped on generations, wall-clock seconds, and tree size
  (`fishbowl/sandbox.py`'s `Limits`), checked before each generation,
  not trusted to self-regulate.
- No primitive vocabulary it can extend, and no fitness definition it
  can redefine. It recurses over HOW it mutates, never over WHAT it's
  capable of computing or WHAT counts as success — the second one
  especially is deliberately off the table (classic reward-hacking
  territory). If either door ever opens, that's a decision to make
  explicitly and in writing here, not something to slide into.

## Foveated vision

There's no motorized camera, so pan/tilt is digital: a small window
("fovea", `fishbowl/fovea.py`) is cropped from the real full frame,
and the organism's own `pan`/`tilt` output trees decide where that
window moves next, bounded to a small step per frame and clamped
inside the real image. Same shape as active/foveated vision models in
both biology (the eye doesn't process its whole field at high
resolution at once) and machine learning (Mnih et al.'s "Recurrent
Models of Visual Attention" — a small glimpse window, and choosing
where to look next is itself a trainable action). This is what makes
CONSPEC's "seek" (below) a real, actionable drive instead of a passive
correlation.

Each `response`/`pan`/`tilt` tree sees the CURRENT fovea vector AND
the PREVIOUS one (`run_vision.py`'s `evaluate_genome`) — added
2026-09-24 after an external audit found the original memoryless,
single-frame input made it structurally impossible for the response
tree to represent the frame-DIFFERENCE reflex signals it's graded
against. Whether it evolves to actually use the difference is still up
to it; it just isn't impossible anymore.

Real self-stimulation gotcha, found and fixed the same day: grading
used to run on the fovea's OWN cropped, panned view, which meant
simply MOVING the fovea across a static scene manufactured apparent
motion/luminance-change out of nothing (the same way saccades would
look like the world moved without a real eye's corollary-discharge
correction). All reflex/conspec grading now runs on the FULL,
un-foveated frame instead (`run_vision.py`'s `_world_vectors`),
computed once per run, independent of any genome's fovea path — see
that function's own docstring.

## Reflexes and drives (fitness, never input)

FIXED, never-evolved signals (`fishbowl/reflexes.py`) grade fitness —
the organism never sees these as input, only gets scored against them:

- **luminance_change** — global brightness change. The most primitive
  orienting response that exists.
- **motion_energy** — coherent whole-field CHANGE, direction-blind.
  Renamed from "optomotor" 2026-09-24 after an external audit caught
  the overclaim -- a real optomotor response is direction-selective;
  this can't tell rightward from leftward, only "something changed."
- **directional_motion** — a real, simplified Hassenstein-Reichardt
  correlator (the classic model of insect motion detection): computes
  WHICH WAY something moved, not just that it moved. See
  `reflexes.py`'s module docstring for the actual math.
- **loom** — a real, simplified approximation of tau (time-to-contact
  from expansion rate) — the reflex behind flinch/startle responses to
  an approaching object, studied down to the single-neuron level in
  locusts. A real threat/food asymmetry is built into how these are
  weighted in `run_vision.py`: missing a loom event costs more than
  missing ordinary motion, mirroring the real evolutionary stakes.

A fifth module, `fishbowl/conspec.py`, is a DRIVE rather than a
reflex — named directly after Morton & Johnson's (1991) CONSPEC/
CONLERN theory of newborn face perception. CONSPEC is the crude,
innate template that biases orientation toward face-like
configurations before any learning; CONLERN is the process that later
learns to refine it. That split is this project's own architecture:
`conspec.py` is CONSPEC (fixed, human-written), and the genome's own
evolved response is standing in for CONLERN. It also carries real
habituation (`conspec.Habituation`) — a being detected repeatedly with
no accompanying threat costs progressively less to keep reacting to
("humans and cats are harmless noise... commensal beings," in the user's
own framing), while a being detected alongside a real threat-level
event never habituates the same way.

**Real orienting pressure**, added 2026-09-24 (the user, watching a real
deployed feed: "this cat's been there the whole time, but the fovea's
too primitive to evolve to lock on it"). Until this point, nothing
rewarded pan/tilt for actually moving TOWARD anything:
- **optokinetic pursuit** — does the fovea's real movement direction
  correlate with real detected motion direction? A real reflex,
  found across nearly all motile visual animals, that stabilizes gaze
  on moving things.
- **seek** — when CONSPEC detects a being strongly enough, is the
  fovea actually near it? Makes CONSPEC's own "seek" drive literal
  (its docstring used to call this an "honest limitation" — there IS a
  real pan/tilt actuator now) and matches the real CONSPEC/CONLERN
  literature more closely: real newborns orient head/eyes toward
  face-like stimuli, not just look longer at them.

Both are still purely REWARDS — same discipline as everything else
here: neither forces a specific movement or a specific tracking style
(smooth vs. saccadic is left for evolution to discover, not dictated).

## Curriculum

Real, established, long-running public video sources, staged easiest
to hardest by how evolutionarily ancient the real reflex is — see
`tools/fetch_curriculum_videos.py`. Deployed runs watch these **live**
(`run_vision.py live` — see its own `LIVE_SOURCES`), resolved fresh
via `yt-dlp -g` at the start of every bounded run and never downloaded
or cached to disk — genuinely new real content every restart, not the
same short loop replaying forever. `fetch_curriculum_videos.py`
remains for local/offline dev testing against a fixed clip, but is no
longer what the deployed service watches. Plus Tanzania's own native
camera feed (facing a low-activity bedroom — lights on/off, a person
moving through frame) as a future option once the curated stages are
solid; that source stays local to Tanzania and is subject to the
strict no-persistence rule above without exception.

## Repository map

| Path                              | Role |
|------------------------------------|------|
| `fishbowl/blocks.py`              | Safe operation vocabulary + pure tree evaluator |
| `fishbowl/genome.py`              | Multi-channel genome + two-level mutation logic |
| `fishbowl/retina.py`              | Frame -> fixed-size numeric vector |
| `fishbowl/fovea.py`               | Digital pan/tilt |
| `fishbowl/reflexes.py`            | luminance/optomotor/loom -- fixed, fitness-only |
| `fishbowl/conspec.py`             | Being-detection drive + habituation |
| `fishbowl/video_source.py`        | Real frame capture -- file, live device, or resolved live stream URL |
| `fishbowl/sandbox.py`             | Hard limits, atomic state writes, request log |
| `fishbowl/task.py`                | The original synthetic sanity-check task |
| `run_vision.py`                   | Real entry point: evolve against real video |
| `tools/fetch_curriculum_videos.py`| yt-dlp wrapper for curated clips |
| `state/evolution_log.jsonl`       | Append-only real run history, one JSON object per line (not a JSON array -- see `sandbox.py`'s `log_generation`) |
| `tools/viewer.py`                 | Unauthenticated LAN HTTP viewer for `state/live_status.json` |
| `tools/resource_handler.py`       | External CPUQuota supervisor (hunger/disgust/extinction) |
| `state/requests.json`             | Self-surfaced, human-reviewed-only requests |

## Status

2026-09-24: deployed and running 24/7 on Tanzania (systemd, see
`deploy/`) against real live video, with a LAN-facing viewer at
`tools/viewer.py`. An independent external audit the same day found
the ~18k-generation fitness plateau up to that point was NOT a
mutation-strategy problem — it traced to structural bugs (stale
fitness comparisons across restarts, a memoryless response tree graded
against frame-difference signals, a mutation operator that could
collapse an entire tree in one step, and a self-stimulation loophole
in how reflexes were graded). All four are fixed as of this commit;
see `run_vision.py`'s module docstring and `fishbowl/genome.py`'s
`_reroll_subtree` for the specifics. The temporal-input fix changes
`n_vars`, so this redeploy starts evolution over from a fresh genome —
expected, not a bug (old genomes couldn't reference the new inputs
anyway). Known, not-yet-fixed gaps from the same audit: the
observability viewer has no auth, `evolution_log.json` is rewritten in
full every generation (real disk-write volume concern for a years-long
run), and a failed live-stream resolve retries with no backoff.
