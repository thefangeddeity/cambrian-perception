# cambrian-perception

A recursively self-improving organism that evolves its own perception
from raw video — no pretrained models, sandboxed, reflex-driven,
growing its own visual machinery from scratch. The point is to watch
what it comes up with, not to deploy it for anything.

Named for the real thing it's meant to echo: watching a Cambrian-era
nervous system slowly discover what real eyes are for, under real
evolutionary pressure, on hardware that can iterate millions of times
faster than 400 million years of actual selection.

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
- Mutate not just its output trees but the WEIGHTS that decide how it
  mutates itself — that's the recursive part: it's not just improving
  its answer, it's improving how it decides to change its answer.
  Deliberately bounded to two levels (the trees, and the mutation-
  strategy weights that shape how they change) plus one scalar meta-
  mutation rate governing how fast those weights drift — not unbounded
  meta-meta-regress.
- Log a structured, plain-language request when it hits one of its own
  ceilings repeatedly (`fishbowl/sandbox.py`'s `log_request`). Purely
  observational — it cannot act on a request or change a ceiling. A
  human, or the observation tool being scoped separately, reads
  requests and decides.

**What it cannot do, by construction, not by policy:**
- No `eval`, `exec`, `os.system`, `subprocess`, or dynamic `import`
  anywhere in this codebase. The block-tree interpreter in `blocks.py`
  is the only thing that ever "runs" an organism-authored program, and
  it only ever does arithmetic on numpy arrays.
- No network access beyond reading its own configured local video
  source — no arbitrary host is ever reachable.
- No filesystem access beyond `state/`, and only through one
  whitelisted writer (`sandbox.py`'s `_write_json_atomic`).
- **Raw video frames are never written to disk.** Frames are streamed
  from `video_source.py`, held in memory for one run, and discarded.
  Only derived numeric signals (retina vectors reduced to a 12x12
  grid, reflex scores, fitness, mutation history) ever reach `state/`.
  This matters more than the other rules here: one of the real camera
  sources this runs against faces the user's own bedroom. Nothing
  reconstructable into an actual image of it should ever exist on
  disk, even transiently, even in a debug log.
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

## Reflexes and drives (fitness, never input)

Three FIXED, never-evolved signals (`fishbowl/reflexes.py`) grade
fitness — the organism never sees these as input, only gets scored
against them:

- **luminance_change** — global brightness change. The most primitive
  orienting response that exists.
- **optomotor** — coherent whole-field motion.
- **loom** — a real, simplified approximation of tau (time-to-contact
  from expansion rate) — the reflex behind flinch/startle responses to
  an approaching object, studied down to the single-neuron level in
  locusts. A real threat/food asymmetry is built into how these are
  weighted in `run_vision.py`: missing a loom event costs more than
  missing ordinary motion, mirroring the real evolutionary stakes.

A fourth module, `fishbowl/conspec.py`, is a DRIVE rather than a
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

## Curriculum

Real, established, long-running public video sources, staged easiest
to hardest by how evolutionarily ancient the real reflex is — see
`tools/fetch_curriculum_videos.py`. Plus Tanzania's own native camera
feed (facing a low-activity bedroom — lights on/off, a person moving
through frame) once the curated stages are solid; that source stays
local to Tanzania and is subject to the strict no-persistence rule
above without exception.

## Repository map

| Path                              | Role |
|------------------------------------|------|
| `fishbowl/blocks.py`              | Safe operation vocabulary + pure tree evaluator |
| `fishbowl/genome.py`              | Multi-channel genome + two-level mutation logic |
| `fishbowl/retina.py`              | Frame -> fixed-size numeric vector |
| `fishbowl/fovea.py`               | Digital pan/tilt |
| `fishbowl/reflexes.py`            | luminance/optomotor/loom -- fixed, fitness-only |
| `fishbowl/conspec.py`             | Being-detection drive + habituation |
| `fishbowl/video_source.py`        | Real frame capture, file or live device |
| `fishbowl/sandbox.py`             | Hard limits, atomic state writes, request log |
| `fishbowl/task.py`                | The original synthetic sanity-check task |
| `run_vision.py`                   | Real entry point: evolve against real video |
| `tools/fetch_curriculum_videos.py`| yt-dlp wrapper for curated clips |
| `state/evolution_log.json`        | Append-only real run history |
| `state/requests.json`             | Self-surfaced, human-reviewed-only requests |

## Status

2026-09-24: real, verified end-to-end run against a synthetic test
clip (fitness 0.0 -> 3.56 over 60 generations, habituation state
tracking correctly, full log output) -- the whole pipeline (video ->
fovea -> retina -> genome -> reflexes/conspec -> fitness -> mutation
-> logging) works, not a stub. Deploying to Tanzania next for a real
run against real curated footage. No observation UI yet; that's being
scoped as its own follow-on so it isn't designed in a rush alongside
the sandbox itself.
