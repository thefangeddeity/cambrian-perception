# fishbowl

A recursively self-improving brain that programs itself inside a
sandbox with hard, real limits — to see what it does, not to make it
useful for anything in particular. The whole point is to watch, not
to deploy.

## The fishbowl boundary

This is the part that has to be right before anything else matters.
The brain lives entirely inside `fishbowl/`, and everything it can
touch is enumerated here — if something isn't listed, it can't do it.

**What the brain can do:**
- Compose and mutate small trees of pure, numeric operations (see
  `fishbowl/blocks.py`) — arithmetic, comparison, a protected
  conditional. No loops or recursion exist in this language; a tree
  is finite and is walked exactly once per evaluation, so evaluating
  it always terminates. There's no way to write an infinite loop in
  it, not "there's a timeout in case you do."
- Mutate not just its task program but the WEIGHTS that decide how it
  mutates itself (`fishbowl/genome.py`) — that's the "recursive" part:
  it's not just improving its answer, it's improving how it decides
  to change its answer. This is deliberately bounded to two levels
  (task program, and the mutation-strategy weights that shape how the
  task program changes) plus one scalar meta-mutation rate governing
  how fast those weights themselves drift — not unbounded meta-meta-
  regress. Said plainly: this is real, working recursive self-
  modification, not a marketing claim for something bigger.
- Log a structured, plain-language "request" when it hits one of its
  own ceilings repeatedly (`fishbowl/sandbox.py`'s `log_request()`) —
  e.g. "hit MAX_TREE_NODES on 8 of the last 10 generations." This is
  purely observational. The brain cannot act on a request, grant
  itself more budget, or change a ceiling. A human (or whatever
  observation tool gets built next) reads requests and decides.

**What the brain cannot do, by construction, not by policy:**
- No `eval`, `exec`, `os.system`, `subprocess`, or dynamic `import`
  anywhere in this codebase. The block-tree interpreter in
  `blocks.py` is the only thing that ever "runs" a brain-authored
  program, and it only ever does arithmetic on numpy arrays.
- No network access. Nothing in this repo opens a socket.
- No filesystem access beyond `state/` in this repo, and only through
  one whitelisted writer (`sandbox.py`'s `_write_json_atomic()`) — no
  arbitrary path ever reaches a brain-influenced value.
- No unbounded loops, recursion, or wall-clock time. Every run is
  hard-capped on generations, wall-clock seconds, and tree size
  (`fishbowl/sandbox.py`'s `Limits`), checked BEFORE each generation
  starts, not trusted to self-regulate.
- No access to real-world data. The task it evolves against
  (`fishbowl/task.py`) is a synthetic, fixed, hidden function with no
  external dependency — there's nothing "real" for it to affect even
  if every other constraint somehow failed.

If a future change would require loosening any of the above, that's a
decision to make explicitly and in writing here, not something to
quietly work around because the sandbox is in the way. The sandbox
being in the way sometimes is the point.

## What it actually does right now

`run_fishbowl.py` runs N generations of an evolutionary loop:

1. Propose a mutation to the current genome (task tree, or — with
   probability drawn from the genome's own mutation weights — the
   mutation weights themselves).
2. Score the mutated genome against `task.py`'s hidden function on a
   fixed evaluation set.
3. Keep it if it scores better than the current genome; discard it
   otherwise.
4. Log the generation (fitness, what mutated, accepted or not, the
   current mutation-weight snapshot) to `state/evolution_log.json`.
5. If a ceiling got hit, log a request to `state/requests.json`.

That's the whole loop. No task-specific cleverness beyond that — the
interesting part is watching what the mutation weights themselves
drift toward over a long run, not the task score itself.

## Repository map

| Path                        | Role |
|------------------------------|------|
| `fishbowl/blocks.py`        | The safe operation vocabulary + pure tree evaluator |
| `fishbowl/genome.py`        | Task tree + mutation-weight genome, and the two-level mutation logic |
| `fishbowl/task.py`          | The synthetic, fixed evaluation task |
| `fishbowl/sandbox.py`       | Hard limits, atomic state writes, the request log |
| `run_fishbowl.py`           | Entry point |
| `state/evolution_log.json`  | Append-only real run history |
| `state/requests.json`       | Self-surfaced, human-reviewed-only requests |

## Status

2026-09-24: initial version. Runs, self-modifies (including its own
mutation weights), logs, and respects every ceiling in `sandbox.py`
-- verified live before this was committed. No observation UI yet;
that's being scoped as its own follow-on so it isn't designed in a
rush alongside the sandbox itself.
