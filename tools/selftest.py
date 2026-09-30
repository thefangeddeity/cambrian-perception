"""
The organism's self-test: the checks its development was verified with,
kept with it, so a broken change can't reach a living organism unnoticed.
Every installer runs it before restarting, and keeps the running code if it
fails. Offline (no stream, no network, no GPU); writes only to a temporary
folder; a couple of minutes on a small laptop.

    python tools/selftest.py           # exit 0 = all passed
"""
from __future__ import annotations

import json
import math
import pathlib
import random
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

TMP = pathlib.Path(tempfile.mkdtemp(prefix="cambrian-selftest-"))
from fishbowl import sandbox  # noqa: E402

for _k in dir(sandbox):  # every state path into the temporary folder: the host's state is never touched
    _v = getattr(sandbox, _k)
    if (_k.endswith("_PATH") or _k.endswith("_DIR")) and isinstance(_v, pathlib.Path):
        setattr(sandbox, _k, TMP / _v.name)
sandbox.STATE_DIR = TMP

import run_vision as RV  # noqa: E402
from fishbowl import blocks, controller as C, fovea, genome as G, live, livelife, organism as O, state as S  # noqa: E402
from fishbowl.field import FieldSignals  # noqa: E402
from tools import seed as seeds  # noqa: E402

RESULTS: list = []
CHECKS: list = []  # registered here, run by main() -- never on import (a worker process re-imports this file)


def check(name):
    def wrap(fn):
        CHECKS.append((name, fn))
        return fn
    return wrap


def run_checks() -> None:
    for name, fn in CHECKS:
        t0 = time.time()
        try:
            fn()
            RESULTS.append((name, True, f"{time.time() - t0:.1f} s"))
        except Exception as e:  # a failure is reported, and the rest still run
            RESULTS.append((name, False, f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}"))
        print(f"{'ok  ' if RESULTS[-1][1] else 'FAIL'}  {name}  ({RESULTS[-1][2].splitlines()[0] if RESULTS[-1][1] else 'see below'})", flush=True)


def founder(seed=3, **kw):
    g = G.random_genome(random.Random(seed), n_vars=RV.TREE_PLAIN_INPUTS, receptors=fovea.DEFAULT_RECEPTORS)
    for name, fn in seeds.SEED_SETS.items():
        fn(g, random.Random(name))
    for k, v in kw.items():
        setattr(g, k, v)
    return g


_SCENE: dict = {}


def ride(n, start=0, speed=6):
    # its world is made on first use, never on import: a worker process re-imports
    # this file, and OpenCV run while importing broke forked workers (macOS)
    if not _SCENE:
        rng = np.random.default_rng(1)
        _SCENE["world"] = cv2.GaussianBlur((rng.random((900, 8000)) * 255).astype(np.uint8), (7, 7), 2)
        _SCENE["dash"] = cv2.GaussianBlur((rng.random((90, 640)) * 255).astype(np.uint8), (7, 7), 2)
    WORLD, DASH = _SCENE["world"], _SCENE["dash"]
    out = []
    for t in range(start, start + n):
        f = WORLD[100:460, 40 + speed * t:680 + speed * t].copy()
        f[270:] = DASH
        out.append(f)
    return out


@check("genome: round trip, clone, seeds idempotent, kappa only at laying")
def _():
    g = founder()
    d = json.loads(json.dumps(g.to_dict()))
    g2 = G.Genome.from_dict(d)
    assert g2.to_dict() == g.to_dict(), "round trip changed the genome"
    assert g.clone().to_dict() == g.to_dict(), "clone changed the genome"
    again = [fn(g, random.Random(n)) for n, fn in seeds.SEED_SETS.items()]
    assert not any(again), f"a seed set applied twice: {again}"
    egg = g.laid_egg(random.Random(5))
    assert [k for k in d if d[k] != egg.to_dict()[k]] == ["kappa"]
    assert 0.0 < egg.kappa < 1.0
    assert not any("kappa" in op for op in G.TASK_OPS)


@check("perception tree: 2000 leaf mutations, finite and identical after save/load")
def _():
    g = founder()
    leaf = lambda i: blocks.Node(kind="cell", index=i % 4, kx=i - 3, ky=3 - i)  # noqa: E731
    g.trees["response"] = blocks.Node(kind="op", op="add", children=[
        blocks.Node(kind="op", op="add", children=[leaf(0), leaf(1)]), blocks.Node(kind="op", op="max", children=[leaf(2), leaf(5)])])
    rng = random.Random(7)
    ret = np.random.default_rng(0).random((3, 4, 22, 22))
    X = np.random.default_rng(1).random((3, RV.TREE_PLAIN_INPUTS))
    for i in range(2000):
        g._mutate_pool(rng, "response")
        t = g.trees["response"]
        y = t.evaluate(X, ret)
        assert np.all(np.isfinite(y)), f"non-finite at {i}"
        assert np.allclose(blocks.Node.from_dict(json.loads(json.dumps(t.to_dict()))).evaluate(X, ret), y), f"save/load changed it at {i}"


@check("brain: energy bound on founders, growth, layers, channels")
def _():
    top = C.founder_units_max()
    sizes = [C.MosquitoBrain.random(random.Random(i)).n_hidden for i in range(300)]
    assert min(sizes) >= C.MIN_HIDDEN and max(sizes) <= top
    b = C.MosquitoBrain.random(random.Random(1), hidden=top - 5)
    rng = random.Random(2)
    while b.grow_unit(rng):
        pass
    b2 = C.MosquitoBrain.random(random.Random(3), hidden=8)
    while b2.duplicate_layer(rng):
        b2.layers[-1]["gate"][0] = 0.5
    while b2.grow_channel(rng):
        pass
    assert b.cost_if() <= C.MAX_THINK_FACTOR + 1e-9 and b2.cost_if() <= C.MAX_THINK_FACTOR + 1e-9


@check("body: the kappa rule, starvation, age, torpor, the cyst")
def _():
    b = S.MosquitoState(); b.kappa = 0.8
    assim = 0.0
    for _ in range(2000):
        b.gut = 0.5
        assim += 0.5 * S.GUT_CAP * (1 - math.exp(-10.0 / S.DIGEST_TAU_S)) * (1 - S.SDA_FRACTION)
        b.update(0, 0, 0, 0.0, dt=1, pace=1, dt_seconds=10.0)
    assert abs(b.repro - 0.2 * assim) < 1e-6 * assim, "the kappa rule"
    s = S.MosquitoState()
    for _ in range(40000):
        s.gut = 0.0
        s.update(0, 0, 0, 0.0, dt=1, pace=1, dt_seconds=10.0)
        if s.death():
            break
    assert s.death() == "starvation", "a starved body must die"
    t = S.MosquitoState(); t.energy = t.glycogen = t.reserve = t.gut = 0.0; t.wasting = 0.5; t.torpid = True
    for _ in range(2000):
        t.update(0, 0, 0, 0.0, dt=1, pace=1, dt_seconds=10.0)
    assert t.wasting == 0.5 and t.death() is None, "torpor must not waste"
    c = S.MosquitoState(); c.glycogen = 1.0
    before = c.sugar
    c.encyst(S.CYST_COST)
    assert c.encysted == 1.0 and abs(before - c.sugar - S.CYST_COST / S.STORE_EFFICIENCY) < 1e-6
    c.revive()
    assert c.encysted == 0.0 and c.protectant == 0.0


def live_organism(g, frames, colour=False):
    org = O.Organism(g, None, None, 100.0, 10.0, colour=colour, prey=True, record=False)
    fs = FieldSignals(); fs.fps = 10
    for f in frames:
        fc = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR) if colour else None
        sig, sh = fs.step(f)
        org.frame(f, sig, [[0, 0.9, 0.4, 0.3, 0.46, 0.9]], fc, sh)
    return org


@check("organism: frames at both ends of brain size and eye size, colour on")
def _():
    for h, rec in ((1, fovea.DEFAULT_RECEPTORS), (C.founder_units_max(), fovea.MAX_RECEPTORS)):
        g = founder(colour_channels=2)
        g.receptors = rec
        g.brain = C.MosquitoBrain.random(random.Random(5), hidden=h)
        org = live_organism(g, ride(40), colour=True)
        assert org.k == 40 and len(org.brain.tree_view()) == C.TREE_HIDDEN


@check("evolution's scoring: the batch path, in a worker and here, the same")
def _():
    frames = ride(40)
    w = RV.World(frames, RV._world_vectors(frames), 10.0, [[] for _ in frames])
    g = founder()
    here = RV.evaluate_genome(g, *w.at_pace(1), 150.0, S.MosquitoState().to_dict(), 10.0, w.prey, None, None, 0.0)[0]
    assert math.isfinite(here)
    pool = RV._Workers(1)
    try:
        assert pool.publish(w)
        there = pool.submit(g, 150.0, S.MosquitoState().to_dict(), 10.0, None, 0.0).result(timeout=600)[0]
        assert abs(here - there) < 1e-9, f"worker {there} vs here {here}"
    finally:
        pool.close()


@check("a live body: 10 s on a fake feed, no error, its status written")
def _():
    g = founder()

    class Feed:
        def __init__(s): s.t0 = time.time()
        def frames_per_second(s): return 10.0
        def since(s, index):
            n = int((time.time() - s.t0) * 10); start = n - 1 if index is None else index + 1
            return [(k, ride(1, k)[0], [], None, s.t0 + k / 10) for k in range(max(0, start), n)]

    from fishbowl.metrics import HourlyMetrics
    life = livelife.LiveLife(Feed(), g, None, None, 100.0, 0.0, 1, TMP / "live_actor.json", HourlyMetrics(), 10.0)
    time.sleep(10)
    life.stop()
    assert life.error is None, f"{life.error}\n{life.error_trace}"
    d = json.loads((TMP / "live_actor.json").read_text())
    for key in ("tally", "life", "senses", "oxygen"):
        assert key in d, f"no {key} in its status"


@check("an adoption into fewer Kenyon cells keeps living")
def _():
    old = live_organism(founder(kc=256, felt_terrain=1, lookahead=1, learning_rate=0.3), ride(30))
    new = O.Organism(founder(kc=64, felt_terrain=1, lookahead=1, learning_rate=0.3), old.body.to_dict(), livelife.memory_of(old),
                     100.0, 10.0, colour=False, prey=True, record=False)
    livelife._carry(old, new)
    fs = FieldSignals(); fs.fps = 10
    for f in ride(30, 30):
        sig, sh = fs.step(f)
        new.frame(f, sig, [[0, 0.9, 0.4, 0.3, 0.46, 0.9]], None, sh)


@check("statistics: Student's t tail, the development test")
def _():
    for t, df, want in ((2.920, 2, 0.05), (4.303, 2, 0.025), (2.015, 5, 0.05)):
        assert abs(RV._t_sf(t, df) - want) < 5e-4, (t, df, RV._t_sf(t, df))
    assert not RV._developed([0.3]) and RV._developed([0.4, 0.5, 0.45, 0.55, 0.5]) and not RV._developed([0.5, -0.4, 0.6, -0.3])


@check("migration never keeps a lineage from starting")
def _():
    from tools import fleet
    frames = ride(30)
    w = RV.World(frames, RV._world_vectors(frames), 10.0, [[] for _ in frames])
    f = founder()
    saved = fleet.discover

    def boom(names):
        raise OSError("no fleet")
    fleet.discover = boom
    try:
        assert RV._migrate(w, f, RV.TREE_PLAIN_INPUTS) is f
    finally:
        fleet.discover = saved


@check("viewer: every <script> parses (node --check), where node exists")
def _():
    node = shutil.which("node")
    if not node:
        return
    import re
    src = (ROOT / "tools" / "viewer.py").read_text(encoding="utf-8")
    for i, blk in enumerate(re.findall(r"<script>(.*?)</script>", src, re.S)):
        f = TMP / f"s{i}.js"
        f.write_text(blk, encoding="utf-8")
        r = subprocess.run([node, "--check", str(f)], capture_output=True, text=True)
        assert r.returncode == 0, f"script {i}: {r.stderr.strip()[:300]}"


def main() -> int:
    run_checks()
    failed = [r for r in RESULTS if not r[1]]
    for name, ok, note in failed:
        print(f"FAIL  {name}\n      " + note.replace("\n", "\n      "))
    print(f"{len(RESULTS) - len(failed)} of {len(RESULTS)} passed.")
    shutil.rmtree(TMP, ignore_errors=True)
    return 1 if failed else 0


if __name__ == "__main__":
    threading.current_thread().name = "selftest"
    raise SystemExit(main())
