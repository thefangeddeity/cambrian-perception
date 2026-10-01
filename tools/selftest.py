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
        sandbox.SETTINGS_PATH.write_text('{"hive": true}')
        assert sandbox.hive()
        assert RV._migrate(w, f, RV.TREE_PLAIN_INPUTS) is f
    finally:
        fleet.discover = saved
        sandbox.SETTINGS_PATH.unlink(missing_ok=True)


@check("solo (no hive): it never asks the fleet; the installers' settings keep what is there")
def _():
    from tools import fleet, settings
    saved = fleet.discover

    def asked(names):
        raise AssertionError("a solo organism asked the fleet")
    fleet.discover = asked
    try:
        assert not sandbox.hive()  # no settings: solo
        f = founder()
        assert RV._migrate(None, f, RV.TREE_PLAIN_INPUTS) is f
    finally:
        fleet.discover = saved
    p = TMP / "settings-test.json"
    settings.main([str(p), "keep", "source=0", "viewer_port=8090"])
    s = json.loads(p.read_text())
    assert s == {"hive": False, "source": "0", "viewer_port": 8090}, s
    settings.main([str(p), "true"])
    settings.main([str(p), "keep", "source=rtsp://x"])
    s = json.loads(p.read_text())
    assert s["hive"] is True and s["source"] == "rtsp://x" and s["viewer_port"] == 8090, s


@check("a power cut: damaged checkpoints resume its founder, a damaged history its previous copy")
def _():
    from fishbowl import state as S_
    ck = {"genome": founder().to_dict(), "generation": 7}
    sandbox.save_checkpoint(ck)
    sandbox.save_checkpoint({**ck, "generation": 8})  # now a .prev too
    assert sandbox.load_checkpoint()["generation"] == 8
    sandbox.CHECKPOINT_PATH.write_text('{"genome": {"tr')  # torn mid-write
    assert sandbox.load_checkpoint()["generation"] == 7  # its previous copy
    sandbox.FOUNDER_PATH.write_text(json.dumps({**ck, "generation": 0}))
    sandbox.CHECKPOINT_PREV_PATH.write_text("")  # both damaged
    got = sandbox.load_checkpoint()
    assert got is not None and got["generation"] == 0, got  # its own founder, never a crash loop
    assert not sandbox.CHECKPOINT_PATH.exists() and list(TMP.glob("damaged-*/checkpoint.json")), "damaged files kept aside"
    assert sandbox.life_history()[-1]["event"] == "resumed after damage"
    n = len(sandbox.life_history())
    sandbox.record_life({"event": "test"})
    sandbox.LIFE_HISTORY_PATH.write_text("[{")  # torn
    assert len(sandbox.life_history()) == n, "its previous copy"
    sandbox.FOUNDER_PATH.unlink()
    # a stream failing while the machine has no network is not counted against it
    assert sandbox.source_failed("https://stream.invalid/x") is False
    del S_


@check("the hive: a peer's genome is held to this host's own limits")
def _():
    from tools import fleet
    good = founder().to_dict()
    assert fleet.vet_genome(good) is None, fleet.vet_genome(good)
    bad_op = json.loads(json.dumps(good))
    name = next(iter(bad_op["trees"]))
    bad_op["trees"][name] = {"kind": "op", "op": "__import__", "children": []}
    assert fleet.vet_genome(bad_op)
    big_const = json.loads(json.dumps(good))
    big_const["trees"][name] = {"kind": "const", "value": 1e300}
    assert fleet.vet_genome(big_const)
    deep = {"kind": "const", "value": 1.0}
    for _i in range(200):
        deep = {"kind": "op", "op": "neg", "children": [deep]}
    too_deep = json.loads(json.dumps(good))
    too_deep["trees"][name] = deep
    assert fleet.vet_genome(too_deep)
    nan_brain = json.loads(json.dumps(good))
    nan_brain["brain"]["bias_h"][0] = float("nan")
    assert fleet.vet_genome(json.loads(json.dumps(nan_brain).replace("NaN", "NaN")))
    assert fleet.vet_genome("not a genome")


@check("a second organism (parked): its paths are its own, and the three derivations agree")
def _():
    code = ("import json; from fishbowl import sandbox as s; from tools import viewer as v, resource_handler as r; "
            "print(json.dumps([s.STATE_DIR.name, v.STATE_DIR.name, r.STATE_DIR.name, str(s._RUNTIME_DIR), str(v._RUNTIME_DIR), "
            "v.DEFAULT_PORT, v.ORGANISM_UNIT, r.SERVICE_NAME]))")
    for inst, want in (("", ["state", 8090, "cambrian-perception.service"]), ("b", ["state-b", 8091, "cambrian-perception-b.service"])):
        env = {**__import__("os").environ, "CAMBRIAN_INSTANCE": inst}
        env.pop("CAMBRIAN_RUNTIME_DIR", None)
        out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
        assert out.returncode == 0, out.stderr[-800:]
        s_dir, v_dir, r_dir, s_rt, v_rt, port, unit, svc = json.loads(out.stdout.strip().splitlines()[-1])
        assert s_dir == v_dir == r_dir == want[0], (s_dir, v_dir, r_dir)
        assert s_rt == v_rt, (s_rt, v_rt)
        assert port == want[1] and unit == svc == want[2], (port, unit, svc)


@check("its diet from the host's settings: a teddy bear as food, people always food")
def _():
    from fishbowl import prey
    saved = (prey.PREY_CLASSES, prey.NECTAR_COCO, prey.NECTAR_OIV7)
    try:
        assert prey.COCO_NAMES[0] == "person" and prey.COCO_NAMES[58] == "potted plant" and prey.COCO_NAMES[77] == "teddy bear"
        assert prey.diet()["food"][0] == "person" and "potted plant" in prey.diet()["nectar"]  # the defaults (no settings)
        sandbox.SETTINGS_PATH.write_text(json.dumps({"diet": {"food": ["cat", "Teddy Bear", "unicorn"], "nectar": ["banana", "cat", "Flower"]}}))
        problems = prey._apply_diet()
        assert prey.PREY_CLASSES == {0: "person", 15: "cat", 77: "teddy bear"}, prey.PREY_CLASSES
        assert prey.NECTAR_COCO == {46} and prey.NECTAR_OIV7 == {"Flower"}, (prey.NECTAR_COCO, prey.NECTAR_OIV7)
        assert any("unicorn" in p for p in problems) and any("'cat' is food already" in p for p in problems), problems
        bear = [[77, 0.9, 0.4, 0.4, 0.6, 0.9]]
        assert prey.hosts_only(bear) == bear and prey.things_only(bear) == []
    finally:
        prey.PREY_CLASSES, prey.NECTAR_COCO, prey.NECTAR_OIV7 = saved
        sandbox.SETTINGS_PATH.unlink(missing_ok=True)


@check("a saved organism (.cambrioid): a round trip, and every kind of bad file refused")
def _():
    import io
    import zipfile
    from tools import organism_file as of
    src, dst = TMP / "save-src", TMP / "save-dst"
    src.mkdir(); dst.mkdir()
    ck = {"genome": founder().to_dict(), "total_generation": 42, "saved_at": time.time()}
    (src / "checkpoint.json").write_text(json.dumps(ck))
    (src / "founder.json").write_text(json.dumps(ck))
    buf = io.BytesIO()
    np.savez_compressed(buf, lens=np.array([2]), codes=np.array([1, 2]))
    (src / "episodes.npz").write_bytes(buf.getvalue())
    (dst / "checkpoint.json").write_text(json.dumps({**ck, "total_generation": 7}))  # the one it replaces
    blob, m = of.save_bytes(src, "a cute one")
    assert m["name"] == "a cute one" and m["generation"] == 42
    m2, backup = of.install(blob, dst)
    assert json.loads((dst / "checkpoint.json").read_text())["total_generation"] == 42
    assert json.loads((backup / "checkpoint.json").read_text())["total_generation"] == 7, "the replaced one kept"
    hist = json.loads((dst / "life_history.json").read_text())
    assert hist[-1]["event"] == "loaded" and hist[-1]["name"] == "a cute one"
    assert sandbox.LIFE_HISTORY_PATH.parent == TMP, "install put sandbox's paths back"

    def rezip(files, manifest=None):
        out = io.BytesIO()
        with zipfile.ZipFile(out, "w") as z:
            z.writestr("manifest.json", json.dumps(manifest or {**m, "files": {k: of._sha(v) for k, v in files.items()}}))
            for k, v in files.items():
                z.writestr(k, v)
        return out.getvalue()
    _, files = of.read(blob)
    bad = {
        "tampered": rezip({**files, "checkpoint.json": files["checkpoint.json"].replace(b"42", b"43")}, m),
        "a foreign op": rezip({**files, "checkpoint.json": json.dumps({**ck, "genome": {**ck["genome"], "trees": {"response": {"kind": "op", "op": "__import__", "children": []}}}}).encode()}),
        "an extra member": rezip({**files, "run.sh": b"rm -rf /"}),
        "a pickle": rezip({**files, "episodes.npz": (lambda b: (np.save(b, np.array([object()], dtype=object), allow_pickle=True), b.getvalue())[1])(io.BytesIO())}),
        "not a zip": b"hello",
        "a newer format": rezip(files, {**m, "format": 99}),
        # its bytes damaged where its members are (past the first local header): refused, never a crash
        "damaged": blob[:200] + bytes(b ^ 0x5A for b in blob[200:600]) + blob[600:],
        "cut short": blob[:len(blob) // 2],
    }
    # folders that don't exist yet are made: a new folder given, and a file in a new folder
    made = of.save(src, TMP / "new" / "Games" / "Lifeforms" / "Cambrioids")
    assert made.parent == TMP / "new" / "Games" / "Lifeforms" / "Cambrioids" and made.suffix == ".cambrioid", made
    named = of.save(src, TMP / "other" / "deep" / "mine.cambrioid")
    assert named == TMP / "other" / "deep" / "mine.cambrioid" and named.exists(), named
    (src / "checkpoint.json").write_text(json.dumps({"genome": ck["genome"]}))  # no generation recorded
    assert "?" not in of.default_name(src) and of.default_name(src).endswith(".cambrioid"), of.default_name(src)
    for why, b in bad.items():
        try:
            of.read(b)
        except of.Refused:
            continue
        raise AssertionError(f"{why}: not refused")


@check("its output survives a code page without jīng (Windows' cp1252 log): no crash")
def _():
    # 2026-09-30: printing "Jīng" into a supervisor's cp1252 log crashed 7elwe's
    # organism at every start. Each entry point's own guard, in a child whose
    # output is cp1252, must print the treasures and exit cleanly.
    import os
    env = {**os.environ, "PYTHONIOENCODING": "cp1252", "PYTHONUTF8": "0"}
    code = "import sys; sys.path.insert(0, %r); import run_vision as R; R.utf8_stdio(); print('Jīng 精 · qì 氣 · shén 神')" % str(ROOT)
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, timeout=120)
    assert out.returncode == 0, out.stderr.decode("utf-8", "replace")[-600:]
    assert "Jīng 精".encode("utf-8") in out.stdout, out.stdout[-200:]
    for f in ("tools/viewer.py", "tools/cambrian_service.py", "tools/organism_file.py", "tools/cambrian_ctl.py", "tools/fleet.py"):
        src = (ROOT / f).read_text(encoding="utf-8")
        assert "reconfigure(" in src.split('if __name__ == "__main__":', 1)[1][:600], f"{f}: no output guard at its entry"


@check("one person split by a chair back is one host; two people side by side stay two")
def _():
    from fishbowl import prey
    # Ariana, 2026-09-30: her upper body and her whole self, overlapping 0.69 of their union
    split = [[0, 0.62, 0.49, 0.09, 0.65, 0.56], [0, 0.48, 0.49, 0.09, 0.66, 0.73]]
    kept = prey.drop_contained(split)
    assert kept == [split[0]], kept  # the surer box stays
    pair = [[0, 0.8, 0.10, 0.2, 0.30, 0.9], [0, 0.7, 0.28, 0.2, 0.48, 0.9]]  # two people, shoulders touching
    assert len(prey.drop_contained(pair)) == 2
    mixed = [[0, 0.9, 0.2, 0.2, 0.6, 0.9], [56, 0.6, 0.25, 0.5, 0.55, 0.85]]  # a person in a chair: different classes
    assert len(prey.drop_contained(mixed)) == 2


@check("a camera, by number or by path, is opened as a camera; a stream as a stream")
def _():
    # 2026-09-30: a by-id camera path went to FFmpeg and never gave a frame (Tanzania)
    from fishbowl import video_source as vs
    for cam in (0, 1, "0", "1", "/dev/video0", "/dev/v4l/by-id/usb-DJKCVA1G4EKI5L_HP_TrueVision_HD_Camera_0001-video-index0"):
        assert vs.capture_kind(cam) == "camera", cam
    for stream in ("https://example.org/live.m3u8", "rtsp://127.0.0.1:8554/cam", "media/clip.mp4"):
        assert vs.capture_kind(stream) in ("ffmpeg", "pipe"), stream


@check("its horizon organ finds a drawn floor's horizon (letterboxed too) and a sea's under a palm, and its organism hears it")
def _():
    import math
    from fishbowl import organs
    # a tiled floor seen by a camera turned 35 degrees: its horizon 40% down
    h, w, hy, f = 180, 320, 0.4, 250.0
    img = np.full((h, w), 90, np.uint8)
    c, sn = math.cos(math.radians(35)), math.sin(math.radians(35))

    def P(X, Z):
        Xc, Zc = c * X - sn * Z, sn * X + c * Z
        return (w / 2 + f * Xc / Zc, hy * h + f / Zc) if Zc > 0.3 else None
    for i in range(-20, 21):
        for a, b in [((i, z), (i, z + 0.5)) for z in np.arange(0.5, 40, 0.5)] + [((x, i), (x + 0.5, i)) for x in np.arange(-20, 20, 0.5)]:
            p, q = P(*a), P(*b)
            if p and q and max(abs(p[0]), abs(q[0]), abs(p[1]), abs(q[1])) < 5000:
                cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), 200, 1, cv2.LINE_AA)
    ests = [organs.horizon(img, None, np.random.default_rng(k)) for k in range(5)]
    ys = [e["y"] for e in ests if e is not None]
    assert len(ys) >= 4 and abs(float(np.median(ys)) - hy) < 0.06, ests
    assert all(abs(math.degrees(e["roll"])) < 5 for e in ests if e is not None), ests
    # letterboxed (a stream's black bars, whose edges look like horizons): the same answer
    boxed = np.vstack([np.zeros((30, w), np.uint8), img, np.zeros((30, w), np.uint8)])
    ys_boxed = [e["y"] for e in (organs.horizon(boxed, None, np.random.default_rng(k)) for k in range(5)) if e]
    assert len(ys_boxed) >= 4 and abs(float(np.median(ys_boxed)) * (h + 60) - 30 - hy * h) < 0.06 * h, ys_boxed
    # a sea's edge at 45% down, and a palm's fronds converging on a crown above
    # the frame: the fronds' point is out of view, the sea line is the horizon
    sea = np.full((h, w), 200, np.uint8)
    sea[int(0.45 * h):] = 110
    for k in range(14):
        a = math.radians(35 + 2.5 * k)
        cv2.line(sea, (330, -40), (int(330 - 150 * math.cos(a)), int(-40 + 150 * math.sin(a))), 40, 2, cv2.LINE_AA)
    sea = np.clip(sea.astype(int) + np.random.default_rng(0).normal(0, 3, sea.shape), 0, 255).astype(np.uint8)
    e = organs.horizon(sea, None, np.random.default_rng(0))
    assert e is not None and abs(e["y"] - 0.45) < 0.03, e
    # the organism takes it as its horizon (its learned one is the fallback)
    o = O.Organism(founder(), None, None, 150.0, 15.0)
    for _k in range(5):
        o.hear_horizon(hy)
    assert o.horizon() == hy and o.learned_horizon() is None
    # an organ that scatters its estimates over half the frame is not believed
    o2 = O.Organism(founder(), None, None, 150.0, 15.0)
    for v in [0.1, 0.9, -0.2, 0.7, 0.3, 1.1, 0.0, 0.6, 0.4] * 3:
        o2.hear_horizon(v)
    assert o2.cv_horizon is None and o2.horizon() == o2.learned_horizon()


@check("its organs run on across a live feed's snapshots: the same outputs, each frame stepped once")
def _():
    from fishbowl import organs
    rng = np.random.default_rng(3)
    base = cv2.GaussianBlur((rng.random((220, 400)) * 255).astype(np.uint8), (5, 5), 0)
    frames = [np.ascontiguousarray(base[10:190, k:k + 320]) for k in range(40)]  # a slow pan
    on = {"horizon": True, "odometry": True, "place": True}
    whole = organs.series(frames, on)
    organs._RUNNING = None
    organs.series_running(frames[:30], 100, None, on)
    steps = organs._RUNNING.organs.t
    later = organs.series_running(frames[10:], 110, None, on)
    assert organs._RUNNING.organs.t == steps + 10  # only the 10 new frames were stepped
    for k in whole:
        assert np.allclose(np.nan_to_num(later[k]), np.nan_to_num(whole[k][10:])), k
    # a source switch restarts the feed's indices: other frames at old indices start it over
    organs.series_running([np.ascontiguousarray(f[::-1]) for f in frames[:5]], 100, None, on)
    assert organs._RUNNING.organs.t == 5
    organs._RUNNING = None


@check("back to the camera works while another process holds the choice file open (Windows)")
def _():
    import tempfile
    import threading
    from tools import viewer
    p = pathlib.Path(tempfile.mkdtemp()) / "selected_source.json"
    p.write_text("{}")
    f = open(p)  # held, as the organism reading it or a virus scanner would
    threading.Timer(0.4, f.close).start()
    viewer._held(lambda: p.unlink())
    assert not p.exists()


@check("a reset asked twice in a row is done once (a second would wipe the fresh founder)")
def _():
    from tools import viewer
    viewer._LAST_RESET.clear()
    assert viewer._reset_refused("amnesia") is None
    assert viewer._reset_refused("amnesia") is not None      # the repeat
    assert viewer._reset_refused("randomize") is None        # another kind is its own
    viewer._LAST_RESET["amnesia"] -= viewer.RESET_REPEAT_S + 1
    assert viewer._reset_refused("amnesia") is None          # later, it may be asked again
    viewer._LAST_RESET.clear()


@check("a founder is born with open stacked layers, within what it can carry (arithmetic and time)")
def _():
    import random as _r
    from fishbowl import controller as C
    from tools import seed
    for k in range(12):
        b = C.MosquitoBrain.random(_r.Random(k))
        seed.seed_layers(b, _r.Random(k))
        top = C.layers_max(b.n_hidden, b.weights_ih.shape[1], b.weights_ho.shape[0], bool(b.apical))
        assert 1 <= len(b.layers) <= top, (b.n_hidden, len(b.layers), top)
        assert all(float(l["gate"][0]) != 0.0 for l in b.layers)
        assert b.cost_if() <= C.MAX_THINK_FACTOR + 1e-9
        assert len(b.layer_hidden) == len(b.layers)
    # evolution can't stack past the clock's bound either
    b = C.MosquitoBrain.random(_r.Random(0), hidden=1)
    top = C.layers_max(1)
    b.layers = [{"W": np.zeros((1, 1)), "U": np.zeros((1, 1)), "b": np.zeros(1), "gate": np.ones(1)} for _ in range(top)]
    b.reset_hidden()
    assert not b.duplicate_layer(_r.Random(0))


@check("its road organ reads a drawn road: which way it bends, a crest from a dip, where it is on it, rolled too")
def _():
    import math
    from fishbowl import organs
    W, H = 320, 180
    F = (W / 2) / math.tan(math.radians(organs.PRIOR_HFOV_DEG) / 2)

    def render(curve=0.0, crest=0.0, roll_deg=0.0, offset=0.0, pitch_deg=4.0):
        img = np.full((H, W), 80, np.float32)
        th, ro = math.radians(pitch_deg), math.radians(roll_deg)
        ct, st, cr, sr = math.cos(th), math.sin(th), math.cos(ro), math.sin(ro)

        def proj(X, Z):
            Y = 1.0 + crest * Z * Z / 2
            dy, dz = Y * ct - Z * st, Y * st + Z * ct
            lu, lv = F * X / dz, F * dy / dz
            return W / 2 + lu * cr - lv * sr, H / 2 + lu * sr + lv * cr
        for x0, dashed in ((-1.4, False), (0.0, True), (1.4, False)):
            Zs = np.linspace(1.2, 40, 160)
            pts = [proj(x, z) for x, z in zip(x0 - offset + curve * Zs ** 2 / 2, Zs)]
            for k in range(len(pts) - 1):
                (a, b), (c, d) = pts[k], pts[k + 1]
                if not (dashed and (k // 4) % 2) and max(abs(a), abs(b), abs(c), abs(d)) < 5000:
                    cv2.line(img, (int(a), int(b)), (int(c), int(d)), 230, 2, cv2.LINE_AA)
        img = np.clip(img + np.random.default_rng(0).normal(0, 3, img.shape), 0, 255).astype(np.uint8)
        return organs.road(img, {"y": (H / 2 - F * math.tan(th)) / H, "roll": ro}, None, np.random.default_rng(0))
    e = render(); assert e and abs(e["heading"]) < 0.08 and abs(e["curve"]) < 0.008, e
    e = render(curve=0.01); assert e and e["curve"] > 0, e
    e = render(curve=-0.01); assert e and e["curve"] < 0, e
    e = render(crest=0.01); assert e and e["crest"] > 0, e
    e = render(crest=-0.01); assert e and e["crest"] < 0, e
    e = render(offset=0.5); assert e and 0.2 < e["offset"] < 0.8, e
    for r in (30, 60):
        e = render(roll_deg=r); assert e and abs(e["heading"]) < 0.08, (r, e)
    # it needs a horizon its organism believes, and says nothing without one: no road in a blank frame
    assert organs.road(np.full((H, W), 80, np.uint8), {"y": 0.4, "roll": 0.0}, None, np.random.default_rng(0)) is None


@check("host arousal: a host in view holds off sleep as far as the trait says; a layer costs energy by its gate")
def _():
    import random as _r
    from fishbowl import controller as C
    from fishbowl.state import MosquitoState as BodyState
    def tried(arousal, host, asleep=False, pressure=0.5):
        b = BodyState()
        b.sleep_pressure, b.asleep = pressure, 1.0 if asleep else 0.0
        b.circ_phase = np.pi  # its clock at midnight
        b.set_sleep(True, host=host, host_arousal=arousal)
        return b.asleep >= 0.5
    assert tried(0.0, True)                # no arousal: the night gate as ever
    assert not tried(1.0, True)            # full arousal, a host in view: stays up
    assert tried(1.0, False)               # no host: sleeps
    assert not tried(1.0, True, asleep=True)  # asleep, a host comes: wakes
    assert tried(1.0, True, pressure=0.99)    # but collapse wins
    g = founder()
    g.host_arousal = 0.37
    assert G.Genome.from_dict(g.to_dict()).host_arousal == 0.37 and g.clone().host_arousal == 0.37
    b = C.MosquitoBrain.random(_r.Random(0), hidden=8)
    b.layers = [{"W": np.zeros((8, 8)), "U": np.zeros((8, 8)), "b": np.zeros(8), "gate": np.array([gt])} for gt in (1.0, 0.1)]
    b.reset_hidden()
    assert b.energy_factor() < b.think_factor() and abs(b.think_factor() - b.energy_factor() - 0.9 * (2 * 64 + 8) / C.REFERENCE_MACS) < 1e-9


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
    for _s in (sys.stdout, sys.stderr):  # UTF-8 into the installers' logs (Windows' cp1252 has no jīng)
        try:
            _s.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
        except (AttributeError, ValueError, OSError):
            pass
    threading.current_thread().name = "selftest"
    raise SystemExit(main())
