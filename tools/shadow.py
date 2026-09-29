"""
Shadow-test a rule change before it reaches the fleet (a 2026-09-29 panel,
after self-driving research's "shadow mode"; Gelman, Gregg).

    python tools/shadow.py v0.2.0            # that version against the working tree
    python tools/shadow.py v0.2.0 HEAD       # two versions
    python tools/shadow.py v0.2.0 --hosts tina,tanzania --rounds 2

Every organism of the fleet (fetched as tools/fleet.py does) is scored under
both versions of the code on the SAME captured frames at the same prices, and
the table shows how each version judges it and how it behaves (prey eaten,
time asleep). A rule change that makes good organisms look worse, or changes
behaviour you didn't intend, shows up here before any host runs it.

The snapshot is captured once into a temporary file so both versions read
the same frames (a few hundred small frames; deleted afterwards). Each
version runs from its own git worktree, in its own process.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

# Runs inside each version's checkout: builds the world from the saved frames
# and scores each organism with that version's own code.
RUNNER = r'''
import json, sys, numpy as np
sys.path.insert(0, ".")
import run_vision as RV
from fishbowl import genome as G
z = np.load(sys.argv[1], allow_pickle=True)
frames = list(z["grey"]); colour = list(z["colour"]) if "colour" in z else None
prey = json.loads(str(z["prey"])); fps = float(z["fps"]); host_rate = float(z["host_rate"])
world = RV.World(frames, np.array(z["vectors"]), fps, prey, colour)
out = {}
for name, path in json.loads(sys.argv[2]).items():
    c = json.load(open(path, encoding="utf-8"))
    memory = RV.memory_from_checkpoint(c) if hasattr(RV, "memory_from_checkpoint") else None
    if memory is not None and hasattr(RV, "_for_evaluation"):
        memory = RV._for_evaluation(memory)
    g = G.Genome.from_dict(c["genome"])
    fit, breakdown, info = RV.evaluate_genome(g, *world.at_pace(1), RV.REFERENCE_QUOTA_PCT, c.get("body"), fps, world.prey, memory, world.colour, host_rate)
    asleep = info.get("asleep") or []
    out[name] = {"fitness": float(fit), "mean_prey": info.get("mean_prey"),
                 "asleep_share": (sum(1 for a in asleep if a) / len(asleep)) if asleep else None}
print(json.dumps(out))
'''


def _worktree(ref: str, where: Path) -> Path:
    if ref in ("", "WORKTREE"):
        return ROOT
    d = where / f"code-{ref.replace('/', '_')}"
    subprocess.run(["git", "-C", str(ROOT), "worktree", "add", "--detach", "-f", str(d), ref], check=True,
                   capture_output=True)
    return d


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("old", help="the version to compare against (a tag, branch or commit)")
    ap.add_argument("new", nargs="?", default="WORKTREE", help="the new version (default: the working tree)")
    ap.add_argument("--hosts", help="comma-separated host names (default: discover on the Tailnet)")
    ap.add_argument("--source", help="the stream to judge on (default: what the fleet is watching)")
    args = ap.parse_args()

    import fleet
    import numpy as np
    from fishbowl import hostspeed
    hosts = fleet.discover([h.strip() for h in args.hosts.split(",")] if args.hosts else None)
    for h in hosts:
        fleet.fetch(h)
    hosts = [h for h in hosts if h["checkpoint"]]
    if not hosts:
        print("No organisms found.")
        return 1
    url = args.source
    if not url:
        for h in hosts:
            raw = fleet._get(f"http://{'127.0.0.1' if h['local'] else h['name']}:{fleet.PORT}/sources")
            url = (json.loads(raw).get("selected_url") if raw else None) or url
            if url:
                break
    if not url:
        print("No stream selected on any host: pass --source <url>.")
        return 1
    tmp = Path(tempfile.mkdtemp(prefix="shadow-"))
    try:
        print(f"Capturing a snapshot from {url} ...", flush=True)
        world = fleet.capture(url)
        snap = tmp / "snapshot.npz"
        extra = {"colour": np.array(world.colour)} if world.colour is not None else {}
        np.savez(snap, grey=np.array(world.frames), vectors=np.array(world.vectors), prey=json.dumps(world.prey),
                 fps=world.fps, host_rate=hostspeed.sec_per_mac(), **extra)
        orgs = {}
        for h in hosts:
            p = tmp / f"{h['host']}.json"
            p.write_bytes(h["checkpoint_bytes"])
            orgs[h["host"]] = str(p)
        results = {}
        for ref in (args.old, args.new):
            code = _worktree(ref, tmp)
            print(f"Scoring under {ref if ref != 'WORKTREE' else 'the working tree'} ...", flush=True)
            r = subprocess.run([sys.executable, "-c", RUNNER, str(snap), json.dumps(orgs)], cwd=code,
                               capture_output=True, text=True, timeout=3600)
            if r.returncode != 0:
                print(r.stderr[-2000:])
                return 1
            results[ref] = json.loads(r.stdout.strip().splitlines()[-1])
        old, new = results[args.old], results[args.new]
        print(f"\n{'host':<10} {'fitness old':>12} {'new':>9} {'change':>8}   {'prey old':>8} {'new':>6}   {'asleep old':>10} {'new':>6}")
        for name in orgs:
            o, n = old[name], new[name]
            f = lambda v: "--" if v is None else f"{v:.3f}"  # noqa: E731
            print(f"{name:<10} {o['fitness']:>12.4f} {n['fitness']:>9.4f} {n['fitness'] - o['fitness']:>+8.4f}   "
                  f"{f(o['mean_prey']):>8} {f(n['mean_prey']):>6}   {f(o['asleep_share']):>10} {f(n['asleep_share']):>6}")
        return 0
    finally:
        for d in tmp.glob("code-*"):
            subprocess.run(["git", "-C", str(ROOT), "worktree", "remove", "--force", str(d)], capture_output=True)
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
