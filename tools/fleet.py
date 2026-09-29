"""
Breeding across the fleet: find the organisms on the LAN / Tailnet, judge
them fairly, and clone one onto others.

    python tools/fleet.py                   # breed: you pick the parent and the targets
    python tools/fleet.py --auto            # shifting balance: nature picks (see below)
    python tools/fleet.py --hosts a,b,c     # these hosts instead of discovering them
    python tools/fleet.py --rounds 3        # tournaments on 3 separate snapshots

Judging (a 2026-09-28 panel; Gelman): a host's own fitness can't be compared
with another's -- each prices energy by its own CPU and sees its own frames.
So every organism, whole (genome, body, memory), is scored on the SAME
captured frames at the SAME prices: a tournament. Frames are captured into
RAM from the stream the fleet is watching; nothing is written.

Breed (artificial selection, like Darwin's pigeon fanciers): the ranking is
shown; you choose the parent (the winner by default) and tick the hosts to
clone it onto; nothing happens without typing "yes".

Auto (Sewall Wright's shifting balance, voted 12-2): 3 tournaments on 3
separate snapshots; if the best organism beats the worst in every one, the
best is cloned onto the worst's host -- at most one replacement per run.
Hosts named with --protect are never replaced (none by default). Run it by
hand or on a timer; every replaced lineage is backed up first.

Cloning a target: stop its organism (it saves), back up its state to
state/backup-<time>-before-<parent>/, write the parent's checkpoint and
episodes, start it. Linux and macOS targets over ssh (the host's own name),
this machine directly; a remote Windows target prints the steps instead.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PORT = 8090


def _get(url: str, timeout: float = 4.0) -> bytes | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.read()
    except Exception:  # unreachable, no viewer, not an organism: not in the fleet
        return None


def discover(names: list[str] | None) -> list[dict]:
    """Hosts whose viewer answers /organism/info: the named ones, else every
    online Tailscale peer and this machine."""
    if not names:
        names = ["localhost"]
        try:
            out = subprocess.run(["tailscale", "status", "--json"], capture_output=True, text=True, timeout=10).stdout
            st = json.loads(out)
            for peer in (st.get("Peer") or {}).values():
                if peer.get("Online"):
                    names.append((peer.get("HostName") or peer.get("DNSName", "").split(".")[0]).lower())
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    found, seen = [], set()
    for name in names:
        raw = _get(f"http://{name}:{PORT}/organism/info", 3.0)
        if not raw:
            continue
        info = json.loads(raw)
        host = info["hostname"].split(".")[0].lower()
        if host in seen or not info.get("has_checkpoint"):
            continue
        seen.add(host)
        local = host == socket.gethostname().split(".")[0].lower()
        found.append({"name": name if name != "localhost" else host, "host": host, "local": local, **info})
    return found


def fetch(h: dict) -> None:
    """Its organism of record: checkpoint (genome, body, memory) and episodes."""
    base = f"http://{'127.0.0.1' if h['local'] else h['name']}:{PORT}"
    raw = _get(base + "/organism/checkpoint", 30.0)
    h["checkpoint_bytes"] = raw
    h["checkpoint"] = json.loads(raw) if raw else None
    h["episodes_bytes"] = _get(base + "/organism/episodes", 30.0)


def capture(url: str, frames: int = 600):
    """One snapshot of the stream, in RAM: a World the organisms are judged in."""
    import run_vision as RV
    from fishbowl import prey as prey_lib, video_source
    src = RV._resolve_live_url(url) if "youtu" in url else url
    det = prey_lib.PreyDetector()
    feed = video_source.LiveFeed(src, detector=det if det.available else None, window=frames)
    try:
        if not feed.wait_for(frames, timeout=600):
            raise RuntimeError("the stream never filled a snapshot")
        grey, vectors, _, boxes, colour = feed.snapshot()
        return RV.World(grey, vectors, feed.frames_per_second(), boxes, colour)
    finally:
        feed.close()


def score(h: dict, world, host_rate: float) -> float:
    import run_vision as RV
    from fishbowl import genome as G
    c = h["checkpoint"]
    g = G.Genome.from_dict(c["genome"])
    memory = RV._for_evaluation(RV.memory_from_checkpoint(c))
    fit, _, _ = RV.evaluate_genome(g, *world.at_pace(1), RV.REFERENCE_QUOTA_PCT, c.get("body"), world.fps,
                                   world.prey, memory, world.colour, host_rate)
    return float(fit)


def tournament(hosts: list[dict], url: str, rounds: int) -> None:
    from fishbowl import hostspeed
    host_rate = hostspeed.sec_per_mac()
    for h in hosts:
        h["scores"] = []
    for r in range(rounds):
        print(f"Capturing snapshot {r + 1} of {rounds} from {url} (in RAM) ...", flush=True)
        world = capture(url)
        for h in hosts:
            h["scores"].append(score(h, world, host_rate))
            print(f"  {h['host']:>10}: {h['scores'][-1]:.4f}", flush=True)
    for h in hosts:
        h["mean"] = sum(h["scores"]) / len(h["scores"])


def table(hosts: list[dict]) -> None:
    print(f"\n{'#':>2}  {'host':<10} {'platform':<8} {'tournament':>10}  rounds          own fitness (not comparable)")
    for i, h in enumerate(hosts, 1):
        own = (h["checkpoint"] or {}).get("best_fitness")
        rounds = " ".join(f"{s:.3f}" for s in h["scores"])
        print(f"{i:>2}  {h['host']:<10} {h['platform']:<8} {h['mean']:>10.4f}  {rounds:<15} {own if own is None else round(own, 3)}")


def _ssh(host: str, cmd: str, stdin: bytes | None = None) -> None:
    subprocess.run(["ssh", host, cmd], input=stdin, check=True, timeout=600)


def clone(parent: dict, target: dict) -> None:
    """Parent's organism onto target's host: stop, back up, write, start."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    backup = f"backup-{stamp}-before-{parent['host']}"
    state = target["state_dir"]
    files = {"checkpoint.json": parent["checkpoint_bytes"]}
    if parent.get("episodes_bytes"):
        files["episodes.npz"] = parent["episodes_bytes"]
    print(f"Cloning {parent['host']} onto {target['host']} (its lineage backed up to state/{backup}) ...", flush=True)
    if target["local"]:
        install = Path(state).parent
        py = install / (".venv/Scripts/python.exe" if os.name == "nt" else ".venv/bin/python")
        ctl = install / "tools" / "cambrian_ctl.py"
        subprocess.run([str(py), str(ctl), "--stop"], check=True)
        b = Path(state) / backup
        b.mkdir(parents=True)
        for f in ("checkpoint.json", "checkpoint.prev.json", "episodes.npz"):
            if (Path(state) / f).exists():
                shutil.copy2(Path(state) / f, b / f)
        for f in ("checkpoint.prev.json", "episodes.npz"):
            (Path(state) / f).unlink(missing_ok=True)
        for f, data in files.items():
            (Path(state) / f).write_bytes(data)
        subprocess.run([str(py), str(ctl), "--start"], check=True)
        return
    if target["platform"] == "win32":
        print(f"  {target['host']} is a remote Windows host: copy the parent's checkpoint.json (and episodes.npz) into "
              f"{state} there with its organism stopped (cambrian --stop / --start). Skipped.")
        return
    # Linux / macOS over ssh: files to a temp dir there first, then the swap
    tmp = f"/tmp/fleet-{stamp}"
    _ssh(target["name"], f"mkdir -p {tmp}")
    with tempfile.TemporaryDirectory() as d:
        for f, data in files.items():
            (Path(d) / f).write_bytes(data)
            subprocess.run(["scp", "-q", str(Path(d) / f), f"{target['name']}:{tmp}/{f}"], check=True, timeout=600)
    q = lambda p: "'" + p.replace("'", "'\\''") + "'"  # noqa: E731  (quote a path for the remote shell)
    S, B = q(state), q(f"{state}/{backup}")
    if target["platform"] == "linux":
        stop, start, sudo, own = ("sudo systemctl stop cambrian-perception", "sudo systemctl start cambrian-perception",
                                  "sudo ", f"sudo chown cambrian:cambrian {S}/checkpoint.json {S}/episodes.npz 2>/dev/null; ")
    else:  # macOS: its own control script, as the logged-in user
        inst = q(str(Path(state).parent))
        stop = f"{inst}/.venv/bin/python {inst}/tools/cambrian_ctl.py --stop"
        start = f"{inst}/.venv/bin/python {inst}/tools/cambrian_ctl.py --start"
        sudo, own = "", ""
    script = (f"set -e; {stop}; {sudo}mkdir -p {B}; "
              f"for f in checkpoint.json checkpoint.prev.json episodes.npz; do [ -f {S}/$f ] && {sudo}cp -p {S}/$f {B}/ || true; done; "
              f"{sudo}rm -f {S}/checkpoint.prev.json {S}/episodes.npz; "
              + "".join(f"{sudo}cp {tmp}/{f} {S}/{f}; " for f in files)
              + f"{own}rm -rf {tmp}; {start}")
    _ssh(target["name"], script)


def _log(entry: dict) -> None:
    from fishbowl import sandbox
    with open(sandbox.STATE_DIR / "fleet_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--hosts", help="comma-separated host names (default: discover on the Tailnet)")
    ap.add_argument("--auto", action="store_true", help="shifting balance: clone the clear winner onto the clear loser")
    ap.add_argument("--protect", default="", help="comma-separated hosts --auto never replaces (default: none)")
    ap.add_argument("--rounds", type=int, default=None, help="tournament snapshots (default 1 to breed, 3 for --auto)")
    ap.add_argument("--source", help="the stream to judge them on (default: what the fleet is watching)")
    ap.add_argument("--dry-run", action="store_true", help="judge and report, change nothing")
    args = ap.parse_args()
    rounds = args.rounds or (3 if args.auto else 1)

    hosts = discover([h.strip() for h in args.hosts.split(",")] if args.hosts else None)
    if len(hosts) < 2:
        print(f"Found {len(hosts)} organism(s); need at least two.")
        return 1
    print("Organisms:", ", ".join(f"{h['host']} ({h['platform']})" for h in hosts))
    for h in hosts:
        fetch(h)
    hosts = [h for h in hosts if h["checkpoint"]]
    url = args.source
    if not url:
        for h in hosts:
            raw = _get(f"http://{'127.0.0.1' if h['local'] else h['name']}:{PORT}/sources")
            sel = json.loads(raw).get("selected_url") if raw else None
            if sel:
                url = sel
                break
    if not url:
        print("No stream selected on any host: pass --source <url> to judge them on.")
        return 1
    tournament(hosts, url, rounds)
    hosts.sort(key=lambda h: -h["mean"])
    table(hosts)

    if args.auto:
        protect = {p.strip().lower() for p in args.protect.split(",") if p.strip()}
        best, worst = hosts[0], next((h for h in reversed(hosts) if h["host"] not in protect and h is not hosts[0]), None)
        decision = None
        if worst is not None and all(b > w for b, w in zip(best["scores"], worst["scores"])):
            decision = (best, worst)
        entry = {"t": time.time(), "mode": "auto", "rounds": rounds, "source": url,
                 "scores": {h["host"]: h["scores"] for h in hosts},
                 "cloned": [decision[0]["host"], decision[1]["host"]] if decision else None, "dry_run": args.dry_run}
        if decision is None:
            print("\nNo clear loser (the best must beat it in every round): nothing replaced.")
        else:
            print(f"\n{decision[0]['host']} beat {decision[1]['host']} in all {rounds} rounds: shifting balance clones it there.")
            if not args.dry_run:
                clone(*decision)
        _log(entry)
        return 0

    # breed: the fancier chooses
    if args.dry_run:
        return 0
    pick = input(f"\nParent [1 = {hosts[0]['host']}, the tournament winner]: ").strip() or "1"
    parent = hosts[int(pick) - 1]
    others = [(i, h) for i, h in enumerate(hosts, 1) if h is not parent]
    print("Clone it onto:", ", ".join(f"{i} {h['host']}" for i, h in others))
    chosen = [s.strip() for s in input("Hosts (numbers, comma-separated; blank = none): ").split(",") if s.strip()]
    targets = [hosts[int(i) - 1] for i in chosen if int(i) - 1 < len(hosts) and hosts[int(i) - 1] is not parent]
    if not targets:
        print("Nothing chosen: no changes.")
        return 0
    print(f"\n{parent['host']} will replace: " + ", ".join(t["host"] for t in targets) + " (each backed up first).")
    if input('Type "yes" to proceed: ').strip().lower() != "yes":
        print("Cancelled: no changes.")
        return 0
    for t in targets:
        clone(parent, t)
    _log({"t": time.time(), "mode": "breed", "parent": parent["host"], "targets": [t["host"] for t in targets],
          "scores": {h["host"]: h["scores"] for h in hosts}, "source": url})
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
