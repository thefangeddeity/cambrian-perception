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
state/backup-<time>-before-<parent>/ (the newest 3 such backups are kept per
host; other backups are never touched), write the parent's checkpoint and
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
KEEP_BACKUPS = 3  # fleet-made backups kept per host (the newest); a lineage's own named backups are never touched


# The most a peer may send for one file (H, 2026-09-30: 20x the largest
# checkpoint measured in the fleet, 0.83 MB on 7elwe). A reply larger than
# this is not an organism of this program, and is never parsed.
MAX_PEER_BYTES = 16 * 1024 * 1024


def _get(url: str, timeout: float = 4.0) -> bytes | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            body = r.read(MAX_PEER_BYTES + 1)
            return body if len(body) <= MAX_PEER_BYTES else None
    except Exception:  # unreachable, no viewer, not an organism: not in the fleet
        return None


def vet_genome(data) -> str | None:
    """Why a genome from ANOTHER machine must not be used here, or None if it
    may (a 2026-09-30 review of the hive: pull-only, data-only).

    A genome is data -- trees of the fixed numeric ops in fishbowl/blocks.py
    and a brain's weights; nothing in it is ever executed as code -- but a
    peer could still send one this program would never make: an unknown op
    (a crash), a constant outside MAX_CONST, NaN weights, a tree past the
    sandbox's ceilings, or a brain past the energy bound (a CPU hog). Each
    is checked here against the SAME limits this host's own evolution obeys;
    a genome that fails is skipped as if the peer were offline."""
    import math
    import numpy as np
    from fishbowl import blocks, controller, genome as G
    from fishbowl.sandbox import Limits
    lim = Limits()
    kinds = {"var", "cell", "pool", "edge", "const", "op"}
    if not isinstance(data, dict) or not isinstance(data.get("trees"), dict):
        return "no trees"
    for name, tree in data["trees"].items():
        count, stack = 0, [(tree, 1)]
        while stack:  # iterative: a hostile nesting can't exhaust Python's recursion
            node, depth = stack.pop()
            count += 1
            if count > lim.max_tree_nodes or depth > lim.max_tree_depth:
                return f"tree {name} past the ceilings ({lim.max_tree_nodes} nodes, depth {lim.max_tree_depth})"
            if not isinstance(node, dict) or node.get("kind") not in kinds:
                return f"tree {name}: an unknown node kind"
            for k in ("index", "value", "kx", "ky", "angle", "bend"):
                v = node.get(k, 0)
                if not isinstance(v, (int, float)) or not math.isfinite(v):
                    return f"tree {name}: a non-numeric or non-finite {k}"
            kids = node.get("children") or []
            if not isinstance(kids, list):
                return f"tree {name}: malformed children"
            if node["kind"] == "op":
                if node.get("op") not in blocks.OPS or len(kids) != blocks.OPS[node["op"]][0]:
                    return f"tree {name}: an unknown op or the wrong number of arguments"
            if node["kind"] == "const" and abs(node.get("value", 0.0)) > blocks.MAX_CONST:
                return f"tree {name}: a constant outside +/-{blocks.MAX_CONST}"
            stack.extend((c, depth + 1) for c in kids)
    try:
        g = G.Genome.from_dict(data)
    except Exception as e:
        return f"unreadable ({type(e).__name__})"
    b = g.brain
    arrays = [b.weights_ih, b.weights_hh, b.weights_ho, b.bias_h, b.bias_o] + [a for l in b.layers for a in l.values()]
    if not all(np.isfinite(a).all() for a in arrays):
        return "non-finite brain weights"
    if b.cost_if() > controller.MAX_THINK_FACTOR * (1 + 1e-9):
        return f"a brain past the energy bound ({b.cost_if():.1f} > {controller.MAX_THINK_FACTOR:.0f}x)"
    return None


def discover(names: list[str] | None) -> list[dict]:
    """Hosts whose viewer answers /organism/info: the named ones, else every
    online Tailscale peer and this machine."""
    # a peer's name -> its Tailscale address (asked by address: no MagicDNS
    # needed). This machine is asked as 127.0.0.1, a name no peer can take (a
    # tailnet device calling itself "localhost" -- a WSL instance on 7elwe --
    # took this machine's place in the list, and 7elwe went missing from its own).
    THIS = "127.0.0.1"
    addr = {}
    if not names:
        names = [THIS]
        # the Tailscale command: on the PATH (Linux, Windows), else inside the
        # macOS app, where it isn't on the PATH
        for ts in ("tailscale", "/Applications/Tailscale.app/Contents/MacOS/Tailscale"):
            try:
                out = subprocess.run([ts, "status", "--json"], capture_output=True, text=True, timeout=10).stdout
                st = json.loads(out)
            except (OSError, ValueError, subprocess.SubprocessError):
                continue
            for peer in (st.get("Peer") or {}).values():
                if peer.get("Online"):
                    n = (peer.get("HostName") or peer.get("DNSName", "").split(".")[0]).lower()
                    ips = [i for i in (peer.get("TailscaleIPs") or []) if ":" not in i]
                    if not ips:
                        continue
                    key = n if n not in addr and n != THIS else f"{n}@{ips[0]}"  # a duplicate name keeps its own address
                    names.append(key)
                    addr[key] = ips[0]
            break
    def probe(name: str) -> list[dict]:
        """One host's organisms, asked by its Tailscale address where known."""
        out = []
        at = addr.get(name, name)
        # A host's first organism on PORT; a second one (PARKED feature,
        # docs/second-organism.md) on PORT + 1, asked only when the first names
        # it in "siblings" -- probing 8091 everywhere cost a 3 s timeout on a
        # host whose firewall drops it (7elwe), for an organism that isn't there.
        ports = [PORT]
        while ports:
            port = ports.pop(0)
            raw = _get(f"http://{at}:{port}/organism/info", 3.0)
            if not raw:
                continue
            try:
                info = json.loads(raw)
            except ValueError:
                continue
            if not isinstance(info, dict) or not isinstance(info.get("hostname"), str):
                continue  # something answered, but not an organism's viewer
            if port == PORT and "b" in (info.get("siblings") or []):
                ports.append(PORT + 1)  # only "b" is supported (the viewer's DEFAULT_PORT)
            out.append((name, at, port, info))
        return out

    # every host asked at once: a device on the tailnet that silently drops the
    # request costs its 3 s timeout in parallel, not in turn
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max(1, min(16, len(names)))) as pool:
        answers = [a for per_host in pool.map(probe, names) for a in per_host]
    found, seen = [], set()
    this = socket.gethostname().split(".")[0].lower()
    here = os.environ.get("CAMBRIAN_INSTANCE", "").strip()
    for name, at, port, info in answers:
        host = info["hostname"].split(".")[0].lower()
        inst = info.get("instance") or ""
        key = f"{host}:{inst}"
        if key in seen or not info.get("has_checkpoint"):
            continue
        seen.add(key)
        local = host == this and inst == here  # this very organism (the other on this host is a peer)
        if not local and not info.get("hive", True):  # a solo organism is no one's peer (an older viewer reports no "hive": it predates solo)
            continue
        label = host if not inst else f"{host}-{inst}"
        found.append({"name": host if name in (THIS, "localhost") or "@" in name else name, "addr": at,
                      "port": port, "host": label, "local": local, **info})
    return found


def fetch(h: dict) -> None:
    """Its organism of record: checkpoint (genome, body, memory) and episodes."""
    base = f"http://{'127.0.0.1' if h['local'] else h.get('addr') or h['name']}:{h.get('port', PORT)}"
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
        for old in sorted(Path(state).glob("backup-*-before-*"), key=lambda d: d.stat().st_mtime)[:-KEEP_BACKUPS]:
            shutil.rmtree(old, ignore_errors=True)
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
              + f"{own}rm -rf {tmp}; "
              + f"ls -dt {S}/backup-*-before-* 2>/dev/null | tail -n +{KEEP_BACKUPS + 1} | while read d; do {sudo}rm -rf \"$d\"; done; "
              + f"{start}")
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
    for h in list(hosts):  # held to the same limits as a migrant before anything is judged or cloned
        why = vet_genome(h["checkpoint"].get("genome"))
        if why:
            print(f"{h['host']}: its genome is refused -- {why}.")
            hosts.remove(h)
    url = args.source
    if not url:
        for h in hosts:
            raw = _get(f"http://{'127.0.0.1' if h['local'] else h.get('addr') or h['name']}:{h.get('port', PORT)}/sources")
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
    for _s in (sys.stdout, sys.stderr):  # a console that can't show jīng or 精 shows "?", never a crash
        try:
            _s.reconfigure(errors="replace")
        except (AttributeError, ValueError, OSError):
            pass
    raise SystemExit(main())
