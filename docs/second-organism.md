# A second organism per host (parked, 2026-09-30)

**Status: parked, built as far as it can be without switching it on, never
enabled on any host.** This note is for whoever picks it up: what exists, how
to switch it on, and what must be solved first. Nothing here runs today; every
host has one organism, exactly as before.

## Why it was parked

Asked for so the fleet's four hosts could hold eight organisms. A resource
panel (Gregg, Poettering, Russinovich, Gelman) advised against it, for now:

- Only one host of four had the room (Tanzania: 8 cores, ~5 GB free with one
  organism using ~1 GB plus ~120-150 MB per worker). Tina (3.3 GB), 7elwe
  (runs memory-short: its worst restart loop was memory pressure) and Ariana
  (dual-core, capped at half a core) did not.
- On Tanzania the two would share one USB hard disk: two checkpoint writers,
  two live bodies -- contention of the kind behind the fleet's worst watchdog
  kills.
- Two organisms on one machine and one stream share every disturbance
  (restarts, stalls, CPU spikes): weak replicates of each other. The four
  hosts already give four.
- The fleet needed undisturbed running more than new organisms.

Sensing each other and sharing energy (reciprocity, after Wilkinson) does NOT
need this: it can work between the hosts' organisms across machines. It is
parked separately (docs/design-ontogeny.md, "Parked").

## What exists

One switch, the environment variable `CAMBRIAN_INSTANCE`:

| | unset (every host today) | `CAMBRIAN_INSTANCE=b` |
|---|---|---|
| state | `state/` | `state-b/` |
| RAM runtime folder | `/dev/shm/cambrian-perception/` | `/dev/shm/cambrian-perception-b/` |
| viewer port | 8090 | 8091 |
| service | `cambrian-perception.service` | `cambrian-perception-b.service` |

- Derived in `fishbowl/sandbox.py` (the organism), `tools/viewer.py` and
  `tools/resource_handler.py` (which don't import sandbox): the three must stay
  equal. `tools/selftest.py` checks they do.
- The code, venv, models and `cambrian.json` (the hive switch) are shared.
- The fleet (`tools/fleet.py discover`) asks port 8091 only when the host's
  first organism names a second in its `/organism/info` `"siblings"` (a
  `state-b/` with a checkpoint). Two organisms on one host are peers of each
  other in the hive, like any two hosts' organisms.
- Units, a polkit rule: `deploy/second-organism/` -- NOT installed by
  `deploy/install.sh`. The second organism watches `live` (the stream chosen in
  its own viewer, else the built-in web streams), never the camera: a camera
  opens for one process at a time, and the first organism holds it.
- Only `b` is supported: a third needs a port scheme first.

## Open gaps -- close these before any host runs two

1. **Memory double-counting.** Each resource handler sizes its organism's
   MemoryHigh/MemoryMax from what the whole host can spare. Two handlers would
   each hand out the same headroom. They must split it (for example in
   proportion to each one's measured RSS), or one handler must manage both.
2. **CPU.** Each unit has its own `CPUQuota`; together they may take twice one
   organism's share. Decide the host's total, then split it.
3. **`cambrian` command.** `--status`, `--tui` and the rest know only the
   first organism (`--tui` can switch to 8091 with `p`).
4. **Installer.** `deploy/install.sh` installs only the first organism's
   units. Installing the second is by hand (below) until it's decided whether
   the installer should take `--second-organism`.
5. **Windows and macOS.** Nothing: their supervisors (`tools/cambrian_service.py`,
   the macOS login agent) run one organism.
6. **The disk.** On a host whose root is a USB hard disk, measure the two
   organisms' checkpoint and episode writes together before trusting it.

## Switching it on (Linux, by hand), once the gaps are closed

```sh
R=/srv/cambrian/cambrian-perception
sudo install -m 644 $R/deploy/second-organism/*.service $R/deploy/second-organism/*.timer /etc/systemd/system/
sudo install -m 644 $R/deploy/second-organism/50-cambrian-b.rules /etc/polkit-1/rules.d/
sudo -u cambrian mkdir -p $R/state-b
sudo systemctl daemon-reload
sudo systemctl enable --now cambrian-viewer-b.service
# choose its stream at http://<host>:8091/ first, then:
sudo systemctl enable --now cambrian-perception-b.service cambrian-resource-handler-b.timer
```

It is born as any new organism is: a fresh founder, saved at birth
(`state-b/founder.json`). To remove it: disable and remove the four units and
the rule; `state-b/` holds its lineage (keep it or delete it).
