"""
Export the teacher-free model for a light CV module (the laptop-livecam-lite
repo's "light cv"): its model card (run_vision.py writes champion.json) as a
bundle -- the genome, its frozen learned parts, their out-of-sample scores, and
the rules they were made under. The full CV module keeps the whole
self-teaching organism instead (upstream, rebased as agreed).

    python tools/export_lite.py <champion.json> <bundle.json>

It refuses a card without learned parts or scores: nothing ships unmeasured.
The adoption bar is the light module's own (scores, not verdicts); it runs the
bundle with fishbowl/live.py (LiveActor, which applies the learned parts).
"""
from __future__ import annotations

import json
import socket
import sys
import time
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    card = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    learned, scores = card.get("learned"), card.get("scores") or {}
    measured = {k: v for k, v in scores.items() if v}
    if not learned or not measured:
        print("refused: the card has no learned parts or no out-of-sample scores yet")
        return 1
    parts = [k for k in ("food", "danger", "heads", "distilled") if learned.get(k)]
    bundle = {**card, "export": {"exported_at": time.time(), "exported_by": socket.gethostname(),
                                 "teacher_free_parts": parts, "unmeasured": [k for k, v in scores.items() if not v]}}
    Path(sys.argv[2]).write_text(json.dumps(bundle, separators=(",", ":")), encoding="utf-8")
    print(f"exported generation {card.get('generation')} from {card.get('host')} (rules {card.get('rules_version')}): parts {parts}")
    for k, v in measured.items():
        print(f"  {k}: mean error {v['mae']} +- {v['mae_se']} (n {v['n']}, effective {v['n_eff']}), correlation {v['corr']}")
    fit = (card.get("fitness") or {}).get("out_of_sample")
    print(f"  fitness out of sample: {fit}" if fit else "  fitness out of sample: not yet measured")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
