"""Robustness sweeps: perception error on the controller's input, and an out-of-distribution envelope.

    python3 scripts/sweep.py                       # all sweeps, 200 episodes per point
    python3 scripts/sweep.py --episodes 50 --only position

Sweeps (each point gets a fresh, non-overlapping seed range: 100000 * sweep_index + 1000 * point_index):
    position   pos σ ∈ {0,2,5,8,12,16,20} mm, yaw 0
    yaw        yaw σ ∈ {0,2,5,8,12,16,20} °, pos 0
    combined   paired (k mm, k °)
    size       size σ ∈ {0,5,10,20,30} %, pose perfect (secondary axis)
    ood        size and mass envelope × {1.0,1.25,1.5,1.75,2.0}, zero noise
    yaw_wide   yaw σ ∈ {30,45,60,90} °: extends the yaw sweep, which is still ~96% at 20° (the yaw cliff lies beyond)

This is a MEASUREMENT. Nothing here tunes the controller. Writes out/robustness_<ts>.json and per-point custody logs
under out/sweep_<ts>/. Plot with scripts/plot_robustness.py.
"""

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone

import numpy as np

import scene

LEVELS = [0, 2, 5, 8, 12, 16, 20]
SWEEPS = {
    "position": [{"pos_mm": v, "yaw_deg": 0} for v in LEVELS],
    "yaw": [{"pos_mm": 0, "yaw_deg": v} for v in LEVELS],
    "combined": [{"pos_mm": v, "yaw_deg": v} for v in LEVELS],
    "size": [{"size_pct": v} for v in (0, 5, 10, 20, 30)],
    "ood": [{"mult": v} for v in (1.0, 1.25, 1.5, 1.75, 2.0)],
    "yaw_wide": [{"pos_mm": 0, "yaw_deg": v} for v in (30, 45, 60, 90)],
}
SWEEP_INDEX = {"position": 1, "yaw": 2, "combined": 3, "size": 4, "ood": 5, "yaw_wide": 6}


def _run_point(sweep, i, point, episodes, out_dir):
    # Imported inside the worker so each process builds its own MuJoCo model.
    from evaluate import run_evaluation
    from perception import NoiseSpec

    noise = NoiseSpec(point.get("pos_mm", 0) / 1000, float(np.radians(point.get("yaw_deg", 0))),
                      point.get("size_pct", 0) / 100)
    mult = point.get("mult", 1.0)
    seed = 100000 * SWEEP_INDEX[sweep] + 1000 * i
    log = os.path.join(out_dir, f"{sweep}_{i}_custody.jsonl")
    results, summary, config = run_evaluation(episodes, seed, noise, size_mult=mult, mass_mult=mult, log_path=log,
                                              quiet=True, progress=False)
    return {"sweep": sweep, "index": i, "point": point, "seed_start": seed, "config": config, "summary": summary,
            "episodes": [r.to_dict() for r in results]}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--only", nargs="*", choices=list(SWEEPS), help="run a subset of the sweeps")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = scene.OUT_DIR / f"sweep_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(s, i, p) for s in (args.only or SWEEPS) for i, p in enumerate(SWEEPS[s])]
    print(f"{len(jobs)} points x {args.episodes} episodes on {args.workers} workers -> {out_dir.name}", flush=True)

    t0 = time.time()
    points = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(_run_point, s, i, p, args.episodes, str(out_dir)) for s, i, p in jobs]
        for f in as_completed(futs):
            r = f.result()
            o = r["summary"]["overall"]
            print(f"[{time.time() - t0:6.0f}s] {r['sweep']:<9} {json.dumps(r['point']):<34} "
                  f"{100 * o['rate']:5.1f}%  [{100 * o['ci95'][0]:4.1f}, {100 * o['ci95'][1]:5.1f}]  "
                  f"chain {'ok' if r['config']['custody_chain_intact'] else 'BROKEN'}", flush=True)
            points.append(r)
    points.sort(key=lambda r: (SWEEP_INDEX[r["sweep"]], r["index"]))

    out = scene.OUT_DIR / f"robustness_{stamp}.json"
    commit = points[0]["config"]["git_commit"] if points else None
    out.write_text(json.dumps({"timestamp_utc": stamp, "git_commit": commit, "episodes_per_point": args.episodes,
                               "wall_time_s": time.time() - t0, "sweeps": {k: SWEEPS[k] for k in (args.only or SWEEPS)},
                               "points": points}, indent=1))
    print(f"wrote {out.relative_to(scene.ROOT)} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
