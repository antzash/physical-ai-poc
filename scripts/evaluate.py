"""Headless N-episode evaluation of the scripted pick-and-place pipeline under domain randomisation.

    python3 scripts/evaluate.py                     # 100 episodes, seed 0
    python3 scripts/evaluate.py --episodes 500 --seed 1000
    python3 scripts/evaluate.py --pos-noise-mm 5 --yaw-noise-deg 5   # perception error on the controller's input
    python3 scripts/evaluate.py --size-mult 1.5 --mass-mult 1.5      # out-of-distribution envelope

Writes out/eval_<timestamp>.json (config, summary, every episode) and its own custody log, and prints a table.
Episode i uses seed `--seed + i`. Any single episode replays exactly (same draws, same physics) with
`python3 scripts/episode.py --episodes 1 --seed <seed> --slot <slot>`, both taken from the JSON.
"""

import argparse
import json
import subprocess
from pathlib import Path
import time
from collections import Counter
from datetime import datetime, timezone

import mujoco
import numpy as np

import scene
from controller import PHASES
from episode import IntakeStation
from perception import NoiseSpec

Z95 = 1.959964


def wilson(k, n, z=Z95):
    """Wilson score interval for a binomial proportion; well-behaved near 0 and 1, unlike the normal approximation."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def rate(k, n):
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "rate": (k / n) if n else float("nan"), "ci95": [lo, hi]}


def _git_commit():
    try:
        return subprocess.check_output(["git", "describe", "--always", "--dirty"], cwd=scene.ROOT, text=True).strip()
    except Exception:
        return None


def summarise(results):
    n = len(results)
    ok = [r for r in results if r.success]
    summary = {"overall": rate(len(ok), n)}
    summary["by_class"] = {
        c: rate(sum(r.success for r in results if r.object_class == c), sum(r.object_class == c for r in results))
        for c in scene.ITEM_CLASSES
    }
    summary["by_slot"] = {
        s: rate(sum(r.success for r in results if r.slot == s), sum(r.slot == s for r in results))
        for s in scene.SLOT_NAMES
    }
    phase_order = PHASES + ["VERIFY"]
    fails = Counter(r.failure_phase for r in results if not r.success)
    summary["failures_by_phase"] = {p: fails[p] for p in sorted(fails, key=lambda p: phase_order.index(p)
                                                                 if p in phase_order else 99)}
    summary["failure_reasons"] = dict(Counter(f"{r.failure_phase}: {r.failure_reason}" for r in results
                                              if not r.success).most_common())
    t_ok = np.array([r.cycle_time for r in ok]) if ok else np.array([np.nan])
    t_all = np.array([r.cycle_time for r in results])
    summary["cycle_time_s"] = {
        "successful_mean": float(np.mean(t_ok)), "successful_p95": float(np.percentile(t_ok, 95)),
        "all_mean": float(np.mean(t_all)), "all_p95": float(np.percentile(t_all, 95)),
        "note": "simulated seconds from HOME to return-to-home (or to the failure)",
    }
    err = np.array([r.placement_error for r in ok]) if ok else np.array([np.nan])
    summary["placement_error_m"] = {"mean": float(np.mean(err)), "p95": float(np.percentile(err, 95)),
                                    "max": float(np.max(err)),
                                    "note": "horizontal distance, item centre to slot centre, successful episodes"}
    failed = [r for r in results if not r.success]
    summary["failed_but_in_slot"] = {"k": sum(r.final_in_slot for r in failed), "n": len(failed),
                                     "note": "failed episodes whose item nevertheless ended at rest inside the target "
                                             "slot (e.g. dropped during INSERT). Not counted as success."}
    grasped = [r for r in results if r.grasped]
    summary["grasp_rate"] = rate(len(grasped), n)
    summary["grasp_slip"] = {**rate(sum(r.slipped for r in grasped), len(grasped)),
                             "note": "of episodes with a verified grasp after LIFT, fraction that then lost "
                                     "two-finger contact before release"}
    return summary


def _pct(r):
    if r["n"] == 0:
        return "   n/a"
    return f"{100 * r['rate']:5.1f}%  [{100 * r['ci95'][0]:4.1f}, {100 * r['ci95'][1]:5.1f}]  ({r['k']}/{r['n']})"


def print_table(summary, config):
    print("\n" + "=" * 78)
    print(f"EVALUATION  episodes={config['episodes']}  seeds {config['seed']}..{config['seed'] + config['episodes'] - 1}"
          f"  commit={config['git_commit']}  wall={config['wall_time_s']:.0f}s")
    print(f"perception: {config['perception']}   size×{config['size_mult']:g}  mass×{config['mass_mult']:g}")
    print("=" * 78)
    print(f"{'overall success':<22}{_pct(summary['overall'])}")
    print(f"{'grasp success':<22}{_pct(summary['grasp_rate'])}")
    print(f"{'grasp slip (of grasped)':<22}{_pct(summary['grasp_slip'])}")
    print("-" * 78)
    print("success by object class            95% CI (Wilson)")
    for c, r in summary["by_class"].items():
        print(f"  {c:<20}{_pct(r)}")
    print("success by slot")
    for s, r in summary["by_slot"].items():
        print(f"  {s:<20}{_pct(r)}")
    print("-" * 78)
    print("failures by phase")
    if not summary["failures_by_phase"]:
        print("  (none)")
    for p, k in summary["failures_by_phase"].items():
        print(f"  {p:<20}{k}")
    for reason, k in summary["failure_reasons"].items():
        print(f"    {k:3d} x {reason}")
    print("-" * 78)
    ct, pe = summary["cycle_time_s"], summary["placement_error_m"]
    print(f"cycle time (successful)  mean {ct['successful_mean']:.2f}s  p95 {ct['successful_p95']:.2f}s   "
          f"(all: mean {ct['all_mean']:.2f}s  p95 {ct['all_p95']:.2f}s)")
    print(f"placement error          mean {pe['mean'] * 1000:.1f} mm  p95 {pe['p95'] * 1000:.1f} mm  "
          f"max {pe['max'] * 1000:.1f} mm")
    print(f"custody chain            {'intact' if config['custody_chain_intact'] else 'BROKEN'}  "
          f"({config['custody_events']} events, {config['custody_log']})")
    print("=" * 78)


def run_evaluation(episodes, seed, noise=NoiseSpec(), size_mult=1.0, mass_mult=1.0, log_path=None, quiet=True,
                   progress=True):
    """Run `episodes` randomised episodes with seeds seed..seed+episodes-1. Returns (results, summary, config)."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = log_path or scene.OUT_DIR / f"eval_{stamp}_custody.jsonl"
    station = IntakeStation(log_path, echo=not quiet)
    results = []
    t0 = time.time()
    for i in range(episodes):
        r = station.run_episode(seed + i, noise=noise, size_mult=size_mult, mass_mult=mass_mult)
        results.append(r)
        if progress:
            mark = "ok  " if r.success else f"FAIL {r.failure_phase}"
            print(f"[{i + 1:4d}/{episodes}] seed {r.seed:<6} {r.object_class:<9} {r.slot}  {mark}", flush=True)
    wall = time.time() - t0
    chain_bad = station.log.verify()
    config = {
        "episodes": episodes, "seed": seed, "timestamp_utc": stamp, "git_commit": _git_commit(),
        "mujoco_version": mujoco.__version__, "wall_time_s": wall,
        "custody_log": str(Path(log_path).resolve().relative_to(scene.ROOT)) if str(Path(log_path).resolve()).startswith(
            str(scene.ROOT)) else str(log_path),
        "custody_events": len(station.log), "custody_chain_intact": chain_bad is None,
        "grasping": "contact physics only (no weld / kinematic attachment)",
        "perception": noise.describe(), "pos_noise_mm": noise.pos_sigma_m * 1000,
        "yaw_noise_deg": float(np.degrees(noise.yaw_sigma_rad)), "size_noise_pct": noise.size_sigma_frac * 100,
        "size_mult": size_mult, "mass_mult": mass_mult,
    }
    return results, summarise(results), config


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quiet", action="store_true", help="suppress per-event custody log output")
    parser.add_argument("--pos-noise-mm", type=float, default=0.0, help="per-axis σ of the controller's position error")
    parser.add_argument("--yaw-noise-deg", type=float, default=0.0, help="σ of the controller's yaw error")
    parser.add_argument("--size-noise-pct", type=float, default=0.0, help="per-axis σ of the size estimate error")
    parser.add_argument("--size-mult", type=float, default=1.0, help="runtime multiplier on drawn item sizes (OOD)")
    parser.add_argument("--mass-mult", type=float, default=1.0, help="runtime multiplier on drawn item masses (OOD)")
    args = parser.parse_args()

    noise = NoiseSpec(args.pos_noise_mm / 1000, float(np.radians(args.yaw_noise_deg)), args.size_noise_pct / 100)
    results, summary, config = run_evaluation(args.episodes, args.seed, noise, args.size_mult, args.mass_mult,
                                              quiet=args.quiet)
    out_json = scene.OUT_DIR / f"eval_{config['timestamp_utc']}.json"
    out_json.write_text(json.dumps({"config": config, "summary": summary,
                                    "episodes": [r.to_dict() for r in results]}, indent=2))
    print_table(summary, config)
    print(f"wrote {out_json.relative_to(scene.ROOT)}")


if __name__ == "__main__":
    main()
