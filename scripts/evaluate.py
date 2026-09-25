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


REFUSAL_CAUSES = ["no_decode", "no_case_match", "cabinet_full", "verify_mismatch", "verify_no_read"]


def _stats(values):
    v = np.array([x for x in values if x is not None], dtype=float)
    if v.size == 0:
        return {"mean": float("nan"), "p95": float("nan"), "max": float("nan"), "n": 0}
    return {"mean": float(v.mean()), "p95": float(np.percentile(v, 95)), "max": float(v.max()), "n": int(v.size)}


def summarise(results):
    """Phase 1 metrics. Misfiles first: an item at rest in any slot other than the one its TRUE id routes to."""
    n = len(results)
    summary = {"misfile": {**rate(sum(r.misfile for r in results), n),
                           "note": "item at rest in a slot other than the one its true ID routes to (ground truth)"}}
    summary["misread"] = {**rate(sum(r.decoded_id is not None and r.decoded_id != r.true_item_id for r in results), n),
                          "note": "intake decode returned a string other than the true item ID"}
    summary["completion"] = {**rate(sum(r.success for r in results), n),
                             "note": "filed and verified in the routed slot without human intervention"}
    summary["refusal"] = rate(sum(r.outcome == "refused" for r in results), n)
    summary["refusal_by_cause"] = {c: rate(sum(r.refusal_cause == c for r in results), n) for c in REFUSAL_CAUSES}
    summary["execution_failure"] = rate(sum(r.outcome == "failed" for r in results), n)
    phase_order = PHASES + ["VERIFY"]
    fails = Counter(r.failure_phase for r in results if r.outcome == "failed")
    summary["failures_by_phase"] = {p: fails[p] for p in sorted(fails, key=lambda p: phase_order.index(p)
                                                                 if p in phase_order else 99)}
    summary["failure_reasons"] = dict(Counter(f"{r.failure_phase}: {r.failure_reason}" for r in results
                                              if r.outcome == "failed").most_common())
    # Scan rates, from the controller's own scan records.
    intake = [r.scan.get("intake", {}).get("decoded") is not None for r in results if "intake" in r.scan]
    with_rescan = [bool(r.scan.get("intake", {}).get("decoded") or r.scan.get("rescan", {}).get("decoded"))
                   for r in results if "intake" in r.scan]
    ver = [r for r in results if "verify" in r.scan]
    ver_first = [r.scan["verify"].get("decoded") is not None for r in ver]
    ver_total = [bool(r.scan["verify"].get("decoded") or r.scan.get("verify_retry", {}).get("decoded")) for r in ver]
    summary["scan"] = {
        "intake_first": rate(sum(intake), len(intake)),
        "intake_with_rescan": rate(sum(with_rescan), len(with_rescan)),
        "rescans": sum("rescan" in r.scan for r in results),
        "verify_first": rate(sum(ver_first), len(ver_first)),
        "verify_with_retry": rate(sum(ver_total), len(ver_total)),
        "verify_match_of_read": rate(sum(r.scan.get("verify_match") is True for r in ver),
                                     sum(r.scan.get("verify_match") is not None for r in ver)),
    }
    grasped = [r for r in results if r.grasped]
    summary["grasp_rate"] = rate(len(grasped), sum(r.outcome != "refused" or r.grasped for r in results))
    summary["grasp_slip"] = {**rate(sum(r.slipped for r in grasped), len(grasped)),
                             "note": "of verified grasps, fraction that then lost two-finger contact"}
    summary["by_class"] = {}
    for c in scene.ITEM_CLASSES:
        rc = [r for r in results if r.object_class == c]
        summary["by_class"][c] = {
            "completion": rate(sum(r.success for r in rc), len(rc)),
            "misfile": rate(sum(r.misfile for r in rc), len(rc)),
            "refusal": rate(sum(r.outcome == "refused" for r in rc), len(rc)),
            "grasp_slip": rate(sum(r.slipped for r in rc if r.grasped), sum(r.grasped for r in rc)),
            "cycle_time_s": _stats([r.cycle_time for r in rc if r.success]),
            "placement_error_m": _stats([r.placement_error for r in rc if r.success]),
            "carry_tilt_deg": _stats([r.max_carry_tilt_deg for r in rc if r.grasped]),
        }
    summary["by_cabinet"] = {cab: rate(sum(r.success for r in results if (r.correct_location or "").startswith(cab)),
                                       sum((r.correct_location or "").startswith(cab) for r in results))
                             for cab in scene.CABINETS}
    summary["cycle_time_s"] = {**_stats([r.cycle_time for r in results if r.success]),
                               "note": "simulated seconds, IDLE to HOME, filed episodes"}
    summary["placement_error_m"] = _stats([r.placement_error for r in results if r.success])
    summary["carry_tilt_deg"] = _stats([r.max_carry_tilt_deg for r in grasped])
    summary["overall"] = summary["completion"]  # alias kept for the Phase 0B plotting and sweep code
    return summary


def _pct(r):
    if r["n"] == 0:
        return "   n/a"
    return f"{100 * r['rate']:5.1f}%  [{100 * r['ci95'][0]:4.1f}, {100 * r['ci95'][1]:5.1f}]  ({r['k']}/{r['n']})"


def print_table(summary, config):
    print("\n" + "=" * 86)
    print(f"EVALUATION  episodes={config['episodes']}  seeds {config['seed']}..{config['seed'] + config['episodes'] - 1}"
          f"  commit={config['git_commit']}  wall={config['wall_time_s']:.0f}s")
    print(f"perception: {config['perception']}   size×{config['size_mult']:g}  mass×{config['mass_mult']:g}")
    print("=" * 86)
    print(f"{'MISFILES':<26}{_pct(summary['misfile'])}   <- headline safety metric, target 0")
    print(f"{'misreads (decode != true)':<26}{_pct(summary['misread'])}")
    print(f"{'completion (filed)':<26}{_pct(summary['completion'])}")
    print(f"{'refused':<26}{_pct(summary['refusal'])}")
    for c, r in summary["refusal_by_cause"].items():
        if r["k"]:
            print(f"    {c:<22}{_pct(r)}")
    print(f"{'execution failures':<26}{_pct(summary['execution_failure'])}")
    for reason, k in summary["failure_reasons"].items():
        print(f"    {k:3d} x {reason}")
    print("-" * 86)
    sc = summary["scan"]
    print(f"scan  intake first attempt  {_pct(sc['intake_first'])}")
    print(f"      intake incl. re-scan  {_pct(sc['intake_with_rescan'])}   ({sc['rescans']} re-scans)")
    print(f"      verify first attempt  {_pct(sc['verify_first'])}")
    print(f"      verify incl. retry    {_pct(sc['verify_with_retry'])}")
    print(f"      verify match of reads {_pct(sc['verify_match_of_read'])}")
    print(f"grasp success             {_pct(summary['grasp_rate'])}")
    print(f"grasp slip (of grasped)   {_pct(summary['grasp_slip'])}")
    print("-" * 86)
    print("by content class           completion                        misfile   cycle(s)  place(mm)  tilt(deg)")
    for c, r in summary["by_class"].items():
        print(f"  {c:<10}{_pct(r['completion']):>44}  {r['misfile']['k']:>3}/{r['misfile']['n']:<4}"
              f"{r['cycle_time_s']['mean']:7.1f}  {1000 * r['placement_error_m']['mean']:7.1f}  "
              f"{r['carry_tilt_deg']['mean']:7.1f}")
    print("by cabinet (true route)")
    for cab, r in summary["by_cabinet"].items():
        print(f"  {cab:<10}{_pct(r)}")
    print("-" * 86)
    ct, pe, tl = summary["cycle_time_s"], summary["placement_error_m"], summary["carry_tilt_deg"]
    print(f"cycle time (filed)       mean {ct['mean']:.1f}s  p95 {ct['p95']:.1f}s")
    print(f"placement error          mean {pe['mean'] * 1000:.1f} mm  p95 {pe['p95'] * 1000:.1f} mm  max {pe['max'] * 1000:.1f} mm")
    print(f"carry tilt (CoM offset)  mean {tl['mean']:.1f}°  p95 {tl['p95']:.1f}°  max {tl['max']:.1f}°")
    print(f"custody chain            {'intact' if config['custody_chain_intact'] else 'BROKEN'}  "
          f"({config['custody_events']} events, {config['custody_log']})")
    print("=" * 86)


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
