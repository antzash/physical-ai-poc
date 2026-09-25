"""Phase 1 charts: completion vs pose error, and what happens to every item (filed / refused / failed / misfiled).

    python3 scripts/plot_phase1.py out/robustness_<ts>.json

Writes out/p1_completion_curve.png, out/p1_outcomes_position.png and out/p1_outcomes_ood.png.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import scene  # noqa: E402
from plot_robustness import (AXIS, CLASS_STYLE, GRID, INK, INK2, MUTED, OVERALL, PERCEPTION_BAND, REGION,  # noqa: E402
                             SURFACE, _style_axes)

# Validated with the dataviz palette validator (adjacent CVD dE >= 21.5). Amber is below 3:1 contrast, so every
# segment is also direct-labelled when large enough and the legend is always present.
OUTCOME_STYLE = [
    ("filed", "#2a78d6", "filed"),
    ("refused", "#eda100", "refused (fail closed)"),
    ("failed", "#4a3aa7", "execution failure (not filed)"),
    ("misfile", "#d03b3b", "MISFILED"),
]


def _points(data, sweep, key):
    return sorted((p for p in data["points"] if p["sweep"] == sweep), key=lambda p: p["point"][key])


def _counts(p):
    eps = p["episodes"]
    c = {"filed": sum(e["success"] for e in eps), "misfile": sum(e["misfile"] for e in eps),
         "refused": sum(e["outcome"] == "refused" and not e["misfile"] for e in eps)}
    c["failed"] = len(eps) - c["filed"] - c["misfile"] - c["refused"]
    return c, len(eps)


def completion_curve(data, out):
    pts = _points(data, "position", "pos_mm")
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=SURFACE)
    _style_axes(ax)
    ax.axvspan(*PERCEPTION_BAND, color=REGION, zorder=0, linewidth=0)
    ax.text(sum(PERCEPTION_BAND) / 2, 4, "typical RGB-D pose-estimator error\non a known rigid object (3–10 mm)",
            ha="center", va="bottom", color=INK2, fontsize=9, linespacing=1.3)
    xs = [p["point"]["pos_mm"] for p in pts]
    for cls, (color, marker) in CLASS_STYLE_P1.items():
        s = [p["summary"]["by_class"][cls]["completion"] for p in pts]
        y = [100 * r["rate"] for r in s]
        ax.fill_between(xs, [100 * r["ci95"][0] for r in s], [100 * r["ci95"][1] for r in s], color=color,
                        alpha=0.07, linewidth=0, zorder=1)
        ax.plot(xs, y, color=color, linewidth=2, marker=marker, markersize=6, markeredgecolor=SURFACE,
                markeredgewidth=1.5, label=cls, zorder=3)
    o = [p["summary"]["completion"] for p in pts]
    ax.fill_between(xs, [100 * r["ci95"][0] for r in o], [100 * r["ci95"][1] for r in o], color=OVERALL, alpha=0.08,
                    linewidth=0, zorder=1)
    ax.plot(xs, [100 * r["rate"] for r in o], color=OVERALL, linewidth=4.5, marker="o", markersize=7,
            markeredgecolor=SURFACE, markeredgewidth=1.5, label="overall", zorder=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{x:g}" for x in xs])
    ax.set_xlabel("Position error σ per axis (mm)", color=INK2, fontsize=11)
    ax.set_ylabel("Completion (filed, verified)", color=INK2, fontsize=11)
    mis = sum(p["summary"]["misfile"]["k"] for p in pts)
    n = sum(p["summary"]["misfile"]["n"] for p in pts)
    ax.set_title("Evidence intake: completion vs position error", loc="left", color=INK, fontsize=14,
                 fontweight="bold", pad=22)
    ax.text(0, 1.02, f"{pts[0]['summary']['completion']['n']} episodes per point · 95% Wilson bands · "
                     f"misfiles across the sweep: {mis} of {n}", transform=ax.transAxes, color=MUTED, fontsize=9.5)
    handles, labels_ = ax.get_legend_handles_labels()
    order = [labels_.index("overall")] + [labels_.index(c) for c in CLASS_STYLE_P1]
    fig.legend([handles[i] for i in order], [labels_[i] for i in order], loc="lower center",
               bbox_to_anchor=(0.5, 0.0), frameon=False, labelcolor=INK2, fontsize=10, ncol=5)
    fig.subplots_adjust(left=0.1, right=0.97, top=0.86, bottom=0.2)
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def outcomes(data, sweep, key, xlabel, title, out):
    pts = _points(data, sweep, key)
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=SURFACE)
    _style_axes(ax)
    xs = list(range(len(pts)))
    bottom = [0.0] * len(pts)
    for name, color, label in OUTCOME_STYLE:
        vals = []
        for p in pts:
            c, n = _counts(p)
            vals.append(100 * c[name] / n)
        # 2 px surface gap between stacked segments (linewidth in the surface colour)
        ax.bar(xs, vals, bottom=bottom, width=0.6, color=color, edgecolor=SURFACE, linewidth=1.5, label=label,
               zorder=2)
        for x, v, b in zip(xs, vals, bottom):
            if v >= 7:
                ax.text(x, b + v / 2, f"{v:.0f}%", ha="center", va="center", fontsize=8.5,
                        color="#ffffff" if name != "refused" else INK, zorder=3)
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{p['point'][key]:g}" for p in pts])
    ax.set_xlabel(xlabel, color=INK2, fontsize=11)
    ax.set_ylabel("Share of episodes", color=INK2, fontsize=11)
    mis = sum(_counts(p)[0]["misfile"] for p in pts)
    n = sum(_counts(p)[1] for p in pts)
    ax.set_title(title, loc="left", color=INK, fontsize=14, fontweight="bold", pad=22)
    ax.text(0, 1.02, f"{_counts(pts[0])[1]} episodes per point · misfiled: {mis} of {n}", transform=ax.transAxes,
            color=MUTED, fontsize=9.5)
    fig.legend(loc="lower center", bbox_to_anchor=(0.5, 0.0), frameon=False, labelcolor=INK2, fontsize=10, ncol=4)
    fig.subplots_adjust(left=0.1, right=0.97, top=0.86, bottom=0.2)
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


CLASS_STYLE_P1 = dict(zip(scene.ITEM_CLASSES, CLASS_STYLE.values()))


def main():
    data = json.loads(Path(sys.argv[1]).read_text())
    plt.rcParams["font.family"] = ["Helvetica", "Arial", "DejaVu Sans"]
    completion_curve(data, scene.OUT_DIR / "p1_completion_curve.png")
    outcomes(data, "position", "pos_mm", "Position error σ per axis (mm)",
             "What happens to each item as pose error grows", scene.OUT_DIR / "p1_outcomes_position.png")
    outcomes(data, "ood", "mult", "Size and mass envelope multiplier (× training range)",
             "What happens to each item out of distribution", scene.OUT_DIR / "p1_outcomes_ood.png")
    print("wrote out/p1_completion_curve.png, out/p1_outcomes_position.png, out/p1_outcomes_ood.png")


if __name__ == "__main__":
    main()
