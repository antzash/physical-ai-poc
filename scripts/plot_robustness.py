"""Plot a robustness sweep: success rate vs perception error (and vs the OOD envelope).

    python3 scripts/plot_robustness.py                              # newest out/robustness_*.json
    python3 scripts/plot_robustness.py out/robustness_A.json out/robustness_B.json   # merge sweeps

`yaw_wide` points are drawn as a continuation of the `yaw` sweep.

Writes out/robustness_curve.png (position sweep, the headline chart) plus robustness_yaw.png,
robustness_combined.png, robustness_size.png, robustness_ood.png and a 2x2 robustness_overview.png.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import scene  # noqa: E402

# Reference categorical palette, slots 1-4 (validated with the dataviz validator: adjacent CVD ΔE ≥ 9.1).
# Aqua and yellow sit below 3:1 on the surface, so every class also carries its own marker shape.
CLASS_STYLE = {
    "box": ("#2a78d6", "o"),
    "bag": ("#eb6834", "s"),
    "cylinder": ("#1baf7a", "^"),
    "folder": ("#eda100", "D"),
}
OVERALL = "#0b0b0b"
SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
REGION = "#f0efec"

PERCEPTION_BAND = (3, 10)  # mm
PERCEPTION_LABEL = "typical RGB-D pose-estimator error\non a known rigid object (3–10 mm)"

SWEEP_AXIS = {
    "position": ("pos_mm", "Position error σ per axis (mm)"),
    "yaw": ("yaw_deg", "Yaw error σ (degrees)"),
    "combined": ("pos_mm", "Paired error σ (k mm position + k° yaw)"),
    "size": ("size_pct", "Size-estimate error σ per axis (%)"),
    "ood": ("mult", "Size and mass envelope multiplier (× training range)"),
}
TITLES = {
    "position": "Success vs position error",
    "yaw": "Success vs yaw error",
    "combined": "Success vs combined position + yaw error",
    "size": "Success vs size-estimate error",
    "ood": "Success vs out-of-distribution item size and mass",
}


def _series(points, key, cls=None):
    xs, ys, lo, hi = [], [], [], []
    for p in points:
        s = p["summary"]["overall"] if cls is None else p["summary"]["by_class"][cls]
        if s["n"] == 0:
            continue
        xs.append(p["point"][key])
        ys.append(100 * s["rate"])
        lo.append(100 * s["ci95"][0])
        hi.append(100 * s["ci95"][1])
    return xs, ys, lo, hi


def _style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.set_ylim(0, 102)
    ax.set_yticks(range(0, 101, 20))
    ax.set_yticklabels([f"{v}%" for v in range(0, 101, 20)])
    ax.grid(True, color=GRID, linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=10, length=0)


def draw(ax, data, sweep, compact=False):
    wanted = ("yaw", "yaw_wide") if sweep == "yaw" else (sweep,)
    points = sorted((p for p in data["points"] if p["sweep"] in wanted), key=lambda p: p["point"][SWEEP_AXIS[sweep][0]])
    key, xlabel = SWEEP_AXIS[sweep]
    _style_axes(ax)
    if sweep in ("position", "combined"):
        ax.axvspan(*PERCEPTION_BAND, color=REGION, zorder=0, linewidth=0)
        ax.text(sum(PERCEPTION_BAND) / 2, 4, PERCEPTION_LABEL, ha="center", va="bottom", color=INK2,
                fontsize=8 if compact else 9, linespacing=1.3)
    for cls, (color, marker) in CLASS_STYLE.items():
        xs, ys, lo, hi = _series(points, key, cls)
        ax.fill_between(xs, lo, hi, color=color, alpha=0.07, linewidth=0, zorder=1)
        ax.plot(xs, ys, color=color, linewidth=2, marker=marker, markersize=6, markeredgecolor=SURFACE,
                markeredgewidth=1.5, solid_capstyle="round", solid_joinstyle="round", label=cls, zorder=3)
    # The overall line is drawn heavier and underneath, so a class that tracks it stays visible on top.
    xs, ys, lo, hi = _series(points, key)
    ax.fill_between(xs, lo, hi, color=OVERALL, alpha=0.08, linewidth=0, zorder=1)
    ax.plot(xs, ys, color=OVERALL, linewidth=4.5, marker="o", markersize=7, markeredgecolor=SURFACE,
            markeredgewidth=1.5, solid_capstyle="round", label="overall", zorder=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{x:g}" for x in xs])
    if sweep == "yaw" and max(xs) > 20:
        ax.set_xticks([x for x in xs if x in (0, 5, 12, 20) or x > 20])
        ax.set_xticklabels([f"{x:g}" for x in ax.get_xticks()])
        ax.axvline(20, color=AXIS, linewidth=0.8, zorder=0)
        ax.text(19, 4, "planned range 0–20°", color=MUTED, fontsize=8.5 if compact else 9, ha="right", va="bottom")
        ax.text(21, 4, "extension 30–90°", color=MUTED, fontsize=8.5 if compact else 9, ha="left", va="bottom")
    ax.set_xlabel(xlabel, color=INK2, fontsize=10 if compact else 11)
    ax.set_ylabel("Success rate", color=INK2, fontsize=10 if compact else 11)
    n = points[0]["summary"]["overall"]["n"] if points else 0
    ax.set_title(TITLES[sweep], loc="left", color=INK, fontsize=12 if compact else 14, fontweight="bold", pad=22)
    ax.text(0, 1.02, f"{n} randomised episodes per point · shaded bands are 95% Wilson intervals",
            transform=ax.transAxes, color=MUTED, fontsize=8.5 if compact else 9.5)
    return ax


def _legend(fig, ax):
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index("overall")] + [labels.index(c) for c in CLASS_STYLE]
    fig.legend([handles[i] for i in order], [labels[i] for i in order], loc="lower center",
               bbox_to_anchor=(0.5, 0.0), frameon=False, labelcolor=INK2, fontsize=10, ncol=5, handlelength=2.2,
               columnspacing=1.2)


def main():
    paths = [Path(a) for a in sys.argv[1:]] or [max(scene.OUT_DIR.glob("robustness_*.json"))]
    data = {"points": [p for path in paths for p in json.loads(path.read_text())["points"]]}
    present = {p["sweep"] for p in data["points"]}
    plt.rcParams["font.family"] = ["Helvetica", "Arial", "DejaVu Sans"]

    names = {"position": "robustness_curve.png", "yaw": "robustness_yaw.png",
             "combined": "robustness_combined.png", "size": "robustness_size.png", "ood": "robustness_ood.png"}
    for sweep, fname in names.items():
        if sweep not in present:
            continue
        fig, ax = plt.subplots(figsize=(8, 5), facecolor=SURFACE)
        draw(ax, data, sweep)
        _legend(fig, ax)
        fig.subplots_adjust(left=0.1, right=0.97, top=0.86, bottom=0.2)
        fig.savefig(scene.OUT_DIR / fname, dpi=200, facecolor=SURFACE)
        plt.close(fig)
        print(f"wrote out/{fname}")

    panels = [s for s in ("position", "yaw", "combined", "ood") if s in present]
    if len(panels) == 4:
        fig, axes = plt.subplots(2, 2, figsize=(14, 9.5), facecolor=SURFACE)
        for ax, sweep in zip(axes.flat, panels):
            draw(ax, data, sweep, compact=True)
        _legend(fig, axes.flat[0])
        fig.subplots_adjust(left=0.06, right=0.98, top=0.93, bottom=0.1, hspace=0.45, wspace=0.18)
        fig.savefig(scene.OUT_DIR / "robustness_overview.png", dpi=170, facecolor=SURFACE)
        plt.close(fig)
        print("wrote out/robustness_overview.png")


if __name__ == "__main__":
    main()
