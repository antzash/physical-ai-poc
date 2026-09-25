"""Evidence bag labels: a real Code128 barcode of the item ID plus a human-readable line.

    python3 scripts/labels.py                # (re)generate the label pool in models/panda/labels/

The officer applies this label; the robot only reads it. Bars are drawn from python-barcode's Code128 module string
at an exact integer number of texture pixels per module, so the texture never limits the decode; the camera does.
"""

from pathlib import Path

from barcode import Code128
from PIL import Image, ImageDraw, ImageFont

import scene

LABEL_DIR = scene.ROOT / "models" / "panda" / "labels"
LABEL_SIZE_MM = (56.0, 40.0)  # along the bag's long axis (x), across it (y); matches the label geom in the scene
BARCODE_SPAN_MM = 52.0  # symbol plus quiet zones, across the label's long side
QUIET_MODULES = 10  # Code128 minimum quiet zone each side
PX_PER_MODULE = 8
BAR_HEIGHT_FRAC = 0.66
FONT = "/System/Library/Fonts/Menlo.ttc"

# The pool: pre-generated labels selected per episode by material. Item IDs follow EV-2026-00XXXX.
POOL_SIZE = 24
POOL_IDS = [f"EV-2026-00{1000 + 37 * i:04d}" for i in range(POOL_SIZE)]
# Special labels for refusal paths: a registered-format ID that has no case record, and a damaged label.
UNREGISTERED_ID = "EV-2026-009999"
DAMAGED_ID = "EV-2026-001184"  # a real ID, printed on a label whose barcode is scuffed through


def modules(item_id):
    return Code128(item_id).build()[0]


def module_mm(item_id):
    return BARCODE_SPAN_MM / (len(modules(item_id)) + 2 * QUIET_MODULES)


def render_label(item_id, damaged=False):
    """Label image, width along the label's long side. Returns a PIL image."""
    mods = modules(item_id)
    px_per_mm = PX_PER_MODULE / module_mm(item_id)
    w, h = round(LABEL_SIZE_MM[0] * px_per_mm), round(LABEL_SIZE_MM[1] * px_per_mm)
    img = Image.new("RGB", (w, h), (250, 250, 247))
    d = ImageDraw.Draw(img)
    x0 = (w - (len(mods) + 2 * QUIET_MODULES) * PX_PER_MODULE) // 2 + QUIET_MODULES * PX_PER_MODULE
    top = round(0.07 * h)
    bottom = top + round(BAR_HEIGHT_FRAC * h)
    for i, bit in enumerate(mods):
        if bit == "1":
            d.rectangle([x0 + i * PX_PER_MODULE, top, x0 + (i + 1) * PX_PER_MODULE - 1, bottom], fill=(10, 10, 10))
    font = ImageFont.truetype(FONT, round(0.13 * h))
    tw = d.textlength(item_id, font=font)
    d.text(((w - tw) / 2, bottom + round(0.04 * h)), item_id, font=font, fill=(10, 10, 10))
    if damaged:
        # A scuff/tear through the bars: the human-readable line survives, the symbol does not.
        d.polygon([(0.30 * w, top - 5), (0.58 * w, top - 5), (0.50 * w, bottom + 5), (0.22 * w, bottom + 5)],
                  fill=(250, 250, 247))
        for k in range(6):
            y = top + (k + 0.5) * (bottom - top) / 6
            d.line([(0.18 * w, y), (0.62 * w, y + 12)], fill=(120, 110, 90), width=5)
    return img


def label_file(i):
    return LABEL_DIR / f"label_{i:02d}.png"


def generate_pool():
    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    for i, item_id in enumerate(POOL_IDS):
        render_label(item_id).save(label_file(i))
    render_label(UNREGISTERED_ID).save(LABEL_DIR / "label_unregistered.png")
    render_label(DAMAGED_ID, damaged=True).save(LABEL_DIR / "label_damaged.png")
    return LABEL_DIR


if __name__ == "__main__":
    out = generate_pool()
    print(f"wrote {POOL_SIZE} + 2 labels to {out}; module {module_mm(POOL_IDS[0]):.3f} mm, "
          f"{len(modules(POOL_IDS[0])) + 2 * QUIET_MODULES} modules with quiet zones")
