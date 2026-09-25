"""Intake scan rate under image degradation (camera blur + sensor noise), on rendered frames.

    python3 scripts/scan_degradation.py [--episodes 100]

MuJoCo renders a perfect pinhole image (no defocus, motion blur, sensor noise or glare), so the clean read rate is a
simulation ceiling. This measures how the read rate, and crucially the misread rate, change as the image degrades.
A misread (decoding a DIFFERENT valid ID) would be a routing hazard; a no-read is safe (it ends in a refusal).
Writes out/scan_degradation.json.
"""

import argparse
import json

import mujoco
import numpy as np
from PIL import Image, ImageFilter

import labels
import scanner
import scene
from randomise import Randomiser

LEVELS = [(0.0, 0.0), (0.6, 2.0), (1.0, 4.0), (1.4, 6.0), (1.8, 8.0), (2.2, 10.0), (2.8, 12.0)]  # (blur px, noise sd)


def degrade(img, blur_px, noise_sd, rng):
    out = Image.fromarray(img)
    if blur_px > 0:
        out = out.filter(ImageFilter.GaussianBlur(blur_px))
    a = np.asarray(out, dtype=np.float32)
    if noise_sd > 0:
        a = a + rng.normal(0, noise_sd, a.shape)
    return np.clip(a, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed", type=int, default=90000)
    args = ap.parse_args()
    m, d = scene.load()
    rnd = Randomiser(m)
    r = mujoco.Renderer(m, scene.SCANNER_RES[1], scene.SCANNER_RES[0])
    frames = []
    for k in range(args.episodes):
        seed = args.seed + k
        scene.reset_home(m, d)
        p = rnd.apply(d, seed)
        idx = seed % labels.POOL_SIZE
        rnd.set_label(p.object_class, f"label_{idx:02d}")
        mujoco.mj_step(m, d, nstep=150)
        r.update_scene(d, camera=scene.SCANNER_CAM)
        frames.append((r.render(), labels.POOL_IDS[idx]))
    rng = np.random.default_rng(0)
    rows = []
    for blur, noise in LEVELS:
        ok = mis = 0
        for img, truth in frames:
            text = scanner.decode(degrade(img, blur, noise, rng)).text
            ok += text == truth
            mis += text is not None and text != truth
        rows.append({"blur_px": blur, "noise_sd": noise, "read": ok, "misread": mis, "n": len(frames)})
        print(f"blur {blur:.1f} px (={blur / (0.278 / 0.135):.2f} modules), noise sd {noise:4.1f}: read {ok}/{len(frames)}, "
              f"misread {mis}")
    (scene.OUT_DIR / "scan_degradation.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
