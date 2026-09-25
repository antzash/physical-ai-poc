"""Barcode reading from rendered camera pixels (zxing-cpp). The first real perception in the pipeline.

decode(rgb) -> ScanResult. The input is ONLY an RGB image from a camera render; nothing here may consult the
simulator's item identity. If nothing decodes, that is a genuine scan failure and the caller must refuse (fail closed).

Omnidirectional reading: a linear barcode only tolerates a small skew along a straight scanline, and the bag can lie
at any yaw. Like an omnidirectional intake scanner, we (1) locate the label by its bar texture, (2) estimate the bar
orientation from image gradients and rotate the crop so the bars are vertical, then decode; (3) if that misses, fall
back to a 15-degree rotation search.
"""

import time
from dataclasses import dataclass, field

import numpy as np
import zxingcpp
from PIL import Image

BLOCK = 24  # px, block size for locating bar texture
CONTRAST_STD = 55.0  # grey-level std above which a block looks like bars
PAD = 0.35  # crop padding as a fraction of the found region size
SEARCH_STEP_DEG = 15


@dataclass
class ScanResult:
    text: str | None
    method: str  # "oriented", "search", or "none"
    angle_deg: float | None
    candidates: int
    seconds: float
    attempts: int = 0
    region: tuple | None = None  # (x0, y0, x1, y1) in the input image
    extra: dict = field(default_factory=dict)

    @property
    def ok(self):
        return self.text is not None


def _grey(rgb):
    return rgb[..., :3].astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)


def locate(grey):
    """Bounding boxes of clusters of high-contrast blocks (bar texture), largest first."""
    h, w = grey.shape
    bh, bw = h // BLOCK, w // BLOCK
    blocks = grey[: bh * BLOCK, : bw * BLOCK].reshape(bh, BLOCK, bw, BLOCK)
    hot = blocks.std(axis=(1, 3)) > CONTRAST_STD
    seen = np.zeros_like(hot)
    boxes = []
    for sy, sx in zip(*np.nonzero(hot)):
        if seen[sy, sx]:
            continue
        stack, cells = [(sy, sx)], []
        seen[sy, sx] = True
        while stack:
            y, x = stack.pop()
            cells.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < bh and 0 <= nx < bw and hot[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
        if len(cells) < 4:
            continue
        ys, xs = zip(*cells)
        y0, y1, x0, x1 = min(ys) * BLOCK, (max(ys) + 1) * BLOCK, min(xs) * BLOCK, (max(xs) + 1) * BLOCK
        py, px = int(PAD * (y1 - y0)), int(PAD * (x1 - x0))
        boxes.append((len(cells), (max(0, x0 - px), max(0, y0 - py), min(w, x1 + px), min(h, y1 + py))))
    return [b for _, b in sorted(boxes, reverse=True)]


def bar_angle(grey):
    """Dominant gradient direction (deg) in a crop: perpendicular to the bars. Doubled-angle averaging."""
    gy, gx = np.gradient(grey)
    mag2 = gx * gx + gy * gy
    ang = np.arctan2(gy, gx)
    c, s = (mag2 * np.cos(2 * ang)).sum(), (mag2 * np.sin(2 * ang)).sum()
    return float(np.degrees(0.5 * np.arctan2(s, c)))


def _read(img):
    res = zxingcpp.read_barcodes(img, formats=zxingcpp.BarcodeFormat.Code128)
    return res[0].text if res else None


def decode(rgb, max_candidates=3):
    t0 = time.time()
    grey = _grey(np.asarray(rgb))
    boxes = locate(grey)[:max_candidates]
    attempts = 0
    for box in boxes:
        x0, y0, x1, y1 = box
        crop = Image.fromarray(np.clip(grey[y0:y1, x0:x1], 0, 255).astype(np.uint8))
        grad = bar_angle(np.asarray(crop, np.float32))
        # Rotate so the gradient (normal to the bars) lies along the image x axis: bars become vertical.
        for method, angles in (("oriented", [grad]), ("search", range(0, 180, SEARCH_STEP_DEG))):
            for a in angles:
                attempts += 1
                text = _read(crop.rotate(a, expand=True, fillcolor=255, resample=Image.BILINEAR))
                if text is not None:
                    return ScanResult(text, method, float(a), len(boxes), time.time() - t0, attempts, box)
    return ScanResult(None, "none", None, len(boxes), time.time() - t0, attempts, None)


class Cameras:
    """The robot's cameras. The only way the robot perceives identity is through these rendered pixels."""

    def __init__(self, model):
        import mujoco

        import scene

        self._mujoco, self._scene, self.model = mujoco, scene, model
        w, h = scene.SCANNER_RES
        self.scanner = mujoco.Renderer(model, h, w)
        ww, wh = scene.WRIST_RES
        self.wrist = mujoco.Renderer(model, wh, ww)
        self.last_image = {}  # camera name -> last rendered frame (for the video overlay)

    def _render(self, renderer, data, cam):
        renderer.update_scene(data, camera=cam)
        img = renderer.render()
        self.last_image[cam] = img
        return img

    def scan_intake(self, data):
        cam = self._scene.SCANNER_CAM
        r = decode(self._render(self.scanner, data, cam))
        r.extra["camera"] = cam
        return r

    def scan_wrist(self, data):
        """Try both wrist cameras; return the first decode (or the last no-read)."""
        r = None
        for cam in self._scene.WRIST_CAMS:
            r = decode(self._render(self.wrist, data, cam))
            r.extra["camera"] = cam
            if r.ok:
                return r
        return r

    def close(self):
        self.scanner.close()
        self.wrist.close()
