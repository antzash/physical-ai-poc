"""The perception seam: the ONLY item state the controller is allowed to use.

    error = draw_error(noise_spec, rng, n_size)                       # once per episode, at reset
    obs = observe(model, data, item_body, item_geom, error)          # at every controller read site

EVALUATION BOUNDARY — do not erode this.
Noise belongs on the controller's input only. Everything that JUDGES the outcome must keep reading ground truth
straight from mjData: grasp_ok / grasp_sample / grasp_verified / _check_hold / finger_contact_forces in
controller.py, the PLACED / VERIFIED checks and placement error in episode.py, and anything in evaluate.py. If an
evaluator ever reads an ItemObservation, success and failure stop meaning anything.

The noise perturbs the robot's BELIEF, never the physics: the item stays exactly where the simulator put it.

The error is drawn ONCE per episode and cached in a frozen PerceptionError. Real perception gives one estimate with
one fixed error for the whole attempt; re-sampling at each call site would average the error away across APPROACH,
DESCEND and grasp-height selection and measure something close to no noise at all. That is why observe() takes an
already-drawn error rather than an rng: a fresh draw per call is impossible by construction.

This is also the interface a pose estimator plugs into later: replace `observe` with a model that returns an
ItemObservation from images.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class NoiseSpec:
    pos_sigma_m: float = 0.0  # per-axis std dev of the position error (x, y and z)
    yaw_sigma_rad: float = 0.0  # std dev of the yaw error
    size_sigma_frac: float = 0.0  # per-axis std dev of the relative size error (0.05 = 5 %)

    @property
    def is_zero(self):
        return self.pos_sigma_m == 0 and self.yaw_sigma_rad == 0 and self.size_sigma_frac == 0

    def describe(self):
        if self.is_zero:
            return "perfect state (no perception error)"
        parts = []
        if self.pos_sigma_m:
            parts.append(f"position σ {self.pos_sigma_m * 1000:g} mm")
        if self.yaw_sigma_rad:
            parts.append(f"yaw σ {np.degrees(self.yaw_sigma_rad):g}°")
        if self.size_sigma_frac:
            parts.append(f"size σ {self.size_sigma_frac * 100:g}%")
        return "pose error: " + ", ".join(parts)


@dataclass(frozen=True)
class PerceptionError:
    """One episode's fixed perception error. Recorded in the episode result so failures replay exactly."""
    pos_offset: tuple = (0.0, 0.0, 0.0)  # metres, world frame
    yaw_offset: float = 0.0  # radians
    size_scale: tuple = (1.0, 1.0, 1.0)  # multiplicative, per geom-size axis

    def to_dict(self):
        return {"pos_offset_mm": [round(v * 1000, 3) for v in self.pos_offset],
                "yaw_offset_deg": round(float(np.degrees(self.yaw_offset)), 3),
                "size_scale": [round(v, 5) for v in self.size_scale]}


ZERO_ERROR = PerceptionError()


def draw_error(spec: NoiseSpec, rng: np.random.Generator) -> PerceptionError:
    """Draw the episode's perception error. Call exactly once per episode."""
    if spec.is_zero:
        return ZERO_ERROR
    pos = rng.normal(0.0, spec.pos_sigma_m, 3) if spec.pos_sigma_m else np.zeros(3)
    yaw = float(rng.normal(0.0, spec.yaw_sigma_rad)) if spec.yaw_sigma_rad else 0.0
    # Clip so an extreme draw can never produce a non-positive size estimate.
    size = np.clip(1.0 + rng.normal(0.0, spec.size_sigma_frac, 3), 0.2, None) if spec.size_sigma_frac else np.ones(3)
    return PerceptionError(tuple(float(v) for v in pos), yaw, tuple(float(v) for v in size))


@dataclass(frozen=True)
class ItemObservation:
    """What the controller believes about the item. Assumes the item rests upright/flat (no tilt estimate)."""
    position: np.ndarray = field(compare=False)  # (3,) body centre
    yaw: float
    size: np.ndarray = field(compare=False)  # (n,) geom half-sizes as the geom stores them
    vertical_half_extent: float  # believed half-height of the item above/below its centre
    error: PerceptionError


def _vertical_half_extent(geom_type, size):
    # Cylinder sizes are (radius, half-height); box and ellipsoid sizes are (x, y, z) half-extents.
    return float(size[1] if geom_type == "cylinder" else size[2])


def observe(model, data, item_body, item_geom, error: PerceptionError, geom_type) -> ItemObservation:
    """Ground-truth pose and size, plus this episode's cached perception error."""
    R = data.xmat[item_body].reshape(3, 3)
    true_yaw = float(np.arctan2(R[1, 0], R[0, 0]))
    true_size = np.array(model.geom_size[item_geom], dtype=float)
    n = 2 if geom_type == "cylinder" else 3
    size = true_size[:n] * np.asarray(error.size_scale[:n])
    return ItemObservation(
        position=np.array(data.xpos[item_body], dtype=float) + np.asarray(error.pos_offset),
        yaw=true_yaw + error.yaw_offset,
        size=size,
        vertical_half_extent=_vertical_half_extent(geom_type, size),
        error=error,
    )
