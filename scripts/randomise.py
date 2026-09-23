"""Per-episode domain randomisation, applied by mutating mjModel fields in place (no XML recompilation).

Randomised: active object class, size, mass, friction, pose on the counter, and key-light direction/position/colour.
All draws come from one seeded numpy Generator, so an episode is exactly replayable from its seed.
"""

from dataclasses import dataclass, asdict

import mujoco
import numpy as np

import scene

# Per-axis size scale ranges relative to the nominal XML size (roughly +-30%). The upper bound of the axis the
# gripper closes across (local y; the cylinder radius) is capped so the item always fits the 80 mm finger stroke,
# and long axes are capped so the item fits a 245 mm bin.
SIZE_SCALE = {
    "box": [(0.7, 1.3), (0.7, 1.2), (0.7, 1.3)],
    "bag": [(0.7, 1.3), (0.7, 1.05), (0.7, 1.3)],
    "cylinder": [(0.7, 1.3), (0.7, 1.3)],  # radius, half-height
    "folder": [(0.7, 1.25), (0.7, 1.1), (0.7, 1.3)],
}
MASS_RANGE = {  # kg
    "box": (0.10, 0.80),
    "bag": (0.05, 0.60),
    "cylinder": (0.15, 0.90),
    "folder": (0.05, 0.40),
}
FRICTION_RANGE = (0.6, 1.4)  # sliding friction; governs pad-item contact because items have contact priority
POSE_X = (0.40, 0.60)
POSE_Y = (-0.25, 0.05)
LIGHT_POS_JITTER = 0.4  # m (sets the shadow-map frustum; the key light is directional)
LIGHT_DIR_JITTER = 0.3  # added to the horizontal components of the unit direction, then renormalised
LIGHT_DIFFUSE_SCALE = (0.8, 1.2)  # overall intensity
LIGHT_TINT = (0.95, 1.05)  # small independent per-channel colour shift

PARK_Z = -3.0  # well below the room floor (z = -0.8)


@dataclass
class EpisodeParams:
    seed: int
    object_class: str
    size: list
    mass: float
    friction: float
    pose_xy: list
    yaw: float
    light_pos: list
    light_dir: list
    light_diffuse: list

    def to_dict(self):
        return asdict(self)


class Randomiser:
    """Snapshots the nominal model values once, so every episode is drawn relative to nominal (no compounding)."""

    def __init__(self, model):
        self.m = model
        self.geoms = {c: scene.geom_id(model, scene.item_geom(c)) for c in scene.ITEM_CLASSES}
        self.bodies = {c: scene.body_id(model, scene.item_body(c)) for c in scene.ITEM_CLASSES}
        self.qadr = {c: model.jnt_qposadr[scene.joint_id(model, scene.item_joint(c))] for c in scene.ITEM_CLASSES}
        self.dadr = {c: model.jnt_dofadr[scene.joint_id(model, scene.item_joint(c))] for c in scene.ITEM_CLASSES}
        self.nom_size = {c: model.geom_size[g].copy() for c, g in self.geoms.items()}
        self.nom_friction = {c: model.geom_friction[g].copy() for c, g in self.geoms.items()}
        self.nom_contype = {c: int(model.geom_contype[g]) for c, g in self.geoms.items()}
        self.nom_conaff = {c: int(model.geom_conaffinity[g]) for c, g in self.geoms.items()}
        self.light = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_LIGHT, "key")
        self.nom_light_pos = model.light_pos[self.light].copy()
        self.nom_light_dir = model.light_dir[self.light].copy()
        self.nom_light_diffuse = model.light_diffuse[self.light].copy()
        self._scratch = mujoco.MjData(model)  # mj_setConst overwrites the MjData it is given

    def _set_geometry(self, cls, size, mass):
        m, g, b = self.m, self.geoms[cls], self.bodies[cls]
        if cls == "cylinder":
            r, h = size
            m.geom_size[g][:2] = size
            m.geom_rbound[g] = np.hypot(r, h)
            m.geom_aabb[g] = [0, 0, 0, r, r, h]
            inertia = [mass * (3 * r * r + 4 * h * h) / 12] * 2 + [mass * r * r / 2]
        else:
            m.geom_size[g] = size
            m.geom_aabb[g] = [0, 0, 0, *size]
            a, bb, c = size
            if cls == "bag":  # ellipsoid
                m.geom_rbound[g] = max(size)
                inertia = [mass * (bb * bb + c * c) / 5, mass * (a * a + c * c) / 5, mass * (a * a + bb * bb) / 5]
            else:  # box
                m.geom_rbound[g] = np.linalg.norm(size)
                inertia = [mass * (bb * bb + c * c) / 3, mass * (a * a + c * c) / 3, mass * (a * a + bb * bb) / 3]
        m.body_mass[b] = mass
        m.body_inertia[b] = inertia
        # Derived fields that MuJoCo does NOT recompute when geom_size changes. A stale body BVH box makes the
        # collision midphase cull contacts for enlarged items, which then sink into the counter and get ejected.
        assert m.body_bvhnum[b] == 1, "item bodies must have exactly one geom"
        m.bvh_aabb[m.body_bvhadr[b]] = m.geom_aabb[g]
        dof = m.jnt_dofadr[m.body_jntadr[b]]
        m.dof_length[dof + 3:dof + 6] = m.geom_rbound[g]

    def _rest_height(self, cls):
        s = self.m.geom_size[self.geoms[cls]]
        return float(s[1] if cls == "cylinder" else s[2])

    def _park(self, data, cls):
        """Move an inactive item far below the floor, stop it, and disable its collisions and gravity."""
        g, b, q, dv = self.geoms[cls], self.bodies[cls], self.qadr[cls], self.dadr[cls]
        self.m.geom_contype[g] = 0
        self.m.geom_conaffinity[g] = 0
        self.m.body_gravcomp[b] = 1.0
        i = scene.ITEM_CLASSES.index(cls)
        data.qpos[q:q + 7] = [i * 0.5, 0.0, PARK_Z, 1, 0, 0, 0]
        data.qvel[dv:dv + 6] = 0

    def _activate(self, data, cls, xy, yaw):
        g, b, q, dv = self.geoms[cls], self.bodies[cls], self.qadr[cls], self.dadr[cls]
        self.m.geom_contype[g] = self.nom_contype[cls]
        self.m.geom_conaffinity[g] = self.nom_conaff[cls]
        self.m.body_gravcomp[b] = 0.0
        z = scene.COUNTER_TOP_Z + self._rest_height(cls) + 0.001
        data.qpos[q:q + 7] = [xy[0], xy[1], z, np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        data.qvel[dv:dv + 6] = 0

    def apply(self, data, seed, object_class=None):
        """Randomise the model and place the items. Call after resetting data to the home keyframe."""
        rng = np.random.default_rng(seed)
        cls = object_class or scene.ITEM_CLASSES[rng.integers(len(scene.ITEM_CLASSES))]
        scales = np.array([rng.uniform(lo, hi) for lo, hi in SIZE_SCALE[cls]])
        nominal = self.nom_size[cls][: len(scales)]
        size = nominal * scales
        if cls == "bag":
            # A sealed bag lies flat; an ellipsoid taller than it is wide would stand on edge and roll over.
            size[2] = min(size[2], 0.85 * size[1])
        mass = float(rng.uniform(*MASS_RANGE[cls]))
        friction = float(rng.uniform(*FRICTION_RANGE))
        xy = [float(rng.uniform(*POSE_X)), float(rng.uniform(*POSE_Y))]
        yaw = float(rng.uniform(-np.pi, np.pi))
        light_pos = self.nom_light_pos + rng.uniform(-LIGHT_POS_JITTER, LIGHT_POS_JITTER, 3) * [1, 1, 0.25]
        light_dir = self.nom_light_dir + rng.uniform(-LIGHT_DIR_JITTER, LIGHT_DIR_JITTER, 3) * [1, 1, 0]
        light_dir /= np.linalg.norm(light_dir)
        light_diffuse = self.nom_light_diffuse * rng.uniform(*LIGHT_DIFFUSE_SCALE) * rng.uniform(*LIGHT_TINT, 3)
        light_diffuse = np.clip(light_diffuse, 0, 1)

        for c in scene.ITEM_CLASSES:
            if c != cls:
                self._park(data, c)
        self._set_geometry(cls, size, mass)
        self.m.geom_friction[self.geoms[cls]][0] = friction
        self.m.light_pos[self.light] = light_pos
        self.m.light_dir[self.light] = light_dir
        self.m.light_diffuse[self.light] = light_diffuse
        self._activate(data, cls, xy, yaw)
        mujoco.mj_setConst(self.m, self._scratch)
        mujoco.mj_forward(self.m, data)
        return EpisodeParams(int(seed), cls, size.round(5).tolist(), round(mass, 4), round(friction, 4),
                             [round(v, 4) for v in xy], round(yaw, 4), light_pos.round(3).tolist(),
                             light_dir.round(3).tolist(), light_diffuse.round(3).tolist())
