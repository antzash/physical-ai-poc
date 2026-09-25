"""Per-episode domain randomisation, applied by mutating mjModel fields in place (no XML recompilation).

Every evidence unit is a SEALED EVIDENCE BAG (a box pouch); what varies is the content class inside it.
Randomised: content class, bag size, mass, centre-of-mass offset, friction, pose on the counter, and key-light
direction/position/colour. All draws come from one seeded numpy Generator, so an episode is exactly replayable.

The centre of mass is offset along the bag's long axis (contents settle to one end), by moving body_ipos. The
controller grasps at the bag's geometric centre, so an offset CoM tips the bag in the grasp. That is intended.
"""

from dataclasses import dataclass, asdict

import mujoco
import numpy as np

import scene

# Per-axis scale ranges for the bag (length x, width y, thickness z) relative to the nominal XML size.
# Width (the axis the gripper closes across) stays <= ~67 mm so every bag fits the 80 mm finger stroke. Length is
# kept long enough that the label fits at one end clear of the hand (see LABEL_* below).
SIZE_SCALE = {
    "phone": [(1.0, 1.1), (0.85, 1.12), (0.8, 1.25)],
    "blade": [(0.96, 1.07), (0.85, 1.15), (0.8, 1.2)],
    "garment": [(1.0, 1.1), (0.85, 1.05), (0.8, 1.25)],
    "carton": [(1.05, 1.12), (0.85, 1.08), (0.8, 1.2)],
}
MASS_RANGE = {  # kg, bag plus contents
    "phone": (0.15, 0.40),
    "blade": (0.10, 0.45),
    "garment": (0.10, 0.50),
    "carton": (0.20, 0.85),
}
# Centre-of-mass offset along the long axis, as a fraction of the bag's half-length (sign drawn separately).
COM_OFFSET_FRAC = {
    "phone": (0.15, 0.45),
    "blade": (0.10, 0.40),
    "garment": (0.00, 0.12),
    "carton": (0.15, 0.45),
}
# The label sits on the top face at the +x end, starting at least LABEL_CLEAR_X from the grasp centre (the hand
# body spans +-32 mm along x over the grasp centre) so the wrist camera can see it during the verify scan.
LABEL_CLEAR_X = 0.045
LABEL_END_MARGIN = 0.004
FRICTION_RANGE = (0.6, 1.4)  # sliding friction; governs pad-item contact because items have contact priority
# Torsional friction (MuJoCo: a length; max torque = coefficient x normal force) derived from the sliding friction and
# the Franka fingertip pad's contact patch (~17 mm square -> effective friction radius ~0.38 x 17 mm = 6.5 mm).
# Phase 0 used a flat 0.02 m, which resists ~0.8 N*m at 21 N per pad and made an offset centre of mass physically
# irrelevant (bags tilted ~1 deg in the grasp); see NOTES.md, Phase 1 Task A.
PAD_PATCH_RADIUS = 0.0065
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
    com_offset: float  # metres along the bag's long axis (body frame x); + is the label end

    def to_dict(self):
        return asdict(self)


class Randomiser:
    """Snapshots the nominal model values once, so every episode is drawn relative to nominal (no compounding)."""

    def __init__(self, model):
        self.m = model
        self.geoms = {c: scene.geom_id(model, scene.item_geom(c)) for c in scene.ITEM_CLASSES}
        self.bodies = {c: scene.body_id(model, scene.item_body(c)) for c in scene.ITEM_CLASSES}
        self.contents = {c: scene.geom_id(model, f"item_{c}_contents") for c in scene.ITEM_CLASSES}  # visual only
        self.labels = {c: scene.geom_id(model, scene.item_label(c)) for c in scene.ITEM_CLASSES}  # visual only
        self.nom_contents = {c: model.geom_size[g].copy() for c, g in self.contents.items()}
        for b in self.bodies.values():
            model.body_sameframe[b] = 0  # CoM may leave the body origin (see scene.build_spec)
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

    def _set_geometry(self, cls, size, mass, com_offset, size_mult):
        m, g, b = self.m, self.geoms[cls], self.bodies[cls]
        m.geom_size[g] = size
        m.geom_aabb[g] = [0, 0, 0, *size]
        m.geom_rbound[g] = np.linalg.norm(size)
        a, bb, c = size
        # Explicit box inertia about the centre of mass (unchanged from Phase 0), with the CoM itself offset.
        m.body_mass[b] = mass
        m.body_inertia[b] = [mass * (bb * bb + c * c) / 3, mass * (a * a + c * c) / 3, mass * (a * a + bb * bb) / 3]
        m.body_ipos[b] = [com_offset, 0.0, 0.0]
        # Contents (visual only) sit where the mass is, kept inside the bag.
        k = self.contents[cls]
        m.geom_size[k] = np.minimum(self.nom_contents[cls] * size_mult, np.asarray(size) - 0.002)
        room = a - m.geom_size[k][0] - 0.002
        m.geom_pos[k] = [float(np.clip(com_offset, -room, room)), 0.0, 0.0]
        # Label (visual only) on the top face at the +x end, clear of the hand's footprint over the grasp centre.
        lab = self.labels[cls]
        lx = a - LABEL_END_MARGIN - m.geom_size[lab][0]
        assert lx - m.geom_size[lab][0] >= LABEL_CLEAR_X - 1e-9, "bag too short for the label to clear the hand"
        m.geom_pos[lab] = [lx, 0.0, c + m.geom_size[lab][2]]
        # Derived fields that MuJoCo does NOT recompute when geom_size changes. A stale body BVH box makes the
        # collision midphase cull contacts for enlarged items, which then sink into the counter and get ejected.
        assert m.body_bvhnum[b] == 1, "item bodies must have exactly one colliding geom"
        m.bvh_aabb[m.body_bvhadr[b]] = m.geom_aabb[g]
        dof = m.jnt_dofadr[m.body_jntadr[b]]
        m.dof_length[dof + 3:dof + 6] = m.geom_rbound[g]

    def _rest_height(self, cls):
        return float(self.m.geom_size[self.geoms[cls]][2])

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

    def apply(self, data, seed, object_class=None, size_mult=1.0, mass_mult=1.0):
        """Randomise the model and place the items. Call after resetting data to the home keyframe.

        size_mult / mass_mult scale the drawn size and mass at runtime for out-of-distribution sweeps. They are
        applied after the draws, so the RNG sequence and the ranges above are unchanged and 1.0 is bit-identical.
        """
        rng = np.random.default_rng(seed)
        cls = object_class or scene.ITEM_CLASSES[rng.integers(len(scene.ITEM_CLASSES))]
        scales = np.array([rng.uniform(lo, hi) for lo, hi in SIZE_SCALE[cls]])
        nominal = self.nom_size[cls][: len(scales)]
        size = nominal * scales * size_mult
        mass = float(rng.uniform(*MASS_RANGE[cls])) * mass_mult
        com_offset = float(rng.choice([-1.0, 1.0]) * rng.uniform(*COM_OFFSET_FRAC[cls]) * size[0])
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
        self._set_geometry(cls, size, mass, com_offset, size_mult)
        self.m.geom_friction[self.geoms[cls]][0] = friction
        self.m.geom_friction[self.geoms[cls]][1] = friction * PAD_PATCH_RADIUS
        self.m.light_pos[self.light] = light_pos
        self.m.light_dir[self.light] = light_dir
        self.m.light_diffuse[self.light] = light_diffuse
        self._activate(data, cls, xy, yaw)
        mujoco.mj_setConst(self.m, self._scratch)
        mujoco.mj_forward(self.m, data)
        return EpisodeParams(int(seed), cls, size.round(5).tolist(), round(mass, 4), round(friction, 4),
                             [round(v, 4) for v in xy], round(yaw, 4), light_pos.round(3).tolist(),
                             light_dir.round(3).tolist(), light_diffuse.round(3).tolist(), round(com_offset, 5))
