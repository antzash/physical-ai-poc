"""Gymnasium environment for learning evidence intake with a reduced Cartesian action space.

Action (Box[-1, 1]^6): TCP delta (dx, dy, dz, dyaw) scaled to at most 1 cm / 0.1 rad per step, a gripper command
(> 0 close, <= 0 open), and a rail velocity command (Phase 1: the arm rides a rail; at most 0.4 m/s). The Cartesian
target is held in the CARRIAGE frame, so moving the rail carries the arm with it. The damped least-squares IK from
scripts/ik.py turns the target into arm joint targets (the rail is not in the IK). Control runs at 20 Hz.

Observation: TCP pose, gripper state, item pose relative to the TCP, item size, class one-hot, target slot position
relative to the TCP and to the item, a two-finger contact flag, and the rail position and setpoint. State only; no
vision, and the target slot is given (the barcode/routing workflow lives in scripts/, not in this env).

Status: plumbing only. See NOTES.md (M7) for why no policy is trained here.
"""

import sys
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import ik  # noqa: E402
import scene  # noqa: E402
from controller import finger_contact_forces  # noqa: E402
from randomise import Randomiser  # noqa: E402

PHYSICS_STEPS = 25
MAX_STEPS = 600  # 30 s of simulated time (a rail traverse to CAB-C adds ~4 s)
RAIL_STEP = 0.02  # m per env step = 0.4 m/s
MAX_DPOS = 0.01
MAX_DYAW = 0.1
WORKSPACE_LO = np.array([0.05, -0.40, 0.005])
WORKSPACE_HI = np.array([0.75, 0.60, 0.45])
ITEM_BOUNDS_LO = np.array([-0.2, -1.2, -0.05])
ITEM_BOUNDS_HI = np.array([0.8, 1.4, 0.8])
LIFT_HEIGHT = 0.03
# MuJoCo's box-pad contacts flicker: one pad can drop out of the active set for single physics steps while an item
# is plainly held (NOTES.md, M5/M7). "Held" is therefore measured over the env step's physics substeps (each finger
# in contact for at least HELD_FRACTION of them), and a drop needs DROP_GRACE_STEPS consecutive un-held env steps.
HELD_FRACTION = 0.3
DROP_GRACE_STEPS = 2

R_REACH = 1.0
R_GRASP_BONUS = 5.0
R_CARRY = 2.0
R_PLACE_BONUS = 50.0
R_DROP = -10.0
R_TIME = -0.01


class EvidenceIntakeEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": int(1 / (PHYSICS_STEPS * 0.002))}

    def __init__(self, render_mode=None, max_steps=MAX_STEPS):
        self.m, self.d = scene.load()
        self.randomiser = Randomiser(self.m)
        self.ik = ik.IK(self.m)
        self.tcp = scene.site_id(self.m, "tcp")
        self.arm_act = np.array([scene.actuator_id(self.m, a) for a in scene.ARM_ACTUATORS])
        self.grip_act = scene.actuator_id(self.m, scene.GRIPPER_ACTUATOR)
        self.arm_qadr = scene.arm_qpos_adr(self.m)
        self.finger_qadr = np.array([self.m.jnt_qposadr[scene.joint_id(self.m, j)] for j in scene.FINGER_JOINTS])
        self.rail_act = scene.actuator_id(self.m, scene.RAIL_ACTUATOR)
        self.rail_qadr = self.m.jnt_qposadr[scene.joint_id(self.m, scene.RAIL_JOINT)]
        self.rail_range = self.m.jnt_range[scene.joint_id(self.m, scene.RAIL_JOINT)]
        self.max_steps = max_steps
        self.render_mode = render_mode
        self._renderer = None

        self.action_space = spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32)
        obs_dim = 3 + 2 + 2 + 3 + 3 + 3 + 4 + 3 + 3 + 1 + 2
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)

    # ---- helpers -------------------------------------------------------------------------------------------------

    def _item_state(self):
        pos = self.d.xpos[self.item_body].copy()
        R = self.d.xmat[self.item_body].reshape(3, 3)
        return pos, float(np.arctan2(R[1, 0], R[0, 0])), float(R[2, 2])

    def _simulate(self):
        """Advance one env step; return whether the item was held (both fingers) over the substeps."""
        contact = np.zeros(2)
        for _ in range(PHYSICS_STEPS):
            mujoco.mj_step(self.m, self.d)
            contact += finger_contact_forces(self.m, self.d, self.item_geom) > 0.1
        self._held_now = bool(np.all(contact >= HELD_FRACTION * PHYSICS_STEPS))
        return self._held_now

    def _in_slot(self, pos):
        return bool(np.all(np.abs(pos - self.slot_center) <= self.slot_half))

    def _obs(self):
        tcp = self.d.site_xpos[self.tcp]
        yaw = ik.tcp_yaw(self.d, self.tcp)
        item, iyaw, tilt = self._item_state()
        opening = float(np.mean(self.d.qpos[self.finger_qadr]) / 0.04)
        rel_yaw = 2 * (iyaw - yaw)  # the gripper is symmetric under a half turn
        size = np.zeros(3)
        size[: len(self.item_size)] = self.item_size
        onehot = np.eye(len(scene.ITEM_CLASSES))[scene.ITEM_CLASSES.index(self.item_cls)]
        return np.concatenate([
            tcp, [np.sin(yaw), np.cos(yaw)], [opening, self.grip_cmd], item - tcp, [np.sin(rel_yaw), np.cos(rel_yaw),
                                                                                    tilt],
            size, onehot, self.slot_center - tcp, self.slot_center - item, [float(self._held_now)],
            [self._rail_q(), self.rail_set],
        ]).astype(np.float32)

    def _rail_q(self):
        return float(self.d.qpos[self.rail_qadr])

    def _world(self, rel):
        return rel + np.array([0.0, self._rail_q(), 0.0])

    # ---- gym API -------------------------------------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        scene.reset_home(self.m, self.d)
        ep_seed = int(self.np_random.integers(2**31 - 1))
        params = self.randomiser.apply(self.d, ep_seed, options.get("object_class"))
        self.params = params
        self.item_cls = params.object_class
        self.item_size = np.array(params.size)
        self.item_body = scene.body_id(self.m, scene.item_body(self.item_cls))
        self.item_geom = scene.geom_id(self.m, scene.item_geom(self.item_cls))
        slot = options.get("slot") or scene.SLOT_NAMES[int(self.np_random.integers(len(scene.SLOT_NAMES)))]
        self.slot = slot
        self.slot_center, self.slot_half = scene.slot_volume(self.m, self.d, slot)
        mujoco.mj_step(self.m, self.d, nstep=150)  # let the item settle on the counter
        self.rest_z = self.d.xpos[self.item_body][2]
        self.rail_set = self._rail_q()
        self.target_rel = self.d.site_xpos[self.tcp].copy() - np.array([0.0, self._rail_q(), 0.0])
        self.target_yaw = ik.tcp_yaw(self.d, self.tcp)
        self.q_cmd = self.d.qpos[self.arm_qadr].copy()
        self.grip_cmd = -1.0
        self.steps = 0
        self.grasped_once = False
        self.was_held_lifted = False
        self.not_held_steps = 0
        self._held_now = False
        return self._obs(), {"seed": ep_seed, "object_class": self.item_cls, "slot": slot}

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        self.target_rel = np.clip(self.target_rel + a[:3] * MAX_DPOS, WORKSPACE_LO, WORKSPACE_HI)
        self.target_yaw = float(np.clip(self.target_yaw + a[3] * MAX_DYAW, -np.pi, np.pi))
        self.grip_cmd = 1.0 if a[4] > 0 else -1.0
        self.rail_set = float(np.clip(self.rail_set + a[5] * RAIL_STEP, *self.rail_range))
        self.d.ctrl[self.rail_act] = self.rail_set
        res = self.ik.solve(self.d, self._world(self.target_rel), ik.down_quat(self.target_yaw), q_init=self.q_cmd)
        self.q_cmd = res.q
        self.d.ctrl[self.arm_act] = self.q_cmd
        self.d.ctrl[self.grip_act] = scene.GRIPPER_CLOSED if self.grip_cmd > 0 else scene.GRIPPER_OPEN
        held = self._simulate()
        self.steps += 1

        tcp = self.d.site_xpos[self.tcp]
        item, _, _ = self._item_state()
        self.not_held_steps = 0 if held else self.not_held_steps + 1
        lifted = item[2] > self.rest_z + LIFT_HEIGHT
        held_lifted = held and lifted

        reward = R_TIME
        info = {"held": held, "lifted": lifted, "event": None}
        terminated = False
        if not self.grasped_once:
            reward -= R_REACH * np.linalg.norm(item - tcp)
        if held_lifted:
            if not self.grasped_once:
                reward += R_GRASP_BONUS
                info["event"] = "grasp"
            self.grasped_once = True
            reward -= R_CARRY * np.linalg.norm(item - self.slot_center)

        vel = np.zeros(6)
        mujoco.mj_objectVelocity(self.m, self.d, mujoco.mjtObj.mjOBJ_BODY, self.item_body, vel, 0)
        at_rest = np.linalg.norm(vel[3:]) < 0.01 and np.linalg.norm(vel[:3]) < 0.2
        if self.grasped_once and not held and self._in_slot(item) and at_rest:
            reward += R_PLACE_BONUS
            terminated = True
            info["event"] = "success"
        elif (self.was_held_lifted and self.not_held_steps >= DROP_GRACE_STEPS
              and not np.all(np.abs(item[:2] - self.slot_center[:2]) <= self.slot_half[:2])):
            reward += R_DROP
            terminated = True
            info["event"] = "drop"
        elif np.any(item < ITEM_BOUNDS_LO) or np.any(item > ITEM_BOUNDS_HI):
            reward += R_DROP
            terminated = True
            info["event"] = "out_of_workspace"
        self.was_held_lifted = held_lifted or (self.was_held_lifted and self.not_held_steps < DROP_GRACE_STEPS)
        truncated = (not terminated) and self.steps >= self.max_steps
        info["is_success"] = info["event"] == "success"
        return self._obs(), float(reward), terminated, truncated, info

    def render(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.m, 480, 640)
        self._renderer.update_scene(self.d, camera="demo_cam")
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


def scripted_action(env):
    """A hand-written policy that acts ONLY through the env's action interface. Used to prove the reduced
    Cartesian action space, success detection and reward can actually complete the task (a plumbing test)."""
    from controller import FINGERTIP_BELOW_TCP, HAND_ABOVE_TCP, geom_highest_z, geom_lowest_z

    d, m = env.d, env.m
    tcp = d.site_xpos[env.tcp]
    yaw = ik.tcp_yaw(d, env.tcp)
    item, iyaw, _ = env._item_state()
    top = geom_highest_z(m, d, env.item_geom)
    stage = getattr(env, "_stage", "approach")

    def toward(goal, goal_yaw, grip, rail=0.0):
        goal_rel = goal - np.array([0.0, env._rail_q(), 0.0])
        dp = np.clip((goal_rel - env.target_rel) / MAX_DPOS, -1, 1)
        dy = np.clip((ik.nearest_equivalent_yaw(goal_yaw, env.target_yaw) - env.target_yaw) / MAX_DYAW, -1, 1)
        return np.array([*dp, dy, grip, rail], dtype=np.float32), np.linalg.norm(goal - tcp) < 0.008

    grasp_yaw = iyaw + np.pi / 2
    if not hasattr(env, "_grasp_yaw"):
        env._grasp_yaw = ik.nearest_equivalent_yaw(grasp_yaw, yaw)
    center = d.geom_xpos[env.item_geom][2]
    gz = max(center, FINGERTIP_BELOW_TCP + 0.004, top + 0.006 - HAND_ABOVE_TCP)
    if stage == "approach":
        act, done = toward(np.array([item[0], item[1], top + 0.08]), env._grasp_yaw, -1)
        env._stage = "descend" if done else stage
    elif stage == "descend":
        act, done = toward(np.array([item[0], item[1], gz]), env._grasp_yaw, -1)
        if done:
            env._stage, env._t = "close", 0
    elif stage == "close":
        act, _ = toward(env._world(env.target_rel), env.target_yaw, 1)
        env._t += 1
        env._stage = "lift" if env._t > 10 else stage
    elif stage == "lift":  # up to a carry height that clears every locker, over the carriage
        act, done = toward(np.array([0.42, env._rail_q(), 0.45]), env._grasp_yaw, 1)
        env._stage = "traverse" if done else stage
    elif stage == "traverse":  # rail only; the arm target is held in the carriage frame
        station = scene.CABINETS[scene.cabinet_of(env.slot)][1]
        act, _ = toward(env._world(env.target_rel), env.target_yaw, 1,
                        rail=float(np.clip((station - env.rail_set) / RAIL_STEP, -1, 1)))
        if abs(env._rail_q() - station) < 0.002 and abs(env.rail_set - station) < 1e-9:
            env._stage = "carry"
    elif stage == "carry":
        act, done = toward(np.array([env.slot_center[0], env.slot_center[1], 0.30]), np.pi / 2, 1)
        env._stage = "insert" if done else stage
    elif stage == "insert":
        below = tcp[2] - geom_lowest_z(m, d, env.item_geom)
        z = max(env.slot_center[2] - env.slot_half[2] + 0.01 + below, 0.108 + 0.012 - HAND_ABOVE_TCP)
        act, done = toward(np.array([env.slot_center[0], env.slot_center[1], z]), np.pi / 2, 1)
        env._stage = "release" if done else stage
    else:
        act, _ = toward(env._world(env.target_rel), env.target_yaw, -1)
    return act


def reset_scripted(env):
    for attr in ("_stage", "_grasp_yaw", "_t"):
        if hasattr(env, attr):
            delattr(env, attr)
