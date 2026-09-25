"""The evidence intake state machine.

IDLE -> SCAN -> ROUTE -> APPROACH -> DESCEND -> CLOSE -> LIFT -> TRAVERSE -> TRANSIT -> VERIFY_SCAN -> INSERT
     -> RELEASE -> RETREAT -> HOME (arm home, then the rail back to the intake station)

FAIL CLOSED. The destination comes only from the barcode the robot READS (scanner.Cameras) through routing.route().
The controller never receives, and must never reach, the simulator's true item identity: the evaluator uses that to
detect misfiles, so a controller that could see it would make the zero-misfile result meaningless. (item_cls below is
simulator plumbing to resolve which body is on the counter; it must not drive any decision.) Unreadable labels,
unknown IDs, full cabinets and verify mismatches end in REFUSED, never a guess or a fallback:
  SCAN        intake scanner; on no-read, ONE re-scan from a second pose (wrist cameras over the bag); else refuse.
  ROUTE       case lookup -> cabinet and slot; else refuse.
  VERIFY_SCAN wrist re-scan before release: mismatch -> refuse; no-read -> one retry from 3 cm higher -> refuse.
              A refused item in the gripper is carried back and put down on the counter (RETURN_ITEM).

Every phase has an entry action, a completion condition measured on the simulated state (never elapsed time
alone), and a timeout that ends the cycle as a failure attributed to that phase. Grasping is done purely through
contact physics between the finger pads and the item; there is no weld or kinematic attachment.
"""

from dataclasses import dataclass, field

import mujoco
import numpy as np

import ik
import routing
import scene
from perception import ZERO_ERROR, observe

PHASES = ["IDLE", "SCAN", "ROUTE", "APPROACH", "DESCEND", "CLOSE", "LIFT", "TRAVERSE", "TRANSIT", "VERIFY_SCAN",
          "INSERT", "RELEASE", "RETREAT", "RETURN_ITEM", "HOME"]

# Gripper geometry measured from the Menagerie meshes, in the TCP frame (see NOTES.md, M2).
FINGERTIP_BELOW_TCP = 0.0089
HAND_ABOVE_TCP = 0.0374

TIP_CLEARANCE = 0.004  # fingertips stay this far above the counter at the grasp pose
HAND_CLEARANCE = 0.006  # hand body stays this far above a tall item's top at the grasp pose
PREGRASP_CLEARANCE = 0.10  # TCP height above the item's top for APPROACH
SAFE_Z = 0.30  # transit height for the TCP: clears the bin walls with any item hanging below
# TRAVERSE: the arm first lifts the bag to a carry pose over the carriage (clears every locker, backboards are 0.30 m),
# then the rail moves with the arm's joint targets frozen. Rail and arm never move at the same time.
CARRY_X = 0.42  # m in front of the carriage
CARRY_Z = 0.45
RAIL_VMAX = 0.40  # m/s; MuJoCo position actuators have no velocity limit, so the setpoint ramp enforces it
RAIL_TOL = 0.001
RAIL_SETTLE_VEL = 0.002
RESCAN_TCP_ABOVE_TOP = 0.005  # second-pose scan: TCP just above the bag, wrist cameras ~0.11 m over the label
VERIFY_RETRY_RAISE = 0.03
PUTDOWN_CLEARANCE = 0.004  # RETURN_ITEM: bag bottom this far above the counter when opening the gripper
BIN_WALL_TOP_Z = 0.108
WALL_CLEARANCE = 0.012
RELEASE_DROP = 0.010  # item bottom this far above the bin floor at release

CART_SPEED = 0.25  # m/s
SLOW_SPEED = 0.06  # m/s for DESCEND / INSERT
YAW_SPEED = 1.5  # rad/s
CONTROL_EVERY = 4  # physics steps per control update

POS_TOL = 0.006
LIFT_HEIGHT = 0.12
GRASP_HEIGHT_GAIN = 0.04  # item must rise at least this much above its resting height for a verified grasp
PAD_FORCE_THRESHOLD = 2.0  # N per finger to count CLOSE as complete on force
FINGER_SETTLE_VEL = 0.002  # m/s
EMPTY_GRIPPER_Q = 0.003  # finger position (m) below which the gripper has closed on nothing
DROP_GRACE_UPDATES = 25  # consecutive control updates (~0.2 s) without two-finger contact = a drop
GRASP_VERIFY_SAMPLES = 12  # control updates (~0.1 s) over which a grasp is verified at the top of LIFT
GRASP_CONTACT_FRACTION = 0.6  # each finger must be in contact in at least this fraction of those samples


@dataclass
class ControllerEvent:
    kind: str  # phase, scanned, scan_retry, routed, refused, grasp_verified, traversed, verified, released, failed, done
    phase: str
    time: float
    detail: str = ""
    data: dict = field(default_factory=dict)


@dataclass
class CycleResult:
    success: bool  # filed: released into the routed slot after a matching verify scan and a clean return
    failure_phase: str | None
    reason: str
    grasped: bool
    slipped: bool
    duration: float
    events: list = field(default_factory=list)
    outcome: str = "filed"  # "filed", "refused" (fail-closed decision) or "failed" (execution failure)
    refusal_cause: str | None = None  # no_decode, no_case_match, cabinet_full, verify_mismatch, verify_no_read
    scan: dict = field(default_factory=dict)  # intake/verify scan details (decoded strings, cameras, times)


def _smoothstep(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3 - 2 * u)


def geom_lowest_z(model, data, g):
    """Lowest world z of a primitive geom (box, ellipsoid, cylinder) in its current pose."""
    R = data.geom_xmat[g].reshape(3, 3)
    s = model.geom_size[g]
    zc = data.geom_xpos[g][2]
    t = model.geom_type[g]
    if t == mujoco.mjtGeom.mjGEOM_BOX:
        return zc - np.abs(R[2, :]) @ s
    if t == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        return zc - np.sqrt(np.sum((R[2, :] * s) ** 2))
    if t == mujoco.mjtGeom.mjGEOM_CYLINDER:
        cz = abs(R[2, 2])
        return zc - (cz * s[1] + s[0] * np.sqrt(max(0.0, 1 - cz * cz)))
    raise ValueError(f"unsupported geom type {t}")


def geom_highest_z(model, data, g):
    return 2 * data.geom_xpos[g][2] - geom_lowest_z(model, data, g)


def finger_contact_forces(model, data, item_geom):
    """Total normal force (N) between the item geom and each finger body: (left, right)."""
    left, right = scene.body_id(model, "left_finger"), scene.body_id(model, "right_finger")
    forces = np.zeros(2)
    f6 = np.zeros(6)
    for i in range(data.ncon):
        c = data.contact[i]
        if item_geom not in (c.geom1, c.geom2):
            continue
        other = c.geom2 if c.geom1 == item_geom else c.geom1
        body = model.geom_bodyid[other]
        if body in (left, right):
            mujoco.mj_contactForce(model, data, i, f6)
            forces[0 if body == left else 1] += abs(f6[0])
    return forces


def grasp_ok(model, data, item_geom, min_height):
    """Instantaneous grasp predicate: item touching BOTH finger bodies with non-trivial force AND above min_height."""
    f = finger_contact_forces(model, data, item_geom)
    return bool(f[0] > 0.1 and f[1] > 0.1 and data.geom_xpos[item_geom][2] > min_height)


def grasp_sample(model, data, item_geom, min_height):
    f = finger_contact_forces(model, data, item_geom)
    return (f[0] > 0.1, f[1] > 0.1, data.geom_xpos[item_geom][2] > min_height)


def grasp_verified(samples):
    """Windowed grasp verdict scored by the evaluation harness (`CycleResult.grasped`).

    The item must stay above the height threshold at EVERY sample, and each finger must be in contact in most
    samples. A single-sample check is not robust: MuJoCo's box contacts occasionally drop one pad's contact from the
    active set for a single step while the item is plainly held (see NOTES.md, M5). Do not relax the height term.
    """
    s = np.asarray(samples, dtype=bool)
    return bool(s[:, 2].all() and s[:, 0].mean() >= GRASP_CONTACT_FRACTION and s[:, 1].mean() >= GRASP_CONTACT_FRACTION)


class PickPlaceController:
    def __init__(self, model, data, item_cls, bank, cameras, on_event=None, ik_solver=None,
                 perception_error=ZERO_ERROR):
        self.m, self.d = model, data
        self.item_cls = item_cls  # plumbing only (which body is on the counter); never used for a decision
        self.bank = bank  # routing.CabinetBank: logical slot occupancy
        self.cameras = cameras  # scanner.Cameras: the ONLY source of identity
        # The controller's ONLY view of the item: ground truth plus this episode's fixed perception error.
        self.perception_error = perception_error
        self.slot = None  # decided in ROUTE from the decoded barcode, e.g. "CAB-B/slot_2"
        self.station = None
        self.scan_id = None  # what the intake scan decoded
        self.route = None
        self.refusal = None
        self.scan_info = {}
        self.rail_act = scene.actuator_id(model, scene.RAIL_ACTUATOR)
        self.rail_qadr = model.jnt_qposadr[scene.joint_id(model, scene.RAIL_JOINT)]
        self.rail_dadr = model.jnt_dofadr[scene.joint_id(model, scene.RAIL_JOINT)]
        self.on_event = on_event
        self.ik = ik_solver or ik.IK(model)
        self.tcp = scene.site_id(model, "tcp")
        self.item_body = scene.body_id(model, scene.item_body(item_cls))
        self.item_geom = scene.geom_id(model, scene.item_geom(item_cls))
        self.arm_act = np.array([scene.actuator_id(model, a) for a in scene.ARM_ACTUATORS])
        self.grip_act = scene.actuator_id(model, scene.GRIPPER_ACTUATOR)
        self.finger_qadr = np.array([model.jnt_qposadr[scene.joint_id(model, j)] for j in scene.FINGER_JOINTS])
        self.finger_dadr = np.array([model.jnt_dofadr[scene.joint_id(model, j)] for j in scene.FINGER_JOINTS])
        self.arm_qadr = scene.arm_qpos_adr(model)
        self.home_q = model.key_qpos[scene.key_id(model, "home")][self.arm_qadr].copy()

        self.events: list[ControllerEvent] = []
        self.phase = None
        self.phase_start = 0.0
        self.steps = 0
        self.q_cmd = data.qpos[self.arm_qadr].copy()
        self.segments = []
        self.seg_start = 0.0
        self.timeout = 0.0
        self.grasped = False
        self.slipped = False
        self.no_contact_steps = 0
        self.rest_z = data.xpos[self.item_body][2]  # ground truth: used only by the grasp verifier (evaluation)
        self.result: CycleResult | None = None
        self.start_time = data.time
        self._enter("IDLE")

    # ---- helpers -------------------------------------------------------------------------------------------------

    def _emit(self, kind, detail="", **data):
        ev = ControllerEvent(kind, self.phase, self.d.time, detail, data)
        self.events.append(ev)
        if self.on_event:
            self.on_event(ev)

    def _tcp_pose(self):
        return self.d.site_xpos[self.tcp].copy(), ik.tcp_yaw(self.d, self.tcp)

    def _plan(self, waypoints, speed):
        """Queue Cartesian segments through waypoints [(pos, yaw), ...] starting from the current commanded pose."""
        pos, yaw = self._cmd_pose
        self.segments = []
        total = 0.0
        for wp_pos, wp_yaw in waypoints:
            wp_pos = np.asarray(wp_pos, float)
            wp_yaw = ik.nearest_equivalent_yaw(wp_yaw, yaw)
            dur = max(np.linalg.norm(wp_pos - pos) / speed, abs(wp_yaw - yaw) / YAW_SPEED, 0.2)
            self.segments.append((pos, yaw, wp_pos, wp_yaw, total, dur))
            total += dur
            pos, yaw = wp_pos, wp_yaw
        self.seg_start = self.d.time
        self.goal = (pos, yaw)
        self.traj_duration = total
        return total

    def _track(self):
        """Advance along the queued Cartesian path and command the IK solution to the arm servos."""
        t = self.d.time - self.seg_start
        target = self.segments[-1]
        for seg in self.segments:
            if t < seg[4] + seg[5]:
                target = seg
                break
        p0, y0, p1, y1, t0, dur = target
        u = _smoothstep((t - t0) / dur)
        pos = p0 + u * (p1 - p0)
        yaw = y0 + u * (y1 - y0)
        self._cmd_pose = (pos, yaw)
        res = self.ik.solve(self.d, pos, ik.down_quat(yaw), q_init=self.q_cmd)
        self.q_cmd = res.q
        self.d.ctrl[self.arm_act] = self.q_cmd

    def _at_goal(self):
        pos, yaw = self._tcp_pose()
        yaw_err = abs(ik.nearest_equivalent_yaw(yaw, self.goal[1]) - self.goal[1])
        return (self.d.time - self.seg_start >= self.traj_duration
                and np.linalg.norm(pos - self.goal[0]) < POS_TOL and yaw_err < 0.05)

    def _finger_q(self):
        return self.d.qpos[self.finger_qadr]

    def _observe(self):
        return observe(self.m, self.d, self.item_body, self.item_geom, self.perception_error, "box")

    def _enter(self, phase):
        self.phase = phase
        self.phase_start = self.d.time
        self._emit("phase")
        getattr(self, f"_enter_{phase.lower()}")()

    def _fail(self, reason):
        self._emit("failed", reason)
        if self.route is not None and self.route.ok:
            self.bank.release(self.slot)  # the reserved slot was never filled
        self.result = CycleResult(False, self.phase, reason, self.grasped, self.slipped,
                                  self.d.time - self.start_time, self.events, "failed",
                                  self.refusal[0] if self.refusal else None, self.scan_info)

    def _refuse(self, cause, reason):
        """Fail closed: decline to file, log REFUSED, and leave the item for a human handler."""
        self.refusal = (cause, reason, self.phase)
        self._emit("refused", reason, cause=cause)
        if self.route is not None and self.route.ok:
            self.bank.release(self.slot)
        self._enter("RETURN_ITEM" if self.grasped else "HOME")

    def _check_hold(self):
        """Detect the item slipping out of the grasp while carrying it."""
        f = finger_contact_forces(self.m, self.d, self.item_geom)
        if f[0] > 0.1 and f[1] > 0.1:
            self.no_contact_steps = 0
            return True
        self.no_contact_steps += 1
        if self.no_contact_steps > DROP_GRACE_UPDATES:
            self.slipped = True
            self._fail(f"item dropped during {self.phase} (lost two-finger contact)")
            return False
        return True

    # ---- rail (staging axis, never in the IK) ----------------------------------------------------------------------

    def _rail_q(self):
        return float(self.d.qpos[self.rail_qadr])

    def _start_rail(self, target):
        """Begin a speed-capped move of the rail setpoint. Returns the ramp duration."""
        self.rail_from, self.rail_to = self._rail_q(), float(target)
        # smoothstep peaks at 1.5x the mean speed, so this duration caps the setpoint speed at RAIL_VMAX
        self.rail_dur = max(0.3, 1.5 * abs(self.rail_to - self.rail_from) / RAIL_VMAX)
        self.rail_t0 = self.d.time
        return self.rail_dur

    def _step_rail(self):
        """Advance the rail setpoint; True once the carriage is at the target and settled (a real condition)."""
        u = _smoothstep((self.d.time - self.rail_t0) / self.rail_dur)
        self.d.ctrl[self.rail_act] = self.rail_from + u * (self.rail_to - self.rail_from)
        return (u >= 1.0 and abs(self._rail_q() - self.rail_to) < RAIL_TOL
                and abs(self.d.qvel[self.rail_dadr]) < RAIL_SETTLE_VEL)

    # ---- phases --------------------------------------------------------------------------------------------------

    def _start_joint_home(self):
        self.d.ctrl[self.grip_act] = scene.GRIPPER_OPEN
        self.joint_start = self.d.qpos[self.arm_qadr].copy()
        self.joint_dur = max(0.3, np.max(np.abs(self.home_q - self.joint_start)) / 1.0)
        self.joint_t0 = self.d.time
        self.timeout = self.joint_dur + 2.0

    def _step_joint_home(self):
        u = _smoothstep((self.d.time - self.joint_t0) / self.joint_dur)
        self.q_cmd = self.joint_start + u * (self.home_q - self.joint_start)
        self.d.ctrl[self.arm_act] = self.q_cmd
        return u >= 1.0 and np.max(np.abs(self.d.qpos[self.arm_qadr] - self.home_q)) < 0.02

    def _enter_idle(self):
        self._start_joint_home()

    def _update_idle(self):
        if self._step_joint_home():  # arm parked at home, clear of the scanner's view
            self._cmd_pose = self._tcp_pose()
            self._enter("SCAN")

    def _enter_scan(self):
        self.scan_stage = "intake"
        self.timeout = 30.0

    def _update_scan(self):
        if self.scan_stage == "intake":
            r = self.cameras.scan_intake(self.d)
            self.scan_info["intake"] = self._scan_record(r)
            if r.ok:
                return self._scanned(r)
            self._emit("scan_retry", f"no read from {r.extra['camera']}; re-scanning from a second pose")
            obs = self._observe()
            yaw = ik.nearest_equivalent_yaw(obs.yaw + np.pi / 2, self._cmd_pose[1])
            top = obs.position[2] + obs.vertical_half_extent
            dur = self._plan([((obs.position[0], obs.position[1], top + PREGRASP_CLEARANCE), yaw),
                              ((obs.position[0], obs.position[1], top + RESCAN_TCP_ABOVE_TOP), yaw)], CART_SPEED)
            self.timeout = self.d.time - self.phase_start + dur + 3.0
            self.scan_stage = "move"
            return
        if self.scan_stage == "move":
            self._track()
            if self._at_goal():
                r = self.cameras.scan_wrist(self.d)
                self.scan_info["rescan"] = self._scan_record(r)
                if r.ok:
                    return self._scanned(r)
                self._refuse("no_decode", "no barcode read from the intake scanner or the wrist re-scan")

    def _scan_record(self, r):
        return {"camera": r.extra.get("camera"), "decoded": r.text, "sim_time": round(self.d.time, 3),
                "method": r.method, "attempts": r.attempts}

    def _scanned(self, r):
        self.scan_id = r.text
        self._emit("scanned", f"{r.extra['camera']} decoded {r.text}", decoded=r.text, camera=r.extra["camera"])
        self._enter("ROUTE")

    def _enter_route(self):
        self.timeout = 1.0

    def _update_route(self):
        self.route = routing.route(self.scan_id, self.bank)  # the DECODED ID; nothing else
        if not self.route.ok:
            return self._refuse(self.route.refusal, self.route.reason)
        self.slot = self.route.location
        self.station = scene.CABINETS[self.route.cabinet][1]
        self._emit("routed", f"{self.scan_id} -> {self.route.record.case_id} -> {self.route.record.category} -> "
                             f"{self.slot}", location=self.slot, record=self.route.record)
        self._cmd_pose = self._tcp_pose()
        self._enter("APPROACH")

    def _enter_approach(self):
        obs = self._observe()
        top = obs.position[2] + obs.vertical_half_extent
        item_pos = obs.position
        self.grasp_yaw = obs.yaw + np.pi / 2  # close across the bag's short axis
        self.grasp_yaw = ik.nearest_equivalent_yaw(self.grasp_yaw, self._cmd_pose[1])
        dur = self._plan([((item_pos[0], item_pos[1], top + PREGRASP_CLEARANCE), self.grasp_yaw)], CART_SPEED)
        self.timeout = dur + 2.0

    def _update_approach(self):
        self._track()
        if self._at_goal():
            self._enter("DESCEND")

    def grasp_height(self, obs):
        """TCP height for the grasp, per object class, limited by what the gripper geometry allows."""
        center = obs.position[2]
        top = center + obs.vertical_half_extent
        # Grasp at the bag's believed geometric centre. An offset centre of mass is not compensated for: the
        # controller does not know where the contents are, so a heavy end tips the bag in the grasp.
        desired = center
        floor_limit = scene.COUNTER_TOP_Z + FINGERTIP_BELOW_TCP + TIP_CLEARANCE
        hand_limit = top + HAND_CLEARANCE - HAND_ABOVE_TCP  # hand body must stay above the item's top
        return max(desired, floor_limit, hand_limit)

    def _enter_descend(self):
        self.grasp_obs = self._observe()
        item_pos = self.grasp_obs.position
        self.grasp_z = self.grasp_height(self.grasp_obs)
        dur = self._plan([((item_pos[0], item_pos[1], self.grasp_z), self.grasp_yaw)], SLOW_SPEED)
        self.timeout = dur + 2.0

    def _update_descend(self):
        self._track()
        if self._at_goal():
            self._enter("CLOSE")

    def _enter_close(self):
        self.d.ctrl[self.grip_act] = scene.GRIPPER_CLOSED
        self.timeout = 2.0

    def _update_close(self):
        self._track()
        elapsed = self.d.time - self.phase_start
        forces = finger_contact_forces(self.m, self.d, self.item_geom)
        settled = np.max(np.abs(self.d.qvel[self.finger_dadr])) < FINGER_SETTLE_VEL
        if elapsed > 0.15 and (settled or np.all(forces > PAD_FORCE_THRESHOLD)):
            if np.max(self._finger_q()) < EMPTY_GRIPPER_Q:
                self._fail("gripper closed on nothing")
                return
            self._emit("closed", f"finger q={self._finger_q().round(4).tolist()} pad N={forces.round(2).tolist()}")
            self._enter("LIFT")

    def _enter_lift(self):
        self.verify_samples = None
        pos, yaw = self._cmd_pose
        dur = self._plan([((pos[0], pos[1], pos[2] + LIFT_HEIGHT), yaw)], SLOW_SPEED * 2)
        self.timeout = dur + 2.0

    def _update_lift(self):
        self._track()
        if self.verify_samples is None:
            if self._at_goal():
                self.verify_samples = []
            return
        self.verify_samples.append(grasp_sample(self.m, self.d, self.item_geom, self.rest_z + GRASP_HEIGHT_GAIN))
        if len(self.verify_samples) < GRASP_VERIFY_SAMPLES:
            return
        if grasp_verified(self.verify_samples):
            self.grasped = True
            self._emit("grasp_verified")
            self._enter("TRAVERSE")
        else:
            s = np.asarray(self.verify_samples, dtype=bool)
            self._fail(f"grasp not verified after lift (contact L {s[:, 0].mean():.0%}, R {s[:, 1].mean():.0%}, "
                       f"above height {s[:, 2].mean():.0%} of samples)")

    def _enter_traverse(self):
        pos, yaw = self._cmd_pose
        self.traverse_stage = "raise"
        dur = self._plan([((pos[0], pos[1], CARRY_Z), yaw), ((CARRY_X, self._rail_q(), CARRY_Z), yaw)], CART_SPEED)
        rail_dur = 1.5 * abs(self.station - self._rail_q()) / RAIL_VMAX
        self.timeout = dur + rail_dur + 4.0

    def _update_traverse(self):
        if not self._check_hold():
            return
        if self.traverse_stage == "raise":
            self._track()
            if self._at_goal():
                self.traverse_stage = "rail"
                self._start_rail(self.station)
            return
        self.d.ctrl[self.arm_act] = self.q_cmd  # arm held: joint targets frozen while the carriage moves
        if self._step_rail():
            shift = self.rail_to - self.rail_from
            pos, yaw = self._cmd_pose
            self._cmd_pose = (pos + np.array([0.0, shift, 0.0]), yaw)
            self._emit("traversed", f"rail at {self._rail_q():+.3f} m ({scene.cabinet_of(self.slot)} station)")
            self._enter("TRANSIT")

    def _enter_transit(self):
        center, _ = scene.slot_volume(self.m, self.d, self.slot)
        pos, yaw = self._cmd_pose
        # Bins run along world x; align the item's long axis with x, i.e. close the fingers along world y.
        self.slot_yaw = ik.nearest_equivalent_yaw(np.pi / 2, yaw)
        dur = self._plan([
            ((pos[0], pos[1], SAFE_Z), yaw),
            ((center[0], center[1], SAFE_Z), self.slot_yaw),
        ], CART_SPEED)
        self.timeout = dur + 3.0

    def _update_transit(self):
        self._track()
        if not self._check_hold():
            return
        if self._at_goal():
            self._enter("VERIFY_SCAN")

    def _enter_verify_scan(self):
        self.verify_stage = "scan"
        self.timeout = 8.0

    def _update_verify_scan(self):
        if not self._check_hold():
            return
        if self.verify_stage == "scan":
            r = self.cameras.scan_wrist(self.d)
            self.scan_info["verify"] = self._scan_record(r)
            if r.ok:
                return self._verified(r)
            pos, yaw = self._cmd_pose
            self._plan([((pos[0], pos[1], pos[2] + VERIFY_RETRY_RAISE), yaw)], SLOW_SPEED)
            self.verify_stage = "raise"
            return
        self._track()
        if self._at_goal():
            r = self.cameras.scan_wrist(self.d)
            self.scan_info["verify_retry"] = self._scan_record(r)
            if r.ok:
                return self._verified(r)
            self._refuse("verify_no_read", "verify scan: no barcode read from the wrist cameras (two attempts)")

    def _verified(self, r):
        match = r.text == self.scan_id
        self.scan_info["verify_match"] = match
        self._emit("verified", f"{r.extra['camera']} decoded {r.text}: {'MATCH' if match else 'MISMATCH'}",
                   decoded=r.text, camera=r.extra["camera"], match=match)
        if not match:
            return self._refuse("verify_mismatch", f"verify scan read {r.text}, expected {self.scan_id}")
        self._enter("INSERT")

    def release_height(self):
        """TCP height at release: item just above the bin floor, but the hand body kept above the bin walls."""
        center, half = scene.slot_volume(self.m, self.d, self.slot)
        floor_z = center[2] - half[2]  # cabinet geometry is known, not perceived
        # Item-derived part comes from perception: the believed item bottom relative to the commanded grasp height.
        believed_bottom = self.grasp_obs.position[2] - self.grasp_obs.vertical_half_extent
        item_below_tcp = self.grasp_z - believed_bottom
        by_item = floor_z + RELEASE_DROP + item_below_tcp
        by_walls = BIN_WALL_TOP_Z + WALL_CLEARANCE - HAND_ABOVE_TCP
        return max(by_item, by_walls)

    def _enter_insert(self):
        center, _ = scene.slot_volume(self.m, self.d, self.slot)
        dur = self._plan([((center[0], center[1], self.release_height()), self.slot_yaw)], SLOW_SPEED * 2)
        self.timeout = dur + 2.0

    def _update_insert(self):
        self._track()
        if not self._check_hold():
            return
        if self._at_goal():
            self._enter("RELEASE")

    def _enter_release(self):
        self.d.ctrl[self.grip_act] = scene.GRIPPER_OPEN
        self.timeout = 2.0

    def _update_release(self):
        self._track()
        forces = finger_contact_forces(self.m, self.d, self.item_geom)
        if np.min(self._finger_q()) > 0.035 and np.all(forces == 0):
            self._emit("released")
            self._enter("RETREAT")

    def _enter_retreat(self):
        pos, yaw = self._cmd_pose
        dur = self._plan([((pos[0], pos[1], SAFE_Z), yaw)], CART_SPEED)
        self.timeout = dur + 2.0

    def _update_retreat(self):
        self._track()
        if self._at_goal():
            self._enter("HOME")

    def _enter_return_item(self):
        """A refused item in the gripper goes back to the counter, where a handler will deal with it."""
        pos, yaw = self._cmd_pose
        self.item_stage = "raise"
        self._plan([((pos[0], pos[1], CARRY_Z), yaw), ((CARRY_X, self._rail_q(), CARRY_Z), yaw)], CART_SPEED)
        self.timeout = 40.0

    def _update_return_item(self):
        if self.item_stage in ("raise", "rail", "lower") and not self._check_hold():
            return
        if self.item_stage == "raise":
            self._track()
            if self._at_goal():
                self.item_stage = "rail"
                self._start_rail(scene.INTAKE_STATION)
        elif self.item_stage == "rail":
            self.d.ctrl[self.arm_act] = self.q_cmd
            if self._step_rail():
                pos, yaw = self._cmd_pose
                self._cmd_pose = (pos + np.array([0.0, self.rail_to - self.rail_from, 0.0]), yaw)
                g = self.grasp_obs.position
                down = self.grasp_z + PUTDOWN_CLEARANCE
                self._plan([((g[0], g[1], down + PREGRASP_CLEARANCE), self.grasp_yaw), ((g[0], g[1], down), self.grasp_yaw)],
                           CART_SPEED)
                self.item_stage = "lower"
        elif self.item_stage == "lower":
            self._track()
            if self._at_goal():
                self.d.ctrl[self.grip_act] = scene.GRIPPER_OPEN
                self.item_stage = "open"
        elif self.item_stage == "open":
            self._track()
            if np.min(self._finger_q()) > 0.035 and np.all(finger_contact_forces(self.m, self.d, self.item_geom) == 0):
                pos, yaw = self._cmd_pose
                self._plan([((pos[0], pos[1], pos[2] + PREGRASP_CLEARANCE), yaw)], CART_SPEED)
                self.item_stage = "clear"
        else:
            self._track()
            if self._at_goal():
                self._emit("returned", "item left on the intake counter for a handler")
                self._enter("HOME")

    def _enter_home(self):
        self._start_joint_home()
        self.return_stage = "arm"
        self.timeout += 1.5 * abs(scene.INTAKE_STATION - self._rail_q()) / RAIL_VMAX + 2.0

    def _update_home(self):
        if self.return_stage == "arm":
            if self._step_joint_home():
                self.return_stage = "rail"
                self._start_rail(scene.INTAKE_STATION)
            return
        self.d.ctrl[self.arm_act] = self.q_cmd  # arm held at home while the carriage returns to the intake station
        if self._step_rail():
            self._emit("done")
            if self.refusal:
                cause, reason, phase = self.refusal
                self.result = CycleResult(False, phase, reason, self.grasped, self.slipped,
                                          self.d.time - self.start_time, self.events, "refused", cause, self.scan_info)
            else:
                self.result = CycleResult(True, None, "cycle complete", self.grasped, self.slipped,
                                          self.d.time - self.start_time, self.events, "filed", None, self.scan_info)

    # ---- public API ----------------------------------------------------------------------------------------------

    @property
    def done(self):
        return self.result is not None

    def update(self):
        """Call once per physics step, before mj_step."""
        if self.done:
            return
        if self.d.time - self.phase_start > self.timeout:
            self._fail(f"timeout in {self.phase} after {self.timeout:.1f}s")
            return
        if self.steps % CONTROL_EVERY == 0 or self.phase in ("CLOSE", "RELEASE"):
            getattr(self, f"_update_{self.phase.lower()}")()
        self.steps += 1

