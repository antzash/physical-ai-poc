"""One complete evidence intake episode, end to end.

randomise -> SUBMITTED / REGISTERED + slot allocation -> pick-and-place -> PICKED / PLACED -> settle -> VERIFIED,
with FAILED (phase + reason) logged at the point anything goes wrong.

Simplification: only one item is physically in the scene per episode, so the cabinet is physically empty at the
start of every episode. Slot occupancy is tracked logically across episodes by the allocator so all four slots are
exercised; when all four are logically full the allocator is cleared (modelling the cabinet being emptied).
"""

import argparse
from dataclasses import dataclass, asdict

import mujoco
import numpy as np

import controller as ctl
import scene
from logger import ROBOT_ACTOR, SYSTEM_ACTOR, CustodyLog, officer
from perception import NoiseSpec, draw_error
from randomise import Randomiser
from slots import CabinetFull, SlotAllocator

SETTLE_BEFORE = 0.3  # s of physics after placing the item, before the robot starts
SETTLE_AFTER = 1.0  # s after the cycle ends before verification
REST_LIN_VEL = 0.01  # m/s
REST_ANG_VEL = 0.2  # rad/s
MAX_CYCLE_TIME = 40.0  # s of simulated time
BADGES = ["4471", "2208", "3915", "5102"]


@dataclass
class EpisodeResult:
    episode: int
    seed: int
    item_id: str
    case_id: str
    object_class: str
    slot: str
    success: bool
    failure_phase: str | None
    failure_reason: str | None
    grasped: bool
    slipped: bool
    cycle_time: float
    placement_error: float | None
    params: dict
    perception: dict  # noise spec and the episode's drawn (cached) error, for exact replay
    final_in_slot: bool  # ground truth after settling, regardless of verdict (a failed episode can land in-slot)

    def to_dict(self):
        return asdict(self)


def in_volume(point, center, half):
    return bool(np.all(np.abs(np.asarray(point) - center) <= half))


class IntakeStation:
    """Holds the model, log, allocator and randomiser that persist across episodes."""

    def __init__(self, log_path, echo=True, model=None, data=None):
        if model is None:
            model, data = scene.load()
        self.m, self.d = model, data
        self.log = CustodyLog(log_path, echo=echo)
        self.alloc = SlotAllocator()
        self.randomiser = Randomiser(model)
        self.episode_index = 0

    def run_episode(self, seed, object_class=None, frame_cb=None, slot=None, noise=NoiseSpec(), size_mult=1.0,
                    mass_mult=1.0):
        """Run one intake. frame_cb(station, controller_or_None) is called after every physics step.

        `slot` forces the target slot (for exact replay of an evaluation episode: same seed + same slot).
        `noise` perturbs only the controller's observation of the item (perception.py); all judging stays on
        ground truth. `size_mult` / `mass_mult` scale the randomisation envelope at runtime (OOD sweeps).
        """
        m, d = self.m, self.d
        ep = self.episode_index
        self.episode_index += 1
        rng = np.random.default_rng([seed, 1])  # separate stream for IDs so they never perturb physics draws
        item_id = f"EV-2026-{seed:06d}"
        case_id = f"CASE-26-{rng.integers(1000, 9999):04d}"
        badge = BADGES[rng.integers(len(BADGES))]

        scene.reset_home(m, d)
        params = self.randomiser.apply(d, seed, object_class, size_mult=size_mult, mass_mult=mass_mult)
        # Separate stream so the perception error never perturbs the physics draws: at zero noise every episode is
        # bit-identical to Phase 0. Drawn once here and cached for the whole attempt.
        perception_error = draw_error(noise, np.random.default_rng([seed, 2]))
        self.perception_error = perception_error  # exposed for display (record.py); the controller gets it below
        cls = params.object_class
        item_body = scene.body_id(m, scene.item_body(cls))
        item_geom = scene.geom_id(m, scene.item_geom(cls))

        for _ in range(int(SETTLE_BEFORE / m.opt.timestep)):
            mujoco.mj_step(m, d)
            if frame_cb:
                frame_cb(self, None)

        size_txt = "x".join(f"{2 * v * 100:.1f}" for v in params.size)
        self.log.append(officer(badge), "SUBMITTED", item_id, case_id, cls,
                        detail=f"presented at intake counter; {params.mass * 1000:.0f} g")
        if slot is None:
            try:
                slot = self.alloc.allocate(item_id, cls)
            except CabinetFull:
                self.alloc.reset()
                slot = self.alloc.allocate(item_id, cls)
        self.log.append(SYSTEM_ACTOR, "REGISTERED", item_id, case_id, cls, slot,
                        detail=f"allocated {slot}; seed {seed}; size(cm) {size_txt}")

        state = {"placed": False, "fail_logged": False, "release_fail": None}
        center, half = scene.slot_volume(m, d, slot)

        def fail(phase, reason):
            state["fail_logged"] = True
            self.log.append(ROBOT_ACTOR if phase not in ("VERIFY",) else SYSTEM_ACTOR, "FAILED", item_id, case_id,
                            cls, slot, detail=f"{phase}: {reason}")

        def on_event(ev):
            if ev.kind == "grasp_verified":
                h = d.xpos[item_body][2] - params.size[-1]
                self.log.append(ROBOT_ACTOR, "PICKED", item_id, case_id, cls, slot,
                                detail=f"grasp verified: two-finger contact, lifted {h * 100:.1f} cm")
            elif ev.kind == "released":
                p = d.xpos[item_body]
                if in_volume(p[:2], center[:2], half[:2]):
                    state["placed"] = True
                    self.log.append(ROBOT_ACTOR, "PLACED", item_id, case_id, cls, slot,
                                    detail=f"released over {slot}")
                else:
                    state["release_fail"] = "item released outside the slot footprint"
                    fail("RELEASE", state["release_fail"])
            elif ev.kind == "failed":
                fail(ev.phase, ev.detail)

        c = ctl.PickPlaceController(m, d, cls, slot, on_event=on_event, perception_error=perception_error)
        t0 = d.time
        while not c.done:
            if d.time - t0 > MAX_CYCLE_TIME:
                c._fail(f"cycle exceeded {MAX_CYCLE_TIME:.0f}s")
                break
            c.update()
            mujoco.mj_step(m, d)
            if frame_cb:
                frame_cb(self, c)
        r = c.result

        # Keep holding the final arm command while the item settles.
        for _ in range(int(SETTLE_AFTER / m.opt.timestep)):
            mujoco.mj_step(m, d)
            if frame_cb:
                frame_cb(self, c)

        pos = d.xpos[item_body].copy()
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, item_body, vel, 0)
        at_rest = np.linalg.norm(vel[3:]) < REST_LIN_VEL and np.linalg.norm(vel[:3]) < REST_ANG_VEL
        inside = in_volume(pos, center, half)
        placement_error = float(np.linalg.norm(pos[:2] - center[:2]))

        success = False
        failure_phase, failure_reason = r.failure_phase, (None if r.success else r.reason)
        if state["release_fail"]:
            failure_phase, failure_reason = "RELEASE", state["release_fail"]
        elif r.success and state["placed"]:
            if inside and at_rest:
                success = True
                self.log.append(SYSTEM_ACTOR, "VERIFIED", item_id, case_id, cls, slot,
                                detail=f"at rest inside {slot}; offset {placement_error * 1000:.0f} mm from centre")
            else:
                why = "not inside slot volume" if not inside else "not at rest"
                failure_phase, failure_reason = "VERIFY", f"post-settle check failed: {why}"
                fail("VERIFY", failure_reason)
        elif not state["fail_logged"]:
            failure_phase = failure_phase or "UNKNOWN"
            failure_reason = failure_reason or "cycle ended without placement"
            fail(failure_phase, failure_reason)

        if not success:
            self.alloc.release(slot)  # the slot was never filled
        return EpisodeResult(ep, int(seed), item_id, case_id, cls, slot, success, None if success else failure_phase,
                             None if success else failure_reason, r.grasped, r.slipped, round(r.duration, 3),
                             round(placement_error, 4) if success else None, params.to_dict(),
                             {"noise": {"pos_sigma_mm": noise.pos_sigma_m * 1000,
                                        "yaw_sigma_deg": float(np.degrees(noise.yaw_sigma_rad)),
                                        "size_sigma_pct": noise.size_sigma_frac * 100},
                              "error": perception_error.to_dict(), "size_mult": size_mult, "mass_mult": mass_mult},
                             bool(inside and at_rest))


def main():
    parser = argparse.ArgumentParser(description="Run consecutive randomised intake episodes")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0, help="episode i uses seed + i")
    parser.add_argument("--log", default=str(scene.OUT_DIR / "custody_log.jsonl"))
    parser.add_argument("--render", action="store_true", help="save start/end frames to out/m4_*.png")
    parser.add_argument("--slot", choices=scene.SLOT_NAMES, help="force the target slot (exact replay)")
    args = parser.parse_args()

    station = IntakeStation(args.log)
    renderer = mujoco.Renderer(station.m, 720, 1280) if args.render else None
    results = []
    for i in range(args.episodes):
        seed = args.seed + i
        print(f"\n=== episode {i}  seed {seed} ===")
        snaps = {}

        def cb(st, c):
            if renderer is not None and c is not None and "start" not in snaps:
                snaps["start"] = scene.render(st.m, st.d, renderer=renderer)

        res = station.run_episode(seed, frame_cb=cb, slot=args.slot)
        if renderer is not None:
            scene.save_png(snaps["start"], scene.OUT_DIR / f"m4_ep{i:02d}_start.png")
            scene.save_png(scene.render(station.m, station.d, renderer=renderer),
                           scene.OUT_DIR / f"m4_ep{i:02d}_end.png")
        results.append(res)
        err = f"{res.placement_error * 1000:.0f} mm" if res.placement_error is not None else "-"
        print(f"--> {'SUCCESS' if res.success else 'FAIL'} class={res.object_class} slot={res.slot} "
              f"phase={res.failure_phase} t={res.cycle_time:.1f}s err={err} "
              f"mass={res.params['mass']} mu={res.params['friction']} reason={res.failure_reason}")

    n_ok = sum(r.success for r in results)
    print(f"\n{n_ok}/{len(results)} episodes succeeded; log chain verify() -> {station.log.verify()}")


if __name__ == "__main__":
    main()
