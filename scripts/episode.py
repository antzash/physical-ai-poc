"""One complete evidence intake episode, end to end (Phase 1: scan -> route -> traverse -> verify -> file).

This module is both the WORLD and the EVALUATOR, and it is where the identity boundary lives:
  world:     picks the true item (a label from the pool), fills the bag according to that item's case record,
             applies the label, logs the officer's submission (as PENDING-SCAN: the ID is unknown until read).
  robot:     controller.PickPlaceController, given only cameras, the routing bank and the perception seam.
  evaluator: knows the true ID and judges from ground truth whether the item ended in the location that ID routes to.
             A misfile is an item that ends at rest in any slot other than that one. The true ID is NEVER passed to
             the controller; if it were, the zero-misfile result would mean nothing.

Simplification (as in Phase 0): one item is physically present per episode; slot occupancy is tracked logically by
routing.CabinetBank across episodes, and a full locker is emptied between episodes (transfer to long-term storage).
"""

import argparse
from dataclasses import dataclass, asdict

import mujoco
import numpy as np

import controller as ctl
import labels
import routing
import scene
from logger import ROBOT_ACTOR, SYSTEM_ACTOR, CustodyLog, officer
from perception import NoiseSpec, draw_error
from randomise import Randomiser
from scanner import Cameras

SETTLE_BEFORE = 0.3  # s of physics after placing the item, before the robot starts
SETTLE_AFTER = 1.0  # s after the cycle ends before verification
REST_LIN_VEL = 0.01  # m/s
REST_ANG_VEL = 0.2  # rad/s
MAX_CYCLE_TIME = 90.0  # s of simulated time (a refusal with a return-to-counter is the longest path)
CARRY_PHASES = ("LIFT", "TRAVERSE", "TRANSIT", "VERIFY_SCAN", "INSERT")
BADGES = ["4471", "2208", "3915", "5102"]
PENDING = "PENDING-SCAN"


@dataclass
class EpisodeResult:
    episode: int
    seed: int
    true_item_id: str  # WORLD TRUTH (evaluator only)
    decoded_id: str | None  # what the robot read at intake (None if unread)
    case_id: str | None
    object_class: str  # WORLD TRUTH: the content class in the bag
    category: str | None  # from the case record the robot looked up
    routed_location: str | None  # where the robot decided to file it
    correct_location: str | None  # evaluator: where the TRUE id routes, from the same occupancy
    final_location: str | None  # ground truth: the slot volume the item is at rest in after settling, if any
    outcome: str  # "filed", "refused", "failed"
    success: bool  # filed and verified at the routed location
    misfile: bool  # item at rest in a slot other than correct_location (the headline safety metric)
    refusal_cause: str | None
    failure_phase: str | None
    failure_reason: str | None
    grasped: bool
    slipped: bool
    cycle_time: float
    placement_error: float | None
    scan: dict
    bank_before: list  # occupied locations before this episode (for exact replay)
    params: dict
    perception: dict  # noise spec and the episode's drawn (cached) error, for exact replay
    final_in_slot: bool  # ground truth: item at rest inside the ROUTED slot, regardless of verdict
    max_carry_tilt_deg: float  # ground truth: largest bag tilt while carried

    @property
    def slot(self):
        return self.routed_location

    def to_dict(self):
        return asdict(self)


def in_volume(point, center, half):
    return bool(np.all(np.abs(np.asarray(point) - center) <= half))


class IntakeStation:
    """Holds the model, log, cabinet bank, cameras and randomiser that persist across episodes."""

    def __init__(self, log_path, echo=True, model=None, data=None):
        if model is None:
            model, data = scene.load()
        self.m, self.d = model, data
        self.log = CustodyLog(log_path, echo=echo)
        self.bank = routing.CabinetBank()
        self.randomiser = Randomiser(model)
        self.cameras = Cameras(model)
        self.episode_index = 0
        self.intake_number = 0

    # ---- world side ------------------------------------------------------------------------------------------------

    def _choose_item(self, seed, label):
        """WORLD: which item arrives, which label it carries, and what the officer sealed inside."""
        rng = np.random.default_rng([seed, 3])
        if label is None:
            idx = int(rng.integers(labels.POOL_SIZE))
            material, true_id = f"label_{idx:02d}", labels.POOL_IDS[idx]
        elif label == "damaged":
            material, true_id = "label_damaged", labels.DAMAGED_ID
        elif label == "unregistered":
            material, true_id = "label_unregistered", labels.UNREGISTERED_ID
        else:
            material, true_id = f"label_{int(label):02d}", labels.POOL_IDS[int(label)]
        record = routing.CASE_DB.get(true_id)
        content = record.content_class if record else scene.ITEM_CLASSES[int(rng.integers(len(scene.ITEM_CLASSES)))]
        return material, true_id, content

    def run_episode(self, seed, frame_cb=None, noise=NoiseSpec(), size_mult=1.0, mass_mult=1.0, label=None,
                    occupied=None, fault=None):
        """Run one intake. frame_cb(station, controller_or_None) is called after every physics step.

        label:    None (random from the pool), a pool index, "damaged" or "unregistered".
        occupied: list of locations to mark occupied first (exact replay of an evaluation episode).
        fault:    "swap_label_in_transit" (test-only world fault: the label on the carried bag is replaced by another
                  pool label after the pick, to exercise the verify-mismatch refusal).
        noise perturbs only the controller's pose observation; size_mult / mass_mult scale the envelope (OOD).
        """
        m, d = self.m, self.d
        ep = self.episode_index
        self.episode_index += 1
        self.intake_number += 1
        badge = BADGES[int(np.random.default_rng([seed, 1]).integers(len(BADGES)))]

        if occupied is not None:
            self.bank = routing.CabinetBank()
            for loc in occupied:
                self.bank.reserve(loc, "REPLAY")
        else:
            self.bank.empty_full_cabinets()
        bank_before = [loc for loc, who in self.bank.occupant.items() if who is not None]

        material, true_id, content = self._choose_item(seed, label)
        scene.reset_home(m, d)
        params = self.randomiser.apply(d, seed, content, size_mult=size_mult, mass_mult=mass_mult)
        self.randomiser.set_label(content, material)
        perception_error = draw_error(noise, np.random.default_rng([seed, 2]))
        self.perception_error = perception_error  # exposed for display (record.py)
        item_body = scene.body_id(m, scene.item_body(content))

        # Evaluator: where the TRUE id should go, from the same occupancy, without touching the real bank.
        shadow = routing.CabinetBank()
        shadow.occupant = dict(self.bank.occupant)
        truth_route = routing.route(true_id, shadow)
        correct_location = truth_route.location if truth_route.ok else None

        for _ in range(int(SETTLE_BEFORE / m.opt.timestep)):
            mujoco.mj_step(m, d)
            if frame_cb:
                frame_cb(self, None)

        self.log.append(officer(badge), "SUBMITTED", PENDING, "PENDING", "sealed-bag",
                        detail=f"sealed, labelled evidence bag left at the intake hatch (intake #{self.intake_number})")

        st = {"decoded": None, "record": None, "location": None, "placed": False, "logged_end": False}

        def ids():
            rec = st["record"]
            return (st["decoded"] or "UNREAD", rec.case_id if rec else "UNKNOWN",
                    rec.category if rec else "sealed-bag", st["location"])

        def on_event(ev):
            if ev.kind == "scanned":
                st["decoded"] = ev.data["decoded"]
            elif ev.kind == "routed":
                st["record"], st["location"] = ev.data["record"], ev.data["location"]
                scan = c.scan_info.get("rescan") or c.scan_info["intake"]
                rec = st["record"]
                self.log.append(SYSTEM_ACTOR, "REGISTERED", st["decoded"], rec.case_id, rec.category, st["location"],
                                detail=f"barcode {st['decoded']} read by {scan['camera']} at t={scan['sim_time']:.2f}s; "
                                       f"{rec.case_id} ({rec.category}: {rec.description}) -> {st['location']}")
            elif ev.kind == "grasp_verified":
                i, cs, cat, loc = ids()
                self.log.append(ROBOT_ACTOR, "PICKED", i, cs, cat, loc, detail="grasp verified: two-finger contact")
            elif ev.kind == "released":
                i, cs, cat, loc = ids()
                v = c.scan_info.get("verify") if c.scan_info.get("verify", {}).get("decoded") else c.scan_info.get(
                    "verify_retry", {})
                center, half = scene.slot_volume(m, d, loc)
                if in_volume(d.xpos[item_body][:2], center[:2], half[:2]):
                    st["placed"] = True
                    self.log.append(ROBOT_ACTOR, "PLACED", i, cs, cat, loc,
                                    detail=f"verify scan {v.get('camera')} read {v.get('decoded')} at "
                                           f"t={v.get('sim_time', 0):.2f}s: MATCH; released into {loc}")
                else:
                    st["logged_end"] = True
                    self.log.append(ROBOT_ACTOR, "FAILED", i, cs, cat, loc,
                                    detail="RELEASE: item released outside the slot footprint")
            elif ev.kind == "refused":
                st["logged_end"] = True
                i, cs, cat, loc = ids()
                actor = SYSTEM_ACTOR if ev.phase == "ROUTE" else ROBOT_ACTOR
                self.log.append(actor, "REFUSED", i, cs, cat, None,
                                detail=f"{ev.phase}: {ev.detail}; not filed, item left on the intake counter for a "
                                       f"handler")
            elif ev.kind == "failed":
                st["logged_end"] = True
                i, cs, cat, loc = ids()
                self.log.append(ROBOT_ACTOR, "FAILED", i, cs, cat, loc, detail=f"{ev.phase}: {ev.detail}")
            if fault == "swap_label_in_transit" and ev.kind == "grasp_verified":
                other = (labels.POOL_IDS.index(true_id) + 1) % labels.POOL_SIZE if true_id in labels.POOL_IDS else 0
                self.randomiser.set_label(content, f"label_{other:02d}")

        c = ctl.PickPlaceController(m, d, content, self.bank, self.cameras, on_event=on_event,
                                    perception_error=perception_error)
        t0 = d.time
        max_tilt_cos = 1.0
        while not c.done:
            if d.time - t0 > MAX_CYCLE_TIME:
                c._fail(f"cycle exceeded {MAX_CYCLE_TIME:.0f}s")
                break
            c.update()
            mujoco.mj_step(m, d)
            if c.phase in CARRY_PHASES:
                max_tilt_cos = min(max_tilt_cos, abs(d.xmat[item_body][8]))
            if frame_cb:
                frame_cb(self, c)
        r = c.result

        for _ in range(int(SETTLE_AFTER / m.opt.timestep)):
            mujoco.mj_step(m, d)
            if frame_cb:
                frame_cb(self, c)

        # ---- evaluator: ground truth only ------------------------------------------------------------------------
        pos = d.xpos[item_body].copy()
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, item_body, vel, 0)
        at_rest = np.linalg.norm(vel[3:]) < REST_LIN_VEL and np.linalg.norm(vel[:3]) < REST_ANG_VEL
        final_location = None
        if at_rest:
            for loc in scene.LOCATIONS:
                if in_volume(pos, *scene.slot_volume(m, d, loc)):
                    final_location = loc
        misfile = final_location is not None and final_location != correct_location

        loc = st["location"]
        success, placement_error = False, None
        outcome, failure_phase, failure_reason = r.outcome, r.failure_phase, (None if r.success else r.reason)
        inside_routed = loc is not None and final_location == loc
        if r.outcome == "filed":
            if st["placed"] and inside_routed:
                success = True
                center, _ = scene.slot_volume(m, d, loc)
                placement_error = float(np.linalg.norm(pos[:2] - center[:2]))
                i, cs, cat, _ = ids()
                self.log.append(SYSTEM_ACTOR, "VERIFIED", i, cs, cat, loc,
                                detail=f"at rest inside {loc}; offset {placement_error * 1000:.0f} mm from centre")
            else:
                outcome = "failed"
                why = "not inside the routed slot volume" if not inside_routed else "not at rest"
                failure_phase, failure_reason = "VERIFY", f"post-settle check failed: {why}"
                if st["placed"]:
                    i, cs, cat, _ = ids()
                    self.log.append(SYSTEM_ACTOR, "FAILED", i, cs, cat, loc, detail=f"VERIFY: {failure_reason}")
                self.bank.release(loc)
        elif not st["logged_end"]:
            i, cs, cat, l2 = ids()
            self.log.append(ROBOT_ACTOR, "FAILED", i, cs, cat, l2, detail=f"{failure_phase}: {failure_reason}")

        rec = st["record"]
        return EpisodeResult(
            ep, int(seed), true_id, st["decoded"], rec.case_id if rec else None, content,
            rec.category if rec else None, loc, correct_location, final_location, outcome, success, misfile,
            r.refusal_cause if outcome == "refused" else None,
            None if success else failure_phase, None if success else failure_reason, r.grasped, r.slipped,
            round(r.duration, 3), round(placement_error, 4) if success else None, r.scan, bank_before,
            params.to_dict(),
            {"noise": {"pos_sigma_mm": noise.pos_sigma_m * 1000, "yaw_sigma_deg": float(np.degrees(noise.yaw_sigma_rad)),
                       "size_sigma_pct": noise.size_sigma_frac * 100},
             "error": perception_error.to_dict(), "size_mult": size_mult, "mass_mult": mass_mult, "label": material,
             "fault": fault},
            bool(inside_routed), round(float(np.degrees(np.arccos(max_tilt_cos))), 2))


def main():
    parser = argparse.ArgumentParser(description="Run consecutive randomised intake episodes")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0, help="episode i uses seed + i")
    parser.add_argument("--log", default=str(scene.OUT_DIR / "custody_log.jsonl"))
    parser.add_argument("--label", help="force a label: pool index, 'damaged' or 'unregistered'")
    parser.add_argument("--occupied", default=None, help="comma-separated occupied locations (exact replay)")
    args = parser.parse_args()

    station = IntakeStation(args.log)
    results = []
    label = int(args.label) if args.label and args.label.isdigit() else args.label
    occupied = [x for x in args.occupied.split(",") if x] if args.occupied is not None else None
    for i in range(args.episodes):
        seed = args.seed + i
        print(f"\n=== episode {i}  seed {seed} ===")
        res = station.run_episode(seed, label=label, occupied=occupied)
        results.append(res)
        err = f"{res.placement_error * 1000:.0f} mm" if res.placement_error is not None else "-"
        print(f"--> {res.outcome.upper()} {res.object_class} true={res.true_item_id} read={res.decoded_id} "
              f"routed={res.routed_location} correct={res.correct_location} final={res.final_location} "
              f"misfile={res.misfile} t={res.cycle_time:.1f}s err={err} "
              f"{res.refusal_cause or res.failure_reason or ''}")

    n = len(results)
    print(f"\nfiled {sum(r.success for r in results)}/{n}, refused {sum(r.outcome == 'refused' for r in results)}, "
          f"failed {sum(r.outcome == 'failed' for r in results)}, MISFILES {sum(r.misfile for r in results)}; "
          f"log chain verify() -> {station.log.verify()}")


if __name__ == "__main__":
    main()
