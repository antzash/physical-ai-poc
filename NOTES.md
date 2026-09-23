# NOTES — running log of decisions, failures, open items

## M0 — Dependencies and robot model

- MuJoCo 3.13.0, NumPy 2.4.3, Python 3.13.5 on Apple Silicon (M2), CPU only.
- Panda model: MuJoCo Menagerie `franka_emika_panda`, commit `367e3d9884401dcf6f9c27fa69f118992539039f`,
  vendored into `models/panda/` (gitignored except `evidence_room.xml`).
- Stock `scene.xml` compiles: `nq=9, nu=8, nbody=12`, one keyframe `home`.

### Real actuator names and control ranges (from the compiled model, not guessed)

| idx | actuator    | target              | ctrlrange            | notes |
|-----|-------------|---------------------|----------------------|-------|
| 0   | actuator1   | joint1 (position)   | [-2.8973, 2.8973]    | kp 4500, forcerange ±87 |
| 1   | actuator2   | joint2 (position)   | [-1.7628, 1.7628]    | kp 4500, ±87 |
| 2   | actuator3   | joint3 (position)   | [-2.8973, 2.8973]    | kp 3500, ±87 |
| 3   | actuator4   | joint4 (position)   | [-3.0718, -0.0698]   | kp 3500, ±87 |
| 4   | actuator5   | joint5 (position)   | [-2.8973, 2.8973]    | kp 2000, ±12 |
| 5   | actuator6   | joint6 (position)   | [-0.0175, 3.7525]    | kp 2000, ±12 |
| 6   | actuator7   | joint7 (position)   | [-2.8973, 2.8973]    | kp 2000, ±12 |
| 7   | actuator8   | tendon `split`      | [0, 255]             | gripper: **255 = open, 0 = closed**; ctrl maps to 0–0.04 m per finger |

- Joints: `joint1`…`joint7` (hinge, qpos 0–6), `finger_joint1`, `finger_joint2` (slide, 0–0.04 m, qpos 7–8),
  coupled by a joint equality constraint.
- Bodies: `link0`…`link7`, `hand`, `left_finger`, `right_finger`.
- Sites: **none**. Menagerie does not ship a TCP site.
- `home` keyframe qpos: `0 0 0 -1.57079 0 1.57079 -0.7853 0.04 0.04`, ctrl gripper = 255 (open).

### Decisions

- **TCP site.** MJCF cannot reopen a body defined in an included file, so the `tcp` site is added to the `hand`
  body at load time via `mujoco.MjSpec` in `scripts/scene.py`, at `pos="0 0 0.1034"` in the hand frame
  (the standard Franka flange-to-fingertip offset; this lands between the fingertip pads).
  At `home` the TCP is at world (0.554, 0, 0.521), z-axis pointing straight down, fingers opening along world x.
- **Gitignore.** `models/panda/*` is ignored but `evidence_room.xml` is un-ignored, since the brief requires the
  scene file to sit beside `panda.xml`.

## M1 — Evidence room scene

Verified by rendering `out/scene.png` (demo_cam), `out/scene_topdown_cam.png` and `out/m1_sites.png` (slot + TCP
sites visible) and inspecting them; numerically, after 2 s of physics all four items stay put on the counter
(< 0.1 mm penetration) and the arm touches nothing.

### Layout (world frame, metres)
- Panda base at the origin, mounted **on** the counter. Counter top z = 0, room floor z = −0.8.
- Counter spans x ∈ [−0.25, 0.75], y ∈ [−0.45, 0.2]. Intake zone (visual marking only) x ∈ [0.34, 0.66], y ∈ [−0.29, 0.09].
- Cabinet centred at (0.3, 0.4) to the arm's left (+y). Bin floor top z = 0.008, walls 0.10 m tall, back panel 0.30 m.
- Slot volumes (box sites, centre / half-extent read from the model):
  `slot_0` (0.1675, 0.3175), `slot_1` (0.4325, 0.3175), `slot_2` (0.1675, 0.4825), `slot_3` (0.4325, 0.4825);
  all half-extent (0.1225, 0.0725, 0.05), z-centre 0.058. Radial reach 0.36–0.65 m.

### Deviations from the brief, and why
- **Cabinet has no top panel; the "horizontal shelf" is a front/back divider.** The brief asks for a closed cabinet
  with a horizontal shelf *and* top-down insertion; those are geometrically incompatible for the lower row. The 2×2
  grid is therefore four open-topped bins (floor, back panel, front wall, two side walls, one divider each way).
- **Object sizes are dictated by the gripper.** The Panda hand opens to 80 mm total, so each item's grasp dimension
  is kept ≤ ~68 mm even after randomisation. In particular the "folder" is a slim 18 × 6 × 2 cm document wallet, not
  an A4 folder: a real A4 folder lying flat cannot be grasped top-down by a parallel-jaw gripper at all (it would need
  suction or a different end effector). The cylinder is a short 11 cm bottle standing upright, because the hand body
  would hit a taller bottle before the pads reached its mid-height.
- **`home` keyframe is patched at load.** Menagerie's `home` only covers the 9 arm/finger qpos; MuJoCo zero-pads the
  rest, which would teleport all items to the origin. `scene.load()` fills the item part from `qpos0`.
- **Pad friction** (`1.5 0.05 0.0001`) is set on all finger collision geoms via MjSpec at load time.

### Items (nominal, before randomisation)
| body | geom | size (half-extents) | mass |
|---|---|---|---|
| item_box | box | 0.06 × 0.028 × 0.022 | 0.25 kg |
| item_bag | ellipsoid | 0.08 × 0.032 × 0.025 | 0.15 kg |
| item_cylinder | cylinder | r 0.025, half-height 0.055 | 0.40 kg |
| item_folder | box | 0.09 × 0.03 × 0.01 | 0.10 kg |

- Interactive viewer: `mujoco.viewer.launch_passive` needs `mjpython` on macOS, and could not be exercised from this
  headless build session. Only the `--headless` path has been verified.

## M2 — IK and scripted pick-and-place

Verified by `python3 scripts/controller.py` (item_box at its fixed scene pose → `slot_0`), which writes one frame
per phase to `out/m2_*.png`; inspected as a contact sheet. The box is carried purely by pad contact forces
(~19–23 N per pad) and comes to rest inside `slot_0`. Additionally, all 16 class × slot combinations were run at
nominal poses: 16/16 complete and end inside the target slot volume (12–15 s per cycle).

### Gripper geometry (measured from the Menagerie collision meshes, TCP frame, z down)
- Fingertips extend 8.9 mm below the TCP; pads span ±9 mm about it; open pad gap 80 mm.
- Hand body starts 37.4 mm above the TCP and is 208 mm wide along the finger axis, 63 mm across it.
- Consequence: the hand cannot enter a 145 mm bin, so items are released with the hand just above the bin walls
  (drop of up to ~6 cm for flat items). This is the release height the controller computes, not a guess.

### Decisions / fixes
- **Gravity compensation on the arm bodies** (`gravcomp=1` on link1–7, hand, fingers — never on items). Without it
  the Menagerie position servos sag ~7 mm at the TCP. The real Franka compensates gravity in its controller.
- **Gripper stiffness raised from kp=100 to kp=1500 (kv 10→120).** Measured failure first: at stock gains the grip
  is ~1.4 N per pad and the box slips out on LIFT at 0.5 kg and 0.9 kg (`grasp not verified`). At kp=1500 the grip
  is ~21 N per pad (~42 N total, within the real Franka Hand's 70 N continuous rating) and a 0.9 kg box completes.
  The ctrl mapping (0 closed … 255 open) is unchanged; ctrl 0 is commanded, i.e. a genuinely closed target.
- **Back-row collision.** First run: every insert into `slot_2`/`slot_3` timed out. Contact inspection showed the
  hand and right finger hitting `cab_back` (the hand is 208 mm wide along world y at the slot). Fix: the bin's back
  wall is now 0.10 m like the others, and the tall backboard is set 50 mm further back on an extended base.
- **No weld / kinematic grasp anywhere.** No `--kinematic-grasp` flag exists; it was not needed.
- The cylinder is grasped as close to mid-height as the hand allows (hand body 6 mm above the bottle top), which
  for the nominal 11 cm bottle is ~2 cm above its centre.
- Motion: Cartesian straight-line segments with smoothstep timing, IK solved every 4 physics steps (125 Hz),
  warm-started from the previous solution. Transit goes up to TCP z = 0.30 before moving over the cabinet.
- Phase completion is measured (TCP within 6 mm of goal, fingers settled or both pads > 2 N, fingers open and no
  pad contact, etc.) with a per-phase timeout that ends the cycle as a failure attributed to that phase.
- `grasp_ok`: item in contact with **both** finger bodies (> 0.1 N each) **and** lifted ≥ 4 cm above its resting
  height. A drop is declared after ~0.2 s without two-finger contact while carrying.

## M3 — Custody log and slot allocation

- `scripts/logger.py`: JSON Lines, append-only, SHA-256 over the canonical JSON (sorted keys, compact separators)
  of every field except `hash`; `prev_hash` is `null` for the first record. Timestamps are wall-clock UTC with
  microseconds (the time the record was written, not simulation time).
- Opening an existing log re-verifies it and **refuses to append to a broken chain**.
- `python3 tests/test_custody_log.py` — 8 tests pass: clean chain verifies; an edited field is caught at that
  record; an edit whose own hash is recomputed by the forger is caught at the *next* record; deleted and reordered
  records are caught; appends refuse a tampered file; reopening continues the chain; allocator fills 4 slots then
  raises `CabinetFull`.
- `python3 scripts/logger.py --demo` writes a synthetic (no physics) two-item log to `out/custody_log_demo.jsonl`,
  verifies it clean, then edits `slot_id` in record 3 of a copy — `verify()` returns 3.
  `python3 scripts/logger.py --verify <file>` checks any log (exit code 1 on failure).
- **Known limitation, stated in the module docstring:** a hash chain cannot detect truncation of the newest records.
  A deployment would anchor the head hash externally (e.g. periodic signed checkpoints).
- `scripts/slots.py`: first-free allocation behind `choose_slot()`, so a case/class/hazard policy can replace it.

## M4 — Domain randomisation and the episode loop

`python3 scripts/episode.py --episodes 10 --seed 0 --render` → **10/10 succeeded**, 50 custody events, chain intact.
Start/end frames per episode in `out/m4_ep*_{start,end}.png`; a new `cabinet_cam` confirms flat items lie inside
the bins (they are hidden behind the 10 cm bin walls from `demo_cam`).

### Randomisation (all from one seeded Generator; the seed is recorded in the result and the REGISTERED event)
- Class: uniform over the four. Inactive items: parked at z = −3 (below the floor), velocities zeroed, collisions
  disabled (`contype/conaffinity = 0`) and `gravcomp = 1` so they neither fall nor collide.
- Size: per-axis scale ≈ 0.7–1.3 of nominal, with the gripper-closing axis capped (box ≤ 1.2, bag ≤ 1.05,
  folder ≤ 1.1) so every item fits the 80 mm stroke, and long axes capped to fit a 245 mm bin.
  Bags are additionally kept flatter than they are wide (height ≤ 0.85 × width).
- Mass (kg): box 0.10–0.80, bag 0.05–0.60, cylinder 0.15–0.90, folder 0.05–0.40. Inertia recomputed analytically.
- Friction: sliding 0.6–1.4 on the item. **Items have contact `priority=1`**, so this is the coefficient actually
  used against the pads and the counter; with MuJoCo's default max-combination the 1.5 pads would always win and
  friction randomisation would be a no-op.
- Pose: x 0.40–0.60, y −0.25–0.05, yaw uniform over the full circle.
- Key light: direction, position, overall intensity (0.8–1.2) and a slight per-channel tint (0.95–1.05).

### Bugs found by looking, and fixed
1. **Stale collision bounds after resizing.** First 10-episode run: 5/10. Rendering showed a 14 cm bottle
   *knocked over before the gripper arrived*; tracing showed the bottle bouncing up to 0.3 m during the pre-cycle
   settle. Diffing the in-place-mutated model against one compiled with the same size found `bvh_aabb` (the body's
   bounding volume used by the collision midphase) still holding the nominal box, so enlarged items sank into the
   counter and were ejected. Fix: update `bvh_aabb` and `dof_length` alongside `geom_size/rbound/aabb`. This also
   explained two folder failures (pad contacts culled). Spawn test: 199/200 seeds now still within 1.1 mm.
2. **Bag standing on edge.** The remaining unstable spawn was an ellipsoid taller than wide, which rolls over;
   fixed by the flatness constraint above.
3. **Bag rocking forever.** Two episodes failed VERIFY ("not at rest"): position fixed to the millimetre but
   rocking/spinning at ~0.6 rad/s for > 5 s — a rigid-ellipsoid artefact. Rather than relax the at-rest check, the
   bag geom now has torsional and rolling friction (`condim=6`, `friction="… 0.05 0.01"`) standing in for the energy
   a deformable bag dissipates.

Before these fixes, the failures were logged correctly (e.g. `FAILED  CLOSE: gripper closed on nothing`,
`FAILED  TRANSIT: item dropped during TRANSIT (lost two-finger contact)`, `FAILED  VERIFY: post-settle check
failed: not at rest`), with no PLACED/VERIFIED written after a failure.

### Episode semantics
- SUBMITTED (OFFICER:<badge>) → REGISTERED (SYSTEM, slot + seed + size) → PICKED (ROBOT, on `grasp_ok` after LIFT)
  → PLACED (ROBOT, on release with the item over the slot footprint) → VERIFIED (SYSTEM, after 1 s: item centre
  inside the slot volume, linear speed < 1 cm/s and angular < 0.2 rad/s). Any failure → FAILED with phase + reason.
- Only one item is physically present per episode; slot occupancy is tracked logically across episodes (so all
  four slots are exercised) and cleared when all four are full. A failed episode frees its slot.
