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

## M5 — Evaluation harness

`python3 scripts/evaluate.py [--episodes N] [--seed S]` → `out/eval_<ts>.json` + its own custody log + a table.
95% intervals are Wilson score intervals (the normal approximation is meaningless near 100%).

### Headline numbers (code at commit `e893af9`, MuJoCo 3.13.0, CPU only)

| run | seeds | success | 95% CI | grasp slip | placement error (mean / p95 / max) | cycle time (mean / p95) |
|---|---|---|---|---|---|---|
| 100 episodes | 0–99 | **100/100** | [96.3%, 100%] | 0/100 | 1.3 / 3.3 / 7.7 mm | 12.6 / 13.9 s |
| 1000 episodes | 1000–1999 | **1000/1000** | [99.6%, 100%] | 0/1000 | 1.3 / 3.5 / 19.1 mm | 12.7 / 14.2 s |

Per class (1000-episode run): box 230/230 [98.4, 100], bag 258/258 [98.5, 100], cylinder 253/253 [98.5, 100],
folder 259/259 [98.5, 100]. Per slot: 250/250 each. Failures by phase: none. Custody chain intact (5000 events).
Wall time ≈ 0.39 s per episode on the M2 Air.

### What these numbers do and do not mean — read before quoting them
- They measure a **scripted controller with ground-truth object pose read from the simulator**. There is no
  perception and no pose-estimation noise. This is the single biggest gap between this number and a real cell.
- The evaluation is **in-distribution**: the randomisation ranges were chosen (M4) so every item fits the Panda
  gripper and the bins, and the controller's per-class grasp heuristics were written for exactly these four classes.
- One uncluttered item at a time, always resting upright/flat on the counter; no stacking, occlusion, toppled or
  deformable items (the "bag" is a rigid ellipsoid with rolling friction).
- It is simulation contact physics. Sim-to-real transfer is not measured here.
- So the claim is: *under this randomisation envelope, with perfect state, the contact-physics pipeline is reliable
  (≥ 99.6% at 95% confidence) and the evaluation, logging and replay infrastructure works*. It is not a claim about
  real-world grasp success.
- The obvious next measurement is the same harness with injected pose noise (e.g. σ = 5 mm / 5°) and wider or
  out-of-distribution ranges, to find where it breaks. Not done in Phase 0.

### History (kept deliberately)
- First 1000-episode run (seeds 1000–1999, code at `19a7653` + uncommitted harness): **996/1000 (99.6%,
  [99.0, 99.8])**, 4 failures, all at LIFT, "grasp not verified". Exact replay (`--seed/--slot`) and a force trace
  showed all four items were plainly held (~20 N per pad, lifted ~11 cm, 7+ cm above the threshold) and that at
  the single instant of the check one pad's contact had dropped out of MuJoCo's active set (0 N on one side, ~30 N
  on the other — not a physical equilibrium). The verifier was a false negative, not the grasp.
- Fix: grasp verification integrates over ~0.1 s (12 control samples): the item must be above the height threshold
  at **every** sample and each finger in contact in ≥ 60% of samples. Negative test: with the stock kp=100 gripper
  and a 0.9 kg, μ=0.6 box, 3/3 episodes still fail at LIFT with `contact L 0%, R 0%, above height 0%`.
- Exact replay verified: re-running evaluation episodes in isolation with the recorded seed and slot reproduces
  cycle time, placement error and parameters bit-for-bit.
- The 19.1 mm max placement error is a cylinder that landed off-centre but inside the slot volume and at rest.

## M6 — Demo video

`python3 scripts/record.py` → `out/demo.mp4`: **85.7 s, 1280×720, 30 fps, H.264/yuv420p, ~16 MB**; decodes
cleanly with ffmpeg. Verified by extracting stills across the whole video and inspecting them.

- Left 860 px: `demo_cam` rendered offscreen at 1280×720 and cropped, 1× simulated time, with a `cabinet_cam` inset
  so flat items are visibly filed. Overlay states: simulation, contact-physics grasping only, and that only one item
  is simulated per episode (the cabinet resets between episodes, so earlier items are not shown).
- Right 420 px: live custody panel drawn from the real log as it is written — current item/case/class/slot/robot
  phase, the last five events (action badge, actor, timestamp, detail, own hash ← prev hash), newest highlighted,
  and a footer that re-runs `verify_file()` whenever the log grows.
- Episodes: natural (unforced) seeds 1007 box → slot_0, 1008 bag → slot_1, 1002 cylinder → slot_2,
  1001 folder → slot_3, 1004 cylinder → slot_0 — all succeed.
- **Episode 6 is a deliberate, on-screen-labelled fault injection** (seed 1097, a 0.80 kg box): the gripper is
  reset to Menagerie's stock kp=100 (~7% of the tuned grip force). The grasp fails at LIFT, and `FAILED … LIFT: grasp
  not verified` is written and highlighted. The measured success rate is ~100%, so there is no natural failure to
  show; this demonstrates failure *handling*, and is labelled as injected, not presented as a real failure.
- Log timestamps are wall-clock at write time; rendering runs slower than real time, so gaps between timestamps
  are longer than the corresponding 1× video time.
- Fixes found by inspecting stills: slot field read the wrong key; the ✓ glyph is missing from Helvetica (now
  drawn); the item card showed the previous item for 0.3 s at each episode start; the red banner overlapped the
  inset; the post-episode hold frame predated the VERIFIED event.

## M7 — RL scaffold (smoke test only)

Dependencies (`requirements-rl.txt`): gymnasium 1.3.0, stable-baselines3 2.9.0, torch 2.14.0 (CPU used; MPS is
available but pointless for small MLPs).

### `rl/env.py` — `EvidenceIntakeEnv`
- Action `Box[-1,1]^5`: TCP (dx, dy, dz, dyaw) ≤ 1 cm / 0.1 rad per step + gripper open/close; the DLS IK maps the
  Cartesian target to joint servos. 20 Hz control (25 physics steps per env step), 400-step (20 s) episodes.
- Observation (27-D, state only): TCP position and yaw, gripper opening and command, item position relative to TCP,
  item relative yaw (doubled, for the gripper's half-turn symmetry) and tilt, item size, class one-hot, slot centre
  relative to TCP and to item, two-finger contact flag.
- Reward: −distance(TCP→item) until first grasp; +5 on first held-and-lifted; −2·distance(item→slot) while held;
  +50 on the item released, inside the slot volume and at rest; −10 on a drop or the item leaving the workspace;
  −0.01 per step. Termination: success, drop, out-of-workspace; truncation at the step limit.
- The same domain randomisation as the episode loop (M4); a random slot per episode.

### Verification
- `gymnasium.utils.env_checker.check_env` and `stable_baselines3.common.env_checker.check_env`: **pass** (only the
  standard warning about an unbounded observation Box).
- Random actions (5 episodes): returns −117 … −173, per-step reward −0.52 … −0.15 (finite, non-constant).
- **Solvability check through the action interface:** a hand-written policy that acts *only* via the 5-D action
  (`scripted_action` in `rl/env.py`) succeeds **60/60** randomised episodes, mean return −36.9 — so the reduced action
  space, IK, success detection and reward shaping support the full task, and a good policy scores far above random.
- **Contact flicker, again.** The scripted policy's first failures were false "drops": at one physics step one pad
  reads 0 N and the other ~30–38 N (alternating sides) while the item rises in lock-step with the TCP — the same
  MuJoCo pad-contact artefact as M5. "Held" is now measured over all 25 physics substeps of an env step (each finger
  in contact ≥ 30% of them), and a drop needs 2 consecutive un-held env steps. 11/12 → 60/60.
- `python3 rl/train.py` (SAC, 3000 steps): 12 s wall (≈ 240 steps/s incl. gradient updates), 15 episodes, returns
  finite (mean −87), checkpoints at 1k/2k/3k + final written to `out/rl/`, final model reloaded and produces a valid
  action. `--algo ppo --steps 4096`: ≈ 1070 steps/s, 20 episodes, checkpoints + reload OK.
  **0 successes in both smoke runs — expected, and no trained policy is claimed.** No longer run was attempted.

### What convergence would actually need (estimate, stated as such)
- Measured here: one CPU env steps at ~650–1070 steps/s; SAC trains at ~240 steps/s on the M2 Air.
- State-based RL on multi-stage pick-and-place with randomised objects commonly needs on the order of 10⁶–10⁷ steps
  (off-policy) to 10⁷–10⁸ (on-policy) per run, before multiplying by seeds and hyperparameter sweeps. At the rates
  above that is roughly 1–12 h per SAC run or a day+ per PPO run *on one core*, times 5+ seeds and several configs,
  on a fanless laptop that thermally throttles — i.e. days to weeks. Adding vision multiplies cost by orders of
  magnitude (rendering per step).
- The realistic path is GPU-parallel simulation (MuJoCo MJX or MuJoCo Warp; Menagerie already ships
  `mjx_panda.xml`) running thousands of environments on a GPU machine, which brings 10⁸ steps into hours. That is a
  cloud-compute question rather than a code question — the kind of allocation the Enterprise Compute Initiative in
  the funding research is meant to cover.
- Sample-efficiency levers that don't need new compute: bootstrapping from the scripted controller (it is a free,
  ~100% demonstrator for behaviour cloning / demo-augmented RL), curriculum over the randomisation ranges, and
  HER-style goal relabelling for the slot target.
