# NOTES — running log of decisions, failures, open items

## Status after Phase 0B (Tasks A–D complete)

- **The headline is now a known failure boundary, not a bare 100%.** With perfect state the pipeline is reliable
  within its randomisation envelope (1000/1000 in Phase 0; 998/1000 across the Phase 0B zero-noise points).
  With pose error on the controller's input, overall success drops below 95% at ≈ 5.8 mm, below 80% at ≈ 10.6 mm
  and below 50% at ≈ 17.3 mm (per-axis σ). Yaw error is minor: still ≥ 95% up to ≈ 21°. Out-of-distribution
  items fall off a cliff at ≈ ×1.1–1.4 of the size/mass envelope. Details under "Task B" below.
- A perception seam (`scripts/perception.py`) now carries the only item state the controller may use; the evaluator
  provably stays on ground truth (`tests/test_perception.py`).
- The scene reads as an evidence room (grey locker, matte floor, evidence tags, amber bottle, slot labels) with
  physics bit-identical to before.
- The demo (`out/demo_phase0b.mp4`) runs under a stated σ 5 mm / 5° pose error and says so on screen, with the
  success rate measured at that level.
- Still true: no perception model, no learned policy, simulation only, four object classes. Two rare zero-noise
  cylinder release failures are known and deliberately left unfixed during measurement (see Task B).

## Status at the end of Phase 0 (M0–M7 complete)

**Works, verified by rendering, by numbers, or both:**
- The scene loads and renders headless. Nothing intersects, and items rest stably after randomisation
  (199/200 spawn test before the bag flatness constraint; that one bag case is fixed).
- Scripted, contact-physics pick-and-place for all 4 classes into all 4 slots. No weld and no kinematic grasp,
  anywhere in the code.
- The hash-chained custody log: 8 tamper tests pass (edit, re-hashed forgery, deletion, reordering, refusal to
  append to a broken chain).
- Domain randomisation with exact replay from a seed and slot (verified bit-for-bit).
- Evaluation: **1000/1000 episodes, 95% CI [99.6%, 100%]**, and 100/100 on a second seed range, under the conditions
  listed in M5. In short: ground-truth state, in-distribution, scripted controller, simulation only.
- Demo video `out/demo.mp4` (85.7 s) with the live custody panel. The final episode is a labelled fault injection.
- The RL environment passes `check_env`, a scripted policy solves it through the action space 60/60, and SAC/PPO
  smoke runs checkpoint and reload.

**Not done, or not verified:**
- No learned policy. RL was a smoke test only, by design (M7 has the compute estimate).
- No perception. Every success number uses ground-truth item pose.
- The interactive viewer (`mjpython scripts/simulation.py`) was not opened during the build, because the build had
  no GUI session. Only the `--headless` path is verified.
- No robustness sweep yet: no pose noise and no out-of-distribution sizes, masses or poses.
- Sim-to-real is not addressed.

**Recurring lesson:** MuJoCo's pad contacts between the Panda finger boxes and items flicker. One pad can vanish from
the active contact set for a physics step while the item is plainly held. Any grasp or hold predicate must be
evaluated over a time window, never at a single instant. This caused every false failure in M5 and M7.

**Open items / next steps:** pose-noise and out-of-distribution evaluation; a perception stage; GPU-parallel
simulation (MJX / Warp) with demo-bootstrapped RL; multiple physical items so the cabinet visibly fills up; and a
policy for allocating slots by case, item class or hazard category.

---

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

---

# Phase 0B

## Task A — Provenance and backup

- `milestones-backup` pushed to `origin`; annotated tags `m0`–`m7` on the milestone commits, pushed. `m5` sits on
  `9630991` (the end of M5); the headline numbers it cites were produced by the code at `e893af9`, its parent.
- `out/ARTIFACTS.md` (committed via a `.gitignore` exception) lists every artefact worth keeping, the command and
  commit that produced it, size and SHA-256 prefix. It flags four early eval JSONs that are stamped `19a7653` but ran
  on uncommitted harness code (the `-dirty` stamp was only added at `e893af9`).

## Task B — Perception seam and robustness

### The seam (`scripts/perception.py`)
- `draw_error(NoiseSpec, rng)` runs **once per episode** at reset and returns a frozen `PerceptionError` (position
  offset per axis x/y/z, yaw offset, per-axis size scale). `observe(model, data, body, geom, error, kind)` returns
  an `ItemObservation` = ground truth + that cached error. Deviation from the brief's signature: `observe` takes the
  already-drawn error instead of `(noise_spec, rng)`, so per-call re-sampling is impossible by construction.
- The error comes from its own RNG stream (`[seed, 2]`), so physics draws are untouched; it is recorded per episode
  (`perception.error` in the eval JSON) for exact replay.
- Controller reads routed through it: APPROACH target and yaw, DESCEND target, grasp height (believed centre and
  believed half-height), and the item-derived part of the release height (believed bottom relative to the commanded
  grasp height). Bin wall height stays ground truth (the cabinet is known, not perceived).
- **Evaluator stays on ground truth**, and `tests/test_perception.py` (7 tests) proves it: zero error = exact truth;
  the same offset on every read; noise never moves the item (qpos unchanged); the error is drawn exactly once per
  episode; a +20 mm belief offset displaces the APPROACH goal by exactly 20 mm; a 60 mm offset makes the robot miss
  and the episode is judged FAILED with no PLACED/VERIFIED; placement error equals the true distance; the judging
  code (grasp predicates, `_check_hold`, VERIFY block, evaluate.py) contains no observation reads.
- **Zero-noise regression check:** 20 headline-run episodes replayed through the seam — box, bag and folder are
  bit-identical to Phase 0; the 9 cylinders differ by a few ms of cycle time and < 1 mm of placement error (all still
  succeed), because a tall bottle's release height now uses the *believed* bottom rather than the true in-hand
  geometry. This is the seam doing what it should, not a regression.

### Measurement setup
- `python3 scripts/sweep.py` (commit `ed19f01`, clean): 200 episodes per point, fresh seed range per point
  (100000·sweep + 1000·point), 8 worker processes, 450 s for 6200 episodes. `yaw_wide` (30–90°) was added after
  seeing the yaw curve still at 96% at 20°; it ran at `ed19f01-dirty` where the only change is its entry in
  `SWEEPS` (no controller change).
- Position noise is per-axis σ on x, y **and z** (depth error is real), yaw σ, and size σ per axis.
- Data: `out/robustness_20260925T083332Z.json` + `out/robustness_20260925T084337Z.json` (yaw_wide).
  Charts: `python3 scripts/plot_robustness.py <both jsons>` → `out/robustness_curve.png` (position, headline),
  `robustness_yaw.png`, `robustness_combined.png`, `robustness_size.png`, `robustness_ood.png`,
  `robustness_overview.png` (2×2). Custody chains intact at every point.
- Nothing in the controller, grasp heuristics, clearances or randomisation ranges was changed for this measurement.

### Results (overall success, 200 episodes per point)

| σ | 0 | 2 | 5 | 8 | 12 | 16 | 20 |
|---|---|---|---|---|---|---|---|
| position (mm) | 100.0 | 100.0 | 97.5 | 88.0 | 75.5 | 55.5 | 39.0 |
| yaw (°) | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 97.0 | 96.0 |
| combined (k mm + k°) | 100.0 | 100.0 | 97.5 | 83.0 | 61.0 | 40.5 | 35.0 |

Yaw extension: 30° 85.0, 45° 73.5, 60° 63.5, 90° 61.0. Size-estimate σ 0/5/10/20/30 %: 99.5 / 100 / 98.5 / 95.0 / 94.0.
OOD envelope ×1.0/1.25/1.5/1.75/2.0: 99.5 / 86.0 / 38.5 / 12.5 / 2.0.

### Headline findings
- **Where it breaks (position σ per axis, linear interpolation between points):** below **95% at ≈ 5.8 mm**,
  below **80% at ≈ 10.6 mm**, below **50% at ≈ 17.3 mm**. Combined position+yaw: 95% at ≈ 5.5, 80% at ≈ 8.5, 50%
  at ≈ 14.1. Across the 3–10 mm band (the brief's stated range for a competent RGB-D estimator on a known rigid
  object — an assumption, not something measured here) success falls from ~99% to ~82%: **the grasp tolerance sits
  inside, not comfortably below, what real perception would supply.**
- **Position dominates; yaw barely matters.** Yaw σ stays ≥ 95% up to ≈ 21° and crosses 80% only at ≈ 37°; it never
  reaches 50% (plateau ~60%) because the gripper is symmetric under a half turn and cylinders ignore yaw by design.
  Pooled over episodes by the *actual* drawn error: non-cylinder items succeed 100% below 20° of yaw error, 85% at
  30–40°, 37% at 40–50°, ~0% beyond 50°. Items do **not** self-align in the jaws (measured: 26° misalignment before
  CLOSE, 25.5° after) — they are carried crooked and still fit the bin; failure comes when the rotated footprint
  across the jaws exceeds the 80 mm stroke and a finger lands on the item.
- **Tolerance to the actual error:** pooled position-sweep episodes succeed 99% below 2 mm of horizontal error, 97%
  at 2–5, 96% at 5–8, 90% at 8–11, 86% at 11–14, 68% at 14–18, 54% at 18–24 mm. With depth error < 3 mm the grasp
  tolerates ~8 mm horizontally at 100% and 11–14 mm at 94%; depth error beyond ~10 mm hurts on its own.
- **Which class degrades first:** under position noise the **folder** (79% at 8 mm, while others are ~90%) — it is
  only ~6 cm wide in an 8 cm stroke (≈1 cm lateral margin per side) and is pinched a few mm above the counter, so
  both lateral and depth error bite. Next the **cylinder** (64% at 12 mm, 26% at 20 mm) — the tallest item, grasped
  as high as the hand allows, so a depth error drives the hand body into the bottle top; it is also the only class
  hurt by size-estimate error (77% at 30%), since its grasp height depends on believed height. The **bag** degrades
  last (72% at 16 mm): its rounded edges deflect a misplaced finger instead of stopping it on a flat top.
- **Failure mode, not slip:** at every noise level the dominant failure is **DESCEND** (a finger lands on the item;
  e.g. 80 of 122 failures at 20 mm), then CLOSE (closed on nothing / knocked the item over), then LIFT. Slip in
  transit stays rare (≤ 8 per 200). These are real, replayed failures — five were traced individually (finger
  pressing the box top at 210 N, bottle knocked over, depth error driving the hand onto the bottle).
- **OOD cliff:** below 95% at ≈ ×1.08 of the training envelope, 80% at ≈ ×1.28, 50% at ≈ ×1.44; ~0 by ×2.
  Mechanisms: items wider than the 80 mm stroke (DESCEND), items longer than the 245 mm bin (INSERT/RELEASE/VERIFY
  — resting across the walls), and heavier items slipping (24 slips at ×1.5). The envelope was designed right up
  to the gripper and bin limits, so there is almost no geometric headroom.
- **Zero-noise failures are not zero.** 2 of the 1000 zero-noise episodes in the sweep failed, both short, wide
  cylinders: one wedged in the *open* gripper (release declared complete, bottle carried back home, 12° tilt), one
  leaning on a finger so RELEASE timed out. Both fail identically with the Phase 0 release logic, so they are not
  caused by the seam. A windowed release check (as for grasp verification) is the obvious fix; per the Phase 0B rules
  it has **not** been applied during measurement.
- **Secondary stat:** a failed episode can still end in the right slot (e.g. a bag knocked out of the grasp by the bin
  wall during INSERT that falls in). These are counted as failures (no controlled placement) and reported separately
  as `failed_but_in_slot` (e.g. 2/24 at 8 mm, 17/123 at OOD ×1.5).
- Per-slot success under noise is confounded by allocation (a failed episode frees its slot, so the next episode
  reuses slot_0); do not read per-slot rates as a slot effect.

### What this implies for the next phase (not done here)
Position — especially depth — is the axis to buy down: either perception better than ~5 mm, or grasp behaviour that
tolerates it (compliant/force-guarded descent that stops on contact instead of timing out, a wider or adaptive
approach, or a re-observe step before closing). Yaw needs nothing below ~20°. The OOD edge says the envelope has no
headroom: a real intake counter needs either a wider gripper or a policy for refusing items that do not fit.

## Task C — Make it read as an evidence room

Verified by rendering `demo_cam` (`out/scene.png`, `out/c_demo_transit.png`) and `cabinet_cam`
(`out/c_cabinet_cam_bag.png`), and a composed video frame (`out/c_test_inset.png`), and looking at each.

- **Cabinet → steel locker, physics untouched.** The cabinet walls now exist twice. The Phase 0 collision boxes
  are unchanged but moved to geom group 3 (hidden, like Menagerie's collision meshes; group has no physical effect).
  New visual-only geoms (`contype=0 conaffinity=0 density=0`) form a thicker shell: 3 cm panels, side walls raised
  to 0.16 m, a thicker backboard with a cap, locker doors and pull handles on the two faces the demo camera sees,
  and a plinth. Materials are opaque institutional grey with low specularity.
- **Not raised: the front and bin back walls** stay at 0.108 m. The 208 mm-wide hand passes over them at release
  with ~12 mm to spare, so raising them would change the task, not the look.
- **Floor:** the checker texture is replaced by a flat matte grey.
- **Evidence tags:** a thin white box on each item's top face, `contype=0 conaffinity=0 density=0`. Checked in the
  compiled model: item masses and inertias are unchanged and each item body's collision BVH is still one node
  (MuJoCo leaves non-colliding geoms out of it). The randomiser moves the tag onto the resized item's top face each
  episode, but **the tag's size does not scale with the item** (a small item gets a proportionally large tag).
- **Bottle:** amber glass (`0.58 0.32 0.10 0.93`) instead of green.
- **Slot sites:** confirmed from the render, not the XML, that the group-4 slot volumes do not appear in `demo_cam`.
- **Slot labels:** `SLOT 0`–`SLOT 3` are drawn into the cabinet inset in `record.py`, by projecting each bin's front
  edge through `cabinet_cam`'s real pose and field of view. The allocated slot is highlighted.
- **Physics unchanged (strict check):** the same 100 seeds were run on the pre-Task-C scene (`15c23aa`) and the new
  one: 100/100 on both, and **100/100 episodes bit-identical** (cycle time, placement error, slot, parameters).
  The official `evaluate.py --episodes 100 --seed 0` gives 100/100 at zero noise, chain intact.
- Against the Phase 0 baseline on those seeds: box, bag and folder are identical; the 29 cylinders differ (mean
  placement error 1.75 → 2.39 mm, all still successful). That comes from Task B's release height (believed item
  bottom), not from this task.

## Task D — Re-record and documentation

- **Video:** `python3 scripts/record.py --out out/demo.mp4` → 85.9 s, 1280×720, 30 fps; decodes cleanly with
  ffmpeg; copied to `out/demo_phase0b.mp4`. Verified by extracting a contact sheet and full-size stills.
- **Recorded with pose error, because the curve supports it:** the controller sees the item through `perception.py`
  with σ 5 mm per axis + 5° yaw, the low end of the 3–10 mm band, where the measured success is 97.5% [94.3, 98.9]
  over 200 episodes. The header states this level and success rate, looked up from the sweep JSON at record time,
  not typed in, plus each episode's actual drawn error (e.g. −4.6, −11.7, +1.7 mm, yaw −1.3° for the bag).
  It also states contact physics only and one item per episode.
- Same natural seeds as Phase 0 (1007, 1008, 1002, 1001, 1004); no reselection. All five succeed under noise.
  The sixth episode is still the labelled fault injection (fails at LIFT, FAILED event highlighted).
- **Mistake, disclosed:** the re-record wrote to `out/demo.mp4` and overwrote the Phase 0 video, which the Task A
  manifest listed as existing in one place only. It was regenerated from tag `m6` in a clean worktree
  (`out/demo_phase0_m6.mp4`): the same episodes, outcomes and duration, but new wall-clock custody timestamps.
  `record.py` now defaults to a timestamped filename so this cannot recur. Recorded in `out/ARTIFACTS.md`.
- README rewritten around the conditions and the robustness result; `out/ARTIFACTS.md` lists the new video,
  sweep JSONs and charts.

---

# Phase 1 — The evidence intake workflow

Governing rule (PHASE_1_BRIEF §1): **fail closed.** Never guess, never retry indefinitely, never fall back to the
simulator's true item ID; refuse and leave the item for a human. Misfile rate is the headline metric.

## Task A — Sealed bags with contents

- **New scene file** `models/panda/evidence_intake.xml` (derived from `evidence_room.xml`). `evidence_room.xml` and
  `panda.xml` are unchanged, so the Phase 0/0B scene still compiles. The code that runs it is at the new local tag
  `phase0b` (`3556a34`).
- **One unit type, the sealed evidence bag**, in four content classes: `phone`, `blade`, `garment`, `carton`. Each
  class has one body (one active per episode, the others parked as before) with three geoms: the polythene outer bag
  (box, the only colliding geom, translucent), the contents (visual only, massless) and the label (visual only,
  massless, top face at the +x end). Checked in the compiled model: deleting the contents and label geoms leaves
  mass and inertia bit-identical, and each bag's collision BVH is still one node.
- **Sizes are set by the gripper and the verify scan.** Width stays ≤ ~67 mm (80 mm stroke). The label must start
  ≥ 45 mm from the grasp centre (the hand body spans ±32 mm along x above it) so the wrist camera can see it. So
  bags are long, narrow pouches, 20–24 cm by 4–6.7 cm, which is gripper-limited like the Phase 0 folder. A real
  evidence bag is wider.
- **Offset centre of mass**, as a fraction of the half-length along the long axis, sign random: phone and carton
  0.15–0.45, blade 0.10–0.40, garment 0–0.12. Set via `body_ipos`; explicit box inertia about the CoM as before; the
  contents geom moves to the CoM so the heavy end is visible. Recorded as `com_offset` in the episode params.
- **MuJoCo detail:** bodies whose CoM equals their origin compile as "simple"/"sameframe". `mj_setConst` then
  refuses a moved CoM, and kinematics would place it at the origin anyway. Bag bodies are compiled with
  `simple=False` and `body_sameframe` reset. Verified: `xipos − xpos` equals the drawn offset in every episode.
- The controller grasps at the bag's believed **geometric** centre (from the perception seam) and does not know
  where the contents are. Class-specific rules for the old cylinder/folder were removed.

### Finding: the offset CoM had no effect until an unrealistic friction parameter was corrected
- First 60 episodes at zero noise: 60/60, and the heaviest, most offset cartons (0.3 N·m) tilted only ~1° in the
  grasp. The items' torsional friction coefficient, inherited from my Phase 0 scene, was a flat 0.02 m. At 21 N per
  pad that resists ~0.8 N·m, so no plausible contents could tip a bag.
- Replaced by a physically derived value: coefficient = μ × effective friction radius of the ~17 mm square pad
  (≈ 6.5 mm), i.e. 0.004–0.009 m for the randomised μ 0.6–1.4 (MuJoCo's default is 0.005). This removes an
  artificial advantage; it is not tuning for a better number. The same cartons now tilt 4–15° while carried.
- 200 episodes, zero noise, after the correction: **200/200** filed. The tipping shows up as placement error (mean
  1.8 mm, p95 5.2 mm, max 13.9 mm, versus 1.1/2.4/4.4 mm with the 0.02 m coefficient), not failures. The bins
  still accept a bag tilted up to ~15°. Whether it costs success under pose error is measured in Task E. Carry tilt
  is now recorded per episode (`max_carry_tilt_deg`, ground truth) and summarised by `evaluate.py`.
- `rl/env.py`'s scripted policy is generic again and still solves 12/12 through the action space; `check_env` passes.

## Task B — The barcode, actually read

### Static decode first (as instructed)
- `scripts/labels.py` draws a real Code128 of the item ID (`EV-2026-00XXXX`) plus a human-readable line. The bars
  come from python-barcode's module string at exactly 8 texture px per module, so the texture never limits the
  decode. An ID encodes to **167 modules, 187 with quiet zones**. The label is 56 × 40 mm (it must fit the narrowest
  blade bag) with the symbol spanning 52 mm, so **0.278 mm per module**.
- The scanner covers every position a label can take (pose range plus label offset, ~0.44 × 0.54 m), looking
  straight down from 0.70 m. First rendered frame at 4000 × 3260: plain zxing returned nothing at any
  resolution. The crop showed a crisp, correctly oriented barcode rotated ~45°, beyond the skew a straight scanline
  tolerates on a 52 × 26 mm symbol.
- `scripts/scanner.py` reads omnidirectionally, from pixels only: it locates the label by bar texture (high-contrast
  blocks), estimates the bar angle from image gradients (doubled-angle average), rotates the crop so the bars are
  vertical, and decodes with zxing-cpp; a 15° rotation search is the fallback. **First static decode: 4000 × 3260
  frame → `EV-2026-001259`, the ID printed on that label, first attempt, 100 ms.**
- Measured decode vs resolution, 40 random poses per row: 1.24 px/module → 22/40; **1.65 → 40/40**; 2.06 → 40/40;
  2.47 → 40/40. **Zero misreads at every resolution** (a read is either correct or nothing). Chosen: 4000 × 3260
  = 2.06 px/module, ~50 ms per decode, ~100 ms per render.

### The pool and the cameras
- Label pool: 24 IDs (`EV-2026-001000` + 37·i) plus two refusal-path labels: `label_unregistered`
  (`EV-2026-009999`, valid barcode, no case record) and `label_damaged` (a real ID with the symbol scuffed through;
  the human-readable line survives). All are pre-generated PNGs in `models/panda/labels/` (committed), one
  material each in the scene, selected per episode by writing `geom_matid` — the brief's recommended approach;
  runtime `tex_data` overwriting was not needed.
- Label assignment is a world-side act (`Randomiser.set_label`): the officer applied the label.
- `scanner_cam`: fixed, over the intake zone, 4000 × 3260.
- **Verify-scan wrist cameras, a pair.** Either grasp yaw (180° apart) may be chosen, so the label end can be on
  either side of the hand. The cameras sit beyond each x-edge of the hand body, looking down the finger direction,
  at 1280 × 960 (a tenth of the scanner's pixels). The fingers sit at the bag's centre, so they never cover the label.
  First placement (±52 mm, 83 mm above the TCP): verify 8/12 — the images showed labels intact but **cut off at the
  frame edge** (label centres sit 73–89 mm from the grasp centre). Moved to ±75 mm, 110 mm above the TCP: 24/24.

### Scan rate (static, zero pose noise)
- Intake, 200 randomised episodes: **200/200 correct, 0 misreads, 0 no-reads**. Damaged label: 10/10 no-read.
  Unregistered label: 10/10 decode to `EV-2026-009999`. Verify (wrist, at the slot): 24/24.
- **Caveat:** MuJoCo's renderer is idealised (no blur, glare, motion, print defects or dirt), so 100% is a
  simulation ceiling, not a field read rate. Read rate under pose error and an image-degradation check follow in
  Task E.
- The verify-scan-before-release behaviour (B3) and every refusal path live in the state machine (Task D).

## Task C — The rail and the cabinet bank

- **`models/panda/panda_railed.xml`**: a copy of `panda.xml` with `link0` nested in a `rail_carriage` body (25 kg)
  on a slide joint along world y (range ±0.95 m, damping 400), driven by `rail_actuator` (position, kp 20000,
  force ±600 N). `panda.xml` is untouched and the stock Menagerie scene still compiles.
- **Layout** (`evidence_intake.xml`): a rail platform along y carries the robot. The intake counter sits at rail
  station 0, with the same top surface and item area as Phase 0, so picks are unchanged. A visual-only wall with
  the intake hatch stands behind it. Three Phase 0B lockers stand directly in front of their stations: `CAB-A`
  (−0.72 m, narcotics), `CAB-B` (+0.45 m, weapons), `CAB-C` (+0.90 m, general). Twelve slot sites
  (`cab_b_slot_2` ↔ address `CAB-B/slot_2`), plus one inset camera per locker.
- **Reach:** at each cabinet's station, IK converges (≤ 0.5 mm) above and at release height for all 12 slots, with
  no arm–locker contacts. The front and side bin walls stay at 0.108 m (the hand-clearance constraint from Phase
  0B applies to all three).
- **Rail dynamics, a bug found by testing:** the first traverse did not move (peak speed 0). Contact listing showed
  `link0`'s collision mesh resting on the platform top and its friction pinning the carriage. A real carriage rides
  on linear bearings, so the platform–`link0`/carriage contact is excluded. Afterwards: peak 0.40 m/s (the capped
  setpoint ramp), zero final error, settled within 0.35 s of the ramp, arm deviation relative to the carriage
  ≤ 1.2 mm.
- **TRAVERSE phase** (rail not in the IK): after a verified LIFT the arm lifts the bag to a carry pose (TCP 0.45 m,
  clearing the 0.30 m backboards), then the rail ramps to the station with the arm's joint targets frozen, and
  completes on a real condition (within 1 mm and < 2 mm/s) with a timeout. Holds are checked for drops throughout.
  On return: retreat, arm home, then the rail back to the intake station, strictly in sequence.
- **Routing** (`scripts/routing.py`): case record → category → cabinet (narcotics → CAB-A, weapons → CAB-B,
  general → CAB-C), then the first free slot via `choose_slot()`. Fails closed: `no_case_match` and `cabinet_full`
  are refusals, never a fallback. `tests/test_routing.py` (6 tests) covers the 12 addresses, category routing, the
  unregistered ID, garbage strings, full-cabinet refusal and double-allocation. The case DB covers 24 pool IDs
  (6 per content class, from which the category follows) plus the damaged label's ID; `EV-2026-009999` is
  deliberately unregistered.
- **Check:** 12 episodes, one into each of the 12 locations, all via rail traverse: **12/12**, cycles 16–21 s,
  chain intact. `out/p1c_traverse_sheet.png` shows a frame per phase for a CAB-C episode.
- **Transitional state:** until Task D, the episode still allocates locations with the Phase 0 first-free counter
  (`slots.py`) over the 12 addresses; routing from the decoded barcode replaces it in Task D. The old allocator test
  moved out of `test_custody_log.py`; `test_perception.py` uses location addresses.
