# BUILD BRIEF — Phase 0 Simulation

This document is the full build specification for the Phase 0 proof of concept. It is written to be executed by a coding agent in this repository. Read it alongside `CLAUDE.md`, which holds the business and project context.

---

## 0. How to work

**Ground rules. These override any instinct to move fast.**

1. **Work milestone by milestone, in order.** Do not start M2 until M1 is committed and verified. Do not jump ahead to the interesting parts.
2. **Verify visually, not by assumption.** You cannot open the interactive MuJoCo viewer — it needs a GUI session. Every script must therefore have a headless mode that renders a PNG to `out/`. After each milestone, render a frame and *look at it* before claiming the milestone is done. A scene that compiles is not a scene that is correct.
3. **Never assume model internals.** Do not guess actuator names, joint limits, ctrl ranges, body names, or site names for the robot model. Compile the model, print what is actually there, and write code against the real names. Guessed names are the single most common way this build fails.
4. **No silent stubs.** If something cannot be made to work, stop, write what you tried in `NOTES.md`, and move on to the next milestone. Do not leave a function that returns a hard-coded success value.
5. **Commit at the end of every milestone**, with the milestone number in the message. This is so the work can be rolled back one milestone at a time.
6. **Honesty constraints apply to code and docs.** This POC covers four object classes and does not solve general-purpose grasping. Do not write comments, README text, or log output that implies otherwise. If a shortcut is taken (see §8), it must be documented loudly at the point it is taken and in the README.

**Environment:** macOS on Apple Silicon (M2), Python 3.13, MuJoCo 3.13, no NVIDIA GPU, CPU only. Everything must run on this machine without cloud compute.

---

## 1. Target repo layout

```
physical-ai-poc/
├── CLAUDE.md
├── README.md
├── BUILD_BRIEF.md            # this file
├── NOTES.md                  # running log of decisions, failures, open items
├── requirements.txt
├── requirements-rl.txt       # RL deps, kept separate and optional
├── .gitignore
├── models/
│   ├── panda/                # vendored robot model (gitignored, see M0)
│   └── evidence_room.xml     # the scene — lives inside models/panda/, see M1
├── scripts/
│   ├── simulation.py         # entry point: interactive viewer
│   ├── scene.py              # model loading, name discovery, scene helpers
│   ├── ik.py                 # damped least-squares inverse kinematics
│   ├── controller.py         # pick-and-place state machine
│   ├── randomise.py          # domain randomisation
│   ├── logger.py             # chain-of-custody event log
│   ├── slots.py              # cabinet slot allocation
│   ├── episode.py            # one full intake episode, end to end
│   ├── evaluate.py           # headless N-episode evaluation harness
│   └── record.py             # demo video capture with custody-log overlay
├── rl/
│   ├── env.py                # Gymnasium wrapper
│   └── train.py              # training entry point (smoke test only in Phase 0)
└── out/                      # renders, videos, logs, eval results (gitignored)
```

---

## 2. M0 — Dependencies and robot model

**Dependencies.** Set `requirements.txt` to:

```
mujoco>=3.2
numpy
imageio
imageio-ffmpeg
pillow
```

Keep RL dependencies out of the main requirements file. Create `requirements-rl.txt`:

```
gymnasium
stable-baselines3
```

**Robot model.** Use the Franka Emika Panda from MuJoCo Menagerie. It is well-maintained, has a working parallel gripper, and ships a `home` keyframe.

Get it with a sparse checkout so the repo does not carry the whole Menagerie:

```bash
git clone --depth 1 --filter=blob:none --sparse \
  https://github.com/google-deepmind/mujoco_menagerie.git /tmp/menagerie
cd /tmp/menagerie && git sparse-checkout set franka_emika_panda
cp -r /tmp/menagerie/franka_emika_panda ./models/panda
```

Add `models/panda/` to `.gitignore` and put the clone commands in the README setup section, so the repo stays light and the model is reproducible.

**Why the scene file lives inside `models/panda/`:** MuJoCo resolves `meshdir` and asset paths relative to the main model file's directory. Putting `evidence_room.xml` in the same directory as `panda.xml` and its `assets/` folder makes every relative path resolve on the first try. Fighting this with `<compiler meshdir=...>` overrides wastes hours. Accept the slightly odd layout.

So: the scene file is `models/panda/evidence_room.xml`, and `models/evidence_room.xml` (the existing empty placeholder) should be deleted.

**Verify M0** by compiling the stock Menagerie scene and printing the model summary:

```python
import mujoco
m = mujoco.MjModel.from_xml_path("models/panda/scene.xml")
print(m.nq, m.nu, m.nbody)
for i in range(m.nu):
    print(i, mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i), m.actuator_ctrlrange[i])
```

Record the real actuator names and control ranges in `NOTES.md`. Everything downstream uses these.

---

## 3. M1 — The evidence room scene

Create `models/panda/evidence_room.xml`. It includes the Panda and adds the room.

**Required contents:**

- `<include file="panda.xml"/>` for the arm and hand.
- **Global options** tuned for grasping:
  ```xml
  <option timestep="0.002" integrator="implicitfast" cone="elliptic" impratio="10"/>
  ```
  `cone="elliptic"` and a raised `impratio` substantially reduce grasp slippage. This is the difference between a gripper that holds objects and one that drops everything, and it costs nothing.
- **Ground plane** and adequate lighting.
- **Intake counter** — a static box in front of the arm, top surface within comfortable reach. Panda's usable workspace is roughly 0.3–0.7 m radial from the base; put the counter surface around 0.45–0.55 m out and size it so objects land inside reach. Verify reachability rather than trusting these numbers.
- **Cabinet** — a static structure to one side of the arm with **four addressable slots** in a 2×2 grid. Build it from thin static boxes: a back panel, a floor, a top, two side walls, one vertical divider, one horizontal shelf. Slots must be open-fronted and open-topped enough that a top-down insertion is geometrically possible — check this by rendering, not by reasoning about it.
- **Named site per slot**: `slot_0` … `slot_3`, each at the centre of the slot's usable volume, at the height an object should be released from. The controller addresses slots by these site names.
- **Four evidence objects**, all present in the model at all times, all freejoint bodies:
  - `item_box` — box geom, rigid boxed item
  - `item_bag` — ellipsoid geom, approximating a sealed evidence bag
  - `item_cylinder` — cylinder geom, bottle
  - `item_folder` — thin flat box, document folder

  Only one is active per episode; the others get parked out of the way (see M4). Give each a distinct colour so renders are readable.
- **Friction**: give the object geoms and the gripper finger pads generous sliding friction, e.g. `friction="1.5 0.05 0.0001"` on the pads. Tune from there.
- **Cameras**: a fixed `demo_cam` framing the counter, arm and cabinet together in one three-quarter view, and a `topdown_cam` looking down at the counter. The demo camera framing matters — this is the shot that ends up in the pitch video.
- **A `tcp` site** at the tool centre point, between the gripper fingers, roughly where a grasped object's centre should sit. Add it to the hand body if Menagerie does not already provide one. All IK targets this site.

**`scripts/scene.py`** should expose:
- `load()` returning `(model, data)` with the `home` keyframe applied via `mj_resetDataKeyframe`
- name→id lookup helpers for bodies, sites, geoms, actuators
- a discovery function that prints every actuator, joint range, and named site, so the real names are always visible

**`scripts/simulation.py`** is the human entry point: loads the scene, launches `mujoco.viewer.launch_passive`, holds the window open. It must also accept `--headless`, which renders one frame from `demo_cam` to `out/scene.png` and exits.

**Definition of done for M1:** `python3 scripts/simulation.py --headless` writes `out/scene.png`; you have looked at that PNG; the arm, counter, cabinet with four visible slots, and all four objects are present and sensibly positioned; nothing is intersecting anything else; nothing has fallen through the floor. Commit.

---

## 4. M2 — Inverse kinematics and scripted pick-and-place

**`scripts/ik.py` — damped least-squares IK.** Do not add an IK dependency; this is about 60 lines and removes a whole class of install problems.

Algorithm, iterated to convergence or an iteration cap:
1. Compute the site Jacobian with `mujoco.mj_jacSite` for the `tcp` site.
2. Position error = target position − current site position.
3. Orientation error: current site rotation matrix → quaternion (`mju_mat2Quat`), conjugate it (`mju_negQuat`), multiply by the target quaternion (`mju_mulQuat`), convert to a rotation vector (`mju_quat2Vel`).
4. Stack into a 6-vector error. Take only the seven arm DOF columns of the Jacobian.
5. `dq = Jᵀ (J Jᵀ + λ²I)⁻¹ err`, with damping λ around 0.1–0.5. Higher damping is slower but far more stable near singularities — prefer stability here.
6. Clamp `dq` to a maximum step, integrate into the joint configuration, clamp to the real joint ranges from the model.

Expose `solve_ik(model, data, target_pos, target_quat) -> q_arm` operating on a scratch `MjData` so it does not disturb simulation state.

**`scripts/controller.py` — the pick-and-place state machine.** Explicit named phases, each with an entry action, a completion condition, and a timeout. Never advance on a fixed number of steps alone; always have a real condition *and* a timeout that records a failure.

```
HOME → APPROACH → DESCEND → CLOSE → LIFT → TRANSIT → INSERT → RELEASE → RETREAT → HOME
```

- `APPROACH` — move the TCP to a pre-grasp pose directly above the object, gripper pointing down, yaw aligned to the object's long axis. Clearance of roughly 10 cm above the object's top.
- `DESCEND` — descend to the grasp pose. Grasp height depends on object class: grasp a cylinder around its mid-height, a folder near its top face, a box at its centre.
- `CLOSE` — command the gripper actuator closed. Complete when finger joint velocities settle *or* contact force on both pads exceeds a threshold. Do not complete purely on elapsed time.
- `LIFT` — raise vertically. This is the real grasp test.
- `TRANSIT` — move above the target slot site. Use a waypoint clear of the cabinet's top edge so the path does not sweep through the structure; straight-line interpolation in Cartesian space will collide with the cabinet otherwise.
- `INSERT` — descend into the slot.
- `RELEASE` — open the gripper.
- `RETREAT` → `HOME`.

Also expose a `grasp_ok(model, data)` predicate: object geom in contact with both finger pads, and object height above a threshold after `LIFT`. This predicate is what the evaluation harness scores, so it must be honest — do not make it optimistic.

**Definition of done for M2:** with `item_box` at a fixed known pose, the full cycle runs and the box ends up inside a named slot. Render frames during the cycle to `out/m2_*.png` and look at them. Commit.

---

## 5. M3 — Custody log and slot allocation

**`scripts/logger.py` — hash-chained, append-only custody log.** This is a small piece of code that carries a disproportionate amount of the pitch. Get it right.

Each event is a record containing:
- `event_id` — UUID4
- `timestamp` — ISO 8601, UTC, microsecond precision
- `actor` — `OFFICER:<badge>`, `ROBOT:arm-01`, or `SYSTEM`
- `action` — one of `SUBMITTED`, `REGISTERED`, `PICKED`, `PLACED`, `VERIFIED`, `FAILED`
- `item_id`, `case_id`, `object_class`
- `slot_id` — nullable
- `detail` — free-text
- `prev_hash` — hash of the previous event, `null` for the first
- `hash` — SHA-256 over the canonical JSON of every field above except `hash` itself

Write as JSON Lines to `out/custody_log.jsonl`. Append only: never update or delete a record, ever. Provide:
- `CustodyLog.append(...)` returning the written event
- `CustodyLog.verify()` recomputing the whole chain and returning the index of the first tampered record, or `None`
- `CustodyLog.tail(n)` for the video overlay
- Readable one-line terminal output per event

The tamper-evidence property is the point: any edit to a historical record breaks every subsequent hash, and `verify()` finds it. Write a small test that appends several events, mutates one field in the file, and confirms `verify()` catches it. That test is worth showing to an evaluator.

**`scripts/slots.py` — slot allocation.** Track occupancy of `slot_0`…`slot_3`. `allocate(item)` returns the next free slot, or raises when the cabinet is full. Keep the allocation policy trivial for now but isolate it behind a function, since a real deployment would allocate by case, item class, or hazard category.

**Definition of done for M3:** a script run produces a valid custody log; `verify()` passes on a clean log and correctly identifies a deliberately corrupted one. Commit.

---

## 6. M4 — Domain randomisation and the episode loop

**`scripts/randomise.py`.** Randomise at the start of every episode, mutating `mjModel` fields in place rather than recompiling XML — primitive geoms support this and it is far faster:

- **Active object class** — uniform over the four classes. The three inactive objects are parked well below the floor plane and their freejoints zeroed, so they neither render nor collide.
- **Size** — scale `model.geom_size` for the active object within a per-class range, roughly ±30%.
- **Mass** — set `model.body_mass` within a per-class range. Cover a genuinely wide range; a 50 g folder and a 900 g bottle are different grasping problems.
- **Friction** — perturb `model.geom_friction` on the object.
- **Pose on the counter** — randomise x, y within the reachable counter area, and yaw over the full circle. Keep the object fully on the counter surface and inside the arm's reach.
- **Lighting** — perturb light position and diffuse colour. Cosmetic for now, but it matters the moment a vision model enters the loop, and costs nothing to build in early.

Seed the RNG and record the seed in the episode result, so any failure can be replayed exactly. This matters more than it sounds like it does.

**`scripts/episode.py` — one complete intake episode:**
1. Randomise the scene, reset to the `home` keyframe.
2. Log `SUBMITTED` (actor: a simulated officer) and `REGISTERED` (actor: `SYSTEM`), allocate a slot.
3. Run the pick-and-place state machine.
4. Log `PICKED` on a verified grasp, `PLACED` on successful insertion, `FAILED` with the phase name and reason on any failure or timeout.
5. Log `VERIFIED` after a settling period confirms the object is inside the slot volume and at rest.
6. Return a structured result: success flag, failure phase, object class, seed, cycle time, final placement error.

**Definition of done for M4:** ten consecutive randomised episodes run without crashing, and the custody log tells a coherent story for each. Failures are expected and fine — they must be *logged correctly* rather than hidden. Commit.

---

## 7. M5 — Evaluation harness

**`scripts/evaluate.py`** runs N episodes headless (default 100) and reports:

- Overall success rate with a 95% confidence interval
- Success rate broken down by object class
- Failure breakdown by phase — where in the cycle things actually fail
- Mean and p95 cycle time
- Mean placement error
- Grasp-slip rate: grasped and lifted, then dropped before release

Write results to `out/eval_<timestamp>.json` and print a readable table. Support `--seed` and `--episodes`.

This is the most important deliverable in the whole build. A video shows that it worked once; a success rate across 100 randomised episodes, broken down by object class, with failures attributed to phases, is an actual technical claim. It is also exactly what a grant assessor or a technical due-diligence conversation will ask for, and what turns "we built a demo" into "we measured a policy".

Report whatever the number is. A 60% success rate honestly measured and broken down is worth more than a claimed 95% that cannot be reproduced.

**Definition of done for M5:** a 100-episode run completes and produces the JSON and the table. Put the headline numbers in `NOTES.md`. Commit.

---

## 8. M6 — Demo video

**`scripts/record.py`** produces the pitch asset: a 60–90 second MP4 showing the full loop.

- Offscreen render via `mujoco.Renderer` at 1280×720 from `demo_cam`.
- Record several consecutive episodes with different object classes, so the variety is visible.
- Composite a **right-hand side panel** onto each frame with PIL showing the live custody log — the last few events scrolling as they happen, plus the current item ID, case ID, and allocated slot.
- Sub-sample frames to about 30 fps output; do not write a frame per physics step.
- Write to `out/demo.mp4` via `imageio`.

The side panel is the whole point. An arm moving boxes is a robotics demo that any lab can show. An arm moving boxes *while a tamper-evident chain-of-custody record writes itself alongside* is the product. Give the panel real visual care.

Include at least one failed episode if the success rate is below about 90%, with the `FAILED` event visible in the log panel. Showing a handled failure reads as competence to a technical evaluator, and hiding it reads as a sales reel.

**Definition of done for M6:** `out/demo.mp4` exists, plays, and both the arm and the log panel are legible. Commit (the video itself stays gitignored).

---

## 9. M7 — RL scaffold, smoke test only

**Read this section carefully and respect its limits.**

Training a contact-rich manipulation policy from scratch, on CPU, on an M2 Air, with a 7-DOF joint-space action space, will not converge in any useful time. Do not attempt it. Do not leave a training run going in the hope that it works.

What to build instead:

**`rl/env.py`** — a Gymnasium environment wrapping the episode loop, with a deliberately reduced action space:
- **Action**: Cartesian TCP delta `(dx, dy, dz, dyaw)` plus a gripper open/close command. The DLS IK converts this to joint targets. Reducing the action space this way is what makes the problem tractable on CPU.
- **Observation**: TCP pose, gripper state, object pose relative to TCP, object class one-hot, target slot position. State-based only — no vision in Phase 0.
- **Reward**: shaped — a distance term toward the grasp pose, a bonus on verified grasp, a distance term toward the slot while holding, a large bonus on verified placement, a penalty on drops, and a small time penalty.
- **Termination**: success, drop, timeout, or the object leaving the workspace.

**`rl/train.py`** — SAC or PPO from stable-baselines3, configurable steps, checkpointing to `out/rl/`.

**Then run a smoke test only**: a few thousand steps, purely to confirm the environment steps correctly, the reward is finite and non-degenerate, observation and action spaces are consistent, and checkpointing works. Confirm the plumbing. Stop there.

Write in `NOTES.md` what would actually be needed for convergence — realistically parallel environments and a GPU box, which is a cloud-compute question rather than a code question, and which the Enterprise Compute Initiative in the funding research is meant to cover.

**Definition of done for M7:** the env passes a `check_env` call, a short training run executes without error and checkpoints, and the honest limitation is documented. Commit.

---

## 10. Known traps

These are the specific things most likely to consume hours. Handle them as described rather than rediscovering them.

**Asset paths.** Covered in M0: keep the scene file beside `panda.xml`. Do not try to be tidy about this.

**Guessed names.** Covered in §0. Print, don't assume.

**Grasps that slip.** In rough order of effectiveness: set `cone="elliptic"` and `impratio` around 10–20; raise finger-pad friction; make sure the gripper actuator is commanded to a genuinely closed target rather than merely "closer"; check that the grasp height actually puts the pads on the object's body rather than an edge; reduce object mass. If grasps are still unreliable after working through all of these, record the failure mode in `NOTES.md` and continue — a measured 50% grasp rate is a real result, and the evaluation harness exists precisely to surface it.

**The weld shortcut.** A `weld` equality constraint attaching the object to the gripper when grasp conditions are met makes everything work instantly. It is also a kinematic fake that discards the contact physics the entire project claims to be about. Do not use it as the default path. If it is used at all, it must be behind an explicit `--kinematic-grasp` flag, off by default, commented at the point of use, and disclosed in the README. A grant assessor asking how grasping works and finding a weld constraint is a credibility problem that outlasts the demo.

**Cabinet collisions during transit.** Straight-line Cartesian motion from the counter to a slot passes through the cabinet structure. Use an explicit waypoint above the cabinet's top edge.

**IK singularities.** Near-singular configurations produce huge joint velocities. Keep damping high and clamp per-step joint deltas.

**Objects tunnelling through the floor.** If a freejoint object falls through the plane, the timestep is too large for its speed or the geoms are too thin. Reduce the timestep or thicken the geom.

**Offscreen rendering on macOS.** `mujoco.Renderer` works on Apple Silicon, but if it fails, check that no interactive viewer is holding a GL context in the same process. Render in a separate process from the viewer.

---

## 11. What "done" looks like for Phase 0

At the end of this build, the repository should produce, on this laptop, with no cloud compute and no physical hardware:

1. An interactive simulation of an evidence room that a person can open and look around.
2. A robotic arm that picks randomised evidence objects from an intake counter and files them into addressed cabinet slots.
3. A tamper-evident chain-of-custody log written automatically as the work happens, with a verification function that detects any retrospective edit.
4. A measured success rate across 100 randomised episodes, broken down by object class and failure phase.
5. A demo video showing the loop with the custody log writing itself alongside.
6. An RL environment that is correct and ready to train, with the compute requirement honestly stated rather than quietly attempted.

Items 3 and 4 are what distinguish this from a robotics tech demo. Give them proportionate care.
