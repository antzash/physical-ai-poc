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
