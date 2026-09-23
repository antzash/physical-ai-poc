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
