# Physical AI — Evidence Room POC

A Phase 0 simulation proof-of-concept for robotic evidence intake: a Franka Emika Panda arm picks randomised evidence
items from an intake counter and files them into addressed cabinet slots. A tamper-evident chain-of-custody record
writes itself as the work happens, and a headless evaluation harness measures the whole loop across randomised
episodes.

Everything runs on a MacBook Air M2, CPU only, in [MuJoCo](https://mujoco.org) 3.x. There is no physical robot and
no cloud compute.

## What Phase 0 delivers

| | What it is | Status |
|---|---|---|
| Evidence room scene | Panda on an intake counter, a 2×2 cabinet of open-topped bins with named slot sites, four item classes | Works (`scripts/simulation.py`) |
| Pick-and-place | Damped least-squares IK and a phase-by-phase state machine; each phase has a measured completion condition and a timeout | Works; **grasping is pure contact physics — no weld or kinematic attachment** |
| Chain-of-custody log | Append-only JSON Lines with SHA-256 hash chaining; `verify()` finds the first tampered record | Works; 8 tamper tests pass |
| Domain randomisation | Item class, size, mass, friction, pose, lighting; seeded and exactly replayable | Works |
| Evaluation harness | N randomised episodes: success with Wilson 95% CIs, per class, per slot, failures by phase, cycle time, placement error, slip rate | Works |
| Demo video | 1280×720 MP4 with a live custody-log panel beside the simulation | Works (`out/demo.mp4`, ~86 s) |
| RL scaffold | Gymnasium env (Cartesian action space) and SB3 trainer | Plumbing verified; **no policy trained** (see below) |

## Measured results

Scripted controller, randomised episodes, code at commit `e893af9`:

| Run | Success | 95% CI (Wilson) | Grasp slips | Placement error (mean / p95) | Cycle time (mean / p95) |
|---|---|---|---|---|---|
| 100 episodes, seeds 0–99 | 100/100 | 96.3–100% | 0 | 1.3 / 3.3 mm | 12.6 / 13.9 s |
| 1000 episodes, seeds 1000–1999 | 1000/1000 | 99.6–100% | 0 | 1.3 / 3.5 mm | 12.7 / 14.2 s |

Per class (1000-episode run): box 230/230, bag 258/258, cylinder 253/253, folder 259/259. Every slot: 250/250. The
custody chain stayed intact across all 5000 events.

### Read these numbers with their conditions

- **The controller is scripted and reads each item's ground-truth pose from the simulator.** There is no
  perception and no pose noise. This is the biggest gap between these numbers and a real cell.
- **The test set comes from the design distribution.** The randomisation ranges were chosen so every item fits the
  Panda gripper and the bins, and the grasp heuristics were written for exactly these four classes.
- One item at a time on an uncluttered counter, always resting upright or flat.
- This is simulated contact physics. **Sim-to-real transfer is not measured.**
- The honest claim: *within this randomisation envelope, with perfect state information, the contact-physics
  pipeline plus the logging, evaluation and replay infrastructure is reliable.* It is not a claim about real-world
  grasp success, general-purpose grasping, or a learned policy.

An earlier 1000-episode run measured 996/1000. All four failures were a false negative in the grasp verifier, caused
by MuJoCo contact flicker; they were diagnosed by exact replay and fixed. The history is kept in `NOTES.md` (M5).

## Scope and modelling choices (disclosed)

- **Four item classes only:** rigid box (phone box), sealed bag, cylinder (bottle), flat folder/document.
  - The **folder** is a slim 18 × 6 × 2 cm document wallet. A real A4 folder lying flat cannot be grasped top-down by
    a parallel-jaw gripper at all; it would need suction or a different end effector.
  - The **bag** is a rigid ellipsoid with rolling and torsional friction standing in for a deformable bag.
  - The **bottle** stands upright and is at most ~14 cm tall. It is gripped as close to mid-height as the hand allows.
- **Gripper force was raised** from Menagerie's stock setting (~1.4 N per pad, which cannot lift 0.5 kg) to ~21 N per
  pad. The real Franka Hand is rated at 70 N continuous.
- **Gravity compensation is enabled on the arm links**, as on the real Franka controller. It is never applied to items.
- **The cabinet is four open-topped bins.** A closed cabinet with a horizontal shelf would make top-down insertion
  into the lower row impossible. Items are released from just above the bin walls, because the 208 mm-wide Panda
  hand does not fit inside a bin.
- **Only one item is physically simulated per episode.** The cabinet is physically empty at the start of every
  episode. Slot occupancy is tracked logically, so all four slots get exercised.
- **The demo video's final episode is a deliberate fault injection,** labelled as such on screen. The gripper is cut
  back to its stock force, so the video can show a failure being detected and logged. It is not a natural failure.
- **RL:** the Gymnasium environment passes `check_env`. A hand-written policy acting only through its 5-D action
  space solves 60/60 episodes, and SAC/PPO smoke runs checkpoint and reload correctly. Training a policy to
  convergence was not attempted; realistically that needs GPU-parallel simulation. `NOTES.md` (M7) has the estimate.

## Setup

Requirements: Apple Silicon Mac (or any machine that runs MuJoCo), Python 3.13.

```bash
pip3 install -r requirements.txt
```

The Franka Emika Panda model comes from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) and
is not committed. Fetch it with a sparse checkout:

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git /tmp/menagerie
cd /tmp/menagerie && git sparse-checkout set franka_emika_panda && cd -
cp -r /tmp/menagerie/franka_emika_panda/. ./models/panda/
```

The scene file `models/panda/evidence_room.xml` is committed. It must stay beside `panda.xml` so MuJoCo resolves
the asset paths. The build used Menagerie commit `367e3d9`.

Optional RL dependencies (these pull in PyTorch):

```bash
pip3 install -r requirements-rl.txt
```

## Usage

```bash
mjpython scripts/simulation.py                     # interactive viewer (macOS needs mjpython for the passive viewer)
python3 scripts/simulation.py --headless           # render demo_cam to out/scene.png
python3 scripts/controller.py                      # one scripted cycle, box -> slot_0, frames to out/m2_*.png
python3 scripts/episode.py --episodes 10 --seed 0  # randomised intake episodes, custody log to out/custody_log.jsonl
python3 scripts/evaluate.py --episodes 100 --seed 0  # evaluation table + out/eval_<timestamp>.json
python3 scripts/record.py                          # demo video to out/demo.mp4
python3 scripts/logger.py --verify out/custody_log.jsonl   # verify any custody log (exit code 1 if tampered)
python3 scripts/logger.py --demo                   # write a demo log, then catch a deliberate edit
python3 tests/test_custody_log.py                  # tamper-evidence tests
python3 rl/train.py                                # RL smoke test (SAC, 3000 steps) - plumbing only
```

To replay any evaluation episode exactly, take its seed and slot from the eval JSON:

```bash
python3 scripts/episode.py --episodes 1 --seed 1739 --slot slot_1
```

All outputs go to `out/`, which is gitignored.

## Repository layout

```
physical-ai-poc/
├── CLAUDE.md, BUILD_BRIEF.md   project context and build specification
├── NOTES.md                    decisions, failures found and fixed, measured results, open items
├── requirements.txt            mujoco, numpy, imageio, imageio-ffmpeg, pillow
├── requirements-rl.txt         gymnasium, stable-baselines3 (optional)
├── models/panda/               vendored Menagerie Panda (gitignored) + evidence_room.xml (committed)
├── scripts/
│   ├── scene.py                model loading (adds the tcp site, gripper tuning), name lookups, rendering
│   ├── simulation.py           interactive viewer / --headless render
│   ├── ik.py                   damped least-squares IK
│   ├── controller.py           pick-and-place state machine, grasp predicates
│   ├── randomise.py            in-place domain randomisation
│   ├── logger.py               hash-chained custody log
│   ├── slots.py                slot allocation
│   ├── episode.py              one full intake episode
│   ├── evaluate.py             N-episode evaluation
│   └── record.py               demo video with custody panel
├── rl/
│   ├── env.py                  Gymnasium environment
│   └── train.py                SB3 smoke-test trainer
└── tests/test_custody_log.py
```

## Next steps

1. Run the same evaluation with injected pose-estimation noise, then with wider and out-of-distribution
   randomisation, to find where it breaks.
2. Add a perception stage, so the controller no longer reads ground truth.
3. Move to GPU-parallel simulation (MJX / MuJoCo Warp) and train a policy bootstrapped from the scripted
   controller's demonstrations.
4. Close the sim-to-real loop on hardware.
