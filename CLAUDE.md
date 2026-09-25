# Physical AI — Evidence Room POC

## Project Context
This is a Physical AI startup POC. The company builds sim-to-real robotic manipulation pipelines for high-stakes environments. The first use case is police evidence intake and storage — replacing the officer at the evidence room counter who manually logs incoming evidence and stores it in cabinets.

The core thesis:
- Train robotic policies in simulation using domain randomisation and reinforcement learning
- Close the sim-to-real gap through iterative feedback between simulation and real deployment
- The training pipeline itself is the IP, not the hardware

The customer's process: an officer **seals the evidence in a bag and applies a barcode label**, then leaves it at the intake hatch; the robot reads the label, files the bag into the correct cabinet and slot, and records the movement. The robot never handles unsealed evidence — every chain-of-custody-critical act stays human, by design.

## Developer Setup
- MacBook Air M2 (Apple Silicon, 4 performance + 4 efficiency cores) — no NVIDIA GPU, CPU only
- Python 3.13, MuJoCo 3.x (3.13 in use) via pip3; zxing-cpp for barcodes (not pyzbar: Homebrew/zbar is broken here)
- No physical robotic arm — simulation only
- Franka Panda from MuJoCo Menagerie, vendored into `models/panda/` (gitignored; clone commands in README)

## What exists (phases)
- **Phase 0** (tags `m0`–`m7`): scene, DLS IK, scripted contact-physics pick-and-place, hash-chained custody log, domain randomisation, evaluation harness, demo video, RL scaffold.
- **Phase 0B** (tag `phase0b`): perception seam with pose-noise injection, robustness sweeps, locker visuals.
- **Phase 1** (current, `PHASE_1_BRIEF.md`): sealed evidence bags with contents and offset centre of mass; real Code128 labels read from rendered pixels; rail-mounted arm and three lockers (CAB-A narcotics, CAB-B weapons, CAB-C general, 12 slots); routing from the decoded barcode through a case record; verify scan before every release; refusals instead of guesses.

`NOTES.md` is the authoritative log of decisions, bugs found, and every measured number with its conditions. Read it before changing anything. `BUILD_BRIEF.md`, `PHASE_0B_BRIEF.md`, `PHASE_1_BRIEF.md` are the specs.

## Repo Structure
```
physical-ai-poc/
├── CLAUDE.md, README.md, NOTES.md, BUILD_BRIEF.md, PHASE_0B_BRIEF.md, PHASE_1_BRIEF.md
├── requirements.txt, requirements-rl.txt
├── docs/                         headline chart(s) for the README
├── models/panda/                 Menagerie Panda (gitignored) plus committed files:
│   ├── evidence_intake.xml       Phase 1 scene (rail, counter + hatch, 3 lockers, bags, cameras)
│   ├── panda_railed.xml          copy of panda.xml with link0 on a rail carriage (panda.xml is untouched)
│   ├── evidence_room.xml         Phase 0/0B scene, kept unchanged (runs with code at tag phase0b)
│   └── labels/                   generated barcode label textures (scripts/labels.py)
├── scripts/
│   ├── scene.py                  model loading (tcp site, wrist cameras, gripper tuning), names, rail/cabinet constants
│   ├── simulation.py             interactive viewer (mjpython) / --headless render
│   ├── ik.py                     damped least-squares IK (arm only; the rail is never in the IK)
│   ├── controller.py             IDLE→SCAN→ROUTE→…→TRAVERSE→…→VERIFY_SCAN→INSERT→…→HOME state machine
│   ├── perception.py             pose seam: the only item pose the controller may use (+ cached noise)
│   ├── scanner.py                omnidirectional Code128 decoding from camera pixels; robot cameras
│   ├── labels.py                 label generation and the label pool
│   ├── routing.py                case DB, category → cabinet, slot policy, CabinetBank; fails closed
│   ├── randomise.py              in-place domain randomisation (size, mass, CoM offset, friction, pose, light)
│   ├── logger.py                 hash-chained custody log (REFUSED distinct from FAILED)
│   ├── episode.py                world + evaluator for one intake (true identity lives here only)
│   ├── evaluate.py, sweep.py     N-episode evaluation (misfiles first) and robustness sweeps
│   ├── plot_robustness.py, plot_phase1.py, scan_degradation.py
│   └── record.py                 workflow demo video with the live custody panel
├── rl/                           Gymnasium env (6-D action incl. rail) + SB3 smoke-test trainer
├── tests/                        custody log, perception seam, routing, workflow boundary tests
└── out/                          generated artefacts (gitignored except out/ARTIFACTS.md)
```

## Rules that govern the build
- **Fail closed.** If a barcode cannot be read, a case record is missing, a cabinet is full or a verify scan mismatches: log `REFUSED`, leave the item for a human. Never guess, never retry indefinitely, never fall back to simulator ground truth. Misfile rate is the headline safety metric and is reported first.
- **Two boundaries, enforced the same way.** The controller may see item pose only through `perception.py` and item identity only through decoded camera pixels (`scanner.py` → `routing.route`). Ground truth (true pose, true item ID) belongs to the evaluator in `episode.py`. `tests/test_perception.py` and `tests/test_workflow.py` check this; keep them passing.
- **Measurements are not tuning.** During a sweep, do not change heuristics, clearances or ranges to raise a number. Fix genuine false-failure bugs (and say so); tune only after the curve exists.
- **Verify by rendering and looking**, not by assumption; never guess model internals (print them); no silent stubs.
- **Visual-only geoms** (`contype="0" conaffinity="0" density="0"`) must stay massless and non-colliding.
- **Do not modify `models/panda/panda.xml`.** Work in copies so the Phase 0 baseline and tags stay reproducible.
- **Rail and arm never move simultaneously**; the rail is a staging axis with its own TRAVERSE phase.
- **Timestamped output filenames**; never overwrite an existing artefact. Record provenance in `out/ARTIFACTS.md`.
- **Commit per task/milestone on `main`**; don't squash history. Honest scope: four content classes, simulation only, scripted controller, no learned policy — do not overclaim in code, docs or video.
