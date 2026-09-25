# Artefact manifest

`out/` is gitignored, so everything below exists only where it was generated. This file is the one exception that
is committed. It records what each artefact is, which command produced it, and from which code. Copy the files
listed under **Keep** somewhere durable; the checksums let a later copy be matched back to this manifest.

Tags `m0`–`m7` on the remote mark the Phase 0 milestone commits, and `milestones-backup` holds the full milestone
history. `main` carries the same code squashed into `cf84711` ("initial commit - start of project").

**Stamp caveat.** Evaluation JSONs record `git describe --always --dirty`. The `-dirty` suffix was only added at
`e893af9`, so earlier runs are stamped with the last commit even though they ran on uncommitted code. They are
flagged below.

## Keep

### Phase 0 headline evidence

| File | What it is | Produced by | Code | Size | SHA-256 (first 16) |
|---|---|---|---|---|---|
| ~~`demo.mp4`~~ | **Overwritten on 2026-09-25** by the Phase 0B re-record (same filename). The original (`543e5dcdda635c9b`, 16.1 MB) survives only if it was copied out before then. See the regenerated copy on the next row | `python3 scripts/record.py` | `670754e` (m6) | — | — |
| `demo_phase0_m6.mp4` | **Phase 0 video regenerated from tag `m6`** in a clean worktree: the same 6 episodes, outcomes and 85.7 s length. Physics and frames are deterministic; the custody-panel timestamps and hashes differ because they are wall-clock | `python3 scripts/record.py` at `m6` | `670754e` (m6) | 16.1 MB | `e3fcda7bdfd6021e` |
| `demo_phase0_m6_custody.jsonl` | Custody log of that regeneration, 28 events, chain intact | same run | `670754e` | 12.7 KB | — |
| `demo_20260923T125828Z_custody.jsonl` | The custody log written live during the original Phase 0 video. 28 events, chain intact, head hash `5020d00b…` | same run | `670754e` | 12.7 KB | `9a405d5679434b56` |
| `eval_20260923T124308Z.json` | **Headline evaluation: 1000/1000, seeds 1000–1999**, with per-episode params for exact replay | `python3 scripts/evaluate.py --episodes 1000 --seed 1000 --quiet` | `e893af9` (clean) | 931 KB | `514da2427cedafcc` |
| `eval_20260923T124308Z_custody.jsonl` | Its custody log, 5000 events, chain intact | same run | `e893af9` | 2.3 MB | `c5cb9310d35116c7` |
| `eval_20260923T124228Z.json` | Evaluation: 100/100, seeds 0–99 | `python3 scripts/evaluate.py --episodes 100 --seed 0 --quiet` | `e893af9` (clean) | 95 KB | `01db0af03bfd1d18` |
| `eval_20260923T124228Z_custody.jsonl` | Its custody log, 500 events | same run | `e893af9` | 227 KB | `63916e1de0e34e1b` |
| `m5_eval_1000.txt`, `m5_eval_100.txt` | Printed tables for the two runs above | stdout of those runs | `e893af9` | 49 KB / 6 KB | `31713e47a9a1b803` / `68653d1eba6b285f` |

### History worth keeping (it documents a failure found and fixed)

| File | What it is | Produced by | Code | Size | SHA-256 (first 16) |
|---|---|---|---|---|---|
| `eval_20260923T122409Z.json` | First 1000-episode run: **996/1000**. The 4 LIFT failures were contact-flicker false negatives in the grasp verifier (NOTES M5) | `evaluate.py --episodes 1000 --seed 1000` | stamped `19a7653`, actually **uncommitted** harness (pre-window verifier) | 932 KB | `0be4af5e457606cd` |
| `eval_20260923T122409Z_custody.jsonl` | Its custody log, including the 4 FAILED events | same run | same | 2.3 MB | `861b8a01d9d16482` |
| `m5_fail_1739.png`, `m5_fail_1485.png` | topdown_cam stills from replaying two of those false failures | ad-hoc replay script | uncommitted, between m4 and m5 | 138 / 129 KB | `2b14610bba2dd3a6` / `de180d9dc84b1ce2` |
| `custody_log.jsonl` | M4 canonical run: 10 randomised episodes, 50 events, 10/10 | `python3 scripts/episode.py --episodes 10 --seed 0 --render` | `19a7653` (m4) | 23 KB | `97547a1e44b04e9e` |
| `custody_log_demo.jsonl`, `custody_log_demo_tampered.jsonl` | Synthetic tamper demo: clean chain, and a copy with record 3's `slot_id` edited (`verify()` → 3) | `python3 scripts/logger.py --demo` | `11145da` | 4 KB each | `6d37838312f253a0` / `f0988fcc817226f5` |

### Stills

| File | What it is | Code |
|---|---|---|
| ~~`scene.png`~~ | Re-rendered in Task C, so it now shows the Phase 0B locker scene (`b908323`). `d_phase0_regen_last.png` shows the Phase 0 look | — |
| `m1_sites.png` | M1 scene with slot volumes and the TCP site visible. Predates the M2 backboard move | `40b0d85` (m1) |
| `m2_contact_sheet.png` | One frame per phase of the M2 box → slot_0 cycle | `3b61622` (m2) |
| `m6_frame_15s.png`, `m6_frame_last.png` | Stills from the original Phase 0 `demo.mp4` (episode boundary; the fault-injection FAILED event) | `670754e` |

### RL smoke-test outputs (plumbing evidence only; the policies are untrained)

| Path | What it is | Code |
|---|---|---|
| `rl/sac_20260923-213639/` | SAC, 3000 steps: checkpoints at 1k/2k/3k plus the final model, and `monitor.monitor.csv` (15 episodes, 0 successes). 13 MB | `11145da` (m7) |
| `rl/ppo_20260923-213709/` | PPO, 4096 steps: checkpoints plus the final model and monitor CSV (20 episodes, 0 successes). 0.8 MB | `11145da` |

## Safe to discard (superseded or scratch)

- `eval_20260923T122306Z*`, `eval_20260923T123449Z*`, `eval_20260923T123529Z*` and `m5_run*.txt`: pre-commit runs
  that the clean `e893af9` runs above supersede, with identical results.
- `eval_20260923T133933Z*`, `final_check_custody.jsonl`: an 8-episode end-of-build smoke check.
- `demo_20260923T125200Z/125250Z/125547Z_custody.jsonl`, `m6_test*`, `m6_contact_sheet.png`, `m6_frame_14s/76s/83s.png`:
  earlier takes of the video.
- `m2_0*.png`, `m2_1*.png`, `m4_*`, `scene_topdown_cam.png`: working frames used during verification.

## Phase 0B

`record.py` now defaults to `out/demo_<timestamp>.mp4`, so a re-record cannot overwrite an earlier take again.

### Keep

| File | What it is | Produced by | Code | Size | SHA-256 (first 16) |
|---|---|---|---|---|---|
| `demo_phase0b.mp4` (identical copy: `demo.mp4`) | **Phase 0B pitch video**, 85.9 s. New locker scene; the controller sees the item pose with σ 5 mm / 5° error (measured 97.5% [94.3, 98.9] at that level); 5 natural episodes (all succeed) plus the labelled fault injection | `python3 scripts/record.py --out out/demo.mp4` | Task D commit (only the default output filename changed after the run) | 14.8 MB | `9144a87369452b4e` |
| `demo_20260925T085524Z_custody.jsonl` | Its custody log, 28 events, chain intact | same run | same | 12.7 KB | `a0bd3939c2608246` |
| `robustness_20260925T083332Z.json` | **Robustness sweeps**: position, yaw, combined, size, OOD. 27 points × 200 episodes, every episode's parameters and drawn perception error | `python3 scripts/sweep.py` | `ed19f01` (clean) | 8.0 MB | `d9d9436b59670995` |
| `robustness_20260925T084337Z.json` | Yaw extension 30/45/60/90°, 4 × 200 episodes | `python3 scripts/sweep.py --only yaw_wide` | `ed19f01-dirty` (only the new `SWEEPS` entry) | 1.0 MB | `b8dd8726cdb18db9` |
| `sweep_20260925T083332Z/`, `sweep_20260925T084337Z/` | Per-point custody logs of those sweeps (chains intact) | same runs | same | 13 MB / 1.6 MB | — |
| `sweep_run.txt` | Printed progress of the main sweep | same run | `ed19f01` | 3 KB | `29622a3ad9b95200` |
| `robustness_curve.png` | **Headline chart**: success vs position σ, per class plus overall, Wilson bands, 3–10 mm band shaded | `python3 scripts/plot_robustness.py <both jsons>` | `15c23aa` | 175 KB | `66e85ad7a3855c3c` |
| `robustness_overview.png` | 2×2: position, yaw (incl. extension), combined, OOD | same | `15c23aa` | 541 KB | `e0f05dd8067e84da` |
| `robustness_yaw.png`, `robustness_combined.png`, `robustness_ood.png`, `robustness_size.png` | Individual sweep charts | same | `15c23aa` | 89–200 KB | `cc1066682a37dc40` / `aa319ffeabe242f1` / `ece356d15dcf85fc` / `6d5bf73e61fb9a71` |
| `eval_20260925T085152Z.json` | Task C zero-noise check, seeds 0–99: 100/100 | `evaluate.py --episodes 100 --seed 0 --quiet` | stamped `15c23aa-dirty`; the code equals `b908323` apart from record.py | 143 KB | `a69f271c995648d8` |

### Stills from verification (useful, not essential)

`b_zero_noise_failures.png` (the two zero-noise cylinder failures), `b_fail_*.png` (replayed noisy failures),
`c_demo_transit.png`, `c_cabinet_cam_bag.png`, `c_test_inset.png` (Task C scene and slot labels), `d_frame_20s.png`,
`d_contact_sheet.png` (Phase 0B video), `d_phase0_regen_last.png` (regenerated Phase 0 video).

### Safe to discard

`robustness_20260925T083234Z.json` and `sweep_20260925T083234Z/` (10-episode smoke test), `c_test.mp4`,
`eval_20260925T083020Z*` (12-episode noise trial).
