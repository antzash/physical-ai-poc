# PHASE 0B BRIEF — Robustness, provenance, presentation

Follow-up to `BUILD_BRIEF.md`. Phase 0 is complete and measured; read `NOTES.md` first for what was built and what its numbers mean.

This phase does three things: protect the work, find out where the system actually breaks, and make the demo read as an evidence room rather than a physics test.

---

## 0. Ground rules

The rules from `BUILD_BRIEF.md` §0 still apply in full — verify by rendering and looking, never guess model internals, no silent stubs, commit per task.

Three additions specific to this phase:

1. **Do not tune the controller to beat the noise.** Task B is a measurement, not an improvement. The moment you adjust grasp heuristics, approach clearances, or randomisation ranges to raise the number, the measurement is worthless. If you find a bug that produces false failures — as happened twice in Phase 0 with contact flicker — fix that and say so, but do not tune for performance. Tuning is a decision for after the curve exists.
2. **Keep per-task commits on `main` this time.** Phase 0's milestone commits were squashed away; do not repeat that.
3. **Expect the success rate to fall.** A curve that drops from 100% to near zero across the sweep is the successful outcome of this phase. A curve that stays flat at 100% means the noise is not reaching the controller, and you should debug that rather than report it.

---

## 1. Task A — Provenance and backup

Small, and it protects everything else.

1. Push the milestone history that currently exists only on this laptop:
   ```bash
   git push -u origin milestones-backup
   ```
2. Tag each milestone commit on that branch (`m0` … `m7`) with annotated tags and push them, so the build history is browsable rather than buried in a side branch.
3. `out/` is gitignored, which is correct — but that means `demo.mp4`, the evaluation JSONs and the stills exist in exactly one place. Create `out/ARTIFACTS.md` listing the artefacts worth preserving, what produced each one, and the commit they came from. The files themselves get copied out of the repo manually; the manifest is what makes that copy meaningful later.
4. Commit.

---

## 2. Task B — The perception seam and the robustness sweep

This is the substance of the phase.

### B1. Why

Every number in `NOTES.md` was produced with the controller reading the object's true pose straight out of the simulator. There is no perception and no pose error anywhere in the loop. That is why the success rate is 100%, and it is why the 100% does not currently support a claim about grasping — it supports a claim about the infrastructure.

The fix is one function. It is also the exact interface a vision model plugs into in a later phase, so this is architecture, not just test scaffolding.

### B2. The seam

Add `scripts/perception.py` exposing an `ItemObservation` — the pose and dimensions the controller is **allowed** to use:

```
observe(model, data, item_body, item_geom, noise_spec, rng) -> ItemObservation
    position    (3,)   true position + offset
    yaw         float  true yaw + offset
    size        (n,)   true geom size * scale error
```

Then route the controller's four ground-truth reads through it. They are:

| Location | Current read |
|---|---|
| `controller.py` `_enter_approach` (~line 273) | `self.d.xpos[self.item_body]` |
| `controller.py` `_enter_descend` (~line 301) | `self.d.xpos[self.item_body]` |
| `controller.py` `_item_yaw` (~line 218) | item body quaternion |
| `controller.py` `grasp_height` (~line 287) | `self.d.geom_xpos[g][2]` and the geom size |

`release_height()` also reads true geometry via `geom_lowest_z`; treat the item-derived part of it as an observation too, and leave the wall-derived part as ground truth (the cabinet's dimensions are known, not perceived).

### B3. Three details that decide whether this measures anything

**The noise perturbs the observation, never the physics.** The item stays exactly where it is. What changes is the robot's belief about where it is. If you move the actual object, the controller simply reads the new true pose and you have tested nothing.

**Draw the noise once per episode and cache it.** This is the easiest way to get a meaningless result. If fresh noise is sampled at every call site, the controller averages over it across APPROACH, DESCEND and grasp-height selection, and the errors largely cancel — you will measure something close to no noise at all. Real perception gives you one estimate with one fixed error that persists for the whole attempt. Sample the offset at episode reset, store it on the observation object, and apply the same offset every time. Record the drawn offset in the episode result so failures can be replayed exactly.

**The evaluator must keep using ground truth.** Noise belongs on the controller's input only. Everything that *judges* the outcome stays perfect:

- `grasp_ok`, `grasp_sample`, `grasp_verified`, `_check_hold`, `finger_contact_forces`
- the VERIFY checks in `episode.py` (inside slot volume, at rest)
- placement error

If the measurement inherits the same error as the controller, success and failure stop meaning anything. Put a comment to this effect at the top of `perception.py` — it is the kind of boundary that erodes silently during later edits.

### B4. The sweep

Extend `evaluate.py` with `--pos-noise-mm`, `--yaw-noise-deg` and `--size-noise-pct`, then add `scripts/sweep.py` to run the grid and aggregate.

Run three sweeps, 200 episodes per point, fresh seed range per point:

- **Position only** — σ ∈ {0, 2, 5, 8, 12, 16, 20} mm, yaw noise zero
- **Yaw only** — σ ∈ {0, 2, 5, 8, 12, 16, 20}°, position noise zero
- **Combined** — the paired values above

Running the axes separately first is worth the extra few minutes: it tells you *which* error your grasp heuristics are actually sensitive to, which is more actionable than a single blended curve. At 0.39s per episode the whole thing is roughly half an hour of compute.

Then an **out-of-distribution sweep** at zero noise: multiply the size and mass envelopes from `randomise.py` by 1.0, 1.25, 1.5, 1.75, 2.0 (as a runtime multiplier — do not edit the training ranges) and run 200 episodes at each. This finds the geometric cliff where items stop fitting the gripper or the slots, which is a different failure boundary from the perception one and worth knowing separately.

Add `--size-noise-pct` to the position sweep as a secondary axis only if time allows; dimension estimation error matters less than pose error for these grasp heuristics.

### B5. Outputs

`out/robustness_<ts>.json` with per-point results, plus a chart at `out/robustness_curve.png`:

- x-axis: noise σ. y-axis: success rate, 0–100%.
- One line per object class, plus overall in a heavier weight.
- Wilson CI as a shaded band per series — the intervals matter at 200 episodes and the chart should not hide that.
- Shade the 3–10 mm region and label it as the range a competent RGB-D pose estimator on a known rigid object would deliver. This is what turns the chart from a display into an argument: it shows at a glance whether the grasp tolerance is inside or outside what real perception could supply.
- Light gridlines, legible axis labels, readable when dropped into a slide at half width.

Add `matplotlib` to `requirements.txt`.

Write the headline findings into `NOTES.md` as a new section: the σ at which overall success drops below 95%, below 80%, and below 50%; which object class degrades first and the likely reason; which axis (position or yaw) dominates; and where the OOD cliff sits.

**Report whatever the curve shows.** If it collapses at 5 mm, that is the finding, and knowing it is worth considerably more than not knowing it. A stated failure boundary is a stronger technical position than an unqualified success rate, and it is the thing that makes the next phase's priorities obvious.

Commit.

---

## 3. Task C — Make it read as an evidence room

Cosmetic, roughly an hour, and it closes most of the perceived-quality gap. All in `models/panda/evidence_room.xml` unless noted. None of this may change the physics — verify by re-running 100 evaluation episodes at zero noise and confirming the result still matches the Phase 0 baseline.

**Cabinet.** Currently reads as a translucent blue tray: the panels are 5 mm half-thickness and the front and sides are only 10 cm tall, so at the demo camera angle it looks like wireframe. Thicken the panels, raise the sides and front, and switch `cabinet_body` and `cabinet_panel` to an opaque institutional grey with low specularity. It should read as a steel locker. While you are there, confirm the group-4 slot sites are not being rendered in `demo_cam` — check the render, not the XML.

**Floor.** The checkerboard says "physics demo". Replace `floor_tex` with a flat matte material.

**Evidence tags.** Add a small thin white box geom to each item body, positioned on the item's top face. A white tag on a brown carton reads as *tagged evidence* instantly, in a way colour tuning cannot achieve.

Two constraints, both of which will corrupt the measurement if missed: set `contype="0" conaffinity="0"` so the tag never collides, and `density="0"` so it contributes no mass or inertia. The tag will not scale with the item under randomisation — note that in `NOTES.md` rather than trying to solve it.

**Slot labels.** Draw `SLOT 0`–`SLOT 3` into the cabinet inset in `record.py`'s overlay, not into the scene. The panel already knows the slot ID.

**Item palette.** Mostly fine already. The cylinder reads as a green canister rather than a bottle — amber or brown glass would sit better next to the kraft carton and manila folder.

Render `demo_cam` and `cabinet_cam` and look at both before committing.

---

## 4. Task D — Re-record and update the documentation

Do this **after** Task B, so the overlay can state the real robustness position rather than being re-rendered twice.

**Video.** Re-run `record.py` with the new scene. Keep the structure that works — natural episodes plus the labelled fault injection. Update the header overlay so it states the conditions honestly: contact physics only, one item per episode, and now also whether the run uses perfect state or a stated pose noise. If the curve shows the system survives realistic noise, consider recording the demo *with* noise enabled and saying so on screen — a demo that works under stated perception error is a much stronger artefact than one that assumes perfect state.

**Documentation.**
- `NOTES.md` — the robustness section from B5, plus the visual changes and their verification.
- `README.md` — replace the bare 100% with the conditions and the robustness result. The honest headline is roughly: *reliable under this randomisation envelope with perfect state; degrades beyond X mm of pose error; no perception, no learned policy, simulation only.*
- `out/ARTIFACTS.md` — add the new video, curve and sweep JSON.

Commit.

---

## 5. Definition of done

1. Milestone history and tags pushed to the remote; artefact manifest written.
2. A perception seam exists, noise reaches the controller and only the controller, and the evaluator is provably still using ground truth.
3. Three noise sweeps and one OOD sweep completed, with a chart and a written statement of where the system breaks.
4. The scene reads as an evidence room, with the zero-noise baseline unchanged.
5. A re-recorded demo whose on-screen claims match what was actually measured.
6. `NOTES.md` and `README.md` describe the system as it now is, including the new failure boundary.

The output of this phase is not a better number. It is a known one.
