# PHASE 1 BRIEF — The evidence intake workflow

Follow-up to `BUILD_BRIEF.md` and `PHASE_0B_BRIEF.md`. Read `NOTES.md` first.

Phase 0 built a robot that picks objects and files them. This phase builds the robot that runs the **evidence intake workflow** — which is a different and more convincing claim, because the routing decision is made from the item itself rather than handed to the controller.

---

## 0. What is actually being modelled

The real process, as described by the customer:

1. An evidence room has a counter with an intake hatch. Officers leave evidence there; a handler collects it and files it into cabinets inside the room.
2. **The officer seals the evidence in a bag and applies a barcode label**, then places the sealed bag on the counter.
3. The robot picks it up and stores it in **the correct cabinet and the correct slot** within the room.

Two things follow from this, and both should shape the build.

**The robot never handles unsealed evidence.** The officer seals and labels; the robot moves a sealed, labelled unit and records the movement. Every chain-of-custody-critical act stays human. Keep it that way — it is a deliberate design property, not a limitation, and it should be stated in the overlay and the README.

**Routing is data-driven.** The destination comes from the barcode, through a case lookup, to a cabinet and slot. In Phase 0 the slot came from a counter in `slots.py`. That change is the whole point of this phase: the robot is executing a records process, not a motion sequence.

---

## 1. The governing design principle: fail closed

An evidence system that files the wrong item in the wrong slot is worse than one that refuses to file it at all. A refusal costs a handler two minutes. A misfile corrupts a chain of custody and may surface months later in court.

So, throughout this phase:

- If the barcode cannot be decoded, **do not guess, do not retry indefinitely, and never fall back to the simulator's ground-truth item ID**. Attempt a re-scan from a second pose, then log `REFUSED` with the reason and leave the item on the counter for a human.
- If the decoded ID has no matching case record, refuse the same way.
- If a placement cannot be verified, log the failure and do not write `VERIFIED`.

**Misfile rate is the headline safety metric for this phase** and it must be reported separately from success rate. The target is zero. The claim you want at the end is: *failures are always refusals, never errors.* That sentence is worth more to a police customer than any success percentage, so the code has to earn it.

An important consequence: the ground-truth item identity must be available to the *evaluator* (to detect a misfile) and must be structurally unavailable to the *controller*. Same boundary as the perception seam in Phase 0B — enforce it the same way, and put a comment saying so.

---

## 2. Task A — Sealed bags with contents

Replace the four object classes with a single unit type — **a sealed evidence bag** — whose contents vary. This is both more realistic and a better randomisation story: real intake variation is what's inside the bag, not whether it's a bottle or a folder.

**Geometry.** One body per bag with three geoms:

- **Outer bag** — the only colliding geom. Translucent polythene grey. Shape varies with contents.
- **Contents** — visual only (`contype="0" conaffinity="0" density="0"`), a coloured shape suggesting a phone, a knife, a garment, a small carton. This is what makes it read as evidence.
- **Label** — a thin white box on the top face carrying the barcode texture (Task B).

**Content classes** (keep four so the existing per-class reporting still works): `phone`, `blade`, `garment`, `carton`. Each drives the bag's outer dimensions, mass range, and — this is the useful part — **centre-of-mass offset**.

**Offset centre of mass is the main new physical difficulty and is worth doing properly.** Contents settle to one end of a bag; a grasp at the geometric centre then tips. Set `body_ipos` away from the geometric centre by a randomised offset (larger for `phone` and `carton`, smaller for `garment`), and keep the existing explicit inertia computation. This is a genuinely harder grasp than Phase 0's centred rigid bodies, it is realistic, and it will move the success numbers. That is expected.

Keep the existing randomisation machinery — size, mass, friction, pose, lighting — and the careful `bvh_aabb` / `geom_rbound` handling from M4. Add CoM offset to the randomised parameters and to the recorded episode params so replay still works.

---

## 3. Task B — The barcode, actually read

This is the first real perception in the pipeline. Do not stub it.

### B1. Generating the label

Generate a Code128 barcode encoding the item ID (`EV-2026-00XXXX`) plus a human-readable line beneath it, as a PNG.

**Texture swapping.** MuJoCo textures are fixed at compile time, so pre-generate a pool of label textures (24 is plenty) as separate materials in the XML, and select one per episode by writing `model.geom_matid[label_geom]`. This is robust and simple.

Runtime overwriting of `model.tex_data` plus `mjr_uploadTexture` also works and avoids the fixed pool, but it reaches into renderer internals and will be the first thing to break on a MuJoCo upgrade. Use it only if the fixed pool proves limiting, and say so in `NOTES.md`.

### B2. Reading it

**Scanner station, not wrist camera.** Mount a fixed camera above the intake zone looking down at the counter, and scan the bag *where it lies, before the pick*. This is how a real intake station works, it is geometrically simple (fixed distance, known orientation, label facing up), and it puts the routing decision before the motion — which is both the correct logical order and a much clearer story on video.

Add a second, lower-resolution **wrist camera** on the hand for the verify scan in B3.

Decode with **`zxing-cpp`** (`pip install zxing-cpp`). It ships self-contained wheels. Do not use `pyzbar` — it needs the native `zbar` library via Homebrew, and Homebrew on this machine has a broken Xcode toolchain.

Render the scanner view with `mujoco.Renderer`, pass the RGB array to the decoder, and use what comes back. If nothing decodes, that is a genuine scan failure — handle it per §1.

**Resolution is the thing that will bite you.** A Code128 needs roughly 2–3 pixels per narrow bar to decode reliably. Work out the pixels-per-bar the camera actually delivers at the label's size and distance before building anything on top of it: render one frame, decode it, and print the result. If it fails, the levers are label size, camera FOV, and render resolution, in that order. Get a clean decode on a static frame before wiring it into the state machine.

Measure and report the **scan success rate** as its own metric, including under the Phase 0B pose noise. A realistic read rate is a selling point; a claimed 100% is not credible to anyone who has used a barcode scanner.

### B3. Verify scan at the slot

Before release, re-scan with the wrist camera to confirm the item in the gripper is the item the system thinks it is holding. On mismatch or no-read, do not release into the slot — return to the counter and log a refusal.

This is cheap once B2 works, it is exactly the double-check a real custody system would implement, and it is the mechanism that makes the zero-misfile claim structural rather than lucky.

---

## 4. Task C — The rail and the cabinet bank

**Rail.** MJCF cannot reopen a body from an included file, so copy `panda.xml` to `panda_railed.xml` and nest `link0` inside a rail body carrying a slide joint along the cabinet bank. Add a position actuator with a realistic range (roughly ±0.9 m) and velocity limits. Keep `panda.xml` untouched so the Phase 0 scene still compiles and the baseline stays reproducible.

**Do not add the rail to the IK.** Treat it as a staging axis with its own `TRAVERSE` phase: move the rail to the target cabinet's station position, settle, then run the arm motion with the rail held. This is how rail-mounted industrial cells typically work, it avoids redundancy resolution entirely, and the visible traverse reads well on video. Adding an eighth DOF to the damped least-squares solver buys nothing here and risks the stability you already have.

**Cabinets.** Three lockers — `CAB-A`, `CAB-B`, `CAB-C` — along the rail, each with a 2×2 slot grid, reusing the Phase 0B locker geometry. Twelve addressable locations as `CAB-B/slot_2`.

**Routing.** Replace `slots.py`'s counter with a real routing table: case record → evidence category → cabinet, then first free slot within it. Route by category — narcotics, weapons, general — because it is what a real room does and it communicates instantly on screen. Keep the allocation policy behind one function; it is the kind of thing a customer will want to change.

Note the constraint already found in Phase 0B: the front bin walls cannot be raised because the 208 mm hand clears them by only about 12 mm at release. That applies to all three cabinets.

---

## 5. Task D — The extended state machine

```
IDLE → SCAN → ROUTE → APPROACH → DESCEND → CLOSE → LIFT
     → TRAVERSE → TRANSIT → VERIFY_SCAN → INSERT → RELEASE → RETREAT → HOME
```

New phases: `SCAN` (render, decode, or refuse), `ROUTE` (case lookup → cabinet and slot, or refuse), `TRAVERSE` (rail to station), `VERIFY_SCAN` (wrist re-scan, or return to counter).

Every phase keeps the Phase 0 discipline: a real completion condition plus a timeout, never advancing on elapsed time alone. Add `REFUSED` to the custody log's action vocabulary alongside `FAILED` — they are different outcomes and conflating them destroys the metric that matters. A refusal is the system working correctly.

Custody events gain the scan result: what was decoded, from which camera, at what time, and whether the verify scan matched.

---

## 6. Task E — Re-measure

The Phase 0B harness carries over. Report, at zero noise and across the existing position-error sweep:

- **Misfile rate** — an item released into a slot other than the one its true ID routes to. **Report this first.** Target zero.
- **Refusal rate**, broken down by cause: no decode, no case match, verify mismatch.
- **Completion rate** — filed correctly without human intervention.
- **Scan success rate**, intake and verify separately.
- Grasp success, slip rate, cycle time, placement error, per content class.

Expect completion to be lower than Phase 0's numbers. Offset centre of mass makes grasping genuinely harder and scanning adds a new failure mode. **Do not tune to recover the old number** — the same rule as Phase 0B. A lower completion rate with zero misfiles is a stronger result than a higher one without that guarantee, and it is a more honest description of what an evidence room would actually experience.

Run the same OOD sweep so the envelope stays characterised.

---

## 7. Task F — The video

The current demo shows an arm moving objects. This one should show a workflow. Beats, in order:

1. **The room** — establishing shot: counter with intake hatch, three lockers along the rail.
2. **Arrival** — a sealed, labelled bag appears at the hatch. Overlay names it as officer-sealed and officer-labelled, robot-handled.
3. **The scan** — inset showing the actual scanner camera view, with the decoded string appearing beside it. This is the money shot. It is the moment the demo stops being a manipulation video.
4. **The routing decision** — on the panel: barcode → case → category → `CAB-B`, `slot_2`.
5. **Traverse** — the arm visibly moves along the rail to the right locker.
6. **Verify scan** — wrist view, match confirmed, then release.
7. **The custody record** — the hash-chained log, as now.
8. **One refusal episode** — a damaged or unreadable label, the robot declining to file it, `REFUSED` written to the log, item left for a handler. Label it on screen.

That last beat is not a caveat, it is a feature demonstration. Showing the system refuse rather than guess is what a police customer needs to see, and it is the visible proof of the fail-closed principle.

Keep the stated pose error in the overlay, as Phase 0B established. Keep timestamped output filenames — do not overwrite existing videos.

---

## 8. Traps

**Barcode resolution.** Covered in B2. Get a static decode working before anything else in Task B. If it will not decode at a sensible camera distance, stop and report it rather than shrinking the problem until it works.

**The controller must not see ground truth.** Same boundary as Phase 0B, now extended to identity as well as pose. The evaluator needs the true ID to detect misfiles; the controller must only ever have the decoded one. If the controller can reach the true ID, the zero-misfile result means nothing.

**Visual-only geoms must stay massless.** The contents and label geoms need `density="0"` and no collision, or they will silently alter the mass and inertia that the randomisation is setting.

**Rail and arm should not move simultaneously** in this phase. Sequence them.

**Do not modify `panda.xml`.** Work in a copy so the Phase 0 baseline stays reproducible and the `m0`–`m7` tags keep compiling.

**Re-verify the zero-noise baseline** after the scene changes, as in Phase 0B Task C — though note it will not be bit-identical this time, since the objects themselves have changed. Say so rather than chasing it.

---

## 9. Definition of done

1. Evidence units are sealed bags with visible varied contents and offset centres of mass.
2. A real Code128 barcode is generated, rendered, and decoded from camera pixels — with the decode rate measured, not assumed.
3. Routing comes from the decoded ID through a case record to one of twelve locations across three cabinets.
4. The arm traverses a rail to reach the right cabinet.
5. A verify scan precedes every release.
6. Unreadable labels and mismatches produce logged refusals, never guesses.
7. Misfile rate is measured and reported first among the metrics.
8. A video that shows the workflow — scan, route, traverse, verify, file — including one refusal.
9. `NOTES.md` and `README.md` describe the workflow, the metrics and the new failure modes.

The claim this phase should earn is: *the robot executes the evidence intake process end to end, decides where items go by reading their labels, verifies before it commits, and refuses rather than guessing.* Nothing in the build should be done in a way that makes that sentence untrue.
