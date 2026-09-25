"""Record the Phase 1 workflow video: scan -> route -> traverse -> verify -> file, plus one labelled refusal.

    python3 scripts/record.py                  # out/demo_<ts>.mp4, pose error sigma 5 mm on the controller's input
    python3 scripts/record.py --pos-noise-mm 0 # perfect-state recording

Left 860 px: the room from a camera that tracks the rail carriage, 1x simulated time, briefly paused at the scan,
routing and verify moments to show them. Bottom-left inset: the ACTUAL scanner / wrist camera pixels the decoder read,
or the target locker. Right 420 px: the custody panel drawn from the real log as it is written. Every claim on screen
is read from data: the noise level's measured completion and misfile counts come from the Phase 1 sweep JSON.
Demo seeds are chosen only so the three lockers are all visited (never by outcome); the last episode is a deliberate,
labelled test item with a damaged label, to show a refusal. Timestamped filenames: nothing is overwritten.
"""

import argparse
import json
from datetime import datetime, timezone

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import scene
from episode import IntakeStation
from logger import verify_file
from perception import NoiseSpec
from routing import CASE_DB, CATEGORY_CABINET

FPS = 30
W, H = 1280, 720
SIM_W = 860
PANEL_W = W - SIM_W
INSET_W, INSET_H = 320, 214
SEED_BASE = 51000
DEMO_POS_NOISE_MM = 5.0

BG = (13, 19, 33)
CARD = (27, 36, 54)
CARD_NEW = (38, 50, 74)
TEXT = (226, 232, 240)
MUTED = (140, 152, 172)
FAINT = (88, 100, 122)
ACCENT = (96, 165, 250)
OK_GREEN = (52, 211, 153)
AMBER = (245, 170, 30)
RED = (239, 68, 68)
BADGE = {"SUBMITTED": (100, 116, 139), "REGISTERED": (59, 130, 246), "PICKED": (245, 158, 11),
         "PLACED": (168, 85, 247), "VERIFIED": (34, 197, 94), "FAILED": (220, 50, 50), "REFUSED": (217, 150, 20)}
STAGES = [("SCAN", ("IDLE", "SCAN")), ("ROUTE", ("ROUTE",)),
          ("PICK", ("APPROACH", "DESCEND", "CLOSE", "LIFT")), ("TRAVERSE", ("TRAVERSE", "TRANSIT")),
          ("VERIFY", ("VERIFY_SCAN",)), ("FILE", ("INSERT", "RELEASE", "RETREAT", "HOME"))]


def _font(path, size, index=0):
    try:
        return ImageFont.truetype(path, size, index=index)
    except OSError:
        return ImageFont.load_default()


SANS, MONO = "/System/Library/Fonts/Helvetica.ttc", "/System/Library/Fonts/Menlo.ttc"
F = {"title": _font(SANS, 22, 1), "sub": _font(SANS, 11), "label": _font(SANS, 10, 1), "value": _font(MONO, 14, 1),
     "value_s": _font(MONO, 12), "badge": _font(SANS, 11, 1), "body": _font(SANS, 11), "mono_s": _font(MONO, 10),
     "overlay": _font(SANS, 15, 1), "overlay_s": _font(SANS, 12), "banner": _font(SANS, 17, 1),
     "callout": _font(SANS, 20, 1), "decoded": _font(MONO, 22, 1), "pill": _font(SANS, 10, 1)}


def _trunc(d, text, font, width):
    if d.textlength(text, font=font) <= width:
        return text
    while text and d.textlength(text + "…", font=font) > width:
        text = text[:-1]
    return text + "…"


def measured_claim(pos_mm):
    """Completion and misfiles measured at this pose-error level in the Phase 1 sweep (read, never typed in)."""
    best = None
    for path in sorted(scene.OUT_DIR.glob("robustness_*.json"), reverse=True):
        data = json.loads(path.read_text())
        pts = [p for p in data["points"] if "misfile" in p["summary"]]
        if not pts:
            continue  # a Phase 0B file: no misfile metric
        at = [p for p in pts if p["sweep"] == "position" and p["point"].get("pos_mm") == pos_mm]
        allpos = [p for p in pts if p["sweep"] in ("position", "baseline")]
        if at:
            c = at[0]["summary"]["completion"]
            mis = sum(p["summary"]["misfile"]["k"] for p in allpos)
            n = sum(p["summary"]["misfile"]["n"] for p in allpos)
            best = f"measured: {100 * c['rate']:.1f}% filed at this level; {mis} misfiles in {n} sweep episodes"
            break
    return best or "not measured at this level"


class Panel:
    def __init__(self):
        self.first_seen = {}

    def draw(self, events, info, t, chain_ok, n_events):
        img = Image.new("RGB", (PANEL_W, H), BG)
        d = ImageDraw.Draw(img)
        pad = 16
        d.rectangle([0, 0, PANEL_W, 60], fill=(9, 13, 24))
        d.text((pad, 10), "CHAIN OF CUSTODY", font=F["title"], fill=TEXT)
        d.text((pad, 38), "evidence intake · ROBOT:arm-01 · append-only SHA-256 hash chain", font=F["sub"], fill=MUTED)

        # Workflow strip.
        y = 70
        phase = info.get("phase", "")
        # After a refusal, flag the stage that refused (not wherever the arm is while it returns home).
        at = info.get("refused_phase", phase)
        cur = next((i for i, (_, ph) in enumerate(STAGES) if at in ph), None)
        x = pad - 4
        for i, (name, _) in enumerate(STAGES):
            tw = d.textlength(name, font=F["pill"]) + 14
            if info.get("refused") and i == cur:
                fill, col = RED, (255, 255, 255)
            elif cur is not None and i < cur:
                fill, col = (30, 70, 60), OK_GREEN
            elif i == cur:
                fill, col = ACCENT, (255, 255, 255)
            else:
                fill, col = CARD, FAINT
            d.rounded_rectangle([x, y, x + tw, y + 20], radius=10, fill=fill)
            d.text((x + 7, y + 4), name, font=F["pill"], fill=col)
            x += tw + 5

        # Item and routing card.
        y = 100
        d.rounded_rectangle([pad - 6, y, PANEL_W - pad + 6, y + 148], radius=8, fill=CARD)
        rows = [("ITEM (decoded)", info.get("item_id", "PENDING-SCAN")), ("CASE", info.get("case_id", "—")),
                ("CATEGORY", info.get("category", "—")), ("DESTINATION", info.get("location") or "—")]
        for k, (lab, val) in enumerate(rows):
            xx, yy = pad + 2 + (k % 2) * 196, y + 8 + (k // 2) * 44
            d.text((xx, yy), lab, font=F["label"], fill=MUTED)
            d.text((xx, yy + 14), val, font=F["value"], fill=RED if val.startswith("REFUSED") else TEXT)
        route = info.get("route_line")
        d.text((pad + 2, y + 96), "ROUTING", font=F["label"], fill=MUTED)
        d.text((pad + 2, y + 110), _trunc(d, route or "waiting for the barcode", F["value_s"], PANEL_W - 2 * pad),
               font=F["value_s"], fill=TEXT if route else FAINT)
        d.text((pad + 2, y + 128), f"robot: {phase}", font=F["mono_s"], fill=MUTED)

        # Event log, newest at the bottom.
        y0, y1 = 258, H - 58
        d.text((pad, y0), "EVENT LOG", font=F["label"], fill=MUTED)
        card_h, gap = 62, 6
        shown = events[-((y1 - y0 - 18) // (card_h + gap)):]
        yy = y1 - len(shown) * (card_h + gap)
        for ev in shown:
            age = t - self.first_seen.setdefault(ev["event_id"], t)
            fresh = age < 1.2
            col = BADGE[ev["action"]]
            top = yy + int(max(0.0, 1 - age / 0.25) * 24)
            d.rounded_rectangle([pad - 6, top, PANEL_W - pad + 6, top + card_h], radius=7,
                                fill=CARD_NEW if fresh else CARD, outline=col if fresh else None, width=2)
            d.rectangle([pad - 6, top + 6, pad - 3, top + card_h - 6], fill=col)
            bw = d.textlength(ev["action"], font=F["badge"]) + 12
            d.rounded_rectangle([pad + 3, top + 7, pad + 3 + bw, top + 23], radius=4, fill=col)
            d.text((pad + 9, top + 9), ev["action"], font=F["badge"], fill=(255, 255, 255))
            d.text((pad + 10 + bw, top + 9), ev["actor"], font=F["body"], fill=TEXT)
            ts = ev["timestamp"][11:23]
            d.text((PANEL_W - pad - d.textlength(ts, font=F["mono_s"]), top + 11), ts, font=F["mono_s"], fill=MUTED)
            line = f"{ev['item_id']} · {ev['slot_id'] or '—'} · {ev['detail']}"
            d.text((pad + 3, top + 28), _trunc(d, line, F["body"], PANEL_W - 2 * pad - 6), font=F["body"], fill=MUTED)
            prev = f"#{ev['prev_hash'][:12]}" if ev["prev_hash"] else "genesis"
            d.text((pad + 3, top + 45), f"#{ev['hash'][:12]}  ←  prev: {prev}", font=F["mono_s"], fill=FAINT)
            yy += card_h + gap

        d.rectangle([0, H - 50, PANEL_W, H], fill=(9, 13, 24))
        if chain_ok:
            d.line([(pad, H - 34), (pad + 5, H - 29), (pad + 13, H - 39)], fill=OK_GREEN, width=3)
            d.text((pad + 20, H - 42), f"chain verified · {n_events} events", font=F["overlay"], fill=OK_GREEN)
        else:
            d.text((pad, H - 42), "CHAIN BROKEN", font=F["overlay"], fill=RED)
        d.text((pad, H - 20), "every record hashes the one before it; any edit breaks the chain", font=F["sub"],
               fill=FAINT)
        return img


def _inset_from(img, region, pad_frac=0.6):
    """Crop the decoded region (plus context) from a camera frame, fitted to the inset size."""
    h, w = img.shape[:2]
    if region:
        x0, y0, x1, y1 = region
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        half_w = max(x1 - x0, (y1 - y0) * INSET_W / INSET_H) * (0.5 + pad_frac)
        half_h = half_w * INSET_H / INSET_W
        box = (int(max(0, cx - half_w)), int(max(0, cy - half_h)), int(min(w, cx + half_w)), int(min(h, cy + half_h)))
        img = img[box[1]:box[3], box[0]:box[2]]
    return np.asarray(Image.fromarray(img).resize((INSET_W, INSET_H), Image.LANCZOS))


def compose_sim(frame, lines, inset=None, inset_title=None, inset_caption=None, caption_col=TEXT, banner=None,
                banner_col=(185, 28, 28), callout=None):
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img, "RGBA")
    bw = max(d.textlength(lines[0], font=F["overlay"]), *(d.textlength(t, font=F["overlay_s"]) for t in lines[1:])) + 22
    d.rounded_rectangle([10, 10, 10 + bw, 10 + 24 + 17 * (len(lines) - 1) + 8], radius=6, fill=(9, 13, 24, 200))
    d.text((20, 15), lines[0], font=F["overlay"], fill=TEXT)
    for i, line in enumerate(lines[1:]):
        d.text((20, 37 + 17 * i), line, font=F["overlay_s"], fill=MUTED)
    if inset is not None:
        x0, y0 = 10, H - INSET_H - 40
        img.paste(Image.fromarray(inset), (x0, y0))
        d.rectangle([x0, y0, x0 + INSET_W, y0 + INSET_H], outline=(255, 255, 255, 170), width=1)
        tw = d.textlength(inset_title, font=F["label"]) + 12
        d.rectangle([x0, y0, x0 + tw, y0 + 17], fill=(9, 13, 24, 210))
        d.text((x0 + 6, y0 + 3), inset_title, font=F["label"], fill=TEXT)
        if inset_caption:
            d.rectangle([x0, y0 + INSET_H, x0 + INSET_W, y0 + INSET_H + 30], fill=(9, 13, 24, 230))
            d.text((x0 + 8, y0 + INSET_H + 4), inset_caption, font=F["decoded"], fill=caption_col)
    if callout:
        cw = d.textlength(callout, font=F["callout"]) + 28
        cx = INSET_W + 30 + (SIM_W - INSET_W - 30 - cw) // 2
        d.rounded_rectangle([cx, H - 132, cx + cw, H - 96], radius=8, fill=(9, 13, 24, 225))
        d.text((cx + 14, H - 125), callout, font=F["callout"], fill=TEXT)
    if banner:
        bw2 = d.textlength(banner, font=F["banner"]) + 28
        bx = INSET_W + 30 + (SIM_W - INSET_W - 30 - bw2) // 2
        d.rounded_rectangle([bx, H - 76, bx + bw2, H - 42], radius=6, fill=(*banner_col, 235))
        d.text((bx + 14, H - 68), banner, font=F["banner"], fill=(255, 255, 255))
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", help="default out/demo_<timestamp>.mp4")
    ap.add_argument("--pos-noise-mm", type=float, default=DEMO_POS_NOISE_MM)
    ap.add_argument("--seed-base", type=int, default=SEED_BASE)
    args = ap.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = args.out or str(scene.OUT_DIR / f"demo_{stamp}.mp4")
    log_path = scene.OUT_DIR / f"demo_{stamp}_custody.jsonl"
    st = IntakeStation(log_path, echo=True)
    m, d = st.m, st.d
    noise = NoiseSpec(args.pos_noise_mm / 1000, 0.0)
    noise_line = ("pose: perfect state (controller reads the true pose)" if noise.is_zero else
                  f"pose error σ {args.pos_noise_mm:g} mm on the controller's input · {measured_claim(args.pos_noise_mm)}")

    # Seeds: the first natural seeds (from the base) whose item routes to CAB-A, CAB-B and CAB-C, chosen by
    # destination only, never by outcome. Then one deliberate test item with a damaged label.
    plan, need = [], ["CAB-A", "CAB-B", "CAB-C"]
    seed = args.seed_base
    while need:
        _, true_id, _ = st._choose_item(seed, None)
        cab = CATEGORY_CABINET[CASE_DB[true_id].category]
        if cab in need:
            need.remove(cab)
            plan.append((seed, None))
        seed += 1
    plan.sort(key=lambda s: ["CAB-B", "CAB-A", "CAB-C"].index(
        CATEGORY_CABINET[CASE_DB[st._choose_item(s[0], None)[1]].category]))
    plan.append((seed, "damaged"))

    renderer = mujoco.Renderer(m, H, SIM_W)
    cab_r = mujoco.Renderer(m, INSET_H, INSET_W)
    cam = mujoco.MjvCamera()
    panel = Panel()
    writer = imageio.get_writer(out_path, fps=FPS, codec="libx264", quality=8, pixelformat="yuv420p",
                                macro_block_size=16)
    S = {"t": 0.0, "acc": 0.0, "n": 0, "chain": True, "ep": 0, "seen_events": 0, "info": {}, "inset": None,
         "inset_title": None, "caption": None, "caption_col": TEXT, "banner": None, "banner_col": (185, 28, 28),
         "test_item": False, "cam_y": 0.0, "intro": 0.0}

    def set_camera(t_intro=None):
        rail = float(d.qpos[m.jnt_qposadr[scene.joint_id(m, scene.RAIL_JOINT)]])
        S["cam_y"] += 0.08 * (0.55 * rail + 0.05 - S["cam_y"])  # smooth follow of the carriage
        u = 1.0 if t_intro is None else min(1.0, t_intro / 3.0)
        u = u * u * (3 - 2 * u)
        # From inside the room, behind the rail: lockers along the bank, intake counter and hatch beyond.
        cam.lookat[:] = [0.35, S["cam_y"] if t_intro is None else 0.1, 0.12]
        cam.distance = 3.6 + u * (2.5 - 3.6)
        cam.azimuth = -40 + u * (-22 + 40)
        cam.elevation = -28 + u * (-36 + 28)

    def lines():
        done = S["filed"]
        return ["MuJoCo simulation · rail-mounted Franka Panda · 1× real time, paused at each scan",
                f"episode {S['ep']}/{len(plan)} · filed {done} · the officer seals and labels; the robot moves sealed bags only",
                "destination from the barcode it reads · verify scan before every release · refuses rather than guesses",
                noise_line,
                "contact physics only · one item per episode (lockers reset between episodes)"]

    def compose(callout=None):
        renderer.update_scene(d, camera=cam)
        sim = renderer.render()
        events = st.log.tail(12)
        if len(st.log) != S["n"]:
            S["n"] = len(st.log)
            S["chain"] = verify_file(log_path) is None
        frame = compose_sim(sim, lines(), S["inset"], S["inset_title"], S["caption"], S["caption_col"],
                            S["banner"], S["banner_col"], callout)
        out = Image.new("RGB", (W, H))
        out.paste(frame, (0, 0))
        out.paste(panel.draw(events, S["info"], S["t"], S["chain"], S["n"]), (SIM_W, 0))
        return np.asarray(out)

    def emit(frame, n=1):
        for _ in range(n):
            writer.append_data(frame)
            S["t"] += 1 / FPS

    def hold(seconds, callout=None):
        for _ in range(int(seconds * FPS)):
            emit(compose(callout))

    def on_new_event(c, ev):
        info = S["info"]
        if ev.kind == "scanned":
            rec = c.scan_info.get("rescan") or c.scan_info["intake"]
            img = st.cameras.last_image[rec["camera"]]
            S["inset"] = _inset_from(img, rec["region"], 0.3)
            S["inset_title"] = f"{rec['camera'].upper()} · ACTUAL PIXELS DECODED"
            S["caption"], S["caption_col"] = rec["decoded"], OK_GREEN
            info["item_id"] = rec["decoded"]
            hold(2.8, "barcode read from camera pixels")
        elif ev.kind == "scan_retry":
            rec = c.scan_info["intake"]
            S["inset"] = _inset_from(st.cameras.last_image[rec["camera"]], None, 0)
            S["inset_title"], S["caption"], S["caption_col"] = "SCANNER_CAM", "no read", AMBER
            hold(1.8, "no read at the intake scanner: re-scan from a second pose")
        elif ev.kind == "routed":
            rec = ev.data["record"]
            info.update(case_id=rec.case_id, category=rec.category, location=ev.data["location"],
                        route_line=f"{info['item_id']} → {rec.case_id} → {rec.category} → {ev.data['location']}")
            hold(2.2, f"routed by case category: {rec.category}, to {ev.data['location']}")
            S["inset"] = None
        elif ev.kind == "verified":
            rec = c.scan_info.get("verify_retry") if c.scan_info.get("verify_retry") else c.scan_info["verify"]
            S["inset"] = _inset_from(st.cameras.last_image[rec["camera"]], rec["region"], 0.35)
            S["inset_title"] = f"{rec['camera'].upper()} · VERIFY SCAN"
            match = ev.data["match"]
            S["caption"], S["caption_col"] = (f"{rec['decoded']}  MATCH" if match else f"{rec['decoded']}  MISMATCH",
                                              OK_GREEN if match else RED)
            hold(2.0, "verify scan at the slot: same item, release" if match else "verify scan mismatch")
            S["inset"] = None
        elif ev.kind == "refused":
            info["refused"] = True
            info["refused_phase"] = ev.phase
            info["location"] = "REFUSED"
            info["route_line"] = f"REFUSED ({ev.data.get('cause')}): {ev.detail}"
            S["banner"], S["banner_col"] = "REFUSED: not filed, left on the counter for a handler", (185, 28, 28)
            hold(3.0, "the robot declines to guess")
        elif ev.kind == "grasp_verified":
            loc = info.get("location")
            if loc and loc != "REFUSED":
                cab = scene.cabinet_of(loc)
                S["cab_cam"] = scene.cabinet_camera(cab)

    def frame_cb(stn, c):
        S["acc"] += m.opt.timestep
        if c is not None:
            while S["seen_events"] < len(c.events):
                ev = c.events[S["seen_events"]]
                S["seen_events"] += 1
                if ev.kind == "phase":
                    S["info"]["phase"] = ev.phase
                    if ev.phase in ("INSERT", "RELEASE") and S.get("cab_cam"):
                        pass
                else:
                    set_camera()
                    on_new_event(c, ev)
            if c.phase in ("INSERT", "RELEASE", "RETREAT") and S.get("cab_cam"):
                cab_r.update_scene(d, camera=S["cab_cam"])
                S["inset"], S["inset_title"] = cab_r.render(), f"{S['cab_cam'].replace('cabinet_cam_', 'CAB-').upper()}"
                S["caption"] = None
        if S["acc"] >= 1 / FPS:
            S["acc"] -= 1 / FPS
            set_camera()
            emit(compose())

    S["filed"] = 0
    for i, (seed, label) in enumerate(plan):
        S.update(ep=i + 1, seen_events=0, inset=None, caption=None, banner=None, cab_cam=None,
                 info={"phase": "IDLE", "item_id": "PENDING-SCAN"})
        if label == "damaged":
            S["banner"], S["banner_col"] = "DELIBERATE TEST ITEM: label damaged by the depositor", (120, 90, 20)
        if i == 0:
            scene.reset_home(m, d)
            for k in range(int(3.5 * FPS)):
                set_camera(t_intro=k / FPS)
                emit(compose("intake hatch · rail-mounted arm · three lockers" if k > 20 else None))
        # Arrival beat is carried by the SUBMITTED record and the overlay; the episode begins.
        res = st.run_episode(seed, frame_cb=frame_cb, noise=noise, label=label)
        S["filed"] += int(res.success)
        print(f"--> episode {i + 1}: seed {seed} {res.object_class} true={res.true_item_id} read={res.decoded_id} "
              f"{res.outcome.upper()} {res.routed_location or res.refusal_cause} misfile={res.misfile}")
        set_camera()
        hold(1.0)
    hold(2.0)
    writer.close()
    print(f"wrote {out_path} ({S['t']:.1f} s); custody log {log_path.name}; "
          f"chain {'intact' if verify_file(log_path) is None else 'BROKEN'}")


if __name__ == "__main__":
    main()
