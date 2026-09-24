"""Record the pitch video: consecutive intake episodes with the live chain-of-custody log alongside.

    python3 scripts/record.py                      # out/demo.mp4 (~85 s, 1280x720, 30 fps)
    python3 scripts/record.py --no-fault-demo      # skip the final, clearly labelled fault-injection episode

Frames: the left 860 px is `demo_cam` rendered offscreen by mujoco.Renderer at 1280x720 and cropped, with a
`cabinet_cam` inset; the right 420 px is the custody panel drawn with PIL from the real log as it is written.
Playback is 1x simulated time.
"""

import argparse
from datetime import datetime, timezone

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import scene
from episode import IntakeStation
from logger import verify_file

FPS = 30
W, H = 1280, 720
SIM_W = 860
PANEL_W = W - SIM_W
CROP_X0 = 190  # crop of the 1280-wide demo_cam render that keeps counter, arm and cabinet centred
INSET_W, INSET_H = 288, 162

# Natural (unforced) seeds from the 1000-episode evaluation, one per class, plus a fault-injection episode.
DEMO_SEEDS = [1007, 1008, 1002, 1001, 1004]
FAULT_SEED = 1097  # a 0.80 kg box

BG = (13, 19, 33)
CARD = (27, 36, 54)
CARD_NEW = (38, 50, 74)
TEXT = (226, 232, 240)
MUTED = (140, 152, 172)
FAINT = (88, 100, 122)
ACCENT = (96, 165, 250)
OK_GREEN = (52, 211, 153)
BADGE = {
    "SUBMITTED": (100, 116, 139),
    "REGISTERED": (59, 130, 246),
    "PICKED": (245, 158, 11),
    "PLACED": (168, 85, 247),
    "VERIFIED": (34, 197, 94),
    "FAILED": (239, 68, 68),
}


def _font(path, size, index=0):
    try:
        return ImageFont.truetype(path, size, index=index)
    except OSError:
        return ImageFont.load_default()


SANS = "/System/Library/Fonts/Helvetica.ttc"
MONO = "/System/Library/Fonts/Menlo.ttc"
F = {
    "title": _font(SANS, 22, 1),
    "sub": _font(SANS, 12),
    "label": _font(SANS, 10, 1),
    "value": _font(MONO, 15, 1),
    "value_s": _font(MONO, 13),
    "badge": _font(SANS, 11, 1),
    "body": _font(SANS, 12),
    "mono_s": _font(MONO, 10),
    "overlay": _font(SANS, 15, 1),
    "overlay_s": _font(SANS, 12),
    "banner": _font(SANS, 17, 1),
}


def _truncate(draw, text, font, width):
    if draw.textlength(text, font=font) <= width:
        return text
    while text and draw.textlength(text + "…", font=font) > width:
        text = text[:-1]
    return text + "…"


class Panel:
    def __init__(self):
        self.first_seen = {}  # event_id -> video time when it first appeared (for the highlight)

    def draw(self, events, info, t_video, chain_ok, n_events):
        img = Image.new("RGB", (PANEL_W, H), BG)
        d = ImageDraw.Draw(img)
        pad = 18

        d.rectangle([0, 0, PANEL_W, 64], fill=(9, 13, 24))
        d.text((pad, 12), "CHAIN OF CUSTODY", font=F["title"], fill=TEXT)
        d.text((pad, 40), "evidence intake · ROBOT:arm-01 · append-only, SHA-256 hash chain", font=F["sub"], fill=MUTED)

        # Current item card.
        y = 76
        d.rounded_rectangle([pad - 6, y, PANEL_W - pad + 6, y + 120], radius=8, fill=CARD)
        cols = [("ITEM", info.get("item_id", "—")), ("CASE", info.get("case_id", "—"))]
        for i, (lab, val) in enumerate(cols):
            x = pad + 4 + i * 200
            d.text((x, y + 10), lab, font=F["label"], fill=MUTED)
            d.text((x, y + 24), val, font=F["value"], fill=TEXT)
        cols = [("CLASS", info.get("object_class", "—")), ("SLOT", info.get("slot_id") or "—")]
        for i, (lab, val) in enumerate(cols):
            x = pad + 4 + i * 200
            d.text((x, y + 52), lab, font=F["label"], fill=MUTED)
            d.text((x, y + 66), val, font=F["value"], fill=TEXT)
        d.text((pad + 4, y + 94), "ROBOT", font=F["label"], fill=MUTED)
        phase = info.get("phase", "—")
        colour = BADGE["FAILED"] if phase == "FAILED" else OK_GREEN if phase == "COMPLETE" else ACCENT
        d.ellipse([pad + 50, y + 97, pad + 58, y + 105], fill=colour)
        d.text((pad + 64, y + 92), phase, font=F["value_s"], fill=TEXT)

        # Event feed, newest at the bottom.
        y0, y1 = 210, H - 64
        d.text((pad, y0), "EVENT LOG", font=F["label"], fill=MUTED)
        card_h, gap = 66, 8
        n_fit = (y1 - y0 - 20) // (card_h + gap)
        shown = events[-n_fit:]
        y = y1 - len(shown) * (card_h + gap)
        for ev in shown:
            first = self.first_seen.setdefault(ev["event_id"], t_video)
            age = t_video - first
            fresh = age < 1.2
            slide = int(max(0.0, 1 - age / 0.25) * 30)
            yy = y + slide
            col = BADGE[ev["action"]]
            d.rounded_rectangle([pad - 6, yy, PANEL_W - pad + 6, yy + card_h], radius=7,
                                fill=CARD_NEW if fresh else CARD, outline=col if fresh else None, width=2)
            d.rectangle([pad - 6, yy + 6, pad - 3, yy + card_h - 6], fill=col)
            bw = d.textlength(ev["action"], font=F["badge"]) + 14
            d.rounded_rectangle([pad + 4, yy + 8, pad + 4 + bw, yy + 25], radius=4, fill=col)
            d.text((pad + 11, yy + 10), ev["action"], font=F["badge"], fill=(255, 255, 255))
            d.text((pad + 12 + bw, yy + 10), ev["actor"], font=F["body"], fill=TEXT)
            ts = ev["timestamp"][11:23]
            d.text((PANEL_W - pad - d.textlength(ts, font=F["mono_s"]), yy + 12), ts, font=F["mono_s"], fill=MUTED)
            line2 = f"{ev['item_id']} · {ev['slot_id'] or '—'} · {ev['detail']}"
            d.text((pad + 4, yy + 31), _truncate(d, line2, F["body"], PANEL_W - 2 * pad - 8), font=F["body"],
                   fill=MUTED)
            prev = f"#{ev['prev_hash'][:12]}" if ev["prev_hash"] else "genesis"
            d.text((pad + 4, yy + 49), f"#{ev['hash'][:12]}  ←  prev: {prev}", font=F["mono_s"], fill=FAINT)
            y += card_h + gap

        # Footer.
        d.rectangle([0, H - 52, PANEL_W, H], fill=(9, 13, 24))
        if chain_ok:
            d.line([(pad, H - 36), (pad + 5, H - 31), (pad + 13, H - 41)], fill=OK_GREEN, width=3)
            d.text((pad + 20, H - 44), f"chain verified · {n_events} events", font=F["overlay"], fill=OK_GREEN)
        else:
            d.text((pad, H - 44), "CHAIN BROKEN", font=F["overlay"], fill=BADGE["FAILED"])
        d.text((pad, H - 22), "every record hashes the one before it; any edit breaks the chain", font=F["sub"],
               fill=FAINT)
        return img


def overlay_sim(frame, inset, lines, banner=None):
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img, "RGBA")
    d.rounded_rectangle([12, 12, 12 + 470, 12 + 24 + 18 * (len(lines) - 1) + 8], radius=6, fill=(9, 13, 24, 190))
    d.text((22, 17), lines[0], font=F["overlay"], fill=TEXT)
    for i, line in enumerate(lines[1:]):
        d.text((22, 40 + 18 * i), line, font=F["overlay_s"], fill=MUTED)
    x0, y0 = 12, H - INSET_H - 12
    img.paste(Image.fromarray(inset), (x0, y0))
    d.rectangle([x0, y0, x0 + INSET_W, y0 + INSET_H], outline=(255, 255, 255, 160), width=1)
    d.rectangle([x0, y0, x0 + 80, y0 + 18], fill=(9, 13, 24, 200))
    d.text((x0 + 6, y0 + 3), "CABINET", font=F["label"], fill=TEXT)
    if banner:
        bw = d.textlength(banner, font=F["banner"]) + 28
        free_x0 = INSET_W + 24  # centre the banner in the space right of the cabinet inset
        bx = free_x0 + (SIM_W - free_x0 - bw) // 2
        d.rounded_rectangle([bx, H - 60, bx + bw, H - 24], radius=6, fill=(185, 28, 28, 230))
        d.text((bx + 14, H - 51), banner, font=F["banner"], fill=(255, 255, 255))
    return img


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(scene.OUT_DIR / "demo.mp4"))
    parser.add_argument("--no-fault-demo", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="*", default=DEMO_SEEDS)
    args = parser.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = scene.OUT_DIR / f"demo_{stamp}_custody.jsonl"
    station = IntakeStation(log_path, echo=True)
    m, d = station.m, station.d
    main_r = mujoco.Renderer(m, H, W)
    inset_r = mujoco.Renderer(m, INSET_H, INSET_W)
    panel = Panel()
    writer = imageio.get_writer(args.out, fps=FPS, codec="libx264", quality=8, pixelformat="yuv420p",
                                macro_block_size=16)

    plan = [(s, False) for s in args.seeds] + ([] if args.no_fault_demo else [(FAULT_SEED, True)])
    state = {"t_video": 0.0, "acc": 0.0, "chain_ok": True, "n": 0, "banner": None, "ep": 0, "seed": None,
             "results": [], "last": None, "ep_start_n": 0}

    def compose(phase):
        main_r.update_scene(d, camera="demo_cam")
        sim = main_r.render()[:, CROP_X0:CROP_X0 + SIM_W]
        inset_r.update_scene(d, camera="cabinet_cam")
        inset = inset_r.render()
        events = station.log.tail(12)
        if len(station.log) != state["n"]:
            state["n"] = len(station.log)
            state["chain_ok"] = verify_file(log_path) is None
        info = dict(events[-1]) if len(station.log) > state["ep_start_n"] else {}
        info["phase"] = phase
        done = sum(r.success for r in state["results"])
        lines = ["MuJoCo simulation · Franka Panda · 1× real time",
                 f"episode {state['ep']}/{len(plan)} · seed {state['seed']} · filed so far: {done}",
                 "grasping by contact physics only — no attachment / weld",
                 "one item simulated per episode: the cabinet resets between episodes"]
        frame = overlay_sim(sim, inset, lines, state["banner"])
        out = Image.new("RGB", (W, H))
        out.paste(frame, (0, 0))
        out.paste(panel.draw(events, info, state["t_video"], state["chain_ok"], state["n"]), (SIM_W, 0))
        return np.asarray(out)

    def emit(frame, n=1):
        for _ in range(n):
            writer.append_data(frame)
            state["t_video"] += 1 / FPS
        state["last"] = frame

    def frame_cb(st, c):
        state["acc"] += m.opt.timestep
        if state["acc"] >= 1 / FPS:
            state["acc"] -= 1 / FPS
            if c is None:
                phase = "PRESENTED"
            elif c.done:
                phase = "VERIFYING" if c.result.success else "FAILED"
            else:
                phase = c.phase
            emit(compose(phase))

    for i, (seed, fault) in enumerate(plan):
        state["ep"], state["seed"], state["ep_start_n"] = i + 1, seed, len(station.log)
        a = scene.actuator_id(m, scene.GRIPPER_ACTUATOR)
        saved = (m.actuator_gainprm[a, 0], m.actuator_biasprm[a, 1], m.actuator_biasprm[a, 2])
        if fault:
            # FAULT INJECTION, deliberate and labelled on screen: restore Menagerie's stock gripper stiffness
            # (kp=100, ~1.4 N per pad) so a heavy item cannot be held, to show how a failure is detected and logged.
            state["banner"] = "FAULT INJECTION (deliberate): gripper force cut to ~7%"
            m.actuator_gainprm[a, 0], m.actuator_biasprm[a, 1], m.actuator_biasprm[a, 2] = 100 * 0.04 / 255, -100, -10
        res = station.run_episode(seed, frame_cb=frame_cb)
        m.actuator_gainprm[a, 0], m.actuator_biasprm[a, 1], m.actuator_biasprm[a, 2] = saved
        state["results"].append(res)
        print(f"--> episode {i + 1}: seed {seed} {res.object_class} {res.slot} "
              f"{'SUCCESS' if res.success else 'FAILED at ' + str(res.failure_phase)}{' (fault injection)' if fault else ''}")
        emit(compose("COMPLETE" if res.success else "FAILED"), int(0.8 * FPS))
        state["banner"] = None

    emit(state["last"], int(2.0 * FPS))
    writer.close()
    print(f"wrote {args.out} ({state['t_video']:.1f} s); custody log {log_path.name}; "
          f"chain {'intact' if verify_file(log_path) is None else 'BROKEN'}")


if __name__ == "__main__":
    main()
