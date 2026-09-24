"""Evidence room simulation entry point.

    python3 scripts/simulation.py              # interactive viewer (needs a GUI session; on macOS use `mjpython`)
    python3 scripts/simulation.py --headless   # render demo_cam to out/scene.png and exit
"""

import argparse
import time

import mujoco
import mujoco.viewer

import scene


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--headless", action="store_true", help="render one frame to out/scene.png and exit")
    parser.add_argument("--camera", default="demo_cam")
    parser.add_argument("--settle", type=float, default=1.0, help="seconds of physics to run before rendering")
    parser.add_argument("--discover", action="store_true", help="print model names and ranges")
    args = parser.parse_args()

    model, data = scene.load()
    if args.discover:
        scene.discover(model)

    if args.headless:
        mujoco.mj_step(model, data, nstep=int(args.settle / model.opt.timestep))
        path = scene.save_png(scene.render(model, data, camera=args.camera), scene.OUT_DIR / ("scene.png" if args.camera == "demo_cam" else f"scene_{args.camera}.png"))
        print(f"wrote {path}")
        return

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            start = time.time()
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(max(0.0, model.opt.timestep - (time.time() - start)))


if __name__ == "__main__":
    main()
