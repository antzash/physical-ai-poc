"""Training entry point. In Phase 0 this is run as a SMOKE TEST ONLY (a few thousand steps) to confirm the plumbing:
the env steps, rewards are finite and non-degenerate, spaces are consistent, and checkpoints save and reload.

    python3 rl/train.py                          # SAC, 3000 steps, checkpoints to out/rl/
    python3 rl/train.py --algo ppo --steps 4096

A few thousand steps will NOT produce a useful policy, and none is claimed. See NOTES.md (M7) for what convergence
would actually need.
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from env import EvidenceIntakeEnv  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "out" / "rl"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--algo", choices=["sac", "ppo"], default="sac")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-episode-steps", type=int, default=200, help="shorter episodes for the smoke test")
    args = parser.parse_args()

    run_dir = OUT / f"{args.algo}_{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    raw = EvidenceIntakeEnv(max_steps=args.max_episode_steps)
    check_env(raw, warn=True)
    env = Monitor(raw, str(run_dir / "monitor"), info_keywords=("is_success",))

    common = dict(policy="MlpPolicy", env=env, seed=args.seed, device="cpu", verbose=0)
    if args.algo == "sac":
        model = SAC(learning_starts=500, batch_size=256, **common)
    else:
        model = PPO(n_steps=1024, batch_size=256, **common)

    ckpt = CheckpointCallback(save_freq=args.checkpoint_every, save_path=str(run_dir), name_prefix=args.algo)
    t0 = time.time()
    model.learn(total_timesteps=args.steps, callback=ckpt, progress_bar=False)
    wall = time.time() - t0
    final = run_dir / f"{args.algo}_final"
    model.save(final)

    returns = np.array(env.get_episode_rewards(), dtype=float)
    lengths = np.array(env.get_episode_lengths())
    env.close()
    with open(run_dir / "monitor.monitor.csv") as f:
        rows = list(csv.DictReader(line for line in f if not line.startswith("#")))
    successes = sum(row["is_success"] == "True" for row in rows)
    print(f"algo={args.algo} steps={args.steps} wall={wall:.0f}s ({args.steps / wall:.0f} steps/s) "
          f"episodes={len(returns)}")
    if len(returns):
        print(f"episode return: mean {returns.mean():.2f}  std {returns.std():.2f}  min {returns.min():.2f}  "
              f"max {returns.max():.2f}  finite={bool(np.all(np.isfinite(returns)))}")
        print(f"episode length: mean {lengths.mean():.0f}   successes: {successes}/{len(returns)}")

    reloaded = (SAC if args.algo == "sac" else PPO).load(final, device="cpu")
    obs, _ = raw.reset(seed=123)
    action, _ = reloaded.predict(obs, deterministic=True)
    assert raw.action_space.contains(action.astype(np.float32)), "reloaded policy produced an invalid action"
    ckpts = sorted(p.name for p in run_dir.glob("*.zip"))
    print(f"checkpoints: {ckpts}")
    print(f"reloaded {final.name}.zip and produced a valid action {[round(float(x), 3) for x in action]}")
    print(f"monitor log: {(run_dir / 'monitor.monitor.csv').relative_to(OUT.parent.parent)}")
    print("SMOKE TEST ONLY — no trained policy is claimed.")


if __name__ == "__main__":
    main()
