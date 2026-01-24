from __future__ import annotations
import argparse
import os
import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

from src.mt_rl.env_factory import make_multitask_env, build_mt_tasks
from src.mt_rl.task_sampler import UniformTaskSampler
from src.mt_rl.utils import set_global_seeds


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mt", type=str, default="MT3", choices=["MT3", "MT10"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-episode-steps", type=int, default=500)
    p.add_argument("--episodes-per-task", type=int, default=10)
    p.add_argument("--model-path", type=str, default="")
    return p.parse_args()


def main():
    args = parse_args()
    set_global_seeds(args.seed)

    bench, train_tasks, task_names = build_mt_tasks(args.mt)
    n_tasks = len(train_tasks)

    # Load model path
    model_path = args.model_path
    if not model_path:
        # default best model path
        model_path = f"./models/best_{args.mt.lower()}/best_model.zip"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Build eval env
    sampler = UniformTaskSampler(n_tasks=n_tasks, seed=args.seed + 999)
    env, _ = make_multitask_env(
        mt_name=args.mt,
        seed=args.seed + 999,
        max_episode_steps=args.max_episode_steps,
        task_sampler=sampler,
        render_mode=None,
        task_id_in_obs=False,
    )

    vec_env = DummyVecEnv([lambda: env])
    model = SAC.load(model_path, env=vec_env)

    per_task_success = np.zeros(n_tasks, dtype=np.int32)
    per_task_count = np.zeros(n_tasks, dtype=np.int32)
    per_task_rewards = [[] for _ in range(n_tasks)]

    all_rewards = []
    all_success = 0
    total_eps = 0

    # Evaluate by forcing tasks in order
    for tid in range(n_tasks):
        for _ in range(args.episodes_per_task):
            # Force the sampler to pick tid for this episode
            class _OneShotSampler:
                def sample_task_id(self_non, step=None):
                    return tid

            env.task_sampler = _OneShotSampler()

            obs, info = env.reset()
            done = False
            truncated = False
            ep_rew = 0.0
            ep_success = False

            while not (done or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                ep_rew += float(reward)
                if isinstance(info, dict) and info.get("success", False):
                    ep_success = True

            per_task_rewards[tid].append(ep_rew)
            per_task_success[tid] += int(ep_success)
            per_task_count[tid] += 1

            all_rewards.append(ep_rew)
            all_success += int(ep_success)
            total_eps += 1

    all_rewards = np.array(all_rewards, dtype=np.float32)

    print("\n" + "=" * 70)
    print(f"Evaluation Results - {args.mt}")
    print(f"Model: {model_path}")
    print(f"Episodes per task: {args.episodes_per_task}")
    print("-" * 70)
    print(f"Overall mean reward: {all_rewards.mean():.2f} +/- {all_rewards.std():.2f}")
    print(f"Overall success rate: {all_success}/{total_eps} ({100*all_success/total_eps:.1f}%)")
    print("-" * 70)

    for tid, name in enumerate(task_names):
        sr = per_task_success[tid] / max(per_task_count[tid], 1)
        r = np.array(per_task_rewards[tid], dtype=np.float32)
        print(
            f"[{tid:02d}] {name:20s} | "
            f"success {per_task_success[tid]}/{per_task_count[tid]} ({100*sr:.1f}%) | "
            f"meanR {r.mean():.2f}"
        )

    print("=" * 70)


if __name__ == "__main__":
    main()
