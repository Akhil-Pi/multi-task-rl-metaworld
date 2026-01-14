from __future__ import annotations
import argparse
import os

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym

from src.mt_rl.env_factory import make_multitask_env, build_mt_tasks
from src.mt_rl.task_sampler import UniformTaskSampler
from src.mt_rl.utils import ensure_dir, set_global_seeds


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mt", type=str, default="MT3", choices=["MT3", "MT10"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-episode-steps", type=int, default=500)
    p.add_argument("--episodes-per-task", type=int, default=1)
    p.add_argument("--video-dir", type=str, default="./videos")
    p.add_argument("--model-path", type=str, default="")
    return p.parse_args()


def main():
    args = parse_args()
    set_global_seeds(args.seed)

    bench, train_tasks, task_names = build_mt_tasks(args.mt)
    n_tasks = len(train_tasks)

    model_path = args.model_path or f"./models/best_{args.mt.lower()}/best_model.zip"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    ensure_dir(args.video_dir)

    # We will create a fresh env for each task to keep RecordVideo clean
    for tid, name in enumerate(task_names):
        sampler = UniformTaskSampler(n_tasks=n_tasks, seed=args.seed + 999)

        env, _ = make_multitask_env(
            mt_name=args.mt,
            seed=args.seed + 999,
            max_episode_steps=args.max_episode_steps,
            task_sampler=sampler,
            render_mode="rgb_array",
        )

        # Force this env to always run task tid
        class _FixedSampler:
            def sample_task_id(self_non):
                return tid

        env.task_sampler = _FixedSampler()

        # Wrap RecordVideo (records episodes)
        task_video_dir = os.path.join(args.video_dir, f"{args.mt.lower()}_{tid:02d}_{name}")
        ensure_dir(task_video_dir)

        env = gym.wrappers.RecordVideo(
            env,
            video_folder=task_video_dir,
            episode_trigger=lambda ep: ep < args.episodes_per_task,
            name_prefix=f"{args.mt.lower()}_{name}",
            disable_logger=True,
        )

        vec_env = DummyVecEnv([lambda: env])
        model = SAC.load(model_path, env=vec_env)

        # Run episodes
        for ep in range(args.episodes_per_task):
            obs = vec_env.reset()
            done = [False]
            while not done[0]:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = vec_env.step(action)

        vec_env.close()
        print(f"Saved videos for task [{tid}] {name} to: {task_video_dir}")

    print("All videos recorded.")


if __name__ == "__main__":
    main()
