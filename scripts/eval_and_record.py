from __future__ import annotations
import argparse
import os
import shutil
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

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
    p.add_argument("--record-video", action="store_true", help="Record videos")
    p.add_argument("--video-dir", type=str, default="./videos_eval")
    return p.parse_args()


def main():
    args = parse_args()
    set_global_seeds(args.seed)

    bench, train_tasks, task_names = build_mt_tasks(args.mt)
    n_tasks = len(train_tasks)

    # Load model path
    model_path = args.model_path
    if not model_path:
        model_path = f"./models/sac_{args.mt.lower()}_final.zip"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Build eval env WITH video recording capability if requested
    render_mode = "rgb_array" if args.record_video else None
    
    sampler = UniformTaskSampler(n_tasks=n_tasks, seed=args.seed + 999)
    env, _ = make_multitask_env(
        mt_name=args.mt,
        seed=args.seed + 999,
        max_episode_steps=args.max_episode_steps,
        task_sampler=sampler,
        render_mode=render_mode,
        task_id_in_obs=True,
    )

    # Wrap with video recorder if requested (single folder first)
    temp_video_dir = args.video_dir + "_temp" if args.record_video else None
    if args.record_video:
        os.makedirs(temp_video_dir, exist_ok=True)
        env = RecordVideo(
            env,
            video_folder=temp_video_dir,
            episode_trigger=lambda ep_id: True,  # Record all episodes
            name_prefix=f"{args.mt.lower()}_eval",
        )

    vec_env = DummyVecEnv([lambda: env])
    model = SAC.load(model_path, env=vec_env)

    per_task_success = np.zeros(n_tasks, dtype=np.int32)
    per_task_count = np.zeros(n_tasks, dtype=np.int32)
    per_task_rewards = [[] for _ in range(n_tasks)]

    all_rewards = []
    all_success = 0
    total_eps = 0

    print(f"\nStarting evaluation with {args.episodes_per_task} episodes per task...")
    print(f"Max episode steps: {args.max_episode_steps}")
    print(f"Recording videos: {args.record_video}\n")

    # Track which videos belong to which task
    video_to_task = []

    # Evaluate by forcing tasks in order
    for tid in range(n_tasks):
        print(f"\nEvaluating task [{tid}] {task_names[tid]}...")
        
        for ep_idx in range(args.episodes_per_task):
            # Force the sampler to pick tid for this episode
            class _OneShotSampler:
                def sample_task_id(self_non, step=None):
                    return tid

            # Get the base environment (unwrap from wrappers)
            base_env = env
            while hasattr(base_env, 'env'):
                base_env = base_env.env
            base_env.task_sampler = _OneShotSampler()

            obs, info = env.reset()
            done = False
            truncated = False
            ep_rew = 0.0
            ep_success = False
            steps = 0

            while not (done or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                ep_rew += float(reward)
                steps += 1
                if isinstance(info, dict) and info.get("success", False):
                    ep_success = True

            per_task_rewards[tid].append(ep_rew)
            per_task_success[tid] += int(ep_success)
            per_task_count[tid] += 1

            all_rewards.append(ep_rew)
            all_success += int(ep_success)
            
            # Track which video corresponds to this task
            video_to_task.append((tid, task_names[tid], total_eps))
            total_eps += 1
            
            print(f"  Episode {ep_idx + 1}/{args.episodes_per_task}: "
                  f"{steps} steps, success: {ep_success}, reward: {ep_rew:.2f}")

    all_rewards = np.array(all_rewards, dtype=np.float32)

    print("\n" + "=" * 70)
    print(f"Evaluation Results - {args.mt}")
    print(f"Model: {model_path}")
    print(f"Episodes per task: {args.episodes_per_task}")
    print(f"Max episode steps: {args.max_episode_steps}")
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
    
    env.close()
    
    # Organize videos into task-specific folders
    if args.record_video:
        print(f"\nOrganizing videos into task folders...")
        
        # Create task folders
        for tid, task_name in enumerate(task_names):
            task_folder = os.path.join(args.video_dir, f"{tid:02d}_{task_name}")
            os.makedirs(task_folder, exist_ok=True)
        
        # Move videos to appropriate folders
        if os.path.exists(temp_video_dir):
            video_files = sorted([f for f in os.listdir(temp_video_dir) if f.endswith('.mp4')])
            
            for video_idx, (tid, task_name, _) in enumerate(video_to_task):
                if video_idx < len(video_files):
                    src = os.path.join(temp_video_dir, video_files[video_idx])
                    task_folder = os.path.join(args.video_dir, f"{tid:02d}_{task_name}")
                    dst = os.path.join(task_folder, f"{task_name}_episode_{video_idx % args.episodes_per_task}.mp4")
                    shutil.move(src, dst)
            
            # Remove temp directory
            shutil.rmtree(temp_video_dir)
        
        print(f"All videos saved in separate task folders under: {args.video_dir}")


if __name__ == "__main__":
    main()
