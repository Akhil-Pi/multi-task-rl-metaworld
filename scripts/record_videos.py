import os
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import SAC

from src.mt_rl.env_factory import make_multitask_env
from src.mt_rl.task_sampler import UniformTaskSampler

def main():
    mt = "MT3"
    model_path = "models/sac_mt3_final.zip"
    video_dir = "results/videos_mt3"
    os.makedirs(video_dir, exist_ok=True)

    # Build env
    n_tasks = 3 if mt == "MT3" else 10
    sampler = UniformTaskSampler(n_tasks=n_tasks, seed=0)
    env, task_names = make_multitask_env(
        mt_name=mt,
        seed=0,
        max_episode_steps=200,
        task_sampler=sampler,
        render_mode="rgb_array",  # This should work but might not be passed through
        task_id_in_obs=True,
    )

    # WORKAROUND: Force render_mode on the base environment
    if hasattr(env, 'unwrapped'):
        env.unwrapped.render_mode = "rgb_array"
    
    # Also try setting it on the env directly
    env.render_mode = "rgb_array"

    # Record every episode
    env = RecordVideo(
        env,
        video_folder=video_dir,
        episode_trigger=lambda ep: True,
        name_prefix=f"{mt.lower()}",
    )

    model = SAC.load(model_path)

    for _ in range(n_tasks):
        obs, info = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

    env.close()
    print("Videos saved to:", video_dir)

if __name__ == "__main__":
    main()
