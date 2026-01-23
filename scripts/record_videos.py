import os
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import SAC

from src.mt_rl.env_factory import make_multitask_env
from src.mt_rl.task_sampler import UniformTaskSampler

def main():
    mt = "MT3"
    model_path = "models/sac_mt3_final.zip"
    video_base_dir = "./videos"
    
    model = SAC.load(model_path)
    
    n_tasks = 3 if mt == "MT3" else 10
    
    # Record each task separately
    for task_id in range(n_tasks):
        print(f"\nRecording task {task_id}...")
        
        # Create a sampler that always returns this task_id
        class FixedTaskSampler:
            def __init__(self, task_id):
                self.task_id = task_id
            def sample_task_id(self, step=None):
                return self.task_id
        
        sampler = FixedTaskSampler(task_id)
        
        # Create environment for this specific task
        env, task_names = make_multitask_env(
            mt_name=mt,
            seed=task_id,
            max_episode_steps=200,
            task_sampler=sampler,
            render_mode="rgb_array",
            task_id_in_obs=True,
        )
        
        task_name = task_names[task_id]
        video_folder = f"{video_base_dir}/mt3_{task_id:02d}_{task_name}"
        os.makedirs(video_folder, exist_ok=True)
        
        # Wrap with video recorder
        env = RecordVideo(
            env,
            video_folder=video_folder,
            episode_trigger=lambda ep_id: True,  # Record all episodes
            name_prefix=f"mt3_{task_name}",
        )
        
        # Record 3 episodes per task to show variability
        for episode in range(3):
            print(f"  Episode {episode + 1}/3 for {task_name}")
            obs, info = env.reset()
            done = False
            steps = 0
            
            while not done and steps < 200:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                steps += 1
            
            print(f"    Completed in {steps} steps, success: {info.get('success', False)}")
        
        env.close()
        print(f"Saved videos for task [{task_id}] {task_name} to: {video_folder}")
    
    print("\nAll videos recorded.")

if __name__ == "__main__":
    main()
