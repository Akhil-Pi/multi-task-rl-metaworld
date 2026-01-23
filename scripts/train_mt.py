from __future__ import annotations
import argparse
import os

import numpy as np
import torch
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from src.mt_rl.env_factory import make_multitask_env, build_mt_tasks
from src.mt_rl.task_sampler import UniformTaskSampler, CurriculumTaskSampler
from src.mt_rl.callbacks import CurriculumStepCallback, MultiTaskEvalCallback
from src.mt_rl.utils import set_global_seeds, ensure_dir


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mt", type=str, default="MT3", choices=["MT3", "MT10"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=1_000_000)
    p.add_argument("--max-episode-steps", type=int, default=500)
    p.add_argument("--use-curriculum", action="store_true")
    p.add_argument("--stage-steps", type=str, default="0,500000,1000000")  # MT3 default
    p.add_argument("--logdir", type=str, default="./logs")
    p.add_argument("--savedir", type=str, default="./models")
    p.add_argument("--eval-freq", type=int, default=100_000)
    p.add_argument("--eval-episodes-per-task", type=int, default=3)
    return p.parse_args()


def main():
    args = parse_args()
    set_global_seeds(args.seed)

    ensure_dir(args.logdir)
    ensure_dir(args.savedir)
    ensure_dir(os.path.join(args.savedir, "checkpoints"))

    # Determine number of tasks
    bench, train_tasks, task_names = build_mt_tasks(args.mt)
    n_tasks = len(train_tasks)

    # Curriculum stage steps parsing
    stage_steps = [int(x.strip()) for x in args.stage_steps.split(",") if x.strip()]

    if args.use_curriculum:
        sampler = CurriculumTaskSampler(n_tasks=n_tasks, stage_steps=stage_steps, seed=args.seed)
        sampler_cb = CurriculumStepCallback(sampler)
    else:
        sampler = UniformTaskSampler(n_tasks=n_tasks, seed=args.seed)
        sampler_cb = None

    # Build training env (single env switching tasks each episode)
    env, _ = make_multitask_env(
        mt_name=args.mt,
        seed=args.seed,
        max_episode_steps=args.max_episode_steps,
        task_sampler=sampler,
        render_mode=None,
        task_id_in_obs=True,   # ✅ explicitly enable task-conditioning
    )

    # SB3 likes VecEnv (even DummyVecEnv)
    env = Monitor(env)
    vec_env = DummyVecEnv([lambda: env])
    vec_env = VecMonitor(vec_env)

    eval_env, _ = make_multitask_env(
    mt_name=args.mt,
    seed=args.seed + 1000,
    max_episode_steps=args.max_episode_steps,
    task_sampler=UniformTaskSampler(n_tasks=n_tasks, seed=args.seed + 1000),
    render_mode=None,
    task_id_in_obs=True,
    )

    eval_env = Monitor(eval_env)
    eval_vec_env = DummyVecEnv([lambda: eval_env])
    eval_vec_env = VecMonitor(eval_vec_env)


    # Separate eval env (same task set), deterministic eval
    #from 
    #stable_baselines3.common.monitor
    #import Monitor
    #from
    #stable_baselines3.common.vec_env
    #import DummyVecEnv, VecMonitor

    #build training env
    #env, = make_multitask_env(
     #   my_name = args.mt,
      #  seed = args.seed,

   # max_episode_steps = args.max_episode_steps,
    #    task_sampler = sampler,
     #   render_mode = None,
    #)

    #important: add Monitor BEFORE wrapping in VecENV
    env = Monitor(env)

    #SB3 requires VecEnv
    vec_env = DummyVecEnv[(lambda:env)]

    #IMPORTANT: add VecMonito to log rollout stats
    vec_env = VecMonitor(vec_env)

    # SAC config (solid default for Meta-World)
    model = SAC(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=3e-4,
        buffer_size=1_000_000,
        learning_starts=20_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",
        target_entropy="auto",
        use_sde=False,
        policy_kwargs=dict(
            net_arch=[256, 256, 256],
            activation_fn=torch.nn.ReLU,
            log_std_init=-1.5,
        ),
        tensorboard_log=args.logdir,
        verbose=1,
        device="auto",
        seed=args.seed,
    )

    # Callbacks
    checkpoint_cb = CheckpointCallback(
        save_freq=100_000,
        save_path=os.path.join(args.savedir, "checkpoints"),
        name_prefix=f"sac_{args.mt.lower()}",
        verbose=1,
    )

    # SB3 EvalCallback evaluates using the eval_env; will save best by mean reward
    eval_cb = EvalCallback(
        eval_vec_env,
        best_model_save_path=os.path.join(args.savedir, f"best_{args.mt.lower()}"),
        log_path=os.path.join(args.logdir, f"eval_{args.mt.lower()}"),
        eval_freq=args.eval_freq,
        n_eval_episodes=max(1, args.eval_episodes_per_task * n_tasks),
        deterministic=True,
        verbose=1,
    )

    # Optional multi-task eval stats (logs success_rate if info['success'] exists)
    mt_eval_cb = MultiTaskEvalCallback(
        eval_env=eval_env,
        n_tasks=n_tasks,
        eval_freq=args.eval_freq,
        n_episodes_per_task=args.eval_episodes_per_task,
        deterministic=True,
        verbose=1,
    )

    callbacks = [checkpoint_cb, eval_cb, mt_eval_cb]
    if sampler_cb is not None:
        callbacks.insert(0, sampler_cb)

    print("=" * 70)
    print(f"Training SAC on {args.mt} with {n_tasks} tasks")
    print(f"Task conditioning: one-hot task id appended to observation")
    print(f"Task sampling: {'curriculum' if args.use_curriculum else 'uniform'}")
    print("=" * 70)

    model.learn(total_timesteps=args.steps, callback=callbacks, progress_bar=True)

    # Save final
    final_path = os.path.join(args.savedir, f"sac_{args.mt.lower()}_final")
    model.save(final_path)
    print(f"Saved final model to: {final_path}.zip")


if __name__ == "__main__":
    main()
