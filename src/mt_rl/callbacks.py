from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from stable_baselines3.common.callbacks import BaseCallback


@dataclass
class EvalStats:
    mean_reward: float
    std_reward: float
    success_rate: float


class CurriculumStepCallback(BaseCallback):
    """
    Updates a CurriculumTaskSampler with the current global step.
    """

    def __init__(self, curriculum_sampler, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.curriculum_sampler = curriculum_sampler

    def _on_step(self) -> bool:
        # SB3 maintains num_timesteps
        if hasattr(self.curriculum_sampler, "set_global_step"):
            self.curriculum_sampler.set_global_step(self.num_timesteps)
        return True


class MultiTaskEvalCallback(BaseCallback):
    """
    Periodically evaluates the current policy on each task by forcing task_id.
    Works with EpisodeTaskSwitchWrapper + TaskIDWrapper setup.

    NOTE: This is lightweight and "good enough" for monitoring.
    For final evaluation + videos, use scripts/eval_mt.py and scripts/record_videos.py.
    """

    def __init__(
        self,
        eval_env,
        n_tasks: int,
        eval_freq: int = 50_000,
        n_episodes_per_task: int = 5,
        deterministic: bool = True,
        verbose: int = 1,
    ):
        super().__init__(verbose=verbose)
        self.eval_env = eval_env
        self.n_tasks = int(n_tasks)
        self.eval_freq = int(eval_freq)
        self.n_episodes_per_task = int(n_episodes_per_task)
        self.deterministic = deterministic

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and (self.num_timesteps % self.eval_freq == 0):
            stats = self.evaluate_all_tasks()
            # Log to tensorboard
            self.logger.record("eval/mean_reward", stats.mean_reward)
            self.logger.record("eval/std_reward", stats.std_reward)
            self.logger.record("eval/success_rate", stats.success_rate)
            if self.verbose:
                print(
                    f"[Eval @ {self.num_timesteps}] "
                    f"mean_reward={stats.mean_reward:.2f} "
                    f"success={stats.success_rate*100:.1f}%"
                )
        return True

    def evaluate_all_tasks(self) -> EvalStats:
        rewards = []
        successes = 0
        total_eps = 0

        # We can "force" the task by setting eval_env.env.current_task_id
        # BUT the clean way is: set task directly via underlying set_task
        # We assume eval_env is built as the same multitask env wrapper stack.
        base = self.eval_env
        # unwrap to get access to EpisodeTaskSwitchWrapper and tasks
        # stable-baselines VecEnv not used here; eval_env is a normal env.

        for tid in range(self.n_tasks):
            for _ in range(self.n_episodes_per_task):
                # Force task by overwriting sampler to return tid once
                if hasattr(base, "task_sampler"):
                    old_sampler = base.task_sampler

                    class _OneShotSampler:
                        def sample_task_id(self_non):
                            return tid

                    base.task_sampler = _OneShotSampler()

                obs, info = base.reset()
                done = False
                truncated = False
                ep_rew = 0.0
                ep_success = False

                while not (done or truncated):
                    action, _ = self.model.predict(obs, deterministic=self.deterministic)
                    obs, reward, done, truncated, info = base.step(action)
                    ep_rew += float(reward)
                    if isinstance(info, dict) and info.get("success", False):
                        ep_success = True

                rewards.append(ep_rew)
                successes += int(ep_success)
                total_eps += 1

                if hasattr(base, "task_sampler"):
                    base.task_sampler = old_sampler

        rewards = np.array(rewards, dtype=np.float32)
        return EvalStats(
            mean_reward=float(rewards.mean()) if len(rewards) else 0.0,
            std_reward=float(rewards.std()) if len(rewards) else 0.0,
            success_rate=float(successes / max(total_eps, 1)),
        )
