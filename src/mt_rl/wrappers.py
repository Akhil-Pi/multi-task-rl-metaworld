# src/mt_rl/wrappers.py
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class MultiTaskSwitchEnvWrapper(gym.Env):
    """
    Correct Meta-World multi-task wrapper:
    - Holds one env instance per env_name
    - On reset(), samples a task_id and switches self.env to the matching env
    - Calls env.set_task(task) on the correct env class
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        env_dict: dict[str, gym.Env],
        tasks: list,
        task_names: list[str],
        task_sampler,
        max_episode_steps: int,
        seed: int = 0,
        render_mode=None,  # ← ADDED: accept render_mode parameter
    ):
        super().__init__()
        self.env_dict = env_dict
        self.tasks = tasks
        self.task_names = task_names
        self.task_sampler = task_sampler
        self.max_episode_steps = int(max_episode_steps)
        self.seed = int(seed)

        self.current_task_id = 0
        self.current_env_name = task_names[0]
        self.env = self.env_dict[self.current_env_name]
        self.env.set_task(self.tasks[self.current_task_id])

        # Meta-World MT benchmarks share spaces (same obs/action shapes)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        
        # ✅ FIX: Use provided render_mode or fallback to env's render_mode
        if render_mode is not None:
            self.render_mode = render_mode
        elif hasattr(self.env, 'render_mode'):
            self.render_mode = self.env.render_mode
        else:
            self.render_mode = None

        self._steps = 0
        self.total_steps = 0

    def _apply_task_id(self, task_id: int):
        self.current_task_id = int(task_id)
        task = self.tasks[self.current_task_id]
        env_name = task.env_name

        self.current_env_name = env_name
        self.env = self.env_dict[env_name]

        # set correct task on matching env class
        self.env.set_task(task)

    def reset(self, *, seed=None, options=None):
        self._steps = 0

        task_id = self.task_sampler.sample_task_id(step=self.total_steps)
        self._apply_task_id(task_id)

        out = self.env.reset()
        if isinstance(out, tuple) and len(out) == 2:
            obs, info = out
        else:
            obs, info = out, {}

        info = dict(info)
        info["task_id"] = self.current_task_id
        info["env_name"] = self.current_env_name
        return obs, info

    def step(self, action):
        self._steps += 1
        self.total_steps += 1

        out = self.env.step(action)
        if len(out) == 5:
            obs, reward, terminated, truncated, info = out
        else:
            obs, reward, done, info = out
            terminated, truncated = bool(done), False

        info = dict(info) if info is not None else {}
        info["task_id"] = self.current_task_id
        info["env_name"] = self.current_env_name

        if self._steps >= self.max_episode_steps:
            truncated = True

        return obs, float(reward), bool(terminated), bool(truncated), info

    def render(self):
        if hasattr(self.env, "render"):
            return self.env.render(mode="rgb_array")
        return None

    def close(self):
        for e in self.env_dict.values():
            if hasattr(e, "close"):
                e.close()


class TaskIDObsWrapper(gym.ObservationWrapper):
    """
    Appends one-hot task id to observation.
    """

    def __init__(self, env: gym.Env, n_tasks: int):
        super().__init__(env)
        self.n_tasks = int(n_tasks)

        assert isinstance(env.observation_space, spaces.Box)
        obs_space: spaces.Box = env.observation_space

        low = np.concatenate([obs_space.low, np.zeros(self.n_tasks, dtype=np.float32)], axis=0)
        high = np.concatenate([obs_space.high, np.ones(self.n_tasks, dtype=np.float32)], axis=0)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, observation):
        task_id = getattr(self.env, "current_task_id", 0)
        one_hot = np.zeros(self.n_tasks, dtype=np.float32)
        one_hot[int(task_id)] = 1.0
        obs = np.asarray(observation, dtype=np.float32)
        return np.concatenate([obs, one_hot], axis=0)
