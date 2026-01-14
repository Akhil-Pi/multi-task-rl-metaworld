# src/mt_rl/env_factory.py
from __future__ import annotations

import gymnasium as gym
import metaworld
import numpy as np

MT3_DEFAULT = ["reach-v3", "push-v3", "pick-place-v3"]


def build_mt_tasks(mt_name: str, seed: int = 0, mt3_tasks: list[str] | None = None):
    """
    Returns:
      bench: metaworld benchmark object (MT10 / MT25 / MT50 or MT10 used for MT3-subset)
      tasks: list of Task objects (one per env_name, stable order)
      task_names: list[str] mapping task_id -> env_name
    """
    mt_name = mt_name.upper().strip()

    if mt_name == "MT10":
        bench = metaworld.MT10()
        selected_env_names = sorted(list(bench.train_classes.keys()))

    elif mt_name == "MT3":
        # Meta-World v3.0.0 has no MT3 -> implement as a subset of MT10
        bench = metaworld.MT10()
        selected_env_names = ["reach-v3", "push-v3", "pick-place-v3"]

        # sanity check
        missing = [n for n in selected_env_names if n not in bench.train_classes]
        if missing:
            raise ValueError(
                f"MT3 subset contains envs not in MT10 train_classes: {missing}\n"
                f"Available: {sorted(list(bench.train_classes.keys()))}"
            )

    else:
        raise ValueError("mt_name must be 'MT3' or 'MT10'")

    # Map env_name -> list of tasks (goal variations)
    env_to_tasks: dict[str, list] = {}
    for t in bench.train_tasks:
        env_to_tasks.setdefault(t.env_name, []).append(t)

    tasks = []
    task_names = []
    for env_name in selected_env_names:
        if env_name not in env_to_tasks:
            continue
        # simplest: pick the FIRST task variation deterministically
        tasks.append(env_to_tasks[env_name][0])
        task_names.append(env_name)

    if not tasks:
        raise RuntimeError("No tasks found. Meta-World API mismatch?")

    return bench, tasks, task_names


def _make_single_env_from_benchmark(bench, env_name: str, seed: int, render_mode=None):
    """
    Create a Meta-World env instance from benchmark train_classes.
    """
    env_cls = bench.train_classes[env_name]
    env = env_cls()
    # set seed if supported
    if hasattr(env, "seed"):
        env.seed(seed)
    if hasattr(env, "action_space") and hasattr(env.action_space, "seed"):
        env.action_space.seed(seed)

    # Some Meta-World envs use render_mode via gymnasium; classic ones don't.
    # We keep it compatible by ignoring render_mode if not supported.
    return env

def _make_env_from_cls(env_cls, seed: int, render_mode=None):
    env = env_cls()
    if hasattr(env, "seed"):
        env.seed(seed)
    if hasattr(env, "action_space") and hasattr(env.action_space, "seed"):
        env.action_space.seed(seed)
    return env

def make_multitask_env(
    mt_name: str,
    seed: int,
    max_episode_steps: int,
    task_sampler,
    render_mode=None,
    task_id_in_obs: bool = True,
    mt3_tasks: list[str] | None = None,
):
    bench, tasks, task_names = build_mt_tasks(mt_name, seed=seed, mt3_tasks=mt3_tasks)

    # build one env instance per env_name
    env_dict = {}
    for i, env_name in enumerate(task_names):
        env_cls = bench.train_classes[env_name]
        env = env_cls()

        # seed (best-effort)
        if hasattr(env, "seed"):
            env.seed(seed + i)
        if hasattr(env, "action_space") and hasattr(env.action_space, "seed"):
            env.action_space.seed(seed + i)

        env_dict[env_name] = env

    from .wrappers import MultiTaskSwitchEnvWrapper, TaskIDObsWrapper

    env = MultiTaskSwitchEnvWrapper(
        env_dict=env_dict,
        tasks=tasks,
        task_names=task_names,
        task_sampler=task_sampler,
        max_episode_steps=max_episode_steps,
        seed=seed,
    )

    if task_id_in_obs:
        env = TaskIDObsWrapper(env, n_tasks=len(tasks))

    return env, task_names

