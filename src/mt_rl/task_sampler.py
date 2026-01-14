# src/mt_rl/task_sampler.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


class BaseTaskSampler:
    def __init__(self, n_tasks: int, seed: int = 0):
        self.n_tasks = int(n_tasks)
        self.rng = np.random.default_rng(seed)

    def sample_task_id(self, step: int = 0) -> int:
        raise NotImplementedError

    # ✅ alias for wrappers/callbacks convenience
    def sample(self, step: int = 0) -> int:
        return self.sample_task_id(step=step)


class UniformTaskSampler(BaseTaskSampler):
    """Sample a random task id uniformly each episode."""
    def sample_task_id(self, step: int = 0) -> int:
        return int(self.rng.integers(0, self.n_tasks))


@dataclass
class CurriculumStage:
    """Defines a curriculum stage: active tasks are [0 .. max_task_id] inclusive."""
    start_step: int
    max_task_id: int


class CurriculumTaskSampler(BaseTaskSampler):
    """
    Simple curriculum:
    - early training: sample from first K tasks
    - later: gradually unlock more tasks
    """
    def __init__(self, n_tasks: int, stage_steps: list[int], seed: int = 0):
        super().__init__(n_tasks=n_tasks, seed=seed)

        if len(stage_steps) == 0:
            raise ValueError("stage_steps must be a non-empty list of step thresholds.")

        # Convert thresholds into stages that progressively unlock tasks
        # Example: stage_steps=[0, 500k, 1M] => stages unlock 1/3, 2/3, 3/3 tasks
        stage_steps = list(stage_steps)
        stage_steps = sorted(stage_steps)

        stages: list[CurriculumStage] = []
        for i, s in enumerate(stage_steps):
            # unlock proportionally more tasks each stage
            frac = (i + 1) / len(stage_steps)
            max_id = max(0, min(n_tasks - 1, int(round(frac * (n_tasks - 1)))))
            stages.append(CurriculumStage(start_step=int(s), max_task_id=int(max_id)))

        self.stages = stages

    def _active_max_id(self, step: int) -> int:
        step = int(step)
        active = self.stages[0].max_task_id
        for st in self.stages:
            if step >= st.start_step:
                active = st.max_task_id
            else:
                break
        return int(active)

    def sample_task_id(self, step: int = 0) -> int:
        max_id = self._active_max_id(step)
        return int(self.rng.integers(0, max_id + 1))
