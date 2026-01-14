# Multi-Task RL on Meta-World (MT3 / MT10) with SAC

This repository implements **multi-task reinforcement learning (MTRL)** on the **Meta-World** benchmark using **Stable-Baselines3 (SB3)**.  
We focus on training a **single shared SAC policy** across multiple tasks using:

- **Task conditioning via task identity** (one-hot task ID appended to observation)
- **Uniform task sampling** (baseline)
- **Curriculum task sampling** (optional): gradually unlock tasks during training
- Evaluation + video recording utilities for reporting

---

## Project Scope

We train and evaluate a shared policy on:
- **MT3**: `reach-v3`, `push-v3`, `pick-place-v3`
- **MT10**: 10 manipulation tasks from Meta-World MT10 benchmark

We report:
- Learning curves (TensorBoard)
- Per-task reward and success rates
- Qualitative rollouts via recorded videos

---

## Requirements

### System Requirements
- Python **3.10+**
- MuJoCo installed and working (Meta-World uses MuJoCo)
- A working Meta-World installation

### Python Dependencies
Key packages:
- `metaworld`
- `mujoco`
- `gymnasium`
- `stable-baselines3`
- `torch`
- `tensorboard`

---

## Training (SAC Multi-Task)

1. Train on MT3
python scripts/train_mt.py --mt MT3 --steps 1000000 --max-episode-steps 200 --seed 42

2. Train on MT10
python scripts/train_mt.py --mt MT10 --steps 2000000 --max-episode-steps 200 --seed 42

3. Train with curriculum sampling
python scripts/train_mt.py --mt MT3 --use-curriculum --stage-steps "0,500000,1000000" --steps 1000000 --seed 42

---

## Evaluation
Evaluates a trained model across all tasks and reports:

1. average reward
2. per-task reward
3. per-task success rates (if available in env info)

Example:
python scripts/eval_mt.py --mt MT3 --model-path models/best_mt3/best_model.zip --episodes-per-task 10

---

## Recoding Videos

Record rollout videos per task (saved in results/ or videos/ depending on script).

Example:
python scripts/record_videos.py --mt MT3 --model-path models/best_mt3/best_model.zip --episodes-per-task 3

---

## TensorBoard

To view training curves:
tensorboard --logdir ./logs

Then open the shown URL (typically http://localhost:6006).

Key plots:
rollout/ep_rew_mean : average episode return during training
custom eval logs (if enabled): success rate and per-task metrics