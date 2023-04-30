---
library_name: stable-baselines3
tags:
- HalfCheetah-v3
- deep-reinforcement-learning
- reinforcement-learning
- stable-baselines3
model-index:
- name: ARS
  results:
  - metrics:
    - type: mean_reward
      value: 4046.14 +/- 2253.39
      name: mean_reward
    task:
      type: reinforcement-learning
      name: reinforcement-learning
    dataset:
      name: HalfCheetah-v3
      type: HalfCheetah-v3
---

# **ARS** Agent playing **HalfCheetah-v3**
This is a trained model of a **ARS** agent playing **HalfCheetah-v3**
using the [stable-baselines3 library](https://github.com/DLR-RM/stable-baselines3)
and the [RL Zoo](https://github.com/DLR-RM/rl-baselines3-zoo).

The RL Zoo is a training framework for Stable Baselines3
reinforcement learning agents,
with hyperparameter optimization and pre-trained agents included.

## Usage (with SB3 RL Zoo)

RL Zoo: https://github.com/DLR-RM/rl-baselines3-zoo<br/>
SB3: https://github.com/DLR-RM/stable-baselines3<br/>
SB3 Contrib: https://github.com/Stable-Baselines-Team/stable-baselines3-contrib

```
# Download model and save it into the logs/ folder
python -m rl_zoo3.load_from_hub --algo ars --env HalfCheetah-v3 -orga sb3 -f logs/
python enjoy.py --algo ars --env HalfCheetah-v3  -f logs/
```

## Training (with the RL Zoo)
```
python train.py --algo ars --env HalfCheetah-v3 -f logs/
# Upload the model and generate video (when possible)
python -m rl_zoo3.push_to_hub --algo ars --env HalfCheetah-v3 -f logs/ -orga sb3
```

## Hyperparameters
```python
OrderedDict([('alive_bonus_offset', 0),
             ('delta_std', 0.03),
             ('learning_rate', 0.02),
             ('n_delta', 32),
             ('n_envs', 16),
             ('n_timesteps', 12500000.0),
             ('n_top', 4),
             ('normalize', 'dict(norm_obs=True, norm_reward=False)'),
             ('policy', 'LinearPolicy'),
             ('normalize_kwargs', {'norm_obs': True, 'norm_reward': False})])
```