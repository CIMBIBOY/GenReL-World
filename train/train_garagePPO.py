import time
import mujoco
import mujoco.viewer
import numpy as np
import torch
import gymnasium as gym
from gym.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env

import sys
sys.path.append('/Users/czimbermark/Documents/Reinf/MetaWorld/GenReL-World')
import metaworld
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
import random
from utils.metaworld_env_wrapper import NormalizedEnvWrapper

# mjpython train/train_garagePPO.py

print("------------------------ Run start ------------------------")
# print(metaworld.ML1.ENV_NAMES)  # Check out the available environments

env_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["pick-place-v2-goal-observable"]
env = env_cls(seed=42)

'''
# if using ml1
# Construct the benchmark, sampling tasks
ml1 = metaworld.ML1('pick-place-v2', seed=420)
env = ml1.train_classes['pick-place-v2']()  # Create an environment with task `pick_place`
task = random.choice(ml1.train_tasks)
env.set_task(task)  # Set task
'''

print(type(env))

# Wrap the environment with NormalizedEnvWrapper
normalized_env = NormalizedEnvWrapper(
    env,
    scale_reward=1.0,
    normalize_obs=True,
    normalize_reward=True,
    expected_action_scale=1.0,
    flatten_obs=True,
    obs_alpha=0.001,
    reward_alpha=0.001
)

print(type(normalized_env))
print("Action space:", normalized_env.action_space)

# env = gym.make(normalized_env)

# Create a vectorized environment (required by SB3)
vec_env = make_vec_env(lambda: normalized_env, n_envs=1)

# Policy initialization
policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=[64, 64])

# PPO trainer
trainer = PPO("MlpPolicy", vec_env, verbose=1, policy_kwargs=policy_kwargs)

# Train the model
trainer.learn(total_timesteps=10000)
# Mujoco viewer init
print("Initializing Mujoco viewer...")
with mujoco.viewer.launch_passive(normalized_env.env.model, normalized_env.env.data) as viewer:
    print("Viewer initialized successfully.")
    # Train on num_steps
    start = time.time()
    time.sleep(0.01)
    for _ in range(5):
        obs = normalized_env.reset()
        done = False
        while not done:
            action, _ = trainer.predict(obs)
            obs, reward, done, info, _ = normalized_env.step(action)
            viewer.sync()

    # Log elapsed time
    elapsed_time = time.time() - start
    print("Training time:", elapsed_time)