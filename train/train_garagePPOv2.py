import time
import mujoco
import mujoco.viewer
import numpy as np
import torch
import gymnasium as gym
from gym.wrappers import TimeLimit
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn

import sys
sys.path.append('/Users/czimbermark/Documents/Reinf/MetaWorld/GenReL-World')
import metaworld
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
import random
from utils.metaworld_env_wrapper import NormalizedEnvWrapper
from garage.envs.normalized_env import NormalizedEnv

from garage.tf.algos import PPO
from garage.tf.policies import GaussianMLPPolicy  # Replace with your chosen policy
from garage.tf.baselines import ContinuousMLPBaseline
from garage.sampler import Sampler

# mjpython train/train_garagePPOv2.py

print(metaworld.ML1.ENV_NAMES)  # Check out the available environments

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

# Wrap the environment with NormalizedEnv
normalized_env = NormalizedEnv(
    env,
    scale_reward=1.0,
    normalize_obs=True,
    normalize_reward=True,
    expected_action_scale=1.0,
    flatten_obs=True,
    obs_alpha=0.001,
    reward_alpha=0.001
)

# Now you can use the normalized environment
gym_env = TimeLimit(normalized_env, max_episode_steps=1000)

# Policy initialization
policy = GaussianMLPPolicy(env_spec=gym_env.spec, hidden_sizes=[64, 64])

# Baseline initialization
baseline = ContinuousMLPBaseline(env_spec=gym_env.spec)

# Sampler initialization
sampler = Sampler(gym_env, policy, baseline, max_episode_length=100)

# PPO trainer
trainer = PPO(
    env_spec=gym_env.spec,
    policy=policy,
    baseline=baseline,
    sampler=sampler,
    discount=0.99,
    gae_lambda=1,
    center_adv=True,
    positive_adv=False,
    fixed_horizon=False,
    lr_clip_range=0.01,
    max_kl_step=0.01,
    optimizer="first_order_optimizer",
    optimizer_args={},
    policy_ent_coeff=0.0,
    use_softplus_entropy=False,
    use_neg_logli_entropy=False,
    stop_entropy_gradient=False,
    entropy_method="no_entropy",
    name="PPO",
)

# Mujoco viewer init
print("Initializing Mujoco viewer...")
with mujoco.viewer.launch_passive(gym_env.env.model, gym_env.env.data) as viewer:
    print("Viewer initialized successfully.")
    # Train on num_steps
    start = time.time()
    time.sleep(0.01)
    num_steps = 10000
    for step in range(num_steps):
        # one step of the environment
        observation = gym_env.reset()
        action = trainer.policy.get_action(observation)
        observation, reward, done, info, _ = gym_env.step(action)
        
        # Render the env
        viewer.sync()
        
        # Update the trainer
        trainer.step(observation, action, reward, done, info)
        
        # Check if episode is done
        if done:
            observation = gym_env.reset()
    
    # Log elapsed time
    elapsed_time = time.time() - start
    print("Training time:", elapsed_time)