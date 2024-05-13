import time
import mujoco
import mujoco.viewer
import numpy as np
import torch
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn

import sys
sys.path.append('/Users/czimbermark/Documents/Reinf/MetaWorld/GenReL-World')
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE

sys.path.append("/Users/czimbermark/Documents/Reinf/MetaWorld/GenReL-World/.venv/lib/python3.11/site-packages/")
import torchrl
import gym
import torch
import torch.nn as nn
from torchrl.networks.base import MLPNetwork
from torchrl.objectives.ppo import PPOLoss

# Set up the environment
env = gym.make('MoonLanderContinuous-v2', render = "human")

# Define the policy and value networks
policy_net = MLPNetwork(
    input_size=env.observation_space.shape[0],
    output_size=env.action_space.shape[0],
    hidden_sizes=[64, 64],
    activation=nn.Tanh
)

value_net = MLPNetwork(
    input_size=env.observation_space.shape[0],
    output_size=1,
    hidden_sizes=[64, 64],
    activation=nn.Tanh
)

# Create the PPO loss module
ppo_loss = PPOLoss(
    actor_network=policy_net,
    critic_network=value_net,
    entropy_bonus=True,
    entropy_coef=0.01,
    critic_coef=1.0,
    loss_critic_type="smooth_l1",
    normalize_advantage=True,
    functional=True,
)

# Train the agent
optimizer = torch.optim.Adam(ppo_loss.parameters(), lr=3e-4)

for episode in range(1000):
    obs = env.reset()
    done = False
    while not done:
        action, log_prob, value = ppo_loss.actor_network(obs)
        next_obs, reward, done, _ = env.step(action)
        ppo_loss.eval(
            obs=obs,
            action=action,
            reward=reward,
            next_obs=next_obs,
            done=done,
            value=value,
            log_prob=log_prob,
        )
        obs = next_obs

    ppo_loss.update()
    optimizer.zero_grad()
    loss = ppo_loss()
    loss.backward()
    optimizer.step()

    print(f"Episode {episode}, Reward: {ppo_loss.episode_reward}")