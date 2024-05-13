import time
import mujoco
import mujoco.viewer
import gymnasium as gym
import numpy as np

import sys
sys.path.append('/Users/czimbermark/Documents/Reinf/MetaWorld/GenReL-World')
from models.PPOgym import PPO

print("------------------------ Run start ------------------------")
# python train/traingymPPO.py

if __name__ == "__main__":
    # Choose the specific environment you want to use
    env = gym.make("LunarLanderContinuous-v2", render_mode="human")
    observation, info = env.reset(seed=42)

    print(type(env))

    episodes = 100000
    explore = 10
    # Initialize the PPO trainer
    trainer = PPO(env, num_episodes=episodes, start_explore = explore, wandb_use=True, hidden_size=256, lr=5e-4, gamma=0.99, lam=0.95, clip_range=0.2, batch_size=64)

    trainer.train()
