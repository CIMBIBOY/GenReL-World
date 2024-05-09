import time
import mujoco
import mujoco.viewer
import gymnasium as gym

import sys
sys.path.append('/Users/czimbermark/Documents/Reinf/MetaWorld/GenReL-World')

from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
import numpy as np
# from models.PPO import PPO
# from models.PPOv2 import PPO
from models.PPOgym import PPO

print("------------------------ Run start ------------------------")
# mjpython train/traingymPPO.py

if __name__ == "__main__":
    # Choose the specific environment you want to use
    env = gym.make("LunarLander-v2", render_mode="human")
    observation, info = env.reset(seed=42)

    print(type(env))

    episodes = 1000
    # Initialize the PPO trainer
    trainer = PPO(env, num_episodes=episodes, wandb_use=True, hidden_size=256, lr=3e-6, gamma=0.99, lam=0.95, clip_range=0.2, batch_size=64)

    '''
    # Initialize Mujoco viewer
    print("Initializing Mujoco viewer...")
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        print("Viewer initialized successfully.")   
        # Train the model on each step
        start = time.time()
        # Delay to control the speed of movement
        trainer.train(viewer=viewer)

        # tensorboard --logdir runs
        '''
