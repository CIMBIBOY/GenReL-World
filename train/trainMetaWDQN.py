import time
import mujoco
import mujoco.viewer

import sys
sys.path.append('/Users/czimbermark/Documents/Reinf/MetaWorld/GenReL-World')

from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
import numpy as np
# from models.PPO import PPO
from models.PPOv2 import PPO
# from models.PPOv3 import PPO

print("------------------------ Run start ------------------------")
# mjpython train/trainMetaW.py

if __name__ == "__main__":
    # Choose the specific environment you want to use
    # Choose the specific environment you want to use
    env_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["pick-place-v2-goal-observable"]
    env = env_cls(seed=0)
    env2 = env_cls(seed=0)

    env.reset()
    first_target = env._target_pos
    env.reset()
    second_target = env._target_pos
    assert (first_target == second_target).all()

    env.reset()
    env2.reset()
    assert (env._target_pos == env2._target_pos).all()

    env3 = env_cls(seed=10)
    env.reset()
    env3.reset()
    assert not (env._target_pos == env3._target_pos).all()

    print(type(env))

    episodes = 1000
    # Initialize the PPO trainer
    trainer = PPO(env, num_episodes=episodes, wandb_use=False, hidden_size=128, lr=3e-5, gamma=0.99, lam=0.95, clip_range=0.1, batch_size=32)

    # Initialize Mujoco viewer
    print("Initializing Mujoco viewer...")
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        print("Viewer initialized successfully.")
        # Train the model on each step
        start = time.time()
        # Delay to control the speed of movement
        trainer.train(viewer=viewer)

        # tensorboard --logdir runs
