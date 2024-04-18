import time
import mujoco
import mujoco.viewer

import sys
sys.path.append('/Users/czimbermark/Documents/Reinf/MetaWorld/GenReL-World')

from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
import numpy as np
from models.PPO import SawyerPickPlaceTrainer

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

    # Initialize the PPO trainer
    trainer = SawyerPickPlaceTrainer(env, hidden_size=64, lr=3e-4, gamma=0.99, lam=0.95, clip_range=0.2, num_epochs=10, batch_size=32)

    # Initialize Mujoco viewer
    print("Initializing Mujoco viewer...")
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        print("Viewer initialized successfully.")
        # Train the model on each step
        start = time.time()
        # Delay to control the speed of movement
        trainer.train(num_episodes=10000, viewer=viewer, start=start)

        # tensorboard --logdir runs
