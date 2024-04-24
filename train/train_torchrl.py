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

sys.path.append("/Users/czimbermark/Documents/Reinf/MetaWorld/GenReL-World/.venv/lib/python3.11/site-packages")
import torchrl
from torchrl.algorithms import PPO
from torchrl.policies import MLPPolicy
from torchrl import utils

# mjpython train/train_torchrl.py

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
policy = MLPPolicy(env.observation_space.shape[0], env.action_space.shape[0], hidden_sizes=[64, 64])
trainer = PPO(policy, env, n_epochs=100, batch_size=32)

# Initialize Mujoco viewer
print("Initializing Mujoco viewer...")
with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    print("Viewer initialized successfully.")
    # Train the model on each step
    start = time.time()
    time.sleep(0.01)
    num_steps = 10000
    for step in range(num_steps):
        # Perform one step of the environment
        action = trainer.predict(observation)
        observation, reward, done, info, _ = env.step(action)
        
        # Render the environment
        viewer.sync()
        
        # Update the trainer
        trainer.step(observation, action, reward, done)
        
        # Check if episode is done
        if done:
            observation = env.reset()
    
    # Log elapsed time
    elapsed_time = time.time() - start
    print("Training time:", elapsed_time)