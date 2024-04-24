import time
import mujoco
import mujoco.viewer

import sys
sys.path.append('/Users/czimbermark/Documents/Reinf/MetaWorld/GenReL-World')

from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
import numpy as np
from metaworld.policies.sawyer_pick_place_v2_policy import SawyerPickPlaceV2Policy
from metaworld.envs.asset_path_utils import full_v2_path_for
from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_pick_place_v2 import TrainPickPlacev2
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv
from metaworld.envs import reward_utils


# mjpython test/testMetaW.py

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

# Initialize the policy
policy = SawyerPickPlaceV2Policy()

# Initialize Mujoco viewer
print("Initializing Mujoco viewer...")
with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    print("Viewer initialized successfully.")

    # Training loop
    num_episodes = 100

    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action = policy.get_action(obs)
            print(action)
            action_pos = action[:3]
            env.set_xyz_action(action_pos)
            obs, reward, done, info, _ = env.step(action)
            viewer.sync()

            # Compute the reward based on the object's position and the gripper's interaction
            obj_pos = env._get_pos_objects()
            goal_pos = env._target_pos
            touching_object = env.touching_main_object

            # Custom reward function
            reward = 0.0
            dist_to_goal = np.linalg.norm(obj_pos - goal_pos)
            reward += 1.0 - np.clip(dist_to_goal, 0.0, 1.0)

            if touching_object:
                reward += 1.0

            episode_reward += reward

            # Update the observation with the combined end-effector and object information
            obs = env._get_obs()

            if done or info:
                obs, _ = env.reset()

        print(f"Episode {episode} reward: {episode_reward:.2f}")