import time
import mujoco
import mujoco.viewer

import sys
sys.path.append('/Users/czimbermark/Documents/Reinf/MetaWorld/GenReL-World')

from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
from metaworld.policies import SawyerPickPlaceV2Policy
import numpy as np
from garage.tf.algos import RL2PPO
from garage.tf.optimizers import FirstOrderOptimizer
from garage.sampler import Sampler
from garage.experiment import MetaEvaluator
from garage.tf.policies import GaussianMLPPolicy
from utils.metaworld_env_wrapper import NormalizedEnvWrapper

print("------------------------ Run start ------------------------")
# mjpython train/trainRL2.py

if __name__ == "__main__":
    env_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["pick-place-v2-goal-observable"]
    env = env_cls(seed=0)

    norm_env = NormalizedEnvWrapper(
    env,
    scale_reward=1.0,
    normalize_obs=True,
    normalize_reward=True,
    expected_action_scale=1.0,
    flatten_obs=True,
    obs_alpha=0.001,
    reward_alpha=0.001
    )

    print(type(norm_env))
    print("Action space:", norm_env.action_space)

    policy = GaussianMLPPolicy(env_spec=norm_env.spec, hidden_sizes=[64, 64])
    baseline = None
    sampler = Sampler(env=norm_env, policy=policy, max_path_length=100)
    meta_evaluator = MetaEvaluator(test_tasks=ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE.values())

    rl2_ppo = RL2PPO(
        meta_batch_size=10,
        task_sampler=env_cls,
        env_spec=norm_env,
        policy=policy,
        baseline=baseline,
        sampler=sampler,
        episodes_per_trial=10,
        scope="RL2PPO",
        discount=0.99,
        gae_lambda=1,
        center_adv=True,
        positive_adv=False,
        fixed_horizon=False,
        lr_clip_range=0.01,
        max_kl_step=0.01,
        optimizer_args=dict(),
        policy_ent_coeff=0.0,
        use_softplus_entropy=False,
        use_neg_logli_entropy=False,
        stop_entropy_gradient=False,
        entropy_method='no_entropy',
        meta_evaluator=meta_evaluator,
        n_epochs_per_eval=10,
        name="RL2PPO"
    )

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        rl2_ppo.train(env=env, viewer=viewer, episodes=1000)