# This code is taken from: https://raw.githubusercontent.com/rlworkgroup/garage/af57bf9c6b10cd733cb0fa9bfe3abd0ba239fd6e/src/garage/envs/normalized_env.py
#
# """"An environment wrapper that normalizes action, observation and reward."""
# type: ignore
import gym
import gym.spaces
import gym.spaces.utils
import numpy as np
import akro
from akro.box import Box

class NormalizedEnvWrapper(gym.Wrapper):
    """An environment wrapper for normalization.

    This wrapper normalizes action, and optionally observation and reward.

    Args:
        env (garage.envs.GarageEnv): An environment instance.
        scale_reward (float): Scale of environment reward.
        normalize_obs (bool): If True, normalize observation.
        normalize_reward (bool): If True, normalize reward. scale_reward is
            applied after normalization.
        expected_action_scale (float): Assuming action falls in the range of
            [-expected_action_scale, expected_action_scale] when normalize it.
        flatten_obs (bool): Flatten observation if True.
        obs_alpha (float): Update rate of moving average when estimating the
            mean and variance of observations.
        reward_alpha (float): Update rate of moving average when estimating the
            mean and variance of rewards.

    """

    def __init__(
        self,
        env,
        scale_reward=1.0,
        normalize_obs=False,
        normalize_reward=False,
        expected_action_scale=1.0,
        flatten_obs=True,
        obs_alpha=0.001,
        reward_alpha=0.001,
    ):
        super().__init__(env)

        self._scale_reward = scale_reward
        self._normalize_obs = normalize_obs
        self._normalize_reward = normalize_reward
        self._expected_action_scale = expected_action_scale
        self._flatten_obs = flatten_obs

        self._obs_alpha = obs_alpha
        flat_obs_dim = np.prod(env.observation_space.shape)
        # flat_obs_dim = gym.spaces.utils.flatdim(env.observation_space)
        self._obs_mean = np.zeros(flat_obs_dim)
        self._obs_var = np.ones(flat_obs_dim)

        self._reward_alpha = reward_alpha
        self._reward_mean = 0.0
        self._reward_var = 1.0
        
        self._action_space = Box(
            low=self.env.action_space.low,
            high=self.env.action_space.high,
            dtype=self.env.action_space.dtype
        )

    def _update_obs_estimate(self, obs):
        flat_obs = gym.spaces.utils.flatten(self.env.observation_space, obs)
        self._obs_mean = (
            1 - self._obs_alpha
        ) * self._obs_mean + self._obs_alpha * flat_obs
        self._obs_var = (
            1 - self._obs_alpha
        ) * self._obs_var + self._obs_alpha * np.square(flat_obs - self._obs_mean)

    def _update_reward_estimate(self, reward):
        self._reward_mean = (
            1 - self._reward_alpha
        ) * self._reward_mean + self._reward_alpha * reward
        self._reward_var = (
            1 - self._reward_alpha
        ) * self._reward_var + self._reward_alpha * np.square(
            reward - self._reward_mean
        )

    def _apply_normalize_obs(self, obs):
        """Compute normalized observation.

        Args:
            obs (np.ndarray): Observation.

        Returns:
            np.ndarray: Normalized observation.

        """
        self._update_obs_estimate(obs)
        flat_obs = gym.spaces.utils.flatten(self.env.observation_space, obs)
        normalized_obs = (flat_obs - self._obs_mean) / (np.sqrt(self._obs_var) + 1e-8)
        if not self._flatten_obs:
            normalized_obs = gym.spaces.utils.unflatten(
                self.env.observation_space, normalized_obs
            )
        return normalized_obs

    def _apply_normalize_reward(self, reward):
        """Compute normalized reward.

        Args:
            reward (float): Reward.

        Returns:
            float: Normalized reward.

        """
        self._update_reward_estimate(reward)
        return reward / (np.sqrt(self._reward_var) + 1e-8)

    def reset(self, **kwargs):
        """Reset environment.

        Args:
            **kwargs: Additional parameters for reset.

        Returns:
            tuple:
                * observation (np.ndarray): The observation of the environment.
                * reward (float): The reward acquired at this time step.
                * done (boolean): Whether the environment was completed at this
                    time step.
                * infos (dict): Environment-dependent additional information.

        """
        ret = self.env.reset(**kwargs)
        if self._normalize_obs:
            return self._apply_normalize_obs(ret)
        else:
            return ret


    def step(self, action):
        """Feed environment with one step of action and get result.

        Args:
            action (np.ndarray): An action fed to the environment.

        Returns:
            tuple:
                * observation (np.ndarray): The observation of the environment.
                * reward (float): The reward acquired at this time step.
                * done (boolean): Whether the environment was completed at this
                    time step.
                * infos (dict): Environment-dependent additional information.

        """
        if isinstance(self.action_space, gym.spaces.Box):
            # rescale the action when the bounds are not inf
            lb, ub = self.action_space.low, self.action_space.high
            if np.all(lb != -np.inf) and np.all(ub != -np.inf):
                scaled_action = lb + (action + self._expected_action_scale) * (
                    0.5 * (ub - lb) / self._expected_action_scale
                )
                scaled_action = np.clip(scaled_action, lb, ub)
            else:
                scaled_action = action
        else:
            scaled_action = action
        try:
            next_obs, reward, done, info = self.env.step(scaled_action)
        except Exception as e:
            print(e)

        if self._normalize_obs:
            next_obs = self._apply_normalize_obs(next_obs)
        if self._normalize_reward:
            reward = self._apply_normalize_reward(reward)

        return next_obs, reward * self._scale_reward, done, info
    
    
    def _flatten_obs(self, obs):
        if isinstance(self.observation_space, gym.spaces.Box):
            return obs.flatten()
        else:
            return gym.spaces.utils.flatten(self.observation_space, obs)
    
    
    @property
    def action_space(self):
        return Box(low=self.env.action_space.low, high=self.env.action_space.high, dtype=self.env.action_space.dtype)
    
    @property
    def observation_space(self):
        obs_space = self.env.observation_space
        low = np.clip(obs_space.low, -10, 10)
        high = np.clip(obs_space.high, -10, 10)
        return Box(low, high, dtype=obs_space.dtype)

    @property
    def observation_space_flat_dim(self):
        return gym.spaces.utils.flatdim(self.observation_space)
