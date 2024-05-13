import sys
import torch
import mujoco
import mujoco.viewer

from ppo import PPO
from network import FeedForwardNN
from eval_policy import eval_policy

def train(env, hyperparameters, actor_model, critic_model, wandb_use):
	"""
		Trains the model.
x
		Parameters:
			env - the environment to train on
			hyperparameters - a dict of hyperparameters to use, defined in main
			actor_model - the actor model to load in if we want to continue training
			critic_model - the critic model to load in if we want to continue training

		Return:
			None
	"""	
	print(f"Training", flush=True)

	with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
		
		print("Viewer initialized successfully.") 

		# Create a model for PPO.
		model = PPO(policy_class=FeedForwardNN, env=env, wandb_use=wandb_use, viewer=viewer, **hyperparameters)

		# Tries to load in an existing actor/critic model to continue training on
		if actor_model != '' and critic_model != '':
			print(f"Loading in {actor_model} and {critic_model}...", flush=True)
			model.actor.load_state_dict(torch.load(actor_model))
			model.critic.load_state_dict(torch.load(critic_model))
			print(f"Successfully loaded.", flush=True)
		elif actor_model != '' or critic_model != '': # Don't train from scratch if user accidentally forgets actor/critic model
			print(f"Error: Either specify both actor/critic models or none at all. We don't want to accidentally override anything!")
			sys.exit(0)
		else:
			print(f"Training from scratch.", flush=True)

		# Train the PPO model with a specified total timesteps
		# NOTE: You can change the total timesteps here, I put a big number just because
		# you can kill the process whenever you feel like PPO is converging
		model.learn(total_timesteps=200_000_000)

def test(env, actor_model):
	"""
		Tests the model.

		Parameters:
			env - the environment to test the policy on
			actor_model - the actor model to load in

		Return:
			None
	"""
	print(f"Testing {actor_model}", flush=True)

	# If the actor model is not specified, then exit
	if actor_model == '':
		print(f"Didn't specify model file. Exiting.", flush=True)
		sys.exit(0)

	# Extract out dimensions of observation and action spaces
	obs_dim = env.observation_space.shape[0]
	act_dim = env.action_space.shape[0]

	# Build our policy the same way we build our actor model in PPO
	policy = FeedForwardNN(obs_dim, act_dim)

	# Load in the actor model saved by the PPO algorithm
	policy.load_state_dict(torch.load(actor_model))

	# Evaluate our policy with a separate module, eval_policy, to demonstrate
	# that once we are done training the model/policy with ppo.py, we no longer need
	# ppo.py since it only contains the training algorithm. The model/policy itself exists
	# independently as a binary file that can be loaded in with torch.
	eval_policy(policy=policy, env=env, viewer=True)
