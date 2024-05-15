
"""
	main method for training PPO on Metaworld Pick-Place-V2
	references: https://medium.com/@eyyu/coding-ppo-from-scratch-with-pytorch-part-1-4-613dfc1b14c8
"""

from arguments import get_args
from train_test import train, test
import sys
sys.path.append('/Users/czimbermark/Documents/Reinf/MetaWorld/GenReL-World')
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE

# from scratch
# mjpython PPO_metaW/main.py
# from existing: 
# mjpython PPO_metaW/main.py --actor_model ppo_meta_actor_v7.pth --critic_model ppo_meta_critic_v7.pth

def main(args):
	"""
		The main function to run.

		Parameters:
			args - the arguments parsed from command line

		Return:
			None
	"""
	# NOTE: Here's where you can set hyperparameters for PPO. I don't include them as part of
	# ArgumentParser because it's too annoying to type them every time at command line. Instead, you can change them here.
	# To see a list of hyperparameters, look in ppo.py at function _init_hyperparameters
	hyperparameters = {
				'timesteps_per_batch': 2048, 
				'max_timesteps_per_episode': 200, 
				'gamma': 0.99, 
				'n_updates_per_iteration': 20,
				'lr': 6e-4, 
				'clip': 0.2,
				'render': True,
				'render_every_i': 1
			  }

	# Creates the environment we'll be running. If you want to replace with your own
	# custom environment, note that it must inherit Gym and have both continuous
	# observation and action spaces.
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

	# Train or test, depending on the mode specified
	if args.mode == 'train':
		train(env=env, hyperparameters=hyperparameters, actor_model=args.actor_model, critic_model=args.critic_model, wandb_use=True)
	else:
		test(env=env, actor_model=args.actor_model)

if __name__ == '__main__':
	args = get_args() # Parse arguments from command line
	main(args)
