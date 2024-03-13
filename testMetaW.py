import time
import mujoco
import mujoco.viewer
import metaworld
import random

# Initialize MetaWorld ML1 benchmark
ml1 = metaworld.ML1('pick-place-v2')
env_class = ml1.train_classes['pick-place-v2']

# Create environment instance and set a random task
env = env_class()
task = random.choice(ml1.train_tasks)
env.set_task(task)

# Initialize Mujoco viewer
with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    # Close the viewer automatically after 30 wall-seconds.
    start = time.time()
    while viewer.is_running() and time.time() - start < 30:
        step_start = time.time()

        # Step the MetaWorld environment
        obs = env.reset()  # Reset environment

        # Simple random control logic: Sample random actions within the action space
        action = env.action_space.sample()

        # Send actions to the environment
        obs, reward, done, info, _ = env.step(action)

        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = env.model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
