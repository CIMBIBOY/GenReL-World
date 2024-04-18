import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import time 
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

class SawyerPickPlaceTrainer:
    def __init__(self, env, hidden_size=128, lr=3e-4, gamma=0.99, lam=0.95, clip_range=0.2, num_epochs=10, batch_size=32, epsilon = 0.1):
        self.env = env
        self.state_size = env.observation_space.shape[0] # state metrics
        self.action_size = env.action_space.shape[0] # action metrics 
        self.epsilon = epsilon # exploration value

        # values for broader understanding
        self.episode_successes = deque(maxlen=100) 
        self.goal_pos = deque(maxlen=1)
        self.target_distances = deque(maxlen=100)

        # Actor and critic networks
        self.actor_net = nn.Sequential(
            nn.Linear(self.state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.action_size),
            nn.Tanh()
        )
        self.critic_net = nn.Sequential(
            nn.Linear(self.state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        # opti
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), lr=lr)

        self.gamma = gamma
        self.lam = lam
        self.clip_range = clip_range
        self.num_epochs = num_epochs
        self.batch_size = batch_size

        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)

        # tensorboard - TODO
        self.writer = SummaryWriter()

    # gae fro esstimating the advantage function
    # how much better or worse an action is compared to the expected value of the state
    # goal ->  effective policy updates 
    def compute_gae(self, rewards, values, next_values, dones):
        """Compute Generalized Advantage Estimation (GAE)."""
        advantages = np.zeros_like(rewards)
        last_advantage = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_values[t] * (1 - dones[t]) - values[t]
            advantages[t] = delta + self.gamma * self.lam * last_advantage
            last_advantage = advantages[t]
        return advantages

    # update fucntion to compute state action values
    def update(self, states, actions, rewards, dones, next_states):
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(np.array(actions), dtype=torch.float32)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32)
        dones = torch.tensor(np.array(dones), dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)

        ''' Debug
        print(f"States shape: {states.shape}")
        print(f"Actions shape: {actions.shape}")
        print(f"Rewards shape: {rewards.shape}")
        print(f"Dones shape: {dones.shape}")
        print(f"Next states shape: {next_states.shape}")
        '''

        # Compute values and advantages
        values = self.critic_net(states).squeeze()
        next_values = self.critic_net(next_states).squeeze()
        advantages = self.compute_gae(rewards, values, next_values, dones)

        last_advantage = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_values[t] * (1 - dones[t]) - values[t]
            advantages[t] = delta + self.gamma * self.lam * last_advantage
            last_advantage = advantages[t]

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = torch.tensor(advantages, dtype=torch.float32).unsqueeze(1)

        # print(f"Advantages shape: {advantages.shape}")

        # Update the actor net
        action_probs = self.actor_net(states)
        log_probs = torch.log(action_probs)
        actor_loss = -torch.mean(torch.sum(log_probs * actions, dim=1, keepdim=True) * advantages)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

       
        # Update the critic net
        critic_loss = nn.MSELoss()(values, rewards + self.gamma * next_values * (1 - dones))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    # tensorboard log function - TODO
    def log_metrics(self, episode):
        self.writer.add_scalar("Episode Reward", np.mean(self.episode_rewards), episode)
        self.writer.add_scalar("Episode Length", np.mean(self.episode_lengths), episode)
        self.writer.add_scalar("Episode Success", np.mean(self.episode_successes), episode)

    # def wandb_visual - ToDo

    # train method with envirnoment stepping
    def train(self, num_episodes, viewer, start):
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            episode_success = 0
            states, actions, rewards, dones, next_states = [], [], [], [], []

            if episode == 0: # explartion decay
                self.epsilon = max(0.1, 1.0 - episode / 100)

            while not done:
                # explore first 100
                # if episode < 50:
                    # action = self.env.action_space.sample()
                # Use epsilon-greedy exploration
                if np.random.rand() < self.epsilon:
                    action = self.env.action_space.sample()
                    
                else:
                    '''Actions optimized by PPO '''
                    action = self.actor_net(torch.tensor(state, dtype=torch.float32)).detach().numpy()

                # actual step function 
                next_state, reward, done, info, _ = self.env.step(action)
                viewer.sync()

                # values for broader understanding
                self.goal_pos.append(np.linalg.norm(next_state[-3:]))
                self.target_distances.append(np.linalg.norm(next_state[4:7] - self.env._target_pos))
                self.episode_successes.append(done)

                # interface debug values for broader understanding timed with built in time
                current_time = time.time()
                time.sleep(0.01)
                elapsed_time = current_time - start
                if elapsed_time >= 15:
                    # Debug
                    print(f"Current state: {state}")
                    print(f"Current action: {action}")
                    print(f"Reward: {reward}")
                    print(f"Done: {done}")
                    print(f"Info: {info}")
                    print(f"State goal pos: {self.env._get_pos_goal}")
                    print(f"Next state goal Pos : {self.goal_pos}")
                    # print(f"Target Distance : {self.target_distances}")
                    start = current_time  # Reset the start time

                # useful parameter saving 
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                dones.append(done)
                next_states.append(next_state) 
                episode_reward += reward
                episode_length += 1
                episode_success += done
                state = next_state

                # update call if batch is reached or task is complete
                if len(states) == self.batch_size or done:
                    self.update(states, actions, rewards, dones, next_states)
                    states, actions, rewards, dones, next_states = [], [], [], [], []
                
                if done or info:
                    state, _ = self.env.reset()

            # important parameter adding 
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.episode_successes.append(episode_success)

            # important parameter logging
            if episode % 10 == 0:
                print(f"Episode {episode}, Reward: {episode_reward:.2f}, Average Reward: {np.mean(self.episode_rewards):.2f}, "
                    f"Average Length: {np.mean(self.episode_lengths):.2f}, Average Success: {np.mean(self.episode_successes):.2f}")
                self.log_metrics(episode)   