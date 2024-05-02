import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import time 
import matplotlib.pyplot as plt
import wandb

from .actor_critic import ActorNetwork, CriticNetwork

from torch.utils.tensorboard import SummaryWriter

class PPO:
    def __init__(self, env, num_episodes, wandb='true', hidden_size=128, lr=3e-4, gamma=0.99, lam=0.95, clip_range=0.2, num_epochs=10, batch_size=32, epsilon = 0.1, noise_std=0.1, exploration_decay=1000):
        self.env = env

        if(wandb=='true'):
            self.wandb_run = wandb.init(project="aitclassproject", entity="czimbermark")
            self.wandb_config = wandb.config
            self.wandb_config.num_episodes = num_episodes
            self.wandb_config.batch_size = batch_size

        self.state_size = env.observation_space.shape[0] # state metrics
        self.action_size = env.action_space.shape[0] # action metrics 
        self.batch_size = batch_size

        self.actor_net = ActorNetwork(self.state_size, self.action_size, hidden_size)
        self.critic_net = CriticNetwork(self.state_size, hidden_size)
        self.actor_hidden = (torch.zeros(1, self.batch_size, self.actor_net.lstm.hidden_size),
                         torch.zeros(1, self.batch_size, self.actor_net.lstm.hidden_size))
        self.critic_hidden = (torch.zeros(1, self.batch_size, self.critic_net.lstm.hidden_size),
                          torch.zeros(1, self.batch_size, self.critic_net.lstm.hidden_size))
        self.noise_std = noise_std
        
        self.epsilon = epsilon # exploration value
        self.exploration_decay = exploration_decay
        self.clip_range = clip_range

        # values for broader understanding
        self.episode_successes = deque(maxlen=100) 
        self.goal_pos = deque(maxlen=3)
        self.target_distances = deque(maxlen=100)

        # opti
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), lr=lr)

        self.num_episodes = num_episodes
        self.gamma = gamma
        self.lam = lam
        self.num_epochs = num_epochs

        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)

    # gae fro esstimating the advantage function
    # how much better or worse an action is compared to the expected value of the state
    # goal ->  effective policy updates 
    def compute_gae(self, rewards, values, next_values, dones):
        """Compute Generalized Advantage Estimation (GAE)."""
        advantages = torch.zeros_like(values)
        last_advantage = torch.zeros(self.batch_size, 1, device=values.device)
        for t in reversed(range(values.size(0))):
            delta = rewards[t] + self.gamma * next_values[t] * (1 - dones[t]) - values[t]
            last_advantage = delta + self.gamma * self.lam * last_advantage
            advantages[:, 0] = last_advantage[:, 0]
        print(f"Advantages shape: {advantages.shape}")
        return advantages
    
    def update_actor(self, states, actions, advantages):
        # Compute the current action probabilities
        current_action_probs, self.actor_hidden = self.actor_net(states, self.actor_hidden)

        # Compute the old action probabilities
        with torch.no_grad():
            old_action_probs, _ = self.actor_net(states, self.actor_hidden)

        # Compute the ratio of current and old action probabilities
        ratios = (current_action_probs * actions).sum(dim=1, keepdim=True) / (old_action_probs * actions).sum(dim=1, keepdim=True)

        # Compute the clipped surrogate loss
        surrogate_loss = -torch.mean(torch.clamp(ratios, 1 - self.clip_range, 1 + self.clip_range) * advantages)

        print(f"Surrogate Loss: {surrogate_loss.item()}")
        print(f"Average Policy Ratio: {torch.mean(ratios).item()}")

        return surrogate_loss
    

    def update(self, states, actions, rewards, dones, next_states):
        states = torch.tensor(np.array(states), dtype=torch.float32).unsqueeze(1)
        actions = torch.tensor(np.array(actions), dtype=torch.float32)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32)
        dones = torch.tensor(np.array(dones), dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).unsqueeze(1)

        # Compute values and advantages for the entire batch
        value, self.critic_hidden = self.critic_net(states, self.critic_hidden)
        next_value, _ = self.critic_net(next_states, self.critic_hidden)

        # debug
        print(value.dim())
        print(next_value.dim())
        print(rewards.dim())
        print(dones.dim())
        print(value.shape)
        print(next_value.shape)
        print(rewards.shape)
        print(dones.shape)

        advantages = self.compute_gae(rewards, value, next_value, dones)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Update the critic network
        torch.autograd.set_detect_anomaly(True)
        critic_loss = nn.SmoothL1Loss()(value, rewards + self.gamma * next_value * (1 - dones))
        self.critic_optimizer.zero_grad() 
        critic_loss.backward() # retain_graph=True
        self.critic_optimizer.step()

        print(f"Critic Loss: {critic_loss.item()}")

        # Update the actor network
        surrogate_loss = self.update_actor(states, actions, advantages)
        self.actor_optimizer.zero_grad()
        surrogate_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_net.parameters(), 0.5)
        self.actor_optimizer.step()

        print(f"Surrogate Loss: {surrogate_loss.item()}")
        print(f"Average Policy Ratio: {torch.mean(self.ratios).item()}")
        
    # def wandb_visual 
    def log_metrics(self, episode):
        self.writer.add_scalar("Episode Reward", np.mean(self.episode_rewards), episode)
        self.writer.add_scalar("Episode Length", np.mean(self.episode_lengths), episode)
        self.writer.add_scalar("Episode Success", np.mean(self.episode_successes), episode)
        wandb.log({
            "Episode": episode,
            "Reward": np.mean(self.episode_rewards),
            "Average Reward": np.mean(self.episode_rewards),
            "Average Length": np.mean(self.episode_lengths),
            "Average Success": np.mean(self.episode_successes),
            "Epsilon": self.epsilon,
            "Goal Position": self.goal_pos[-1],
            "Target Distance": self.target_distances[-1]
        })


    # train method with envirnoment stepping
    def train(self, viewer, start):
        total_steps = 0
        for episode in range(self.num_episodes):
            state, _ = self.env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            episode_success = 0
            states, actions, rewards, dones, next_states, destinations, log_probs = [], [], [], [], [], [], []

            # Decay the exploration rate over time using an exponential decay
            self.epsilon = max(0.1, np.exp(-total_steps / self.exploration_decay))

            while not done:
                #  Exploration strategy
                if np.random.rand() < self.epsilon:
                    action = self.env.action_space.sample()
                    log_prob = torch.zeros(self.action_size)
                else:
                    action, log_prob = self.actor_net.forward(torch.tensor(state, dtype=torch.float32), self.actor_hidden)
                    action = action.detach().numpy()
                    print(" ppo action ")
                
                # actual step function 
                next_state, reward, done, info, _ = self.env.step(action)
                viewer.sync()

                # values for broader understanding
                self.goal_pos.append(np.linalg.norm(next_state[-3:]))
                self.target_distances.append(np.linalg.norm(next_state[4:7] - self.env._target_pos))
                self.episode_successes.append(done)

                # interface debug values for broader understanding timed with built in time
                current_time = time.time()
                time.sleep(0.005)
                elapsed_time = current_time - start
                if elapsed_time >= 9:
                    print(f"Current action: {action}, \nLast 5 reward: {rewards[-5:]},"
                           f"\nLast 2 finish position: {destinations[-2:]}, \nGoal position: {self.goal_pos}")
                    start = current_time  # Reset the start time

                # useful parameter saving 
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                dones.append(done)
                next_states.append(next_state) 
                log_probs.append(log_prob)

                episode_reward += reward
                episode_length += 1
                episode_success += done
                state = next_state
                total_steps += 1

                # update call if batch is reached or task is complete
                if len(states) == self.batch_size or done:
                    self.update(states, actions, rewards, dones, next_states)
                    states, actions, rewards, dones, next_states, log_probs = [], [], [], [], [], []
                
                if done or info:
                    destinations.append(state[:3])
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
                wandb.log({
                    "Episode": episode,
                    "Reward": episode_reward,
                    "Average Reward": np.mean(self.episode_rewards),
                    "Average Length": np.mean(self.episode_lengths),
                    "Average Success": np.mean(self.episode_successes)
                })  