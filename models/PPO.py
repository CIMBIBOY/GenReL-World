import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import time 
import matplotlib.pyplot as plt
import wandb

from torch.utils.tensorboard import SummaryWriter


class PPO:
    def __init__(self, env, num_episodes, wandb_use=True, hidden_size=128, lr=3e-4, gamma=0.99, lam=0.95, clip_range=0.2, num_epochs=10, batch_size=1, epsilon = 0.1, noise_std=0.1, exploration_decay=-0.0005):
        self.env = env

        self.wandb_use = wandb_use
        if(self.wandb_use==True):
            self.wandb_run = wandb.init(project="aitclassproject", entity="czimbermark")
            self.wandb_config = wandb.config
            self.wandb_config.num_episodes = num_episodes
            self.wandb_config.batch_size = batch_size

        self.state_size = env.observation_space.shape[0] # state metrics
        self.action_size = env.action_space.shape[0] # action metrics 
        self.batch_size = batch_size

        self.epsilon = epsilon # exploration value
        self.clip_range = clip_range
        self.exploration_decay = exploration_decay

        # values for broader understanding
        self.episode_successes = deque(maxlen=100) 
        self.goal_pos = deque(maxlen=3)
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
        self.noise_std = noise_std
        self.advantages = None

        # opti
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), lr=lr)

        self.actor_loss = None
        self.critic_loss = None

        self.num_episodes = num_episodes
        self.gamma = gamma
        self.lam = lam
        self.num_epochs = num_epochs

        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)

        self.writer = SummaryWriter()

    # gae fro esstimating the advantage function
    # how much better or worse an action is compared to the expected value of the state
    # goal ->  effective policy updates 
    def compute_gae(self, rewards, values, next_values, dones):
        """Compute Generalized Advantage Estimation (GAE)."""
        advantages = torch.zeros_like(values)
        gae = 0

        for t in reversed(range(values.size(0))):
            delta = rewards[t] + self.gamma * next_values[t, 0] * torch.logical_not(dones[t]) - values[t, 0]
            gae = delta + self.gamma * self.lam * gae
            advantages[t, 0] = gae

        return advantages

    def update(self, states, actions, rewards, dones, next_states):
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(np.array(actions), dtype=torch.float32)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32)
        dones = torch.tensor(np.array(dones), dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)

        # Compute values and advantages
        values = self.critic_net(states)
        next_values = self.critic_net(next_states)
        self.advantages = self.compute_gae(rewards, values, next_values, dones)
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)
        self.advantages = self.advantages.clone().detach().requires_grad_(True)

        # Compute the current action probabilities
        current_action_probs = self.actor_net(states)

        # Compute the old action probabilities
        with torch.no_grad():
            old_action_probs = self.actor_net(states)

        # Compute the ratio of current and old action probabilities
        ratios = (current_action_probs * actions).sum(dim=1, keepdim=True) / (old_action_probs * actions).sum(dim=1, keepdim=True)

        # Compute the clipped surrogate loss
        self.actor_loss = -torch.mean(torch.clamp(ratios, 1 - self.clip_range, 1 + self.clip_range) * self.advantages)

        # Update the actor network
        self.actor_optimizer.zero_grad()
        self.actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_net.parameters(), 0.5)
        self.actor_optimizer.step()

        # Update the critic network
        self.critic_loss = nn.MSELoss()(values, rewards.unsqueeze(1) + self.gamma * next_values * torch.logical_not(dones).unsqueeze(1))
        self.critic_optimizer.zero_grad()
        self.critic_loss.backward()
        self.critic_optimizer.step()


     # def wandb_visual 
    def log_metrics(self, episode):
        self.writer.add_scalar("Average Reward", np.mean(self.episode_rewards), episode)
        self.writer.add_scalar("Average Length", np.mean(self.episode_lengths), episode)
        self.writer.add_scalar("Average Success", np.mean(self.episode_successes), episode)
        self.writer.add_scalar("Epsilon", self.epsilon, episode)
        self.writer.add_scalar("Critic Loss", self.critic_loss.item(), episode)
        self.writer.add_scalar("Actor Loss", self.actor_loss.item(), episode)
        wandb.log({
        "Average Reward": np.mean(self.episode_rewards),
        "Average Length": np.mean(self.episode_lengths),
        "Average Success": np.mean(self.episode_successes),
        "Epsilon": self.epsilon,
        "Advantages": self.advantages.mean().item(),
        }, step=episode)

        # Plot the actor and critic losses
        wandb.log({
            "Critic Loss": {
                "value": self.critic_loss.item()
            },
            "Actor Loss": {
                "value": self.actor_loss.item()
            }
        }, commit=False)

        # Plot the episode rewards and lengths
        wandb.log({
            "Episode Reward": {
                "value": np.mean(self.episode_rewards)
            },
            "Episode Length": {
                "value": np.mean(self.episode_lengths)
            }
        }, commit=False)

        # Commit the logged data to Wandb
        wandb.log({})

    def log_metrics_info(self, steps, batch_reward, rewards):
        self.writer.add_scalar("Average Reward", np.mean(self.episode_rewards), steps)
        self.writer.add_scalar("Average Length", np.mean(self.episode_lengths), steps)
        self.writer.add_scalar("Average Success", np.mean(self.episode_successes), steps)
        self.writer.add_scalar("Epsilon", self.epsilon, steps)
        self.writer.add_scalar("Critic Loss", self.critic_loss.item(), steps)
        self.writer.add_scalar("Actor Loss", self.actor_loss.item(), steps)
        # Log advantages
        self.writer.add_scalar("Advantages", self.advantages.mean().item(), steps)
        wandb.log({
        "Batch Reward": np.mean(batch_reward),
        "Avarage Reward": np.mean(rewards),
        "Epsilon": self.epsilon,
        "Advantages": self.advantages.mean().item(),
        }, step=steps)

        # Plot the actor and critic losses
        wandb.log({
            "Critic Loss": {
                "value": self.critic_loss.item()
            },
            "Actor Loss": {
                "value": self.actor_loss.item()
            }
        }, commit=False)

        # Commit the logged data to Wandb
        wandb.log({})


    # train method with envirnoment stepping
    def train(self, viewer):
        total_steps = 0
        log_steps = 0
        for episode in range(self.num_episodes):
            state, _ = self.env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            episode_success = 0
            states, actions, rewards, dones, next_states, destinations = [], [], [], [], [], []
            first_ppo = False

            while not done:
                if total_steps < 10000:
                    action = self.env.action_space.sample()
                else:
                    # Decay the exploration rate using inverse square root decay
                    self.epsilon = max(0.1, np.exp(self.exploration_decay * total_steps))
                    if(first_ppo == False):
                            print(f"Epsilon init value: {self.epsilon}")
                    #  Exploration strategy
                    if np.random.rand() < self.epsilon:
                        action = self.env.action_space.sample()
                    else:
                        if(first_ppo == False):
                            print("---------- First Policy Optimalization ----------")
                            first_ppo = True
                        action = self.actor_net(torch.tensor(state, dtype=torch.float32)).detach().numpy()
                        action += np.random.normal(0, self.noise_std, size=self.action_size)
                        action = np.clip(action, self.env.action_space.low, self.env.action_space.high)

                
                # actual step function 
                next_state, reward, done, info, _ = self.env.step(action)
                viewer.sync()
                time.sleep(0.008)

                # useful parameter saving 
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                dones.append(done)
                next_states.append(next_state) 

                # values for broader understanding
                self.goal_pos.append(np.linalg.norm(next_state[-3:]))
                self.target_distances.append(np.linalg.norm(next_state[4:7] - self.env._target_pos))
                episode_reward += reward
                episode_length += 1
                episode_success += done
                state = next_state
                total_steps += 1

                # update call if batch is reached or task is complete
                if len(states) == self.batch_size or done:
                    self.update(states, actions, rewards, dones, next_states)
                    states, actions, rewards, dones, next_states = [], [], [], [], []
                
                if done or info:
                    log_steps += 1
                    if self.wandb_use == True:
                        self.log_metrics_info(log_steps, rewards, episode_reward)
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
                if(self.wandb_use == True):
                    self.log_metrics(episode)