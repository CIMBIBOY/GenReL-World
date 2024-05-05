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
    def __init__(self, env, num_episodes, wandb_use=True, hidden_size=128, lr=3e-4, gamma=0.99, lam=0.95, clip_range=0.2, num_epochs=10, batch_size=32, epsilon = 0.1, entropy_coef=0.01, noise_std=0.1, exploration_decay=-5e-6):
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

        self.actor_net = ActorNetwork(self.state_size, self.action_size, hidden_size)
        self.critic_net = CriticNetwork(self.state_size, hidden_size)
        self.actor_hidden = (torch.zeros(1, self.batch_size, self.actor_net.lstm.hidden_size),
                            torch.zeros(1, self.batch_size, self.actor_net.lstm.hidden_size))
        self.critic_hidden = (torch.zeros(1, self.batch_size, self.critic_net.lstm.hidden_size),
                          torch.zeros(1, self.batch_size, self.critic_net.lstm.hidden_size))
        self.noise_std = noise_std
        self.advantages = torch.zeros((self.batch_size, 1))
        
        self.epsilon = epsilon # exploration value
        self.exploration_decay = exploration_decay
        self.clip_range = clip_range
        self.avg_return = -float('inf')  # Running average of episodic returns
        self.target_return = 2000.0  # Target average episodic return

        # values for broader understanding
        self.episode_successes = deque(maxlen=100) 
        self.goal_pos = deque(maxlen=3)
        self.target_distances = deque(maxlen=100)

        # opti
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), lr=lr)
        self.critic_loss = None
        self.actor_loss = None

        self.num_episodes = num_episodes
        self.gamma = gamma
        self.lam = lam
        self.entropy_coef = entropy_coef
        self.num_epochs = num_epochs

        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)

    # gae fro esstimating the advantage function
    # how much better or worse an action is compared to the expected value of the state
    # goal ->  effective policy updates 
    def compute_gae(self, rewards, values, next_values, dones):
        """Compute Generalized Advantage Estimation (GAE)."""
        advantages = torch.zeros(self.batch_size, 1, 1, device=values.device)
        last_advantage = torch.zeros(self.batch_size, 1, 1, device=values.device)

        for t in reversed(range(values.size(0))):
            delta = rewards[t] + self.gamma * next_values[t] * torch.logical_not(dones[t]) - values[t]
            last_advantage = delta + self.gamma * self.lam * last_advantage
            advantages[:, 0, 0] = last_advantage[:, 0, 0]

        # print(f"Advantages shape: {advantages.shape}")
        return advantages
    
   
    def update_critic(self, states, actions, rewards, dones, next_states):
        torch.autograd.set_detect_anomaly(True)
        with torch.set_grad_enabled(True):
            # stacking tensors, in lstm batch_first=True so stacking on dim=0 to achive (batc_size, seq_length, feature)
            states = torch.stack([torch.from_numpy(np.array(s)) for s in states], dim=0).float().unsqueeze(1) # adding seq lenght dim
            actions = torch.stack([torch.from_numpy(np.array(a)) for a in actions])
            rewards = torch.stack([torch.from_numpy(np.array(r)) for r in rewards])
            dones = torch.stack([torch.from_numpy(np.array(d)) for d in dones])
            next_states = torch.stack([torch.from_numpy(np.array(ns)) for ns in next_states], dim=0).float().unsqueeze(1)

            ''' debug
            print(states.dim())
            print(actions.dim())
            print(rewards.dim())
            print(dones.dim())
            print(next_states.dim())
            print(states.shape)
            print(actions.shape)
            print(rewards.shape)
            print(dones.shape)
            print(next_states.shape)

            print(f"States shape: {states.shape}")
            print(f"Critic hidden shape: {self.critic_hidden[0].shape}, {self.critic_hidden[1].shape}")
            '''

            # Compute values and advantages for the entire batch
            value, self.critic_hidden = self.critic_net(states, self.critic_hidden)
            next_value, _ = self.critic_net(next_states, self.critic_hidden)

            ''' debug
            print(value.dim())
            print(next_value.dim())
            print(rewards.dim())
            print(dones.dim())
            print(value.shape)
            print(next_value.shape)
            print(rewards.shape)
            print(dones.shape)
            '''

            self.advantages = self.compute_gae(rewards, value, next_value, dones)
            self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

            # Update the critic network
            self.critic_loss = nn.SmoothL1Loss()(value.squeeze(1), rewards.unsqueeze(1) + self.gamma * next_value.squeeze(1) * torch.logical_not(dones).unsqueeze(1)) 
            self.critic_optimizer.zero_grad() 
            self.critic_loss.backward(retain_graph=True) # retain_graph=True
            self.critic_optimizer.step()

            self.critic_hidden = (self.critic_hidden[0].detach(), self.critic_hidden[1].detach())  # Detach the critic hidden state for solving 
            # Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass error

            # print(f"Critic Loss: {self.critic_loss.item()}")


    def update_actor(self, states, actions):
        torch.autograd.set_detect_anomaly(True)
        with torch.set_grad_enabled(True):
            states = torch.stack([torch.from_numpy(np.array(s)) for s in states], dim=0).float().unsqueeze(1)
            actions = torch.stack([torch.from_numpy(np.array(a)) for a in actions])
            
            ''' debug
            print(states.dim())
            print(states.shape)
            '''

            # Compute the current action probabilities
            current_action_probs, self.actor_hidden = self.actor_net(states, self.actor_hidden)

            # Compute the old action probabilities
            with torch.no_grad():
                old_action_probs, _ = self.actor_net(states, (self.actor_hidden[0].detach(), self.actor_hidden[1].detach()))

            # Compute the ratio of current and old action probabilities
            ratios = (current_action_probs * actions).sum(dim=1, keepdim=True) / (old_action_probs * actions).sum(dim=1, keepdim=True)

            # Compute the clipped surrogate loss
            self.actor_loss = -torch.mean(torch.clamp(ratios, 1 - self.clip_range, 1 + self.clip_range) * self.advantages)
            self.actor_loss = self.actor_loss - self.entropy_coef * current_action_probs.mean()

            self.actor_optimizer.zero_grad()
            grads = torch.autograd.grad(self.actor_loss, self.actor_net.parameters(), retain_graph=True)
            for param, grad in zip(self.actor_net.parameters(), grads):
                param.grad = grad
            torch.nn.utils.clip_grad_norm_(self.actor_net.parameters(), 0.5) # retain_graph=True
            self.actor_optimizer.step()

            self.actor_hidden = (self.actor_hidden[0].detach(), self.actor_hidden[1].detach())  # Detach the actor hidden state

            # print(f"Surrogate Loss: {self.actor_loss.item()}")
            # print(f"Average Policy Ratio: {torch.mean(ratios).item()}")
        

    # def wandb_visual 
    def log_metrics(self, episode):
        wandb.log({
        "Episode": episode,
        "Reward": np.mean(self.episode_rewards),
        "Average Reward": np.mean(self.episode_rewards),
        "Average Length": np.mean(self.episode_lengths),
        "Average Success": np.mean(self.episode_successes),
        "Epsilon": self.epsilon,
        "Goal Position": self.goal_pos[-1],
        "Target Distance": self.target_distances[-1],
        "Critic Loss": self.critic_loss.item(),
        "Actor Loss": self.actor_loss.item()
        }, step=episode)

        # Plot the actor and critic losses
        wandb.log({
            "Critic Loss": {
                "value": self.critic_loss.item(),
                "x": episode
            },
            "Actor Loss": {
                "value": self.actor_loss.item(),
                "x": episode
            }
        }, commit=False)

        # Plot the episode rewards and lengths
        wandb.log({
            "Episode Reward": {
                "value": np.mean(self.episode_rewards),
                "x": episode
            },
            "Episode Length": {
                "value": np.mean(self.episode_lengths),
                "x": episode
            }
        }, commit=False)

        # Commit the logged data to Wandb
        wandb.log({})

    def log_metrics_info(self, steps, batch_reward, rewards):
        wandb.log({
        "Batch Reward": np.mean(batch_reward),
        "Avarage Reward": np.mean(rewards),
        "Epsilon": self.epsilon,
        "Target Distance": self.target_distances[-1],
        "Critic Loss": self.critic_loss.item(),
        "Actor Loss": self.actor_loss.item()
        }, step=steps)

        # Plot the actor and critic losses
        wandb.log({
            "Critic Loss": {
                "value": self.critic_loss.item(),
                "x": steps
            },
            "Actor Loss": {
                "value": self.actor_loss.item(),
                "x": steps
            }
        }, commit=False)

        # Commit the logged data to Wandb
        wandb.log({})


    # train method with envirnoment stepping
    def train(self, viewer, start):
        total_steps = 0
        log_steps = 0
        for episode in range(self.num_episodes):
            state, _ = self.env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            episode_success = 0
            states, actions, rewards, dones, next_states, destinations, log_probs = [], [], [], [], [], [], []
            allstates = []
            first_ppo = False

            while not done:
                # step 10000 with random actions
                if total_steps < 200000:
                    action = self.env.action_space.sample()
                    log_prob = torch.zeros(self.action_size)
                else:
                    # Decay the exploration rate using adaptive decay
                    self.epsilon = max(0.1, 1.0 - (self.avg_return / self.target_return))
                    if not first_ppo:
                        print(f"Epsilon init value: {self.epsilon}")
                    #  Exploration strategy
                    if np.random.rand() < self.epsilon:
                        action = self.env.action_space.sample()
                        log_prob = torch.zeros(self.action_size)
                    else:
                        if not first_ppo:
                            print("---------- First Policy Optimization ----------")
                            first_ppo = True
                        states_tensor = torch.stack([torch.from_numpy(np.array(s)) for s in allstates[-32:]], dim=0).float().unsqueeze(1)
                        action, (h_n, c_n) = self.actor_net(states_tensor, self.actor_hidden)
                        action = action.squeeze(1)
                        action = action[31].detach().numpy()
                        self.actor_hidden = (h_n.detach(), c_n.detach())
                        action += np.random.normal(0, self.noise_std, size=self.action_size)
                        action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
                
                # actual step function 
                next_state, reward, done, info, _ = self.env.step(action)
                viewer.sync()
                time.sleep(0.01) # 0.005 for faster
                
                '''
                # interface debug values for broader understanding timed with built in time
                current_time = time.time()
                elapsed_time = current_time - start
                if elapsed_time >= 9:
                    print(f"Current action: {action}, \nLast 5 reward: {rewards[-5:]},"
                           f"\nLast 2 finish position: {destinations[-2:]}, \nGoal position: {self.goal_pos}")
                    start = current_time  # Reset the start time
                '''

                # useful parameter saving 
                states.append(state)
                allstates.append(state)
                actions.append(action)
                rewards.append(reward)
                dones.append(done)
                next_states.append(next_state) 
                log_probs.append(log_prob)

                # values for broader understanding
                self.goal_pos.append(np.linalg.norm(next_state[-3:]))
                self.target_distances.append(np.linalg.norm(next_state[4:7] - self.env._target_pos))
                episode_reward += reward
                episode_length = total_steps
                episode_success += done
                state = next_state
                total_steps += 1

                # depending on looping on info or done
                if episode_length % 10000 == 0 : 
                    self.avg_return = np.mean(episode_reward)
                # self.avg_return = np.mean(self.episode_rewards)

                # update call if batch is reached or task is complete
                if len(states) == self.batch_size or done:
                    self.update_critic(states, actions, rewards, dones, next_states)
                    self.update_actor(states, actions)
                    states, actions, rewards, dones, next_states, log_probs = [], [], [], [], [], []
                
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
            print(f"Episode {episode}, Reward: {episode_reward:.2f}, Average Reward: {np.mean(self.episode_rewards):.2f}, "
                    f"Average Length: {np.mean(self.episode_lengths):.2f}, Average Success: {np.mean(self.episode_successes):.2f}")
            if(self.wandb_use == True):
                self.log_metrics(episode)