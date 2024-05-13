import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import MultivariateNormal
import numpy as np
from collections import deque
import time 
import matplotlib.pyplot as plt
import wandb

from .network.actor_critic import LSTMActorNetwork, LSTMCriticNetwork
from .network.actor_critic_linear import Agent
from .network.GAE import GAE

from torch.utils.tensorboard import SummaryWriter

class PPO:
    def __init__(self, env, num_episodes, start_explore=100, wandb_use=True, hidden_size=128, lr=3e-4, gamma=0.99, lam=0.95, clip_range=0.2, batch_size=32, epsilon = 0.10, entropy_coef=0.01, noise_std=0.1, exploration_decay=-5e-6):
        self.env = env

        self.wandb_use = wandb_use
        if(self.wandb_use==True):
            self.wandb_run = wandb.init(project="aitclassproject", entity="czimbermark")
            self.wandb_config = wandb.config

        self.state_size = env.observation_space.shape[0] # state metrics
        self.action_size = env.action_space.shape[0] # action metrics 
        self.batch_size = batch_size

        ''' For LSTM Agent 
        self.actor_net = LSTMActorNetwork(self.state_size, self.action_size, hidden_size)
        self.critic_net = LSTMCriticNetwork(self.state_size, 1, hidden_size)
        self.actor_hidden = (torch.zeros(1, self.batch_size, self.actor_net.lstm.hidden_size),
                            torch.zeros(1, self.batch_size, self.actor_net.lstm.hidden_size))
        self.critic_hidden = (torch.zeros(1, self.batch_size, self.critic_net.lstm.hidden_size),
                          torch.zeros(1, self.batch_size, self.critic_net.lstm.hidden_size))
        '''
        # For Linear Agent
        self.actor_net = Agent(self.state_size, self.action_size, hidden_size)
        self.critic_net = Agent(self.state_size, 1, hidden_size)
        self.noise_std = noise_std
        
        self.epsilon = epsilon # exploration value
        self.exploration_decay = exploration_decay
        self.start_explore = start_explore
        self.clip_range = clip_range
        self.avg_return = -float('inf')  # Running average of episodic returns
        self.target_return = 1000.0  # Target average episodic return

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
        self.gae = GAE(n_workers=self.batch_size, worker_steps=1, gamma=self.gamma, lambda_=self.lam)

        # Initialize the covariance matrix used to query the actor for actions
        self.cov_var = torch.full(size=(self.action_size,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)

        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)

        self.writer = SummaryWriter()


    def update_critic(self, states, actions, rewards, dones, next_states):
        torch.autograd.set_detect_anomaly(True)
        with torch.set_grad_enabled(True):
            # stacking tensors, in lstm batch_first=True so stacking on dim=0 to achive (batc_size, seq_length, feature)
            states = torch.stack([torch.from_numpy(np.array(s)) for s in states], dim=0).float() # adding seq lenght dim
            actions = torch.stack([torch.from_numpy(np.array(a)) for a in actions]).float()
            rewards = torch.stack([torch.from_numpy(np.array(r)) for r in rewards]).float().unsqueeze(1)
            dones = torch.stack([torch.from_numpy(np.array(d)) for d in dones]).float().unsqueeze(1)
            next_states = torch.stack([torch.from_numpy(np.array(ns)) for ns in next_states], dim=0).float()

            ''' For LSTM 
            # Compute values and advantages for the entire batch
            values, self.critic_hidden = self.critic_net(states, self.critic_hidden)
            next_values, _ = self.critic_net(next_states, self.critic_hidden)
            
            values = values.squeeze(1).float() # removing seq_sen
            next_values = next_values.squeeze(1).float()
            '''
            values = self.critic_net(states)
            next_val = self.critic_net(next_states)
            # values = torch.transpose(values, 0, 1)
            # next_val = torch.transpose(next_val, 0, 1)

            # advantages = rewards - values

            advantages = self.gae(dones.numpy(), rewards.numpy(), values.detach().numpy())
            advantages = torch.from_numpy(advantages).to(values.device)

            advantages = (advantages - advantages.mean() + 1e-10) / (advantages.std() + 1e-10)

            # Update the critic network
            self.critic_loss = nn.MSELoss()(values, (rewards + self.gamma * next_val * torch.logical_not(dones))) 
            self.critic_optimizer.zero_grad() 
            self.critic_loss.backward(retain_graph=True) # retain_graph=True
            self.critic_optimizer.step()

            # detach for LSTMs
            # self.critic_hidden = (self.critic_hidden[0].detach(), self.critic_hidden[1].detach())  # Detach the critic hidden state for solving 
            # Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass error

            # print(f"Critic Loss: {self.critic_loss.item()}")
            return advantages

    def update_actor(self, states, actions, advantages, batch_log_probs):
        torch.autograd.set_detect_anomaly(True)
        with torch.set_grad_enabled(True):
            states = torch.stack([torch.from_numpy(s) for s in states], dim=0).float()
            actions = torch.stack([torch.from_numpy(a) for a in actions], dim=0).float()
            batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float32)
            
            # Calculate the log probabilities of batch actions using most recent actor network.
            # This segment of code is similar to that in get_action()
            mean = self.actor_net(states)
            dist = MultivariateNormal(mean, self.cov_mat)
            curr_log_probs = dist.log_prob(actions)

            ratios = torch.exp(curr_log_probs - batch_log_probs)

            # Calculate surrogate losses.
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_range, 1 + self.clip_range) * advantages

            # Calculate actor and critic losses.
            # NOTE: we take the negative min of the surrogate losses because we're trying to maximize
            # the performance function, but Adam minimizes the loss. So minimizing the negative
            # performance function maximizes it.
            self.actor_loss = (-torch.min(surr1, surr2)).mean()
				
            # Compute the current action probabilities
            # current_action_probs = self.actor_net(states)

            # Compute the old action probabilities
            # with torch.no_grad():
                # old_action_probs = self.actor_net(states)
           
            ''' For LSTM
            with torch.no_grad():
                old_action_probs, _ = self.actor_net(states, (self.actor_hidden[0].detach(), self.actor_hidden[1].detach()))
            '''

            # Compute the ratio of current and old action probabilities
            # ratios = torch.clip(current_action_probs / old_action_probs, 1 - self.clip_range, 1 + self.clip_range)
            # print(advantages.shape)
            # print(ratios.shape)
            # Compute the clipped surrogate loss
            # self.actor_loss = -torch.mean(torch.clamp(ratios, 1 - self.clip_range, 1 + self.clip_range) * advantages) 
            # self.actor_loss = self.actor_loss - self.entropy_coef * current_action_probs.mean()

            self.actor_optimizer.zero_grad()
            grads = torch.autograd.grad(self.actor_loss, self.actor_net.parameters(), retain_graph=True)
            for param, grad in zip(self.actor_net.parameters(), grads):
                param.grad = grad
            torch.nn.utils.clip_grad_norm_(self.actor_net.parameters(), 0.5) # retain_graph=True
            self.actor_optimizer.step()

            # Detach for LSTMs
            # self.actor_hidden = (self.actor_hidden[0].detach(), self.actor_hidden[1].detach())  # Detach the actor hidden state

            # print(f"Surrogate Loss: {self.actor_loss.item()}")
            # print(f"Average Policy Ratio: {torch.mean(ratios).item()}")
        

    # def wandb_visual 
    def log_metrics(self, episode, advantages):
        self.writer.add_scalar("Average Reward", np.mean(self.episode_rewards), episode)
        self.writer.add_scalar("Average Length", np.mean(self.episode_lengths), episode)
        self.writer.add_scalar("Average Success", np.mean(self.episode_successes), episode)
        self.writer.add_scalar("Epsilon", self.epsilon, episode)
        self.writer.add_scalar("Critic Loss", self.critic_loss.item(), episode)
        self.writer.add_scalar("Actor Loss", self.actor_loss.item(), episode)

        batch_advantages = torch.cat(advantages, dim=0)
        mean_advantages = batch_advantages.mean().item()

        wandb.log({
        "Epsilon": self.epsilon,
        "Advantages": mean_advantages,
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

    # train method with envirnoment stepping
    def train(self):
        total_steps = 0
        first_ppo = False
        states, actions, rewards, dones, next_states, advantages, batch_log_probs = [], [], [], [], [], [], []
        allstates, advantages = [], []
        for episode in range(self.num_episodes):
            state, _= self.env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            episode_success = 0
        
            while not done:
                if total_steps < 100 * self.start_explore:
                    action = self.env.action_space.sample()
                    mean = self.actor_net(state)
                    dist = MultivariateNormal(mean, self.cov_mat)
                    log_prob = dist.log_prob(torch.from_numpy(action)).item()
                # Decay the exploration rate using adaptive decay
                #  Exploration strategy
                else: 
                    if np.random.rand() < self.epsilon:
                        action = self.env.action_space.sample()
                        mean = self.actor_net(state)
                        dist = MultivariateNormal(mean, self.cov_mat)
                        log_prob = dist.log_prob(torch.from_numpy(action)).item()
                    else:
                        if not first_ppo:
                            print("---------- First Policy Optimization ----------")
                            first_ppo = True
                        # states_tensor = torch.stack([torch.from_numpy(np.array(s)) for s in allstates[-self.batch_size:]], dim=0).float().unsqueeze(1)
                        mean = self.actor_net(state)

                        dist = MultivariateNormal(mean, self.cov_mat)

                        # Sample an action from the distribution
                        action = dist.sample()

                        # Calculate the log probability for that action
                        log_prob = dist.log_prob(action).item()

                        # Get the action for the last state in the batch
                        # action = action_values[-1, :]
                        action = action.detach().numpy()

                        # Add noise and clip the action
                        # noise = np.random.normal(0, self.noise_std, size=self.action_size)
                        # action += noise
                        action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
                
                # actual step function 
                next_state, reward, done, info, _ = self.env.step(action)
                # time.sleep(0.01) # 0.005 for faster

                # useful parameter saving 
                states.append(state)
                allstates.append(state)
                actions.append(action)
                rewards.append(reward)
                dones.append(done)
                next_states.append(next_state) 
                batch_log_probs.append(log_prob)

                # values for broader understanding
                episode_reward += reward
                episode_length = total_steps
                episode_success += done
                state = next_state
                total_steps += 1

                self.avg_return = np.mean(self.episode_rewards)

                # update call if batch is reached or task is complete
                if len(states) == self.batch_size:
                    advantage = self.update_critic(states, actions, rewards, dones, next_states)
                    advantages.append(advantage)
                    self.update_actor(states, actions, advantage, batch_log_probs)
                    states, actions, rewards, dones, next_states, batch_log_probs = [], [], [], [], [], [] 
                    
                if done or info:
                    state, _ = self.env.reset()
                    
            # important parameter adding    
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.episode_successes.append(episode_success)

            # important parameter logging
            print(f"Episode {episode}, Reward: {episode_reward:.2f}, Average Reward: {np.mean(self.episode_rewards):.2f}, "
                    f"Average Length: {np.mean(self.episode_lengths):.2f}, Average Success: {np.mean(self.episode_successes):.2f}")
            if(self.wandb_use == True):
                self.log_metrics(episode, advantages)