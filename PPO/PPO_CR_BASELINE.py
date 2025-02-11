import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import gymnasium as gym

class ActorCritic(nn.Module):
    def __init__(self, input_shape, action_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        conv_out = self._get_conv_out(input_shape)
        
        self.fc = nn.Sequential(
            nn.Linear(conv_out, 512),
            nn.ReLU()
        )
        self.actor = nn.Linear(512, action_dim * 2)  # Mean and log_std for each action
        self.critic = nn.Linear(512, 1)

    def _get_conv_out(self, shape):
        dummy = torch.zeros(1, *shape)
        return self.conv(dummy).shape[1]

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        actor_out = self.actor(x)
        critic_out = self.critic(x)
        return actor_out, critic_out

class PPOAgent:
    def __init__(self, env, config=None):
        self.env = env
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Default configuration
        self.config = {
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_epsilon': 0.2,
            'entropy_coef': 0.01,
            'lr': 3e-4,
            'update_epochs': 10,
            'batch_size': 64,
            'normalize_advantages': True,
        }
        if config: self.config.update(config)
        
        # Network setup
        self.action_dim = self.env.action_space.shape[0]
        input_shape = (3, 96, 96)  # Channels-first
        self.model = ActorCritic(input_shape, self.action_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config['lr'])
        
        # Memory buffers
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []

    def act(self, state):
        state = torch.tensor(state, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            actor_out, critic_out = self.model(state)
            
        mean, log_std = actor_out.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, min=-20, max=2)
        std = log_std.exp()
        dist = Normal(mean, std)
        
        raw_action = dist.sample()
        log_prob = dist.log_prob(raw_action).sum(dim=-1)
        
        # Apply action transformations
        steering = torch.tanh(raw_action[0, 0])
        gas = torch.sigmoid(raw_action[0, 1])
        brake = torch.sigmoid(raw_action[0, 2])
        action = torch.tensor([steering, gas, brake], device='cpu').numpy()
        
        # Adjust log probability for transformations
        log_prob -= torch.log(1 - steering**2 + 1e-6)
        log_prob -= torch.log(gas * (1 - gas) + 1e-6)
        log_prob -= torch.log(brake * (1 - brake) + 1e-6)
        
        return action, log_prob.item(), critic_out.item()

    def remember(self, state, action, reward, done, log_prob, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def update(self):
        # Convert to tensors
        states = torch.tensor(np.array(self.states), dtype=torch.float32).permute(0, 3, 1, 2).to(self.device)
        actions = torch.tensor(np.array(self.actions), dtype=torch.float32).to(self.device)
        old_log_probs = torch.tensor(np.array(self.log_probs), dtype=torch.float32).to(self.device)
        values = torch.tensor(np.array(self.values), dtype=torch.float32).to(self.device)
        rewards = torch.tensor(np.array(self.rewards), dtype=torch.float32).to(self.device)
        dones = torch.tensor(np.array(self.dones), dtype=torch.float32).to(self.device)

        # Calculate advantages and returns
        advantages = []
        returns = []
        gae = 0
        next_value = 0
        next_done = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_done = 1.0 - dones[t]
                next_value = values[t]
            else:
                next_non_done = 1.0 - dones[t+1]
                next_value = values[t+1]
            
            delta = rewards[t] + self.config['gamma'] * next_value * next_non_done - values[t]
            gae = delta + self.config['gamma'] * self.config['gae_lambda'] * next_non_done * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
        
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        
        if self.config['normalize_advantages']:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Optimize policy for K epochs
        for _ in range(self.config['update_epochs']):
            indices = torch.randperm(len(states))
            
            for start in range(0, len(states), self.config['batch_size']):
                end = start + self.config['batch_size']
                idx = indices[start:end]
                
                batch_states = states[idx]
                batch_actions = actions[idx]
                batch_old_log_probs = old_log_probs[idx]
                batch_advantages = advantages[idx]
                batch_returns = returns[idx]
                
                # Get current policy outputs
                actor_out, critic_out = self.model(batch_states)
                mean, log_std = actor_out.chunk(2, dim=-1)
                log_std = torch.clamp(log_std, min=-20, max=2)
                std = log_std.exp()
                dist = Normal(mean, std)
                
                # Calculate new log probabilities
                raw_actions = torch.zeros_like(batch_actions).to(self.device)
                raw_actions[:, 0] = torch.atanh(batch_actions[:, 0])  # Inverse tanh
                raw_actions[:, 1] = torch.log(batch_actions[:, 1]/(1 - batch_actions[:, 1]))  # Inverse sigmoid
                raw_actions[:, 2] = torch.log(batch_actions[:, 2]/(1 - batch_actions[:, 2]))
                
                new_log_probs = dist.log_prob(raw_actions).sum(dim=-1)
                
                # Apply transformation corrections
                new_log_probs -= torch.log(1 - batch_actions[:, 0]**2 + 1e-6)
                new_log_probs -= torch.log(batch_actions[:, 1] * (1 - batch_actions[:, 1]) + 1e-6)
                new_log_probs -= torch.log(batch_actions[:, 2] * (1 - batch_actions[:, 2]) + 1e-6)
                
                # Calculate ratios
                ratios = (new_log_probs - batch_old_log_probs).exp()
                
                # Policy loss
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.config['clip_epsilon'], 
                                  1 + self.config['clip_epsilon']) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(critic_out.squeeze(), batch_returns)
                
                # Entropy loss
                entropy = dist.entropy().mean()
                
                # Total loss
                loss = policy_loss + 0.5 * value_loss - self.config['entropy_coef'] * entropy
                
                # Update
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()

        # Clear memory
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))