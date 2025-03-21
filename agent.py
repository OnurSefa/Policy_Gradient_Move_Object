import torch
from torch import optim
from models import ActorHigh
import torch.nn.functional as F
import numpy as np


class Agent:
    def __init__(self, device, learning_rate, gamma, epsilon, epsilon_decay_rate, epsilon_decay_steps, entropy_coef=0.01, max_grad_norm=0.5):
        self.model = ActorHigh().to(device)
        self.rewards = []
        self.log_probs = []
        self.entropies = []
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.e = epsilon
        self.edr = epsilon_decay_rate
        self.eds = epsilon_decay_steps
        self.step = 0
        self.device = device
        self.gamma = gamma
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=500, gamma=0.9)

    def decide_action(self, state):
        # state = torch.from_numpy(state).float().to(self.device)
        # action_mean, act_std = self.model(state).chunk(2, dim=-1)
        # action_std = F.softplus(act_std) + self.e + 1e-6
        # eps = torch.randn_like(action_std)
        # dist = torch.distributions.Normal(action_mean, action_std)
        # action = action_mean + eps * action_std
        # log_prob = dist.log_prob(action).sum()
        # self.log_probs.append(log_prob)
        # return action
        state = torch.from_numpy(state).float().to(self.device)
        action_mean, log_std = self.model(state).chunk(2, dim=-1)
        action_std = torch.exp(log_std) + self.e
        dist = torch.distributions.Normal(action_mean, action_std)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        self.log_probs.append(log_prob)
        self.entropies.append(entropy)
        return torch.clamp(action, -1, 1)


    def update_model(self):
        returns = self.compute_returns()
        # returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        log_probs = torch.stack(self.log_probs)
        entropies = torch.stack(self.entropies)
        policy_loss = -(log_probs * returns).mean()
        entropy_loss = -self.entropy_coef * entropies.mean()
        loss = policy_loss + entropy_loss
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()
        self.step += 1
        if self.step % self.eds == 0:
            self.e *= self.edr

        self.log_probs = []
        self.rewards = []
        self.entropies = []
        return loss.item()

    def compute_returns(self):
        if len(self.rewards) > 1:
            rewards = np.array(self.rewards)
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        else:
            rewards = self.rewards
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        return torch.tensor(returns).float().to(self.device)

    def add_reward(self, reward):
        self.rewards.append(reward)
