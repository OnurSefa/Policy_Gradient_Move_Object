import torch
from torch import optim
from models import *
import torch.nn.functional as F
import numpy as np
from collections import deque
import random


class ReplayBuffer:
    """Experience replay buffer for SAC"""

    def __init__(self, capacity=100000, device="cpu"):
        self.buffer = deque(maxlen=capacity)
        self.device = device

    def add(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Sample a batch of experiences"""
        experiences = random.sample(self.buffer, k=min(batch_size, len(self.buffer)))

        # Convert to tensors
        states = torch.FloatTensor(np.array([e[0] for e in experiences])).to(self.device)
        actions = torch.FloatTensor(np.array([e[1] for e in experiences])).to(self.device)
        rewards = torch.FloatTensor(np.array([e[2] for e in experiences])).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array([e[3] for e in experiences])).to(self.device)
        dones = torch.FloatTensor(np.array([float(e[4]) for e in experiences])).unsqueeze(1).to(self.device)

        return states, actions, rewards, next_states, dones

    def clear(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)


class SACAgent:
    """Soft Actor-Critic agent for continuous control"""

    def __init__(
            self,
            state_dim=6,
            action_dim=2,
            hidden_dim=256,
            buffer_size=100000,
            batch_size=256,
            learning_rate=3e-4,
            gamma=0.99,
            tau=0.005,
            alpha=0.2,
            target_update_interval=1,
            automatic_entropy_tuning=True,
            epsilon=0.,
            epsilon_decay_rate=0.995,
            epsilon_decay_steps=1,
            device="cpu"
    ):
        self.device = device
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.target_update_interval = target_update_interval
        self.automatic_entropy_tuning = automatic_entropy_tuning

        # Initialize networks
        self.policy = GaussianPolicy(state_dim, action_dim, hidden_dim).to(device)
        self.critic = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target = QNetwork(state_dim, action_dim, hidden_dim).to(device)

        # Initialize target network with same weights
        self.hard_update(self.critic_target, self.critic)

        # Setup optimizers
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)

        # Setup automatic entropy tuning
        if self.automatic_entropy_tuning:
            # Target entropy is -dim(A) (e.g. -2 for a 2D action space)
            self.target_entropy = -action_dim
            # Optimizable log(alpha) parameter
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=learning_rate)

        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size, device)

        # Training metrics
        self.critic_loss_history = []
        self.policy_loss_history = []
        self.alpha_loss_history = []
        self.mean_q_history = []

        # Update counter
        self.updates = 0

        # Epsilon
        self.epsilon = epsilon
        self.epsilon_decay_rate = epsilon_decay_rate
        self.epsilon_decay_steps = epsilon_decay_steps

    def decide_action(self, state, evaluate=False):
        """Select action from policy"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        if evaluate:
            with torch.no_grad():
                action = self.policy.evaluate(state)
        else:
            with torch.no_grad():
                action, _ = self.policy.sample(state)

        return action.clamp(-1, 1).cpu().numpy()[0]

    def add_experience(self, state, action, reward, next_state, done):
        """Add experience to replay buffer"""
        self.replay_buffer.add(state, action, reward, next_state, done)

    def update_model(self, num_updates=1):
        """Update SAC parameters using a batch from replay buffer"""
        if len(self.replay_buffer) < self.batch_size:
            return 0, 0, 0  # Not enough samples

        total_critic_loss = 0
        total_policy_loss = 0
        total_alpha_loss = 0

        for _ in range(num_updates):
            # Sample batch from buffer
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

            # Get current alpha value
            if self.automatic_entropy_tuning:
                alpha = self.log_alpha.exp()
            else:
                alpha = self.alpha

            # --------------------- Update Critics --------------------- #

            # Get next actions and log probs from current policy
            with torch.no_grad():
                next_actions, next_log_probs, _ = self.policy(next_states)

                # Target Q-values
                next_q1_target, next_q2_target = self.critic_target(next_states, next_actions)
                min_next_q_target = torch.min(next_q1_target, next_q2_target)

                # Entropy regularized target
                next_q_target = rewards + (1 - dones) * self.gamma * (min_next_q_target - alpha * next_log_probs)

            # Current Q estimates
            current_q1, current_q2 = self.critic(states, actions)

            # Compute critic loss (MSE)
            critic_loss = F.mse_loss(current_q1, next_q_target) + F.mse_loss(current_q2, next_q_target)

            # Update critics
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # --------------------- Update Actor --------------------- #

            # Get actions and log probs from current policy
            new_actions, log_probs, _ = self.policy(states)

            # Get Q-values for new actions
            q1_new, q2_new = self.critic(states, new_actions)
            min_q_new = torch.min(q1_new, q2_new)

            # Compute policy loss
            policy_loss = (alpha * log_probs - min_q_new).mean()

            # Update policy
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            # --------------------- Update Alpha --------------------- #

            if self.automatic_entropy_tuning:
                # Compute alpha loss
                alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()

                # Update alpha
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()

                # Update alpha value
                alpha = self.log_alpha.exp()

                total_alpha_loss += alpha_loss.item()
            else:
                alpha_loss = torch.tensor(0.0)

            # --------------------- Update Targets --------------------- #

            self.updates += 1
            if self.updates % self.target_update_interval == 0:
                self.soft_update(self.critic_target, self.critic)

            # --------------------- Update Epsilon --------------------- #
            if self.updates % self.epsilon_decay_steps == 0:
                self.epsilon *= self.epsilon_decay_rate

            # Save metrics
            total_critic_loss += critic_loss.item()
            total_policy_loss += policy_loss.item()
            self.mean_q_history.append(min_q_new.mean().item())

        # Track metrics
        if num_updates > 0:
            self.critic_loss_history.append(total_critic_loss / num_updates)
            self.policy_loss_history.append(total_policy_loss / num_updates)
            if self.automatic_entropy_tuning:
                self.alpha_loss_history.append(total_alpha_loss / num_updates)

        # Return average losses
        avg_critic_loss = total_critic_loss / num_updates if num_updates > 0 else 0
        avg_policy_loss = total_policy_loss / num_updates if num_updates > 0 else 0
        avg_alpha_loss = total_alpha_loss / num_updates if num_updates > 0 else 0

        return avg_critic_loss, avg_policy_loss, avg_alpha_loss

    def soft_update(self, target, source):
        """Soft update for target network parameters"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    @staticmethod
    def hard_update(target, source):
        """Hard update for target network parameters"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def save(self, path):
        """Save model parameters"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'log_alpha': self.log_alpha if self.automatic_entropy_tuning else torch.tensor([self.alpha]),
            'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict() if self.automatic_entropy_tuning else None,
            'updates': self.updates,
        }, path)

    def load(self, path):
        """Load model parameters"""
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])

        if self.automatic_entropy_tuning:
            self.log_alpha = checkpoint['log_alpha']
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
        else:
            self.alpha = checkpoint['log_alpha'].exp().item()

        self.updates = checkpoint['updates']


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

    def update_model_mp(self, states, actions, returns, epsilon):
        # states = states.to(self.device)
        # actions = actions.to(self.device)
        # returns = returns.to(self.device)
        # self.model = self.model.to(self.device)
        action_mean, log_std = self.model(states).chunk(2, dim=-1)
        action_std = torch.exp(log_std) + epsilon
        dist = torch.distributions.Normal(action_mean, action_std)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropies = dist.entropy().sum(dim=-1)
        policy_loss = -(log_probs * returns.flatten()).mean()
        # entropy_loss = -self.entropy_coef * entropies.mean()
        # loss = policy_loss + entropy_loss
        loss = policy_loss
        self.optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()
        return loss.item()

    def compute_returns(self):
        # if len(self.rewards) > 1:
        #     rewards = np.array(self.rewards)
        #     rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        # else:
        #     rewards = self.rewards
        rewards = self.rewards
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        return torch.tensor(returns).float().to(self.device)

    def add_reward(self, reward):
        self.rewards.append(reward)
