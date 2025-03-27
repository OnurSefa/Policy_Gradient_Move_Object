import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np


class QNetwork(nn.Module):
    """Critic network for SAC, estimates Q-values."""

    def __init__(self, state_dim=6, action_dim=2, hidden_dim=256):
        super().__init__()

        # Q1 architecture
        self.q1_network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Q2 architecture (for min double-Q trick)
        self.q2_network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Initialize the final layers with smaller weights
        self._initialize_weights(self.q1_network[-1], 1e-3)
        self._initialize_weights(self.q2_network[-1], 1e-3)

    def forward(self, state, action):
        """
        Returns Q-values from both critics given state and action
        """
        x = torch.cat([state, action], dim=-1)

        q1 = self.q1_network(x)
        q2 = self.q2_network(x)

        return q1, q2

    def q1(self, state, action):
        """
        Returns Q-value from first critic only
        """
        x = torch.cat([state, action], dim=-1)
        return self.q1_network(x)

    def _initialize_weights(self, layer, init_w):
        """Initialize weights for better training stability"""
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight, gain=init_w)
            nn.init.constant_(layer.bias, 0)


class GaussianPolicy(nn.Module):
    """Actor network for SAC, outputs mean and log_std of a Gaussian policy."""

    def __init__(self, state_dim=6, action_dim=2, hidden_dim=256,
                 log_std_min=-20, log_std_max=2, epsilon=1e-6):
        super().__init__()

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Mean and log_std heads
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

        # Initialize with smaller weights
        self._initialize_weights(self.mean_head, 1e-3)
        self._initialize_weights(self.log_std_head, 1e-3)

        # Parameters
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.epsilon = epsilon
        self.action_dim = action_dim

    def forward(self, state):
        """
        Returns sampled action, log probability, and mean action
        """
        features = self.policy_net(state)

        # Get mean and log_std
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        # Create normal distribution
        normal = torch.distributions.Normal(mean, std)

        # Sample using reparameterization trick
        x_t = normal.rsample()

        # Enforcing action bounds using tanh (maps to [-1, 1])
        y_t = torch.tanh(x_t)

        # Calculate log probability, accounting for tanh transformation
        log_prob = normal.log_prob(x_t)

        # Correction for tanh squashing
        # log_prob = log_prob - torch.log(1 - y_t.pow(2) + self.epsilon)
        log_prob = log_prob - 2 * (np.log(2) - x_t - F.softplus(-2 * x_t))
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        # Mean action (for deterministic evaluation)
        mean_action = torch.tanh(mean)

        return y_t, log_prob, mean_action

    def sample(self, state):
        """Sample action with noise (for training)"""
        action, log_prob, _ = self.forward(state)
        return action, log_prob

    def evaluate(self, state):
        """Return deterministic action (for evaluation)"""
        with torch.no_grad():
            _, _, action = self.forward(state)
        return action

    def _initialize_weights(self, layer, init_w):
        """Initialize weights for better training stability"""
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight, gain=init_w)
            nn.init.constant_(layer.bias, 0)

class ActorHigh(nn.Module):
    # def __init__(self, state_dim=6, action_dim=2, hidden_dim=64):
    #     super().__init__()
    #     self.state_handler = nn.Sequential(
    #         nn.Linear(state_dim, hidden_dim),
    #         nn.ReLU(),
    #         nn.Linear(hidden_dim, hidden_dim),
    #         nn.ReLU(),
    #         nn.Linear(hidden_dim, hidden_dim),
    #         nn.ReLU(),
    #         nn.Linear(hidden_dim, hidden_dim),
    #         nn.ReLU(),
    #     )
    #
    #     self.mean_head = nn.Linear(hidden_dim, action_dim)
    #     self.log_std_head = nn.Linear(hidden_dim, action_dim)
    #
    #     nn.init.xavier_uniform_(self.mean_head.weight, gain=0.01)
    #     nn.init.xavier_uniform_(self.log_std_head.weight, gain=0.01)
    #     nn.init.constant_(self.mean_head.bias, 0.0)
    #     nn.init.constant_(self.log_std_head.bias, -1.0)
    #
    # def forward(self, x):
    #     if x.ndim == 1:
    #         x = x.unsqueeze(0)
    #     features = self.state_handler(x)
    #     action_mean = self.mean_head(features)
    #     log_std = self.log_std_head(features)
    #     log_std = torch.clamp(log_std, -20, 2)
    #
    #     return torch.cat([action_mean, log_std], dim=-1)
    def __init__(self):
        super().__init__()
        self.state_handler = nn.Sequential(
            nn.Linear(33, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 4) # x_mean, y_mean, x_var, y_var
        )

    def forward(self, x):
        x = self.state_handler(x)
        return x

