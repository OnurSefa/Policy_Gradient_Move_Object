import torch.nn as nn
import torch
import torch.nn.functional as F


class ActorHigh(nn.Module):
    def __init__(self, state_dim=6, action_dim=2, hidden_dim=256):
        super().__init__()
        # Increase hidden dimensions
        self.state_handler = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Separate heads for mean and log_std
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

        # Initialize the last layer with smaller weights
        nn.init.xavier_uniform_(self.mean_head.weight, gain=0.01)
        nn.init.xavier_uniform_(self.log_std_head.weight, gain=0.01)
        nn.init.constant_(self.mean_head.bias, 0.0)
        nn.init.constant_(self.log_std_head.bias, -1.0)  # Start with smaller std

    def forward(self, x):
        if x.ndim == 1:
            x = x.unsqueeze(0)
        features = self.state_handler(x)
        action_mean = self.mean_head(features)
        log_std = self.log_std_head(features)

        # Bound the log_std to prevent extremely large or small values
        log_std = torch.clamp(log_std, -20, 2)

        return torch.cat([action_mean, log_std], dim=-1)
    # def __init__(self):
    #     super().__init__()
    #     self.state_handler = nn.Sequential(
    #         nn.Linear(6, 128),
    #         nn.ReLU(),
    #         nn.Linear(128, 128),
    #         nn.ReLU(),
    #         nn.Linear(128, 128),
    #         nn.ReLU(),
    #         nn.Linear(128, 128),
    #         nn.ReLU(),
    #         nn.Linear(128, 4) # x_mean, y_mean, x_var, y_var
    #     )
    #
    # def forward(self, x):
    #     if x.ndim == 1:
    #         x = x.unsqueeze(0)
    #     x = self.state_handler(x)
    #     return x

