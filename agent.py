import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import random
import time
import cv2

# --- Agent ---
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, action_dim, grid_size=10):
        super().__init__()

        self.critic = nn.Sequential(
            layer_init(nn.Linear(3 * grid_size * grid_size, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0)
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(3 * grid_size * grid_size, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, action_dim), std=0.01)
        )

    def get_value(self, x):
        x = x.reshape(x.size(0), -1) # Flatten the input to (batch_size, 3 * grid_size * grid_size)
        return self.critic(x / 255.0)

    def get_action_and_value(self, x, action=None):
        x = x.reshape(x.size(0), -1) # Flatten the input to (batch_size, 3 * grid_size * grid_size)
        logits = self.actor(x / 255.0)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x / 255.0)