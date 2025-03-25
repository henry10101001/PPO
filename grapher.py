import matplotlib.pyplot as plt
from agent import Agent
import torch

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# setup
action_dim = 4
grid_sizes = list(range(1, 101))
param_counts = []

# loop over grid sizes
for size in grid_sizes:
    agent = Agent(action_dim=action_dim, grid_size=size)
    total_params = count_parameters(agent)
    param_counts.append(total_params)

# plot
plt.figure(figsize=(10, 6))
plt.plot(grid_sizes, param_counts, marker='o')
plt.title("Total Trainable Parameters vs Grid Size")
plt.xlabel("Grid Size")
plt.ylabel("Total Trainable Parameters")
plt.grid(True)
plt.tight_layout()
plt.show()
