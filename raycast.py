import numpy as np
import math

import numpy as np
import matplotlib.pyplot as plt
import math
import random

# === CONFIG ===
GRID_SIZE = 15
NUM_RAYS = 16
FOV = math.pi / 2  # 90 degrees
MAX_VIEW_DISTANCE = 10.0

# === GRID VALUES ===
EMPTY = 0
WALL = 1
GOAL = 2

# === SETUP WORLD ===
def create_grid(size):
    grid = np.zeros((size, size), dtype=int)
    grid[0, :] = WALL
    grid[-1, :] = WALL
    grid[:, 0] = WALL
    grid[:, -1] = WALL
    return grid

def place_random(grid, value):
    while True:
        x, y = random.randint(1, grid.shape[0] - 2), random.randint(1, grid.shape[1] - 2)
        if grid[y, x] == EMPTY:
            grid[y, x] = value
            return np.array([x + 0.5, y + 0.5])  # center in cell

# === RAYCASTING ===
def cast_ray(grid, origin, angle, max_dist=MAX_VIEW_DISTANCE):
    step_size = 0.1
    x, y = origin
    dx = math.cos(angle)
    dy = math.sin(angle)
    dist = 0.0
    
    while dist < max_dist:
        xi, yi = int(y), int(x)
        if grid[yi, xi] == WALL:
            return dist
        x += dx * step_size
        y += dy * step_size
        dist += step_size
    return max_dist

def get_depth_map(grid, agent_pos, agent_angle):
    depths = []
    start_angle = agent_angle - FOV / 2
    for i in range(NUM_RAYS):
        ray_angle = start_angle + (i / (NUM_RAYS - 1)) * FOV
        dist = cast_ray(grid, agent_pos.copy(), ray_angle)
        depths.append(dist)
    return np.array(depths)

# === VISUALIZATION ===
def render(grid, agent_pos, agent_angle, depths):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Top-down view
    ax1.imshow(grid, cmap='gray_r')
    ax1.set_title("Top-down View")
    ax1.plot(agent_pos[0], agent_pos[1], 'ro')

    # Rays
    start_angle = agent_angle - FOV / 2
    for i, d in enumerate(depths):
        angle = start_angle + (i / (NUM_RAYS - 1)) * FOV
        end_x = agent_pos[0] + d * math.cos(angle)
        end_y = agent_pos[1] + d * math.sin(angle)
        ax1.plot([agent_pos[0], end_x], [agent_pos[1], end_y], 'r-', linewidth=0.5)

    ax1.set_xlim(0, grid.shape[1])
    ax1.set_ylim(grid.shape[0], 0)

    # Bodycam view (1D depth map)
    ax2.bar(range(NUM_RAYS), depths, color='blue')
    ax2.set_ylim(0, MAX_VIEW_DISTANCE)
    ax2.set_title("Agent Vision (Depth Map)")

    plt.tight_layout()
    plt.show()

# === MAIN ===
if __name__ == '__main__':
    grid = create_grid(GRID_SIZE)
    goal_pos = place_random(grid, GOAL)
    agent_pos = place_random(grid, EMPTY)  # we'll track agent separately
    agent_angle = random.uniform(0, 2 * math.pi)

    depths = get_depth_map(grid, agent_pos, agent_angle)
    render(grid, agent_pos, agent_angle, depths)