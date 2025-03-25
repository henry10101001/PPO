# --- Custom GridWorld Environment ---
class GridWorldEnv:
    def __init__(self, grid_size=5):
        self.grid_size = grid_size
        self.agent_pos = None
        self.goal_pos = None
        self.reset()

    def reset(self):
        while True:
            self.agent_pos = np.random.randint(0, self.grid_size, size=2).tolist()
            self.goal_pos = np.random.randint(0, self.grid_size, size=2).tolist()
            if self.agent_pos != self.goal_pos:
                break
        return self._get_obs()

    def step(self, action):
        if action == 0:
            self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
        elif action == 1:
            self.agent_pos[1] = min(self.grid_size - 1, self.agent_pos[1] + 1)
        elif action == 2:
            self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
        elif action == 3:
            self.agent_pos[0] = min(self.grid_size - 1, self.agent_pos[0] + 1)

        done = self.agent_pos == self.goal_pos
        reward = 1.0 if done else 0.0
        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        obs = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)
        ax, ay = self.agent_pos
        gx, gy = self.goal_pos
        obs[ay, ax] = [0, 0, 255]     # agent = red
        obs[gy, gx] = [0, 255, 0]     # goal = green
        return obs.transpose(2, 0, 1)  # to CHW format