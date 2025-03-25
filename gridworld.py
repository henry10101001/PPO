import time
import numpy as np
import cv2
import torch

# --- GridWorld Environment ---
class GridWorldEnv:
    def __init__(self, grid_size=10):
        self.grid_size = grid_size
        self.agent_pos = None
        self.goal_pos = None
        self.reset()

    def reset(self):
        self.agent_pos = np.random.randint(0, self.grid_size, size=2).tolist()
        while True:
            self.goal_pos = np.random.randint(0, self.grid_size, size=2).tolist()
            if self.agent_pos != self.goal_pos:
                break
        return self._get_obs()

    def step(self, action):
        if action == 0:   # up
            self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
        elif action == 1: # down
            self.agent_pos[1] = min(self.grid_size - 1, self.agent_pos[1] + 1)
        elif action == 2: # left
            self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
        elif action == 3: # right
            self.agent_pos[0] = min(self.grid_size - 1, self.agent_pos[0] + 1)

        done = self.agent_pos == self.goal_pos
        if done:
            reward = 1.0  # Goal reached, positive reward
        else:
            reward = -0.01 # Step cost, small negative reward for each step

        return self._get_obs(), reward, done, {}

    def render(self, scale=64):
        img = self._get_obs().transpose(1, 2, 0)  # back to HWC
        img = cv2.resize(img, (self.grid_size * scale, self.grid_size * scale), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("GridWorld", img)
        cv2.waitKey(200)  # 200 ms delay for visible animation

    def render_frame(self, scale=64):
        img = self._get_obs().transpose(1, 2, 0)  # CHW to HWC
        img = cv2.resize(img, (self.grid_size * scale, self.grid_size * scale), interpolation=cv2.INTER_NEAREST)
        return img

    def _get_obs(self):
        obs = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)
        ax, ay = self.agent_pos
        gx, gy = self.goal_pos
        obs[ay, ax] = [0, 0, 255]     # agent = red
        obs[gy, gx] = [0, 255, 0]     # goal = green
        return obs.transpose(2, 0, 1) # to (channels, height, width) format for PyTorch (aka CHW)
    

# --- Vectorized Env Wrapper ---
class VecEnv:
    def __init__(self, num_envs, env_fn, grid_size=10):
        self.envs = [env_fn() for _ in range(num_envs)]
        self.num_envs = num_envs
        self.observation_space = (3, grid_size, grid_size)  # (channels, height, width)
        self.action_space = 4

    def reset(self):
        obs = [env.reset() for env in self.envs]
        return np.stack(obs), [{} for _ in range(self.num_envs)]

    def step(self, actions):
        obs, rews, dones, infos = [], [], [], []
        for i, env in enumerate(self.envs):
            o, r, d, info = env.step(actions[i])
            if d:
                o = env.reset()
            obs.append(o)
            rews.append(r)
            dones.append(d)
            infos.append(info)
        return np.stack(obs), np.array(rews), np.array(dones), np.array(dones), infos

# --- Save Agent Demo as Video ---
def record_agent_demo(agent, save_path="agent_demo.mp4", grid_size=10, device="cpu"):
    env = GridWorldEnv(grid_size=grid_size)
    obs = env.reset()
    done = False
    frames = []

    start_time = time.time()
    instant_time = None
    framecount_prev = None
    while not done:
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            action, _, _, _ = agent.get_action_and_value(obs_tensor)
        obs, reward, done, _ = env.step(action.item())
        frame = env.render_frame()
        frames.append(frame)

        if len(frames) % 5 == 0:
            framecount = len(frames)
            if instant_time is not None and framecount_prev is not None:
                elapsed_time = time.time() - instant_time
                fps = (framecount - framecount_prev) / elapsed_time
                print(f"FPS: {fps:.2f}", end="\r")
            instant_time = time.time()
            framecount_prev = len(frames)

        # print(f"Frame {len(frames)}: Action {action.item()}, Avg FPS: {len(frames) / (time.time() - start_time):.2f}")

    print(f"Rendering time: {time.time() - start_time:.2f} seconds")
    print(f"Total frames: {len(frames)}")

    height, width, _ = frames[0].shape
    out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), 5, (width, height)) # 5 FPS
    for frame in frames:
        out.write(frame)
    out.release()
    print(f"Saved agent demo to {save_path}")
