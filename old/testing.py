from gridworld import GridWorldEnv
import numpy as np

if __name__ == "__main__":
    env = GridWorldEnv()
    obs = env.reset()

    done = False
    while not done:
        action = np.random.choice(4)  # random movement
        obs, reward, done, _ = env.step(action)
        env.render()
        print(f"Action: {action}, Reward: {reward}, Done: {done}")
