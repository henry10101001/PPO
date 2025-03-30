import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import time
import matplotlib.pyplot as plt

from gridworld import GridWorldEnv, VecEnv
from agent import Agent

# --- Hyperparameters and Configuration ---
class Args:
    total_timesteps = 5_000_000
    grid_size = 100
    num_envs = 8
    num_steps = 500
    batch_size = num_envs * num_steps
    num_minibatches = 4
    minibatch_size = batch_size // num_minibatches
    update_epochs = 4
    learning_rate = 2.5e-4
    gamma = 0.99  # Discount factor
    gae_lambda = 0.95  # GAE lambda parameter
    clip_coef = 0.1  # PPO clip coefficient
    ent_coef = 0.02  # Entropy coefficient
    vf_coef = 0.5  # Value function coefficient
    max_grad_norm = 0.5  # Gradient clipping
    norm_adv = True  # Normalize advantages
    clip_vloss = True  # Clip value loss
    anneal_lr = True  # Learning rate annealing
    target_kl = None  # Early stopping KL threshold
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    # --- Environment and Agent Setup ---
    args = Args()
    seed = 4
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f"Using device: {args.device}")

    # Initialize vectorized environment and agent
    envs = VecEnv(args.num_envs, lambda: GridWorldEnv(grid_size=args.grid_size), grid_size=args.grid_size)
    obs_shape = envs.observation_space
    agent = Agent(action_dim=envs.action_space, grid_size=args.grid_size).to(args.device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # --- Storage Setup for PPO ---
    obs = torch.zeros((args.num_steps, args.num_envs) + obs_shape).to(args.device)
    actions = torch.zeros((args.num_steps, args.num_envs)).to(args.device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(args.device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(args.device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(args.device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(args.device)

    # --- Training Loop Setup ---
    global_step = 0
    next_obs, _ = envs.reset()
    next_obs = torch.tensor(next_obs, dtype=torch.float32).to(args.device)
    next_done = torch.zeros(args.num_envs).to(args.device)
    num_updates = args.total_timesteps // args.batch_size

    # --- Tracking and Visualization ---
    avg_ep_returns = []
    max_avg_ep_return = -np.inf
    plt.ion()
    fig, ax = plt.subplots()

    start_time = time.time()
    for update in range(1, num_updates + 1):
        # --- Learning Rate Annealing ---
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        # --- Episode Tracking Variables ---
        completed_episode_steps = []
        uncompleted_steps_per_env = [0] * args.num_envs
        recorded_ep_returns = []
        ep_returns_per_env = [0.0] * args.num_envs
        batch_successes = 0
        batch_fails = 0
        
        # --- Collect Rollouts ---
        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # Get action and value from policy
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # Environment step
            next_obs_np, reward, done, _, _ = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward, dtype=torch.float32).to(args.device) # record tensor of all envs' rewards for this step
            next_obs = torch.tensor(next_obs_np, dtype=torch.float32).to(args.device)
            next_done = torch.tensor(done, dtype=torch.float32).to(args.device)

            # Accumulate reward and check for episode end for each env
            for env_idx in range(args.num_envs):
                ep_returns_per_env[env_idx] += reward[env_idx]
                uncompleted_steps_per_env[env_idx] += 1
                if done[env_idx]:
                    recorded_ep_returns.append(ep_returns_per_env[env_idx])
                    completed_episode_steps.append(uncompleted_steps_per_env[env_idx])
                    ep_returns_per_env[env_idx] = 0.0 # Reset for next episode
                    uncompleted_steps_per_env[env_idx] = 0
                    batch_successes += 1
                elif step == args.num_steps - 1:
                    # If the episode didn't finish, we can consider it a failure
                    # Add a fail, and also add rewards from failed games
                    batch_fails += 1
                    recorded_ep_returns.append(ep_returns_per_env[env_idx])

        # --- Compute Advantages using GAE ---
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(args.device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # --- Prepare Batch Data (flatten the batch) ---
        b_obs = obs.reshape((-1,) + obs_shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # --- PPO Update ---
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        approx_kl_sum = 0
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                # Get updated values from minibatch
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                # Calculate KL divergence and clip fraction
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    approx_kl_sum += approx_kl.item()
                    clipfrac = ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    clipfracs.append(clipfrac)

                # Normalize advantages
                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss = torch.max(
                    -mb_advantages * ratio,
                    -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef),
                ).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                # Total loss and optimization step
                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            # Early stopping if KL threshold is exceeded
            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        # --- Performance Tracking and Visualization ---
        episodes_this_batch = len(recorded_ep_returns)
        avg_ep_return_this_batch = np.mean(recorded_ep_returns) if recorded_ep_returns else 0
        avg_ep_returns.append(avg_ep_return_this_batch)

        # reset recorded episodic returns for each batch
        recorded_ep_returns = []

        avg_approx_kl = approx_kl_sum / (args.update_epochs * args.num_minibatches)
        avg_clipfrac = np.mean(clipfracs)

        # Print training progress
        print(f"Update {update}/{num_updates}, total_steps: {global_step}, # success (done): {batch_successes} out of {episodes_this_batch:.0f}, avg steps when done: {np.mean(completed_episode_steps):.1f}, Avg episodic return: {avg_ep_returns[-1]:.8f}", end="")
        # print(f"Completed episode steps: {completed_episode_steps}")
        
        # Save best model
        if avg_ep_returns[-1] > max_avg_ep_return:
            max_avg_ep_return = avg_ep_returns[-1]
            print(f" 🔥")
            torch.save(agent.state_dict(), "agents/agent.pt")
        else:
            print("")

        # Update visualization
        ax.clear()
        ax.plot(avg_ep_returns)
        ax.set_xlabel("Updates")
        ax.set_ylabel("Average Episodic Return (per batch)")
        plt.pause(0.01)
        ep_returns_per_env = [0.0] * args.num_envs  # Reset episodic returns for each env


    print(f"Training time: {time.time() - start_time:.3f} seconds")
    plt.ioff()
    plt.show()