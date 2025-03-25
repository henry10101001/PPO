import torch
from gridworld import record_agent_demo
from agent import Agent

# --- Load the trained model and run a demo ---
def test_agent(model_path="agents/agent.pt", save_path="agent_demo.mp4", grid_size=10):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading agent from {model_path} on {device}")

    # Initialize model structure
    agent = Agent(action_dim=4, grid_size=grid_size).to(device)
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()

    # Record a video of the agent
    record_agent_demo(agent, save_path=save_path, grid_size=grid_size, device=device)

if __name__ == "__main__":
    grid_size = 15
    model = "agents/agent_15v2.pt"
    test_agent(model_path=model, grid_size=grid_size)