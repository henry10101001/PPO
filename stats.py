from agent import Agent
from prettytable import PrettyTable

def count_parameters(model):
    """
    Counts the total number of trainable parameters in a PyTorch model.

    :param model: PyTorch model
    :return: Total number of parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":

    # Create an instance of the Agent
    action_dim = 4
    grid_size = 100
    agent = Agent(action_dim=action_dim, grid_size=grid_size)

    # Count total parameters
    total_params = count_parameters(agent)
    
    # Create a table to display the parameters
    table = PrettyTable()
    table.field_names = ["Layer", "Parameters"]
    for name, param in agent.named_parameters():
        table.add_row([name, param.numel()])
    table.add_row(["---", "---"])
    table.add_row(["Actor", count_parameters(agent.actor)])
    table.add_row(["Critic", count_parameters(agent.critic)])
    table.add_row(["Total", total_params])
    print(table)
    