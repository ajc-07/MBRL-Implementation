import numpy as np
import torch
import argparse

from env_util import collect_random_transitions, make_env
from model import MLP
from planner import cem_planner, simple_planner
from train import train_model, evaluate_model

def train_test_split(states: torch.Tensor,
                     actions: torch.Tensor,
                     next_states: torch.Tensor,
                     train_ratio: float = 0.8):
    """
    Splits the dataset into training and testing sets based on the specified ratio.

    Args:
        states: Tensor of current states.
        actions: Tensor of actions taken.
        next_states: Tensor of actual next states.
        train_ratio: Ratio of data to be used for training. Default is 0.8 (80% train, 20% test).

    Returns:
        train_states, test_states: Tensors of training and testing states.
        train_actions, test_actions: Tensors of training and testing actions.
        train_next_states, test_next_states: Tensors of training and testing next states.
    """
    num_transitions = states.shape[0]

    # Train test split, randomly shuffle the data and split into 80% train, 20% test
    indices = torch.randperm(num_transitions)
    states = states[indices]
    actions = actions[indices]
    next_states = next_states[indices]

    train_size = int(train_ratio* num_transitions)
    train_states, test_states = states[:train_size], states[train_size:]
    train_actions, test_actions = actions[:train_size], actions[train_size:]
    train_next_states, test_next_states = next_states[:train_size], next_states[train_size:]
    return train_states, test_states, train_actions, test_actions, train_next_states, test_next_states

def prepare_model():
    """
    Prepares the MLP dynamics model by training it on a set of random transitions.

    Returns:
        model: The trained MLP dynamics model.
    """
    # 1. Collect random transitions
    num_transitions = 1000
    transitions = collect_random_transitions(num_transitions)

    # 2. Train MLP dynamics model on random transitions
    model = MLP(input_dim=5, hidden_dim=64, output_dim=4)  # input_dim = state_dim + action_dim = 4 + 1 = 5
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Convert list of tuples into seperate lists
    states, actions, next_states = zip(*transitions) 
    states = torch.tensor(np.asarray(states), dtype=torch.float32)
    actions = torch.tensor(np.asarray(actions), dtype=torch.float32)
    next_states = torch.tensor(np.asarray(next_states), dtype=torch.float32)

    # Train test split
    train_states, test_states, train_actions, test_actions, train_next_states, test_next_states = train_test_split(states, actions, next_states)

    train_model(model, optimizer, train_states, train_actions, train_next_states)
    evaluate_model(model, test_states, test_actions, test_next_states)

    return model

def simple_planner_run():
    """
    Runs the simple planner on the env with a MLP learned dynamics model trained on random transition data.
    """
    model = prepare_model()
    horizon = 10
    num_candidates = 100
    env = make_env(render_mode="human")
    state, info = env.reset()

    cumulative_reward = 0.0
    time = 1
    done = False
    print("Starting simple planner run...")
    while not done:
        action = simple_planner(model, state, horizon, num_candidates)
        next_state, reward, terminated, truncated, info = env.step(action)
        cumulative_reward += reward
        print(f"Time: {time} - Current cumulative reward: {cumulative_reward:.2f}")
        done = terminated or truncated
        state = next_state
        time += 1
    env.close()
    print(f"Simple planner run completed.")
    print(f"Ended on time {time - 1}, angle: {state[1]:.2f}, terminated: {terminated}, truncated: {truncated}.")
    print(f"Total cumulative reward: {cumulative_reward:.2f}")

def cem_planner_run():
    """
    Runs the CEM planner on the env with a MLP learned dynamics model trained on random transition data.
    """
    model = prepare_model()
    horizon = 10
    num_candidates = 100
    env = make_env(render_mode="human")
    state, info = env.reset()

    cumulative_reward = 0.0
    time = 1
    done = False
    print("Starting CEM planner run...")
    while not done:
        action = cem_planner(model, state, horizon, num_candidates)
        next_state, reward, terminated, truncated, info = env.step(action)
        cumulative_reward += reward
        print(f"Time: {time} - Current cumulative reward: {cumulative_reward:.2f}")
        done = terminated or truncated
        state = next_state
        time += 1
    env.close()
    print(f"CEM planner run completed.")
    print(f"Ended on time {time - 1}, angle: {state[1]:.2f}, terminated: {terminated}, truncated: {truncated}.")
    print(f"Total cumulative reward: {cumulative_reward:.2f}")

if __name__ == "__main__":
    # Run argument parser to select which planner to run
    parser = argparse.ArgumentParser()
    parser.add_argument("--planner", choices=["simple", "cem"], default="simple", help="Select the planner to use")

    args = parser.parse_args()

    if args.planner == "simple":
        simple_planner_run()
    elif args.planner == "cem":
        cem_planner_run()