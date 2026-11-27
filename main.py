import numpy as np
import torch
import argparse
import copy
from collections.abc import Callable
from gymnasium.wrappers import RecordVideo

from env_util import collect_random_transitions, make_env
from model import MLP
from planner import cem_planner, simple_planner
from train import train_model, evaluate_model, train_step

PlannerFunction = Callable[[MLP, np.ndarray, int, int], np.ndarray] # Type alias for planner functions
# Takes in same inputs and returns same outputs as simple_planner and cem_planner

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

def planner_run(model: MLP,
                env,
                planner_function: PlannerFunction,
                horizon: int,
                num_candidates: int,
                online_learning: bool = False,
                verbose: bool = False):
    """
    Runs an episode using specified planner on the env with a MLP learned dynamics model trained on random transition data.

    Args:
        model: The MLP dynamics model.
        env: The environment to run the planner on.
        planner_function: The planner function to use (simple_planner or cem_planner).
        horizon: Number of steps to simulate into the future.
        num_candidates: Number of random candidate action sequences to sample.
        online_learning: Whether to enable online learning during the planner run. Default is False.
        verbose: Whether to print detailed information during the planner run. Default is False.
    
    Returns:
        cumulative_reward: The total cumulative reward obtained during the planner run.
    """
    state, info = env.reset()

    if online_learning:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    cumulative_reward = 0.0
    time = 1
    done = False

    if verbose:
        print("Starting planner run...")

    while not done:
        action = planner_function(model, state, horizon, num_candidates)
        next_state, reward, terminated, truncated, info = env.step(action)
        cumulative_reward += reward

        if verbose:
            print(f"Time: {time} - Current cumulative reward: {cumulative_reward:.2f}")

        if online_learning:
            # Update model with new transition
            # Add batch dimension, since train_step takes (1, state_dim) and (1, action_dim) tensors
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action_tensor = torch.tensor(action, dtype=torch.float32).unsqueeze(0)
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
            train_step(model, optimizer, state_tensor, action_tensor, next_state_tensor)

        done = terminated or truncated
        state = next_state
        time += 1

    if verbose:
        print(f"Planner run completed.")
        print(f"Ended on time {time - 1}, angle: {state[1]:.2f}, terminated: {terminated}, truncated: {truncated}.")
        print(f"Total cumulative reward: {cumulative_reward:.2f}")

    return cumulative_reward

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--planner", choices=["simple", "cem"], default="simple", help="Select the planner to use")
    parser.add_argument("--horizon", type=int, default=10, help="Horizon for the planner")
    parser.add_argument("--num_candidates", type=int, default=100, help="Number of candidates for the CEM planner")
    parser.add_argument("--online_learning", action="store_true", help="Enable online learning during planner run")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    parser.add_argument("--num_trials", type=int, default=1, help="Number of trials to run for evaluation")
    parser.add_argument("--video_folder", type=str, default="results/videos", help="Folder to save evaluation videos")

    # Placeholder function for RecordVideo's episode_trigger argument
    def record_episode(episode_id: int) -> bool:
        """Boolean to determine whether or not to record a given episode. For now, record all episodes."""
        return True

    args = parser.parse_args()
    model = prepare_model()

    if args.planner == "simple":
        planner_function = simple_planner
    elif args.planner == "cem":
        planner_function = cem_planner

    # Run eval loop for multiple trials, or run a single trial with human rendering
    if args.num_trials > 1:
        env = make_env(render_mode="rgb_array")
        output_folder = f"{args.video_folder}/{args.planner}_online" if args.online_learning else f"{args.video_folder}/{args.planner}"
        env = RecordVideo(
                env,
                video_folder=output_folder,
                episode_trigger=record_episode,
        )
        rewards = []

        # Note that we don't want the model's online learning to carry over across trials
        # So we copy initial models weights, and reset the model after each trial
        initial_model_weights = copy.deepcopy(model.state_dict())
    
        print(f"\nEvaluating {args.planner} for {args.num_trials} trials...")
    
        for trial in range(args.num_trials):
            model.load_state_dict(initial_model_weights)
            episode_reward = planner_run(model, env, planner_function=planner_function, horizon=args.horizon, num_candidates=args.num_candidates, online_learning=args.online_learning, verbose=args.verbose)
            rewards.append(episode_reward)
            print(f"{args.planner} trial {trial + 1}/{args.num_trials}: reward={episode_reward:.0f}")
    
        env.close()
    
        average_reward = sum(rewards) / len(rewards)
        print(f"Average {args.planner} reward: {average_reward:.2f}")
    elif args.num_trials == 1:
        env = make_env(render_mode="human")
        episode_reward = planner_run(model, env, planner_function=planner_function, horizon=args.horizon, num_candidates=args.num_candidates, online_learning=args.online_learning, verbose=args.verbose)
        env.close()
    else:
        raise ValueError("Number of trials must be at least 1.")