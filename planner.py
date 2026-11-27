import torch
import torch.nn as nn
import numpy as np

from model import MLP

def reward_function(state: torch.Tensor) -> float:
    """
    Reward function for a given state as defined on the official Gymnasium Inverted Pendulum-v5 env.

    Specifically, as cited verbatim:
    The goal is to keep the inverted pendulum stand upright (within a certain angle limit, |angle| < 0.2) for as long as possible - as such, a reward of +1 is given for each timestep that the pole is upright.

    Args:
        state: Tensor of the current state.
    
    Returns:
        reward: The computed reward for the given state.
    """
    angle = state[1] # Recall the second element of state refers to the angle of the pole
    if abs(angle) < 0.2:
        return 1.0
    else:
        return 0.0

@torch.no_grad() # Prevent gradients from being computed, only inference
def simple_planner(model: MLP,
                   current_state: np.ndarray,
                   horizon: int,
                   num_candidates: int) -> np.ndarray:
    """
    Simple planner :
    1. Samples random candidate action sequences
    2. Simulate each sequence through the learned dynamics model
    3. Score each sequence based on predicted future
    4. Keep the best sequence and return the first action of that sequence

    Args:
        model: The MLP dynamics model.
        current_state: Numpy array of the current state.
        horizon: Number of steps to simulate into the future.
        num_candidates: Number of random candidate action sequences to sample.
    
    Returns:
        best_action: The first action of the best candidate sequence. Env.step() expects a numpy array of shape (1,).
    """
    model.eval()
    current_state_tensor = torch.as_tensor(current_state, dtype=torch.float32)

    # Sample random candidate action sequences
    # Recall each action is scalar [-3, 3]
    # So we want a list of num_candidate sequences, each of length horizon, with each action as a (1,) almost-scalar value sampled uniformly from [-3, 3]
    action_sequences = torch.rand((num_candidates, horizon, 1), dtype = current_state_tensor.dtype) * 6 - 3  # shape (num_candidates, horizon, 1)
    # [0, 1] -> [0, 6] -> [-3, 3]

    # Simulate each sequence through the learned dynamics model
    scores = [0.0] * num_candidates  # Initialize scores for each candidate
    for candidate in range(num_candidates):
        state = current_state_tensor.clone()
        cumulative_reward = 0.0
        for t in range(horizon):
            action = action_sequences[candidate, t]
            next_state = model.predict_next_state(state, action)

            # Score each sequence based on predicted future
            # Break when pole falls, since real env would terminate and reset
            if reward_function(next_state) == 0.0:
                break

            cumulative_reward += reward_function(next_state)
            state = next_state
        scores[candidate] = cumulative_reward

    # Keep the best sequence and return the first action of that sequence
    best_candidate_index = int(torch.tensor(scores).argmax().item())
    best_action_tensor = action_sequences[best_candidate_index, 0]  # Return the first
    best_action = best_action_tensor.numpy() # env.step() expects a numpy array of shape (1,)
    return best_action