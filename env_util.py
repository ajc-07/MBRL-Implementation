import gymnasium as gym

def make_env(render_mode: str | None = "human"):
    """
    Creates and returns the Inverted Pendulum-v5 environment from Gymnasium. Really intuitive/simple.
    
    Args:
        render_mode (str): The render mode for the environment. Default is "human".

    Returns:
        gym.Env: The created Inverted Pendulum-v5 environment.
    """
    env = gym.make("InvertedPendulum-v5", render_mode=render_mode) # move to different file so doesn't init env every time we import model
    return env

# https://gymnasium.farama.org/environments/mujoco/inverted_pendulum/
# Two objects, a cart and a pole
# Action space: Box(-3.0, 3.0, (1,), float32)
#   Actions have shape (1,), scalar, value is a float between -3 and 3, representing forward/backward force applied to the cart.
# Observation space: Box(-inf, inf, (4,), float64)
#   Observations have shape (4,), representing the cart's position, pole's angle, cart's velocity, and pole's angular velocity.

def collect_random_transitions(num_transitions: int):
    """
    Collects a list of random transitions from the environment. Random walk down the state space until termination, then reset and continue random walks until we hit the required number of transitions.
    
    Args:
        num_transitions (int): Number of transitions to collect.

    Returns:
        list: A list of tuples, where each tuple is a transition (state, action, next_state).
    """
    env = make_env(render_mode = None) # Don't waste compute on rendering
    transitions = []
    state, info = env.reset()
    for _ in range(num_transitions):
        action = env.action_space.sample()  # Sample a random action
        next_state, reward, terminated, truncated, info = env.step(action) # Take a step in the env using this action

        transition = (state, action, next_state) # Add the new data
        transitions.append(transition)

        # Update state for next iteration, or reset state if end is reached
        if terminated or truncated:
            state, info = env.reset()  # Reset the environment if the episode is done
        else:
            state = next_state  # Update state to next_state for the next iteration
        
    env.close()
    return transitions