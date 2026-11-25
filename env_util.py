import gymnasium as gym

def make_env():
    env = gym.make("InvertedPendulum-v5", render_mode="human") # move to different file so doesn't init env every time we import model
    return env

# https://gymnasium.farama.org/environments/mujoco/inverted_pendulum/
# Two objects, a cart and a pole
# Action space: Box(-3.0, 3.0, (1,), float32)
#   Actions have shape (1,), scalar, value is a float between -3 and 3, representing forward/backward force applied to the cart.
# Observation space: Box(-inf, inf, (4,), float64)
#   Observations have shape (4,), representing the cart's position, pole's angle, cart's velocity, and pole's angular velocity.