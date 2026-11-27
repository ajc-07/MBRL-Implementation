from collections.abc import Callable

import numpy as np
from gymnasium.wrappers import RecordVideo

from env_util import make_env
from main import prepare_model
from model import MLP
from planner import cem_planner, simple_planner

# This file was created by Codex. Simple script that wraps planner functions and loops, with recording.

NUM_TRIALS = 50
HORIZON = 10
NUM_CANDIDATES = 100

PlannerFunction = Callable[[MLP, np.ndarray, int, int], np.ndarray] # Type alias for planner functions
# Takes in same inputs and returns same outputs as simple_planner and cem_planner

def record_episode(episode_id: int) -> bool:
    """Boolean to determine whether or not to record a given episode. For now, record all episodes."""
    return True

def evaluate_planner(planner: PlannerFunction,
                     planner_name: str,
                     video_folder: str,
                     model: MLP) -> float:
    """
    Runs one planner for NUM_TRIALS episodes and returns its average reward.
    """
    # Set up env for video recording
    env = make_env(render_mode="rgb_array")
    # Use built in RecordVideo wrapper
    env = RecordVideo(
        env,
        video_folder=video_folder,
        name_prefix=planner_name.lower().replace(" ", "-"),
        episode_trigger=record_episode,
    )
    rewards = []

    print(f"\nEvaluating {planner_name} for {NUM_TRIALS} trials...")

    for trial in range(NUM_TRIALS):
        state, info = env.reset(seed=trial)
        episode_reward = 0.0
        done = False

        while not done:
            action = planner(model, state, HORIZON, NUM_CANDIDATES)
            next_state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            state = next_state

        rewards.append(episode_reward)
        print(f"{planner_name} trial {trial + 1}/{NUM_TRIALS}: reward={episode_reward:.0f}")

    env.close()

    average_reward = sum(rewards) / len(rewards)
    print(f"Average {planner_name} reward: {average_reward:.2f}")
    return average_reward


def main():
    print("Preparing one trained model for both planner evaluations...")
    model = prepare_model()

    simple_average = evaluate_planner(
        simple_planner,
        "Simple planner",
        "results/videos/simple",
        model,
    )
    cem_average = evaluate_planner(
        cem_planner,
        "CEM planner",
        "results/videos/cem",
        model,
    )

    print("\nPlanner comparison:")
    print(f"Simple planner average reward: {simple_average:.2f}")
    print(f"CEM planner average reward: {cem_average:.2f}")
    print(f"CEM - simple: {cem_average - simple_average:.2f}")


if __name__ == "__main__":
    main()
