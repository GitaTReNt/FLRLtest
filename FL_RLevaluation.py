import gymnasium as gym
from models.agentmodel import Agent


def evaluation(agent, episode):
    success_number = 0

    for _ in range(episode):
        state, done = agent.agentReset()
        done = False

        # Until the agent gets stuck or reaches the goal, keep training it
        while not done:
            done, new_state, reward = agent.predict(state)
            # Update our current state
            state = new_state

            # When we get a reward, it means we solved the game
            success_number += reward
    return success_number / episode


if __name__ == "__main__":
    environment = gym.make("FrozenLake-v1", is_slippery=True, render_mode="rgb_array")
    environment.reset()

    episodes = 1000  # Total number of episode

    agent = Agent(environment)
    agent.load()
    rate = evaluation(agent, episodes)
    print(f"success:{rate * 100}")
