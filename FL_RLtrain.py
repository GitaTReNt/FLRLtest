import gymnasium as gym
from models.agentmodel import Agent


def train(agent, episode):
    for _ in range(episode):
        state, done = agent.agentReset()

        # By default, we consider our outcome to be a failure
        agent.outcome_list.append("Failure")
        # Until the agent gets stuck in a hole or reaches the goal, keep training it
        while not done:
            done, new_state, reward = agent.train(state)
            state = new_state
            if reward:
                agent.outcome_list[-1] = "Success"
            # If we have a reward, it means that our outcome is a success
        agent.epsilon = max(agent.epsilon - agent.epsilon_decay, 0)


if __name__ == "__main__":
    environment = gym.make("FrozenLake-v1", is_slippery=True, render_mode="rgb_array")
    environment.reset()
    episodes = 3000  # Total number of episode

    agent = Agent(environment)

    train(agent, episodes)

    print('Q-table after training:')
    print(agent.Q)
    agent.save()

    agent.resultPaint()
