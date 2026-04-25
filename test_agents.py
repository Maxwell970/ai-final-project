from env import MiniRiskEnv
from agents import RandomAgent, GreedyAgent, run_agent_episode


def test_random_agent():
    print("\n==============================")
    print("TESTING RANDOM AGENT")
    print("==============================")

    env = MiniRiskEnv()
    agent = RandomAgent(env.action_space)

    total_reward, turns = run_agent_episode(env, agent, render=True)

    print("\nRandom Agent Results")
    print(f"Total Reward: {total_reward}")
    print(f"Turns: {turns}")


def test_greedy_agent():
    print("\n==============================")
    print("TESTING GREEDY AGENT")
    print("==============================")

    env = MiniRiskEnv()
    agent = GreedyAgent()

    total_reward, turns = run_agent_episode(env, agent, render=True)

    print("\nGreedy Agent Results")
    print(f"Total Reward: {total_reward}")
    print(f"Turns: {turns}")


if __name__ == "__main__":
    test_random_agent()
    test_greedy_agent()