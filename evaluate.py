import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from env import MiniRiskEnv
from agents import RandomAgent, GreedyAgent


def run_episode(env, agent_type, model=None):
    obs, _ = env.reset()
    done = False

    total_reward = 0
    turns = 0
    won = False

    if agent_type == "random":
        agent = RandomAgent(env.action_space)
    elif agent_type == "greedy":
        agent = GreedyAgent()
    else:
        agent = None

    while not done:
        # PPO returns MultiDiscrete action array
        if agent_type == "ppo":
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = agent.choose_action(obs)

        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        turns += 1
        done = terminated or truncated

        # Win = player owns all territories
        if terminated and np.all(env.owners == 0):
            won = True

    return total_reward, turns, won


def evaluate_agent(agent_type, num_games=100):
    rewards = []
    turns = []
    wins = []

    model = None
    if agent_type == "ppo":
        model = PPO.load("ppo_minirisk")

    for _ in range(num_games):
        env = MiniRiskEnv()
        total_reward, num_turns, won = run_episode(env, agent_type, model)

        rewards.append(total_reward)
        turns.append(num_turns)
        wins.append(won)

    return {
        "agent": agent_type,
        "avg_reward": np.mean(rewards),
        "avg_turns": np.mean(turns),
        "win_rate": np.mean(wins),
    }


def plot_results(results):
    agents = [r["agent"].upper() for r in results]
    rewards = [r["avg_reward"] for r in results]
    turns = [r["avg_turns"] for r in results]
    win_rates = [r["win_rate"] * 100 for r in results]

    # Reward chart
    plt.figure(figsize=(8, 5))
    plt.bar(agents, rewards)
    plt.title("Average Reward by Agent")
    plt.ylabel("Average Reward")
    plt.tight_layout()
    plt.savefig("avg_reward_by_agent.png")
    plt.close()

    # Turns chart
    plt.figure(figsize=(8, 5))
    plt.bar(agents, turns)
    plt.title("Average Turns by Agent")
    plt.ylabel("Average Turns")
    plt.tight_layout()
    plt.savefig("avg_turns_by_agent.png")
    plt.close()

    # Win rate chart
    plt.figure(figsize=(8, 5))
    plt.bar(agents, win_rates)
    plt.title("Win Rate by Agent")
    plt.ylabel("Win Rate (%)")
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig("win_rate_by_agent.png")
    plt.close()


def main():
    num_games = 100

    results = []

    for agent_type in ["random", "greedy", "ppo"]:
        print(f"Evaluating {agent_type} agent...")
        result = evaluate_agent(agent_type, num_games=num_games)
        results.append(result)

    print("\n=== Evaluation Results ===")
    print(f"{'Agent':<10} {'Avg Reward':<15} {'Avg Turns':<15} {'Win Rate':<15}")
    print("-" * 65)

    for r in results:
        print(
            f"{r['agent']:<10}"
            f"{r['avg_reward']:<15.2f}"
            f"{r['avg_turns']:<15.2f}"
            f"{r['win_rate'] * 100:<15.2f}%"
        )

    plot_results(results)

    print("\nCharts saved:")
    print("- avg_reward_by_agent.png")
    print("- avg_turns_by_agent.png")
    print("- win_rate_by_agent.png")


if __name__ == "__main__":
    main()