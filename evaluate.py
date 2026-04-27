import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import matplotlib.pyplot as plt

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks

from env import MiniRiskEnv
from agents import RandomAgent, GreedyAgent


def display_name(agent_type):
    names = {
        "random": "Random",
        "greedy": "Greedy",
        "ppo": "PPO",
        "ppo_greedy_enemy": "PPO vs Greedy Enemy",
    }
    return names.get(agent_type, agent_type)


def run_episode(env, agent_type, model=None):
    obs, _ = env.reset()
    done = False

    total_reward = 0
    turns = 0
    won = False
    final_info = {}

    if agent_type == "random":
        agent = RandomAgent(env.action_space)
    elif agent_type == "greedy":
        agent = GreedyAgent()
    else:
        agent = None

    while not done:
        if agent_type in ["ppo", "ppo_greedy_enemy"]:
            action_masks = get_action_masks(env)
            action, _ = model.predict(
                obs,
                deterministic=True,
                action_masks=action_masks,
            )
        else:
            action = agent.choose_action(obs)

        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        turns += 1
        final_info = info

        done = terminated or truncated

        if terminated and np.all(env.owners == 0):
            won = True

    return total_reward, turns, won, final_info


def evaluate_agent(agent_type, num_games=100):
    rewards = []
    turns = []
    wins = []

    valid_reinforces = []
    invalid_reinforces = []

    successful_attacks = []
    invalid_attacks = []
    total_attacks_attempted = []
    skipped_valid_attacks = []

    model = None
    if agent_type in ["ppo", "ppo_greedy_enemy"]:
        model = MaskablePPO.load("ppo_minirisk")

    for _ in range(num_games):
        if agent_type == "ppo_greedy_enemy":
            env = MiniRiskEnv(enemy_policy="greedy")
        else:
            env = MiniRiskEnv()

        total_reward, num_turns, won, info = run_episode(
            env,
            agent_type,
            model,
        )

        rewards.append(total_reward)
        turns.append(num_turns)
        wins.append(won)

        valid_reinforces.append(info["valid_reinforces"])
        invalid_reinforces.append(info["invalid_reinforces"])

        successful_attacks.append(info["successful_attacks"])
        invalid_attacks.append(info["invalid_attacks"])
        total_attacks_attempted.append(info["total_attacks_attempted"])
        skipped_valid_attacks.append(info["skipped_valid_attacks"])

    total_valid_reinf = np.sum(valid_reinforces)
    total_invalid_reinf = np.sum(invalid_reinforces)

    total_success_attacks = np.sum(successful_attacks)
    total_attack_attempts = np.sum(total_attacks_attempted)

    reinforce_valid_pct = (
        100 * total_valid_reinf /
        max(1, total_valid_reinf + total_invalid_reinf)
    )

    attack_success_pct = (
        100 * total_success_attacks /
        max(1, total_attack_attempts)
    )

    return {
        "agent": agent_type,
        "agent_name": display_name(agent_type),
        "avg_reward": np.mean(rewards),
        "avg_turns": np.mean(turns),
        "win_rate": np.mean(wins),

        "valid_reinforce_pct": reinforce_valid_pct,
        "avg_invalid_reinforces": np.mean(invalid_reinforces),

        "attack_success_pct": attack_success_pct,
        "avg_invalid_attacks": np.mean(invalid_attacks),

        "avg_skipped_valid_attacks": np.mean(skipped_valid_attacks),
    }


def plot_results(results):
    agents = [r["agent_name"] for r in results]
    rewards = [r["avg_reward"] for r in results]
    turns = [r["avg_turns"] for r in results]
    win_rates = [r["win_rate"] * 100 for r in results]

    plt.figure(figsize=(10, 5))
    plt.bar(agents, rewards)
    plt.title("Average Reward by Agent")
    plt.ylabel("Average Reward")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig("avg_reward_by_agent.png")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(agents, turns)
    plt.title("Average Turns by Agent")
    plt.ylabel("Average Turns")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig("avg_turns_by_agent.png")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(agents, win_rates)
    plt.title("Win Rate by Agent")
    plt.ylabel("Win Rate (%)")
    plt.ylim(0, 100)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig("win_rate_by_agent.png")
    plt.close()


def main():
    num_games = 100
    results = []

    for agent_type in ["random", "greedy", "ppo", "ppo_greedy_enemy"]:
        print(f"Evaluating {display_name(agent_type)}...")
        result = evaluate_agent(agent_type, num_games=num_games)
        results.append(result)

    print("\n=== Evaluation Results ===")

    header = (
        f"{'Agent':<22}"
        f"{'Win %':<10}"
        f"{'Avg Rwd':<12}"
        f"{'Turns':<10}"
        f"{'Valid Rein%':<14}"
        f"{'Atk Success%':<15}"
        f"{'Bad Atks':<12}"
        f"{'Skip Good Atk':<14}"
    )

    print(header)
    print("-" * len(header))

    for r in results:
        print(
            f"{r['agent_name']:<22}"
            f"{r['win_rate'] * 100:<10.2f}"
            f"{r['avg_reward']:<12.2f}"
            f"{r['avg_turns']:<10.2f}"
            f"{r['valid_reinforce_pct']:<14.2f}"
            f"{r['attack_success_pct']:<15.2f}"
            f"{r['avg_invalid_attacks']:<12.2f}"
            f"{r['avg_skipped_valid_attacks']:<14.2f}"
        )

    plot_results(results)

    print("\nCharts saved:")
    print("- avg_reward_by_agent.png")
    print("- avg_turns_by_agent.png")
    print("- win_rate_by_agent.png")


if __name__ == "__main__":
    main()