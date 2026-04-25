import random
import numpy as np


class RandomAgent:
    """
    Agent that chooses random actions from the environment action space.
    """

    def __init__(self, action_space):
        self.action_space = action_space

    def choose_action(self, obs):
        return self.action_space.sample()


class GreedyAgent:
    """
    Rule-based baseline for MiniRisk v3.

    Observation format:
    obs = [owners for A-E, troops for A-E]

    Actions:
    0-4 = reinforce territory A-E
    5-14 = attack actions defined in env.attack_actions
    """

    def choose_action(self, obs):
        owners = obs[:5]
        troops = obs[5:]

        attack_actions = [
            (0, 1),  # A -> B
            (0, 3),  # A -> D
            (1, 0),  # B -> A
            (1, 2),  # B -> C
            (1, 4),  # B -> E
            (2, 1),  # C -> B
            (3, 0),  # D -> A
            (3, 4),  # D -> E
            (4, 1),  # E -> B
            (4, 3),  # E -> D
        ]

        # First, attack if a player-owned territory can capture an enemy territory
        best_attack = None
        best_margin = -999

        for i, (attacker, defender) in enumerate(attack_actions):
            if owners[attacker] == 0 and owners[defender] == 1:
                margin = troops[attacker] - troops[defender]
                if margin > 0 and margin > best_margin:
                    best_margin = margin
                    best_attack = 5 + i

        if best_attack is not None:
            return best_attack

        # Otherwise reinforce weakest player-owned territory
        player_owned = np.where(owners == 0)[0]

        if len(player_owned) > 0:
            weakest = player_owned[np.argmin(troops[player_owned])]
            return int(weakest)

        # Fallback
        return random.randint(0, 14)


def run_agent_episode(env, agent, render=True):
    obs, _ = env.reset()
    done = False
    total_reward = 0
    turns = 0

    while not done:
        if render:
            env.render()

        action = agent.choose_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        turns += 1
        done = terminated or truncated

        if render:
            print(f"Action Taken: {action} | Reward: {reward}")

    if render:
        env.render()
        print("Episode finished.")
        print(f"Total Reward: {total_reward}")
        print(f"Turns: {turns}")

    return total_reward, turns