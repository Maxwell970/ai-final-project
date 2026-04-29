import numpy as np


class RandomAgent:
    """
    Random baseline agent for MiniRisk without fortify.
    Action format:
    [reinforce_target, attack_choice]
    """

    def __init__(self, action_space):
        self.action_space = action_space

    def choose_action(self, obs):
        return self.action_space.sample()


class GreedyAgent:
    """
    Greedy rule-based baseline.

    Observation format:
    obs = [owners A-N, troops A-N]

    Action format:
    [reinforce_target, attack_choice]
    """

    def __init__(self):
        self.num_territories = 14

        self.adjacency = {
            0: [1, 2],
            1: [0, 3, 6],
            2: [0, 3, 4],
            3: [1, 2, 11],
            4: [2, 5],
            5: [4],
            6: [1, 7, 9],
            7: [6, 8, 10],
            8: [7],
            9: [6, 10, 11],
            10: [7, 9, 12],
            11: [3, 9, 12, 13],
            12: [10, 11],
            13: [11],
        }

        self.attack_actions = []
        for attacker in range(self.num_territories):
            for defender in self.adjacency[attacker]:
                self.attack_actions.append((attacker, defender))

        self.continents = {
            "Continent 1": [0, 1, 2, 3],
            "Continent 2": [6, 7, 8, 9, 10],
            "Continent 3": [4, 5],
            "Continent 4": [11, 12, 13],
        }

    def choose_action(self, obs):
        owners = obs[: self.num_territories]
        troops = obs[self.num_territories :]

        player_owned = np.where(owners == 0)[0]

        if len(player_owned) == 0:
            return np.array([0, 0], dtype=np.int64)

        reinforce_target = self._choose_reinforce_target(owners, troops, player_owned)
        attack_choice = self._choose_attack(owners, troops)

        return np.array([reinforce_target, attack_choice], dtype=np.int64)

    def _choose_reinforce_target(self, owners, troops, player_owned):
        border_territories = []

        for territory in player_owned:
            territory = int(territory)

            for neighbor in self.adjacency[territory]:
                if owners[neighbor] == 1:
                    border_territories.append(territory)
                    break

        if border_territories:
            return int(min(border_territories, key=lambda t: troops[t]))

        return int(player_owned[np.argmin(troops[player_owned])])

    def _choose_attack(self, owners, troops):
        best_attack_choice = 0
        best_score = -999

        for i, (attacker, defender) in enumerate(self.attack_actions):
            if owners[attacker] == 0 and owners[defender] == 1:
                margin = troops[attacker] - troops[defender]

                if margin > 0:
                    score = margin

                    # Prefer attacks that help complete continents
                    for continent_territories in self.continents.values():
                        if defender in continent_territories:
                            owned_in_continent = sum(
                                owners[t] == 0 for t in continent_territories
                            )
                            score += owned_in_continent * 0.5

                    if score > best_score:
                        best_score = score
                        best_attack_choice = i + 1

        return int(best_attack_choice)


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