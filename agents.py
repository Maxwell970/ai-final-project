import numpy as np


class RandomAgent:
    """
    Random baseline agent for MiniRisk v4.
    Works with MultiDiscrete action space.
    """

    def __init__(self, action_space):
        self.action_space = action_space

    def choose_action(self, obs):
        return self.action_space.sample()


class GreedyAgent:
    """
    Greedy rule-based baseline for MiniRisk v4.

    Action format:
    [reinforce_target, attack_choice, fortify_source, fortify_dest, fortify_amount]

    reinforce_target:
        0-4 = A-E

    attack_choice:
        0 = no attack
        1-10 = attack actions

    fortify_amount:
        0 = move none
        1 = move 1
        2 = move 2
        3 = move 3
        4 = move all possible while leaving 1 behind
    """

    def __init__(self):
        self.num_territories = 5

        self.attack_actions = [
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

        self.adjacency = {
            0: [1, 3],
            1: [0, 2, 4],
            2: [1],
            3: [0, 4],
            4: [1, 3],
        }

    def choose_action(self, obs):
        owners = obs[:5]
        troops = obs[5:]

        # 1. Reinforce weakest player-owned territory
        player_owned = np.where(owners == 0)[0]

        if len(player_owned) == 0:
            return np.array([0, 0, 0, 0, 0], dtype=np.int64)

        reinforce_target = int(player_owned[np.argmin(troops[player_owned])])

        # 2. Choose best valid attack
        attack_choice = 0
        best_margin = -999

        for i, (attacker, defender) in enumerate(self.attack_actions):
            valid_attack = owners[attacker] == 0 and owners[defender] == 1

            if valid_attack:
                margin = troops[attacker] - troops[defender]

                if margin > 0 and margin > best_margin:
                    best_margin = margin
                    attack_choice = i + 1

        # 3. Fortify: move troops from strongest owned territory to weakest adjacent owned territory
        fortify_source = 0
        fortify_dest = 0
        fortify_amount = 0

        if len(player_owned) >= 2:
            strongest = int(player_owned[np.argmax(troops[player_owned])])

            adjacent_owned = [
                t for t in self.adjacency[strongest]
                if owners[t] == 0
            ]

            if len(adjacent_owned) > 0 and troops[strongest] > 2:
                weakest_adjacent = int(min(adjacent_owned, key=lambda t: troops[t]))

                fortify_source = strongest
                fortify_dest = weakest_adjacent
                fortify_amount = 2

        return np.array(
            [
                reinforce_target,
                attack_choice,
                fortify_source,
                fortify_dest,
                fortify_amount,
            ],
            dtype=np.int64,
        )


def run_agent_episode(env, agent, render=True):
    """
    Runs one full episode using the provided agent.
    Returns total reward and number of turns.
    """

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