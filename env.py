import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random


class MiniRiskEnv(gym.Env):
    def __init__(self):
        super().__init__()

        # 5 territories: A, B, C, D, E
        # Layout:
        # A --- B --- C
        # |     |
        # D --- E
        self.num_territories = 5

        self.territory_names = ["A", "B", "C", "D", "E"]

        self.adjacency = {
            0: [1, 3],     # A connects to B, D
            1: [0, 2, 4],  # B connects to A, C, E
            2: [1],        # C connects to B
            3: [0, 4],     # D connects to A, E
            4: [1, 3],     # E connects to B, D
        }

        # Actions:
        # 0-4  = reinforce one territory with both troops
        # 5-14 = attack actions
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

        self.action_space = spaces.Discrete(15)

        # Observation = owners + troop counts
        # owners: 0 = player, 1 = enemy
        self.observation_space = spaces.Box(
            low=0,
            high=50,
            shape=(10,),
            dtype=np.int32,
        )

        self.max_turns = 40
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initial board:
        # A and D are player-owned.
        # B, C, E are enemy-owned.
        self.owners = np.array([0, 1, 1, 0, 1], dtype=np.int32)

        # Starting troops
        self.troops = np.array([4, 3, 3, 4, 3], dtype=np.int32)

        self.turn = 0

        return self._get_obs(), {}

    def _get_obs(self):
        return np.concatenate([self.owners, self.troops]).astype(np.int32)

    def _capture(self, attacker, defender, new_owner):
        """
        If attacker wins, defender territory gets:
        attacking troops - defending troops.
        Attacking territory is left with 1 troop.
        """
        remaining_attackers = self.troops[attacker] - self.troops[defender]

        self.owners[defender] = new_owner
        self.troops[defender] = max(1, remaining_attackers)
        self.troops[attacker] = 1

    def _reinforce_one(self, owner_id, territory, amount=2):
        """
        Reinforce one owned territory with all reinforcement troops.
        """
        if self.owners[territory] == owner_id:
            self.troops[territory] += amount
            return True
        return False

    def _reinforce_split(self, owner_id):
        """
        Enemy helper: split 2 troops across two weakest owned territories,
        or put both on one if only one territory is owned.
        """
        owned = np.where(self.owners == owner_id)[0]

        if len(owned) == 0:
            return

        if len(owned) == 1:
            self.troops[owned[0]] += 2
            return

        sorted_owned = owned[np.argsort(self.troops[owned])]
        self.troops[sorted_owned[0]] += 1
        self.troops[sorted_owned[1]] += 1

    def _player_action(self, action):
        reward = -0.1

        # Actions 0-4: reinforce territory with 2 troops
        if 0 <= action <= 4:
            success = self._reinforce_one(owner_id=0, territory=action, amount=2)
            if success:
                reward += 0.2
            else:
                reward -= 0.5
            return reward

        # Actions 5-14: attack
        attack_index = action - 5
        attacker, defender = self.attack_actions[attack_index]

        valid_attack = (
            self.owners[attacker] == 0
            and self.owners[defender] == 1
            and defender in self.adjacency[attacker]
        )

        if valid_attack:
            if self.troops[attacker] > self.troops[defender]:
                self._capture(attacker, defender, new_owner=0)
                reward += 3.0
            else:
                reward -= 1.0
        else:
            reward -= 0.5

        return reward

    def _enemy_attack_options(self):
        options = []

        for attacker in range(self.num_territories):
            if self.owners[attacker] != 1:
                continue

            for defender in self.adjacency[attacker]:
                if self.owners[defender] == 0:
                    options.append((attacker, defender))

        return options

    def _enemy_turn(self):
        penalty = 0.0

        # Enemy reinforces by 2 troops total each turn.
        # It sometimes splits across weak territories.
        if random.random() < 0.5:
            self._reinforce_split(owner_id=1)
        else:
            enemy_owned = np.where(self.owners == 1)[0]
            if len(enemy_owned) > 0:
                weakest = enemy_owned[np.argmin(self.troops[enemy_owned])]
                self.troops[weakest] += 2

        penalty -= 0.1

        # Enemy may attack after reinforcing.
        possible_attacks = self._enemy_attack_options()

        strong_attacks = [
            (attacker, defender)
            for attacker, defender in possible_attacks
            if self.troops[attacker] > self.troops[defender]
        ]

        if strong_attacks and random.random() < 0.35:
            attacker, defender = random.choice(strong_attacks)
            self._capture(attacker, defender, new_owner=1)
            penalty -= 3.0

        return penalty

    def step(self, action):
        self.turn += 1
        terminated = False
        truncated = False

        reward = self._player_action(action)

        # Player wins if all territories are player-owned
        if np.all(self.owners == 0):
            reward += 20
            terminated = True

        # Enemy acts if player has not already won
        if not terminated:
            reward += self._enemy_turn()

        # Enemy wins if all territories are enemy-owned
        if np.all(self.owners == 1):
            reward -= 20
            terminated = True

        # Timeout penalty discourages stalling
        if self.turn >= self.max_turns:
            reward -= 10
            truncated = True

        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        print(f"\n=== TURN {self.turn} ===")

        for i in range(self.num_territories):
            owner = "Player" if self.owners[i] == 0 else "Enemy"
            print(
                f"{self.territory_names[i]} | "
                f"Owner: {owner:<6} | "
                f"Troops: {self.troops[i]}"
            )

        print("\nMap:")
        print(f"{self.territory_names[0]} --- {self.territory_names[1]} --- {self.territory_names[2]}")
        print("|     |")
        print(f"{self.territory_names[3]} --- {self.territory_names[4]}")
        print("================\n")


if __name__ == "__main__":
    env = MiniRiskEnv()

    obs, _ = env.reset()
    done = False

    print("Starting MiniRisk v3 simulation...")

    while not done:
        env.render()

        action = random.randint(0, env.action_space.n - 1)
        obs, reward, terminated, truncated, info = env.step(action)

        print(f"Action Taken: {action} | Reward: {reward}")

        done = terminated or truncated

    env.render()
    print("Game over.")