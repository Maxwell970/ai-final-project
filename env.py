import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random


class MiniRiskEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self.num_territories = 14
        self.territory_names = list("ABCDEFGHIJKLMN")

        self.adjacency = {
            0: [1, 2],          # A: B, C
            1: [0, 3, 6],       # B: A, D, G
            2: [0, 3, 4],       # C: A, D, E
            3: [1, 2, 11],      # D: B, C, L
            4: [2, 5],          # E: C, F
            5: [4],             # F: E
            6: [1, 7, 9],       # G: B, H, J
            7: [6, 8, 10],      # H: G, I, K
            8: [7],             # I: H
            9: [6, 10, 11],     # J: G, K, L
            10: [7, 9, 12],     # K: H, J, M
            11: [3, 9, 12, 13], # L: D, J, M, N
            12: [10, 11],       # M: K, L
            13: [11],           # N: L
        }

        self.continents = {
            "Continent 1": [0, 1, 2, 3],        # A, B, C, D
            "Continent 2": [6, 7, 8, 9, 10],    # G, H, I, J, K
            "Continent 3": [4, 5],              # E, F
            "Continent 4": [11, 12, 13],        # L, M, N
        }

        self.continent_bonuses = {
            "Continent 1": 3,
            "Continent 2": 4,
            "Continent 3": 2,
            "Continent 4": 2,
        }

        self.attack_actions = []
        for attacker in range(self.num_territories):
            for defender in self.adjacency[attacker]:
                self.attack_actions.append((attacker, defender))

        self.num_attack_actions = len(self.attack_actions)

        # Multi-part PPO action:
        # [reinforce_target, attack_choice, fortify_source, fortify_dest, fortify_amount]
        #
        # reinforce_target: 0-13
        # attack_choice: 0 = no attack, 1-num_attack_actions = attack action
        # fortify_source: 0-13
        # fortify_dest: 0-13
        # fortify_amount:
        #   0 = move none
        #   1 = move 1
        #   2 = move 2
        #   3 = move 3
        #   4 = move all movable troops while leaving 1 behind
        self.action_space = spaces.MultiDiscrete(
            [
                self.num_territories,
                self.num_attack_actions + 1,
                self.num_territories,
                self.num_territories,
                5,
            ]
        )

        # Observation = owners + troops
        self.observation_space = spaces.Box(
            low=0,
            high=100,
            shape=(self.num_territories * 2,),
            dtype=np.int32,
        )

        self.max_turns = 75
        self.starting_troops = 3
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        territories = np.arange(self.num_territories)
        self.np_random.shuffle(territories)

        player_territories = territories[: self.num_territories // 2]

        self.owners = np.ones(self.num_territories, dtype=np.int32)
        self.owners[player_territories] = 0

        self.troops = np.full(
            self.num_territories,
            self.starting_troops,
            dtype=np.int32,
        )

        self.turn = 0

        return self._get_obs(), {}

    def _get_obs(self):
        return np.concatenate([self.owners, self.troops]).astype(np.int32)

    def _calculate_reinforcements(self, owner_id):
        owned_count = int(np.sum(self.owners == owner_id))

        territory_bonus = max(3, owned_count // 3)
        continent_bonus = 0

        for continent_name, territories in self.continents.items():
            if all(self.owners[t] == owner_id for t in territories):
                continent_bonus += self.continent_bonuses[continent_name]

        return territory_bonus + continent_bonus

    def _capture(self, attacker, defender, new_owner):
        remaining_attackers = self.troops[attacker] - self.troops[defender]

        self.owners[defender] = new_owner
        self.troops[defender] = max(1, remaining_attackers)
        self.troops[attacker] = 1

    def _place_reinforcements(self, owner_id, territory):
        reinforcements = self._calculate_reinforcements(owner_id)

        if self.owners[territory] == owner_id:
            self.troops[territory] += reinforcements
            return True, reinforcements

        return False, reinforcements

    def _can_fortify_between(self, owner_id, source, dest):
        if source == dest:
            return False

        if self.owners[source] != owner_id or self.owners[dest] != owner_id:
            return False

        # Simplified Risk-style fortify: adjacent owned territories only.
        return dest in self.adjacency[source]

    def _fortify(self, owner_id, source, dest, amount_choice):
        if not self._can_fortify_between(owner_id, source, dest):
            return False

        movable = self.troops[source] - 1

        if movable <= 0:
            return False

        if amount_choice == 0:
            amount = 0
        elif amount_choice == 1:
            amount = 1
        elif amount_choice == 2:
            amount = 2
        elif amount_choice == 3:
            amount = 3
        else:
            amount = movable

        amount = min(amount, movable)

        if amount <= 0:
            return False

        self.troops[source] -= amount
        self.troops[dest] += amount
        return True

    def _player_turn(self, action):
        reward = -0.1

        reinforce_target, attack_choice, fortify_source, fortify_dest, fortify_amount = action

        reinforce_target = int(reinforce_target)
        attack_choice = int(attack_choice)
        fortify_source = int(fortify_source)
        fortify_dest = int(fortify_dest)
        fortify_amount = int(fortify_amount)

        # 1. Reinforcement phase
        reinforced, reinforcements = self._place_reinforcements(
            owner_id=0,
            territory=reinforce_target,
        )

        if reinforced:
            reward += 0.2
        else:
            reward -= 1.0

        # 2. Attack phase
        if attack_choice == 0:
            reward -= 0.2
        else:
            attack_index = attack_choice - 1
            attacker, defender = self.attack_actions[attack_index]

            valid_attack = (
                self.owners[attacker] == 0
                and self.owners[defender] == 1
                and self.troops[attacker] > self.troops[defender]
            )

            if valid_attack:
                self._capture(attacker, defender, new_owner=0)
                reward += 5.0
            else:
                reward -= 1.0

        # 3. Fortify phase
        if self._fortify(
            owner_id=0,
            source=fortify_source,
            dest=fortify_dest,
            amount_choice=fortify_amount,
        ):
            reward += 0.2

        return reward

    def _enemy_reinforce(self):
        enemy_owned = np.where(self.owners == 1)[0]

        if len(enemy_owned) == 0:
            return

        reinforcements = self._calculate_reinforcements(owner_id=1)

        # Enemy reinforces weakest territory.
        weakest = int(enemy_owned[np.argmin(self.troops[enemy_owned])])
        self.troops[weakest] += reinforcements

    def _enemy_attack_options(self):
        options = []

        for attacker in range(self.num_territories):
            if self.owners[attacker] != 1:
                continue

            for defender in self.adjacency[attacker]:
                if self.owners[defender] == 0:
                    options.append((attacker, defender))

        return options

    def _enemy_attack(self):
        possible_attacks = self._enemy_attack_options()

        strong_attacks = [
            (attacker, defender)
            for attacker, defender in possible_attacks
            if self.troops[attacker] > self.troops[defender]
        ]

        if len(strong_attacks) == 0:
            return False

        # Enemy prefers largest troop advantage.
        attacker, defender = max(
            strong_attacks,
            key=lambda pair: self.troops[pair[0]] - self.troops[pair[1]],
        )

        self._capture(attacker, defender, new_owner=1)
        return True

    def _enemy_fortify(self):
        enemy_owned = np.where(self.owners == 1)[0]

        if len(enemy_owned) < 2:
            return False

        strongest = int(enemy_owned[np.argmax(self.troops[enemy_owned])])

        possible_dests = [
            t for t in self.adjacency[strongest]
            if self.owners[t] == 1
        ]

        if len(possible_dests) == 0:
            return False

        weakest_dest = min(possible_dests, key=lambda t: self.troops[t])

        movable = self.troops[strongest] - 1

        if movable <= 0:
            return False

        amount = min(3, movable)

        self.troops[strongest] -= amount
        self.troops[weakest_dest] += amount

        return True

    def _enemy_turn(self):
        penalty = 0.0

        self._enemy_reinforce()
        penalty -= 0.1

        if self._enemy_attack():
            penalty -= 5.0

        self._enemy_fortify()

        return penalty

    def step(self, action):
        self.turn += 1
        terminated = False
        truncated = False

        reward = self._player_turn(action)

        # Reward shaping: encourage territory control.
        player_territories = int(np.sum(self.owners == 0))
        enemy_territories = int(np.sum(self.owners == 1))

        reward += 0.3 * player_territories
        reward -= 0.15 * enemy_territories

        # Reward shaping: encourage continent control.
        player_continent_bonus = 0
        enemy_continent_bonus = 0

        for continent_name, territories in self.continents.items():
            if all(self.owners[t] == 0 for t in territories):
                player_continent_bonus += self.continent_bonuses[continent_name]
            elif all(self.owners[t] == 1 for t in territories):
                enemy_continent_bonus += self.continent_bonuses[continent_name]

        reward += 0.5 * player_continent_bonus
        reward -= 0.25 * enemy_continent_bonus

        # Player win condition
        if np.all(self.owners == 0):
            reward += 150
            terminated = True

        # Enemy acts if player has not already won
        if not terminated:
            reward += self._enemy_turn()

        # Enemy win condition
        if np.all(self.owners == 1):
            reward -= 150
            terminated = True

        # Timeout penalty
        if self.turn >= self.max_turns:
            reward -= 50
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

        print("\nContinents:")
        for continent_name, territories in self.continents.items():
            letters = [self.territory_names[t] for t in territories]
            print(
                f"{continent_name}: {letters}, "
                f"Bonus: +{self.continent_bonuses[continent_name]}"
            )

        print("\nMap:")
        print("Continent 1: A-B, A-C, B-D, C-D")
        print("Continent 2: G-H-I, G-J, H-K, J-K")
        print("Continent 3: E-F")
        print("Continent 4: L-M, L-N")
        print("Cross-continent: B-G, C-E, D-L, J-L, K-M")
        print("================\n")


if __name__ == "__main__":
    env = MiniRiskEnv()

    obs, _ = env.reset()
    done = False

    print("Starting MiniRisk v5 simulation...")

    while not done:
        env.render()

        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        print(f"Action Taken: {action} | Reward: {reward}")

        done = terminated or truncated

    env.render()
    print("Game over.")