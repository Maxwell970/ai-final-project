import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random


class MiniRiskEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self.num_territories = 5
        self.territory_names = ["A", "B", "C", "D", "E"]

        # Map:
        # A --- B --- C
        # |     |
        # D --- E
        self.adjacency = {
            0: [1, 3],
            1: [0, 2, 4],
            2: [1],
            3: [0, 4],
            4: [1, 3],
        }

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

        # Multi-part action:
        # [reinforce_target, attack_choice, fortify_source, fortify_dest, fortify_amount_choice]
        #
        # reinforce_target: 0-4
        # attack_choice: 0 = no attack, 1-10 = attack action
        # fortify_source: 0-4
        # fortify_dest: 0-4
        # fortify_amount_choice:
        #   0 = move 0
        #   1 = move 1
        #   2 = move 2
        #   3 = move 3
        #   4 = move all possible troops while leaving 1 behind
        self.action_space = spaces.MultiDiscrete([5, 11, 5, 5, 5])

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

        # 0 = player/PPO, 1 = enemy
        self.owners = np.array([0, 1, 1, 0, 1], dtype=np.int32)
        self.troops = np.array([6, 2, 2, 6, 2], dtype=np.int32)
        self.turn = 0

        return self._get_obs(), {}

    def _get_obs(self):
        return np.concatenate([self.owners, self.troops]).astype(np.int32)

    def _capture(self, attacker, defender, new_owner):
        remaining_attackers = self.troops[attacker] - self.troops[defender]

        self.owners[defender] = new_owner
        self.troops[defender] = max(1, remaining_attackers)
        self.troops[attacker] = 1

    def _reinforce(self, owner_id, territory, amount=2):
        if self.owners[territory] == owner_id:
            self.troops[territory] += amount
            return True
        return False

    def _can_fortify_between(self, owner_id, source, dest):
        if source == dest:
            return False

        if self.owners[source] != owner_id or self.owners[dest] != owner_id:
            return False

        # For now, fortify only between adjacent owned territories.
        # This keeps it Risk-like but simple.
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

        # 1. Player reinforcement phase
        if self._reinforce(owner_id=0, territory=int(reinforce_target), amount=2):
            reward += 0.2
        else:
            reward -= 0.5

        # 2. Player attack phase
        if attack_choice == 0:
            reward -= 0.1  # small penalty for skipping attack
        else:
            attack_index = int(attack_choice) - 1
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

        # 3. Player fortify/reorganize phase
        if self._fortify(
            owner_id=0,
            source=int(fortify_source),
            dest=int(fortify_dest),
            amount_choice=int(fortify_amount),
        ):
            reward += 0.2

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

    def _enemy_reinforce(self):
        enemy_owned = np.where(self.owners == 1)[0]

        if len(enemy_owned) == 0:
            return

        weakest = enemy_owned[np.argmin(self.troops[enemy_owned])]
        self.troops[weakest] += 2

    def _enemy_attack(self):
        possible_attacks = self._enemy_attack_options()

        strong_attacks = [
            (attacker, defender)
            for attacker, defender in possible_attacks
            if self.troops[attacker] > self.troops[defender]
        ]

        if strong_attacks and random.random() < 0.35:
            attacker, defender = random.choice(strong_attacks)
            self._capture(attacker, defender, new_owner=1)
            return True

        return False

    def _enemy_fortify(self):
        enemy_owned = np.where(self.owners == 1)[0]

        if len(enemy_owned) < 2:
            return False

        # Move troops from strongest enemy territory to weakest adjacent enemy territory.
        strongest = enemy_owned[np.argmax(self.troops[enemy_owned])]

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

        amount = min(2, movable)

        self.troops[strongest] -= amount
        self.troops[weakest_dest] += amount

        return True

    def _enemy_turn(self):
        penalty = 0.0

        # 1. Enemy reinforcement phase
        self._enemy_reinforce()
        penalty -= 0.1

        # 2. Enemy attack phase
        attacked = self._enemy_attack()
        if attacked:
            penalty -= 3.0

        # 3. Enemy fortify/reorganize phase
        self._enemy_fortify()

        return penalty

    def step(self, action):
        self.turn += 1
        terminated = False
        truncated = False

        reward = self._player_turn(action)
        player_territories = np.sum(self.owners == 0)
        enemy_territories = np.sum(self.owners == 1)

        reward += 0.2 * player_territories
        reward -= 0.1 * enemy_territories

        # Player win condition
        if np.all(self.owners == 0):
            reward += 100
            terminated = True

        # Enemy acts if player has not won
        if not terminated:
            reward += self._enemy_turn()

        # Enemy win condition
        if np.all(self.owners == 1):
            reward -= 20
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

        print("\nMap:")
        print("A --- B --- C")
        print("|     |")
        print("D --- E")
        print("================\n")


if __name__ == "__main__":
    env = MiniRiskEnv()

    obs, _ = env.reset()
    done = False

    print("Starting MiniRisk v4 simulation...")

    while not done:
        env.render()

        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        print(f"Action Taken: {action} | Reward: {reward}")

        done = terminated or truncated

    env.render()
    print("Game over.")