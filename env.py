import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random


class MiniRiskEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self.num_territories = 3
        self.action_space = spaces.Discrete(3)

        self.observation_space = spaces.Box(
            low=0,
            high=30,
            shape=(6,),
            dtype=np.int32,
        )

        self.max_turns = 25
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Board: A --- B --- C
        # 0 = player/PPO, 1 = enemy
        self.owners = np.array([0, 1, 1], dtype=np.int32)

        # Tuned starting point
        self.troops = np.array([6, 2, 2], dtype=np.int32)

        self.turn = 0
        return self._get_obs(), {}

    def _get_obs(self):
        return np.concatenate([self.owners, self.troops]).astype(np.int32)

    def _capture(self, attacker, defender, new_owner):
        """
        Combat rule:
        If attacker wins, defender territory gets:
        attacking_troops - defending_troops

        Example:
        A has 6, B has 2.
        A attacks B.
        B becomes player-owned with 4 troops.
        A is left with 1 troop.
        """
        remaining_attackers = self.troops[attacker] - self.troops[defender]

        self.owners[defender] = new_owner
        self.troops[defender] = max(1, remaining_attackers)
        self.troops[attacker] = 1

    def _player_action(self, action):
        reward = -0.1  # small time penalty

        # Action 0: A attacks B
        if action == 0:
            if self.owners[0] == 0 and self.owners[1] == 1:
                if self.troops[0] > self.troops[1]:
                    self._capture(0, 1, 0)
                    reward += 2.0
                else:
                    reward -= 0.75
            else:
                reward -= 0.25

        # Action 1: B attacks C
        elif action == 1:
            if self.owners[1] == 0 and self.owners[2] == 1:
                if self.troops[1] > self.troops[2]:
                    self._capture(1, 2, 0)
                    reward += 2.0
                else:
                    reward -= 0.75
            else:
                reward -= 0.25

        # Action 2: reinforce weakest player-owned territory
        elif action == 2:
            owned = np.where(self.owners == 0)[0]

            if len(owned) > 0:
                weakest = owned[np.argmin(self.troops[owned])]
                self.troops[weakest] += 1
                reward += 0.1

        return reward

    def _enemy_turn(self):
        penalty = 0.0

        possible_attacks = []

        # Enemy B attacks Player A
        if self.owners[1] == 1 and self.owners[0] == 0:
            possible_attacks.append((1, 0))

        # Enemy C attacks Player B
        if self.owners[2] == 1 and self.owners[1] == 0:
            possible_attacks.append((2, 1))

        strong_attacks = [
            (attacker, defender)
            for attacker, defender in possible_attacks
            if self.troops[attacker] > self.troops[defender]
        ]

        # Enemy sometimes attacks if it can
        if strong_attacks and random.random() < 0.25:
            attacker, defender = random.choice(strong_attacks)
            self._capture(attacker, defender, 1)
            penalty -= 2.0

        else:
            # Enemy reinforces weakest enemy territory
            enemy_territories = np.where(self.owners == 1)[0]

            if len(enemy_territories) > 0:
                weakest = enemy_territories[np.argmin(self.troops[enemy_territories])]
                self.troops[weakest] += 1
                penalty -= 0.1

        return penalty

    def step(self, action):
        self.turn += 1
        terminated = False
        truncated = False

        reward = self._player_action(action)

        # Player wins by owning all territories
        if np.all(self.owners == 0):
            reward += 10
            terminated = True

        # Enemy acts only if player has not won
        if not terminated:
            reward += self._enemy_turn()

        # Enemy wins by owning all territories
        if np.all(self.owners == 1):
            reward -= 10
            terminated = True

        # Timeout penalty discourages PPO from stalling
        if self.turn >= self.max_turns:
            reward -= 5
            truncated = True

        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        territory_names = ["A", "B", "C"]

        print(f"\n=== TURN {self.turn} ===")
        for i in range(3):
            owner = "Player" if self.owners[i] == 0 else "Enemy"
            print(f"{territory_names[i]} | Owner: {owner} | Troops: {self.troops[i]}")
        print("================\n")


if __name__ == "__main__":
    env = MiniRiskEnv()

    obs, _ = env.reset()
    done = False

    print("Starting tuned MiniRisk simulation...")

    while not done:
        env.render()

        action = random.randint(0, 2)
        obs, reward, terminated, truncated, info = env.step(action)

        print(f"Action Taken: {action} | Reward: {reward}")

        done = terminated or truncated

    env.render()
    print("Game over.")