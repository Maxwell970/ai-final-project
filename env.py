import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import time


class MiniRiskEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self.num_territories = 3
        self.action_space = spaces.Discrete(3)

        self.observation_space = spaces.Box(
            low=0,
            high=10,
            shape=(6,),
            dtype=np.int32
        )

        self.max_turns = 20
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.owners = np.array([0, 1, 1], dtype=np.int32)
        self.troops = np.array([5, 2, 1], dtype=np.int32)
        self.turn = 0

        return self._get_obs(), {}

    def _get_obs(self):
        return np.concatenate([self.owners, self.troops]).astype(np.int32)

    def step(self, action):
        self.turn += 1
        reward = -0.1
        terminated = False
        truncated = False

        # Action 0: attack B from A
        if action == 0:
            if self.owners[0] == 0 and self.owners[1] == 1 and self.troops[0] > self.troops[1]:
                self.owners[1] = 0
                self.troops[1] = 1
                self.troops[0] -= 1
                reward = 1

        # Action 1: attack C from B
        elif action == 1:
            if self.owners[1] == 0 and self.owners[2] == 1 and self.troops[1] > self.troops[2]:
                self.owners[2] = 0
                self.troops[2] = 1
                self.troops[1] -= 1
                reward = 1

        # Action 2: reinforce owned territories
        elif action == 2:
            owned = np.where(self.owners == 0)[0]
            reinforce_territory = random.choice(owned)
            self.troops[reinforce_territory] += 1
            reward = 0.2

        # Win condition
        if np.all(self.owners == 0):
            reward = 10
            terminated = True

        # Stop if game takes too long
        if self.turn >= self.max_turns:
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

    print("Starting MiniRisk simulation...")

    while not done:
        env.render()

        action = random.randint(0, 2)
        obs, reward, terminated, truncated, info = env.step(action)

        print(f"Action Taken: {action} | Reward: {reward}")

        done = terminated or truncated
        time.sleep(0.5)

    env.render()
    print("Game over.")