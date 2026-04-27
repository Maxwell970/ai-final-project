import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import random
import numpy as np
import matplotlib.pyplot as plt

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env

from env import MiniRiskEnv


class MixedEnemyMiniRiskEnv(MiniRiskEnv):
    """
    Environment wrapper that randomly chooses the opponent strategy
    at the start of each episode.

    This helps the PPO agent learn a more robust policy rather than
    overfitting to one opponent type.
    """

    def __init__(self, greedy_probability=0.5):
        self.greedy_probability = greedy_probability
        super().__init__(enemy_policy="scripted")

    def reset(self, seed=None, options=None):
        if random.random() < self.greedy_probability:
            self.enemy_policy = "greedy"
        else:
            self.enemy_policy = "scripted"

        return super().reset(seed=seed, options=options)


class RewardCurveCallback(BaseCallback):
    def __init__(self, eval_freq=5_000, n_eval_episodes=20, verbose=1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.timesteps = []
        self.avg_rewards = []

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq == 0:
            avg_reward = self.evaluate_model()
            self.timesteps.append(self.num_timesteps)
            self.avg_rewards.append(avg_reward)

            if self.verbose:
                print(
                    f"Evaluation at {self.num_timesteps} timesteps: "
                    f"Average Reward = {avg_reward:.2f}"
                )

        return True

    def evaluate_model(self):
        rewards = []

        # Evaluate on mixed opponents too
        for _ in range(self.n_eval_episodes):
            env = MixedEnemyMiniRiskEnv(greedy_probability=0.5)
            obs, _ = env.reset()

            done = False
            total_reward = 0

            while not done:
                action_masks = get_action_masks(env)

                action, _ = self.model.predict(
                    obs,
                    deterministic=True,
                    action_masks=action_masks,
                )

                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                done = terminated or truncated

            rewards.append(total_reward)

        return np.mean(rewards)

    def save_reward_curve(self, filename="training_reward_curve_mixed.png"):
        plt.figure(figsize=(8, 5))
        plt.plot(self.timesteps, self.avg_rewards, marker="o")
        plt.title("PPO Learning Curve on MiniRiskEnv - Mixed Opponents")
        plt.xlabel("Training Timesteps")
        plt.ylabel("Average Evaluation Reward")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()


def train_mixed_model():
    check_env(MiniRiskEnv(), warn=True)

    env = MixedEnemyMiniRiskEnv(greedy_probability=0.5)

    model = MaskablePPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0003,
        n_steps=512,
        batch_size=256,
        gamma=0.99,
        ent_coef=0.04,
        device="cpu",
    )

    reward_callback = RewardCurveCallback(
        eval_freq=5_000,
        n_eval_episodes=20,
        verbose=1,
    )

    model.learn(
        total_timesteps=150_000,
        callback=reward_callback,
    )

    model.save("ppo_minirisk_mixed")
    reward_callback.save_reward_curve("training_reward_curve_mixed.png")

    print("Training complete. Model saved as ppo_minirisk_mixed.zip")
    print("Learning curve saved as training_reward_curve_mixed.png")


if __name__ == "__main__":
    train_mixed_model()