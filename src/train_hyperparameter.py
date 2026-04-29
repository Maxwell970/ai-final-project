import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.env_checker import check_env

from env import MiniRiskEnv


CONFIGS = [
    {
        "version": "A",
        "ent_coef": 0.01,
        "batch_size": 128,
        "n_steps": 512,
        "learning_rate": 0.0003,
        "gamma": 0.99,
    },
    {
        "version": "B",
        "ent_coef": 0.03,
        "batch_size": 256,
        "n_steps": 512,
        "learning_rate": 0.0003,
        "gamma": 0.99,
    },
    {
        "version": "C",
        "ent_coef": 0.04,
        "batch_size": 256,
        "n_steps": 512,
        "learning_rate": 0.0003,
        "gamma": 0.99,
    },
]


def evaluate_model(model, num_games=50):
    rewards = []
    turns = []
    wins = []

    for _ in range(num_games):
        env = MiniRiskEnv()
        obs, _ = env.reset()

        done = False
        total_reward = 0
        num_turns = 0

        while not done:
            action_masks = get_action_masks(env)

            action, _ = model.predict(
                obs,
                deterministic=True,
                action_masks=action_masks,
            )

            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            num_turns += 1
            done = terminated or truncated

        rewards.append(total_reward)
        turns.append(num_turns)
        wins.append(np.all(env.owners == 0))

    return {
        "avg_reward": np.mean(rewards),
        "avg_turns": np.mean(turns),
        "win_rate": np.mean(wins) * 100,
    }


def train_config(config, total_timesteps=100_000):
    env = MiniRiskEnv()

    model = MaskablePPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=config["learning_rate"],
        n_steps=config["n_steps"],
        batch_size=config["batch_size"],
        gamma=config["gamma"],
        ent_coef=config["ent_coef"],
        device="cpu",
    )

    model.learn(total_timesteps=total_timesteps)

    model_name = f"ppo_minirisk_hparam_{config['version']}"
    model.save(model_name)

    results = evaluate_model(model, num_games=50)

    return {
        "Version": config["version"],
        "Entropy": config["ent_coef"],
        "Batch Size": config["batch_size"],
        "Learning Rate": config["learning_rate"],
        "Gamma": config["gamma"],
        "Average Reward": round(results["avg_reward"], 2),
        "Average Turns": round(results["avg_turns"], 2),
        "Win Rate": round(results["win_rate"], 2),
        "Model File": model_name + ".zip",
    }


def main():
    check_env(MiniRiskEnv(), warn=True)

    all_results = []

    for config in CONFIGS:
        print(f"\nTraining version {config['version']}...")
        result = train_config(config)
        all_results.append(result)

    df = pd.DataFrame(all_results)

    df["Result"] = df["Average Reward"].rank(ascending=False).astype(int)
    df["Result"] = df["Result"].map({
        1: "best",
        2: "better",
        3: "weaker",
    })

    df.to_csv("hyperparameter_tuning_results.csv", index=False)

    print("\nHyperparameter Tuning Results")
    print(df.to_string(index=False))

    print("\nSaved results to hyperparameter_tuning_results.csv")


if __name__ == "__main__":
    main()