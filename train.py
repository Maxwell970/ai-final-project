import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from env import MiniRiskEnv


def train_model():
    env = MiniRiskEnv()

    check_env(env, warn=True)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0005,
        n_steps=128,
        batch_size=64,
        gamma=0.97,
        ent_coef=0.05,   # encourages exploration
        device="cpu",
    )

    model.learn(total_timesteps=75_000)

    model.save("ppo_minirisk")

    print("Training complete. Model saved as ppo_minirisk.zip")


def test_model():
    env = MiniRiskEnv()
    model = PPO.load("ppo_minirisk")

    obs, _ = env.reset()
    done = False
    total_reward = 0
    turns = 0

    print("\nTesting trained PPO agent...")

    while not done:
        env.render()

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))

        print(f"Action Taken: {int(action)} | Reward: {reward}")

        total_reward += reward
        turns += 1
        done = terminated or truncated

    env.render()
    print("\nPPO test complete.")
    print(f"Total Reward: {total_reward}")
    print(f"Turns: {turns}")


if __name__ == "__main__":
    train_model()
    test_model()