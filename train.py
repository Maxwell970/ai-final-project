from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from env import MiniRiskEnv


def train_model():
    env = MiniRiskEnv()

    # Checks that your custom Gymnasium environment follows the right format
    check_env(env, warn=True)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.001,
        n_steps=64,
        batch_size=32,
        gamma=0.95,
    )

    model.learn(total_timesteps=10_000)

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
    print("PPO test complete.")
    print(f"Total Reward: {total_reward}")
    print(f"Turns: {turns}")


if __name__ == "__main__":
    train_model()
    test_model()