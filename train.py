import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env

from env import MiniRiskEnv


def train_model():
    check_env(MiniRiskEnv(), warn=True)

    # Uses 8 environments at once to speed up training
    env = make_vec_env(MiniRiskEnv, n_envs=8)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0003,
        n_steps=512,
        batch_size=256,
        gamma=0.99,
        ent_coef=0.03,
        device="cpu",
    )

    model.learn(total_timesteps=150_000)

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
        obs, reward, terminated, truncated, info = env.step(action)

        reinforce_target = int(action[0])
        attack_choice = int(action[1])

        print(f"Action Taken: {action} | Reinforce: {env.territory_names[reinforce_target]} | Attack Choice: {attack_choice} | Reward: {reward}")

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