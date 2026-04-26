import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from sb3_contrib import MaskablePPO
from stable_baselines3.common.env_checker import check_env
from sb3_contrib.common.maskable.utils import get_action_masks

from env import MiniRiskEnv


def describe_attack(env, attack_choice):
    if attack_choice == 0:
        return "No attack"

    attacker, defender = env.attack_actions[attack_choice - 1]
    return f"{env.territory_names[attacker]} -> {env.territory_names[defender]}"


def train_model():
    check_env(MiniRiskEnv(), warn=True)

    env = MiniRiskEnv()

    model = MaskablePPO(
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

    model.learn(total_timesteps=100_000)

    model.save("ppo_minirisk")

    print("Training complete. Model saved as ppo_minirisk.zip")


def test_model():
    env = MiniRiskEnv()
    model = MaskablePPO.load("ppo_minirisk")

    obs, _ = env.reset()
    done = False
    total_reward = 0
    turns = 0
    final_info = {}

    print("\nTesting trained Maskable PPO agent...")

    while not done:
        env.render()

        action_masks = get_action_masks(env)
        action, _ = model.predict(
            obs,
            deterministic=True,
            action_masks=action_masks,
        )

        obs, reward, terminated, truncated, info = env.step(action)

        reinforce_target = int(action[0])
        attack_choice = int(action[1])
        attack_desc = describe_attack(env, attack_choice)

        print(
            f"Action Taken: {action} | "
            f"Reinforce: {env.territory_names[reinforce_target]} | "
            f"Attack: {attack_desc} | "
            f"Reward: {reward:.2f}"
        )

        total_reward += reward
        turns += 1
        final_info = info
        done = terminated or truncated

    env.render()

    print("\nMaskable PPO test complete.")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Turns: {turns}")

    print("\nDebug Summary:")
    print(f"Valid Reinforces: {final_info.get('valid_reinforces', 0)}")
    print(f"Invalid Reinforces: {final_info.get('invalid_reinforces', 0)}")
    print(f"Successful Attacks: {final_info.get('successful_attacks', 0)}")
    print(f"Invalid Attacks: {final_info.get('invalid_attacks', 0)}")
    print(f"Total Attacks Attempted: {final_info.get('total_attacks_attempted', 0)}")
    print(f"Skipped Valid Attacks: {final_info.get('skipped_valid_attacks', 0)}")
    print(f"Enemy Successful Attacks: {final_info.get('enemy_successful_attacks', 0)}")


if __name__ == "__main__":
    train_model()
    test_model()