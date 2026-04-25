import random
import numpy as np


class RandomAgent:
    """
    Agent that chooses actions randomly.
    This is useful as a basic baseline.
    """

    def __init__(self, action_space):
        self.action_space = action_space

    def choose_action(self, obs):
        return self.action_space.sample()


class GreedyAgent:
    """
    Simple rule-based Risk agent.

    Observation format:
    obs = [owner_A, owner_B, owner_C, troops_A, troops_B, troops_C]

    Actions:
    0 = attack B from A
    1 = attack C from B
    2 = reinforce
    """

    def choose_action(self, obs):
        owners = obs[:3]
        troops = obs[3:]

        owner_A, owner_B, owner_C = owners
        troops_A, troops_B, troops_C = troops

        # If Player owns A and Enemy owns B, attack B if likely to win
        if owner_A == 0 and owner_B == 1 and troops_A > troops_B:
            return 0

        # If Player owns B and Enemy owns C, attack C if likely to win
        if owner_B == 0 and owner_C == 1 and troops_B > troops_C:
            return 1

        # Otherwise reinforce
        return 2


def run_agent_episode(env, agent, render=True):
    """
    Runs one full episode using the provided agent.
    Returns total reward and number of turns.
    """

    obs, _ = env.reset()
    done = False
    total_reward = 0
    turns = 0

    while not done:
        if render:
            env.render()

        action = agent.choose_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        turns += 1
        done = terminated or truncated

        if render:
            print(f"Action Taken: {action} | Reward: {reward}")

    if render:
        env.render()
        print("Episode finished.")
        print(f"Total Reward: {total_reward}")
        print(f"Turns: {turns}")

    return total_reward, turns