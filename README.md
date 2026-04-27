# MiniRisk PPO Agent

A reinforcement learning project where a PPO agent learns to play a simplified Risk-style strategy game. Using a custom Gymnasium environment, action masking, and benchmark comparisons, the trained agent learns to outperform random and greedy heuristic baselines in territory conquest and game-winning efficiency.

---

# What it Does

MiniRisk PPO Agent is a custom reinforcement learning system built around a simplified version of the board game Risk. The project uses a map with 14 connected territories divided into continents, where the player and enemy compete for  control of the board. A Maskable Proximal Policy Optimization (PPO) agent learns how to reinforce territories, choose legal attacks, expand efficiently, and win games against an adversarial scripted opponent. The environment includes randomized balanced starting positions to reduce overfitting to one opening setup. The trained PPO model is evaluated against two baselines: a random agent and a greedy  agent. Results show that reinforcement learning can discover strong strategic behavior in a turn-based conquest game. 

---

# Quick Start

## Install Dependencies
    pip install -r requirements.txt

## Train the PPO Agent
    python train.py

## Evaluate Performance
    python evaluate.py

## Watch the Agent Play
    python visualize.py

---

# Video Links

Demo Link

Technical Walkthrough Link

---

# Evaluation

The PPO agent was compared against two baselines. A random agent and a greedy agent. The random agent selected actions randomly and the greedy agent reinforces its weakest border and attacks the best immediate target. 

## Metrics Used (PPO vs Scripted Opponent)
- Win Rate
- Average Reward
- Average Turns to Finish
- Valid Reinforcement %
- Attack Success %

| Agent  | Win Rate | Avg Reward | Avg Turns | Valid Reinforce % | Attack Success % | Bad Attacks | Skipped Good Attacks |
|--------|---------:|-----------:|----------:|------------------:|-----------------:|------------:|---------------------:|
| Random | 0.00%    | -235.85    | 30.99     | 30.75%            | 2.51%            | 29.47       | 0.32                 |
| Greedy | 79.00%   | 166.45     | 33.32     | 100.00%           | 100.00%          | 0.00        | 4.06                 |
| PPO    | 100.00%  | 221.55     | 30.52     | 100.00%           | 100.00%          | 0.00        | 3.62                 |

## Generated Charts


### Reward by Agent
<img src="EVIDENCE\avg_reward_by_agent_4-26.png" width="650">

### Win Rate by Agent
<img src="EVIDENCE\win_rate_by_agent_4-26.png" width="650">

### PPO Training Reward Curve
<img src="EVIDENCE\training_reward_curve_4-26.png" width="650">

Average reward increased substantially over the training period, despite random start positions for both the enemy and the player. 



