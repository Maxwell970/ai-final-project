April 24:
- Built PPO
- PPO beats random in turns
- Saved evaluation charts
- Enemy still doesn't take any actions

Currently, the board is A -- B -- C
The player has five troops on a, and the enemy has 2 on B and 1 on C
Also the enemy doesn't take actions. Will fix that now

Now:
Each turn:

Player chooses an action
0 = attack B from A
1 = attack C from B
2 = reinforce a player-owned territory
Attack succeeds only if attacker has more troops
Example: A can capture B only if troops_A > troops_B
If player owns all territories, player wins
Large positive reward
Enemy acts after the player
Enemy can reinforce one enemy territory
Enemy can attack adjacent player territory if it has more troops
If enemy owns all territories, player loses
Large negative reward
Game ends if max turns are reached
This prevents infinite games

Now updating so enemy is harder and actually takes moves
changed timesteps to 75000 to give more time to train for ppo model

changed the reward system to stop model from just waiting out and incentivizing it to win by giving a penalty for timeouts

now, the enemy has 25% to attack if they have more troops

greedy performs well in very simple three territories, either attack or reinforce

this was fixed by using an entropy term of 0.05 to encourage exploration making the ppo as efficient as the greedy algorithm


now changing the game rules:

1. Player reinforcement phase (+2 troops)
2. Player attack action
3. Check win
4. Enemy reinforcement phase (+2 troops)
5. Enemy attack action
6. Check win

changed rules again to allow for troops to be moved around after
also updated rewards to incentivize having control of territories and maintaining control


that worked. now i am changing it to 14 territories with a new map. also random territory starts, and continent + territory control bonuses

things maybe to do in future: let the attacks go on runs, so like attacking and then continuing from that newly conquered territory, adding in randomness to troop loss, changing visualize

for 14 territories, learning rate was slightly decreased, more training, keeping same entropy for encouraging exploration

ppo now performing poorly even with 500,000 training iterations. will try to change rewards to incentivize capturing entire continents for the troop bonus

now also trying balanced start that is the same each time

going to try removing the ability to fortify since action space was likely too big

to speed up training, simultaneously run 8 instances of the model?

maybe logic is wrong with counting troops after attack

main issue now is ppo model choosing invalid attacks and invalid reinforcements even after simplifying action space

