import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

# ======================================================
# VARIABLES
# ======================================================

MAX_TURNS = 75
STARTING_TROOPS = 3
ENEMY_ATTACK_PROBABILITY = 0.25

BASE_TURN_REWARD = -0.05

VALID_REINFORCE_REWARD = 0.2
INVALID_REINFORCE_PENALTY = -1.0

NO_ATTACK_REWARD = -0.05
SKIP_VALID_ATTACK_PENALTY = -1.0
VALID_ATTACK_REWARD = 8.0
PER_TERRITORY_CAPTURE_REWARD = 3.0
INVALID_ATTACK_PENALTY = -1.0

PLAYER_TERRITORY_REWARD = 0.1
ENEMY_TERRITORY_PENALTY = -0.1

NEW_PLAYER_CONTINENT_REWARD = 8.0
NEW_ENEMY_CONTINENT_PENALTY = -6.0

PLAYER_CONTINENT_HOLD_REWARD = 0.05
ENEMY_CONTINENT_HOLD_PENALTY = -0.05

HOARDING_MULTIPLIER = 3.0
HOARDING_PENALTY = -1.0

ENEMY_REINFORCE_PENALTY = -0.05
ENEMY_SUCCESSFUL_ATTACK_PENALTY = -4.0

WIN_REWARD = 100
LOSS_PENALTY = -100
TIMEOUT_PENALTY = -50


class MiniRiskEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self.num_territories = 14
        self.territory_names = list("ABCDEFGHIJKLMN")

        self.adjacency = {
            0: [1, 2],
            1: [0, 3, 6],
            2: [0, 3, 4],
            3: [1, 2, 11],
            4: [2, 5],
            5: [4],
            6: [1, 7, 9],
            7: [6, 8, 10],
            8: [7],
            9: [6, 10, 11],
            10: [7, 9, 12],
            11: [3, 9, 12, 13],
            12: [10, 11],
            13: [11],
        }

        self.continents = {
            "Continent 1": [0, 1, 2, 3],
            "Continent 2": [6, 7, 8, 9, 10],
            "Continent 3": [4, 5],
            "Continent 4": [11, 12, 13],
        }

        self.continent_bonuses = {
            "Continent 1": 3,
            "Continent 2": 4,
            "Continent 3": 2,
            "Continent 4": 2,
        }

        self.attack_actions = []
        for attacker in range(self.num_territories):
            for defender in self.adjacency[attacker]:
                self.attack_actions.append((attacker, defender))

        self.num_attack_actions = len(self.attack_actions)

        # Action format:
        # [reinforce_target, attack_choice]
        #
        # reinforce_target: 0-13
        # attack_choice: 0 = no attack, 1-num_attack_actions = attack action
        self.action_space = spaces.MultiDiscrete(
            [self.num_territories, self.num_attack_actions + 1]
        )

        self.observation_space = spaces.Box(
            low=0,
            high=150,
            shape=(self.num_territories * 2,),
            dtype=np.int32,
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Fixed balanced start:
        # Player owns A, B, E, I, J, K, N
        # Enemy owns C, D, F, G, H, L, M
        self.owners = np.array(
            [0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0],
            dtype=np.int32,
        )

        self.troops = np.full(self.num_territories, STARTING_TROOPS, dtype=np.int32)
        self.turn = 0

        # Debug counters
        self.invalid_reinforces = 0
        self.invalid_attacks = 0
        self.skipped_valid_attacks = 0
        self.successful_attacks = 0
        self.total_attacks_attempted = 0
        self.enemy_successful_attacks = 0
        self.valid_reinforces = 0
        self.no_attack_turns = 0

        return self._get_obs(), {}

    # --------------------------------------------------
    # ACTION MASKING FOR MASKABLE PPO
    # --------------------------------------------------

    def action_masks(self):
        """
        MaskablePPO uses this to avoid illegal actions.

        For MultiDiscrete([14, num_attack_actions + 1]), the mask is flattened:
        first 14 entries = legal reinforce targets
        remaining entries = legal attack choices
        """

        reinforce_mask = np.zeros(self.num_territories, dtype=bool)

        for territory in range(self.num_territories):
            reinforce_mask[territory] = self.owners[territory] == 0

        attack_mask = np.zeros(self.num_attack_actions + 1, dtype=bool)

        # Always allow no attack
        attack_mask[0] = True

        for i, (attacker, defender) in enumerate(self.attack_actions):
            legal_attack = (
                self.owners[attacker] == 0
                and self.owners[defender] == 1
                and self.troops[attacker] > self.troops[defender]
            )

            attack_mask[i + 1] = legal_attack

        return np.concatenate([reinforce_mask, attack_mask])

    # --------------------------------------------------
    # CORE ENVIRONMENT HELPERS
    # --------------------------------------------------

    def _get_obs(self):
        return np.concatenate([self.owners, self.troops]).astype(np.int32)

    def _get_info(self):
        return {
            "invalid_reinforces": self.invalid_reinforces,
            "valid_reinforces": self.valid_reinforces,
            "invalid_attacks": self.invalid_attacks,
            "skipped_valid_attacks": self.skipped_valid_attacks,
            "successful_attacks": self.successful_attacks,
            "total_attacks_attempted": self.total_attacks_attempted,
            "enemy_successful_attacks": self.enemy_successful_attacks,
            "no_attack_turns": self.no_attack_turns,
            "player_territories": int(np.sum(self.owners == 0)),
            "enemy_territories": int(np.sum(self.owners == 1)),
            "player_continents": self._count_controlled_continents(0),
            "enemy_continents": self._count_controlled_continents(1),
        }

    def _count_controlled_continents(self, owner_id):
        count = 0

        for territories in self.continents.values():
            if all(self.owners[t] == owner_id for t in territories):
                count += 1

        return count

    def _continent_bonus_total(self, owner_id):
        total = 0

        for name, territories in self.continents.items():
            if all(self.owners[t] == owner_id for t in territories):
                total += self.continent_bonuses[name]

        return total

    def _calculate_reinforcements(self, owner_id):
        owned = int(np.sum(self.owners == owner_id))
        territory_bonus = max(3, owned // 3)
        continent_bonus = self._continent_bonus_total(owner_id)

        return territory_bonus + continent_bonus

    def _capture(self, attacker, defender, new_owner):
        remaining = self.troops[attacker] - self.troops[defender]

        self.owners[defender] = new_owner
        self.troops[defender] = max(1, remaining)
        self.troops[attacker] = 1

    def _place_reinforcements(self, owner_id, territory):
        if self.owners[territory] != owner_id:
            return False

        self.troops[territory] += self._calculate_reinforcements(owner_id)
        return True

    def _player_has_valid_attack(self):
        for attacker, defender in self.attack_actions:
            if (
                self.owners[attacker] == 0
                and self.owners[defender] == 1
                and self.troops[attacker] > self.troops[defender]
            ):
                return True

        return False

    def _player_turn(self, action):
        reward = BASE_TURN_REWARD
        reinforce_target, attack_choice = map(int, action)

        # Reinforce phase
        if self._place_reinforcements(0, reinforce_target):
            self.valid_reinforces += 1
            reward += VALID_REINFORCE_REWARD
        else:
            self.invalid_reinforces += 1
            reward += INVALID_REINFORCE_PENALTY

        # Attack phase
        if attack_choice == 0:
            self.no_attack_turns += 1

            if self._player_has_valid_attack():
                self.skipped_valid_attacks += 1
                reward += SKIP_VALID_ATTACK_PENALTY
            else:
                reward += NO_ATTACK_REWARD

        else:
            self.total_attacks_attempted += 1

            idx = attack_choice - 1
            attacker, defender = self.attack_actions[idx]

            valid = (
                self.owners[attacker] == 0
                and self.owners[defender] == 1
                and self.troops[attacker] > self.troops[defender]
            )

            if valid:
                prev_enemy = int(np.sum(self.owners == 1))

                self._capture(attacker, defender, 0)

                new_enemy = int(np.sum(self.owners == 1))
                captured = prev_enemy - new_enemy

                self.successful_attacks += 1

                reward += VALID_ATTACK_REWARD
                reward += PER_TERRITORY_CAPTURE_REWARD * captured

            else:
                self.invalid_attacks += 1
                reward += INVALID_ATTACK_PENALTY

        return reward

    # --------------------------------------------------
    # ENEMY LOGIC
    # --------------------------------------------------

    def _enemy_reinforce(self):
        enemy_owned = np.where(self.owners == 1)[0]

        if len(enemy_owned) == 0:
            return

        reinforcements = self._calculate_reinforcements(1)

        border = []
        for territory in enemy_owned:
            territory = int(territory)

            if any(self.owners[n] == 0 for n in self.adjacency[territory]):
                border.append(territory)

        if border:
            target = min(border, key=lambda x: self.troops[x])
        else:
            target = int(enemy_owned[np.argmin(self.troops[enemy_owned])])

        self.troops[target] += reinforcements

    def _enemy_attack(self):
        attacks = []

        for attacker in range(self.num_territories):
            if self.owners[attacker] != 1:
                continue

            for defender in self.adjacency[attacker]:
                if (
                    self.owners[defender] == 0
                    and self.troops[attacker] > self.troops[defender]
                ):
                    attacks.append((attacker, defender))

        if not attacks:
            return False

        if random.random() > ENEMY_ATTACK_PROBABILITY:
            return False

        attacker, defender = max(
            attacks,
            key=lambda p: self.troops[p[0]] - self.troops[p[1]],
        )

        self._capture(attacker, defender, 1)
        self.enemy_successful_attacks += 1

        return True

    def _enemy_turn(self):
        reward = 0.0

        self._enemy_reinforce()
        reward += ENEMY_REINFORCE_PENALTY

        if self._enemy_attack():
            reward += ENEMY_SUCCESSFUL_ATTACK_PENALTY

        return reward

    # --------------------------------------------------
    # STEP
    # --------------------------------------------------

    def step(self, action):
        self.turn += 1
        terminated = False
        truncated = False

        prev_player_cont = self._count_controlled_continents(0)
        prev_enemy_cont = self._count_controlled_continents(1)

        reward = self._player_turn(action)

        player_territories = int(np.sum(self.owners == 0))
        enemy_territories = int(np.sum(self.owners == 1))

        reward += PLAYER_TERRITORY_REWARD * player_territories
        reward += ENEMY_TERRITORY_PENALTY * enemy_territories

        new_player_cont = self._count_controlled_continents(0)
        new_enemy_cont = self._count_controlled_continents(1)

        reward += NEW_PLAYER_CONTINENT_REWARD * (new_player_cont - prev_player_cont)
        reward += NEW_ENEMY_CONTINENT_PENALTY * (new_enemy_cont - prev_enemy_cont)

        reward += PLAYER_CONTINENT_HOLD_REWARD * self._continent_bonus_total(0)
        reward += ENEMY_CONTINENT_HOLD_PENALTY * self._continent_bonus_total(1)

        player_troops = self.troops[self.owners == 0]

        if len(player_troops) > 0:
            if np.max(player_troops) > np.mean(player_troops) * HOARDING_MULTIPLIER:
                reward += HOARDING_PENALTY

        if np.all(self.owners == 0):
            reward += WIN_REWARD
            terminated = True

        if not terminated:
            reward += self._enemy_turn()

        if np.all(self.owners == 1):
            reward += LOSS_PENALTY
            terminated = True

        if self.turn >= MAX_TURNS:
            reward += TIMEOUT_PENALTY
            truncated = True

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self):
        print(f"\n=== TURN {self.turn} ===")

        for i in range(self.num_territories):
            owner = "Player" if self.owners[i] == 0 else "Enemy"
            print(
                f"{self.territory_names[i]} | "
                f"{owner:<6} | "
                f"Troops: {self.troops[i]}"
            )

        print("\nDebug Counters:")
        print(self._get_info())
        print("======================")


if __name__ == "__main__":
    env = MiniRiskEnv()
    obs, _ = env.reset()

    done = False

    while not done:
        env.render()

        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        print("Action:", action)
        print("Reward:", reward)

        done = terminated or truncated