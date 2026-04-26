import pygame
import time
import numpy as np

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks

from env import MiniRiskEnv

WIDTH, HEIGHT = 1400, 850

BLUE = (85, 120, 240)
RED = (195, 80, 75)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (70, 200, 120)
CONTINENT_BORDER = (130, 205, 80)

NAMES = list("ABCDEFGHIJKLMN")

POSITIONS = {
    0: (160, 150), 1: (360, 150), 2: (160, 335), 3: (360, 335),
    4: (160, 570), 5: (160, 735),
    6: (690, 150), 7: (880, 150), 8: (1070, 150),
    9: (690, 335), 10: (880, 335),
    11: (690, 570), 12: (880, 570), 13: (690, 735),
}

EDGES = [
    (0, 1), (0, 2), (1, 3), (1, 6), (2, 3), (2, 4),
    (3, 11), (4, 5), (6, 7), (6, 9), (7, 8), (7, 10),
    (9, 10), (9, 11), (10, 12), (11, 12), (11, 13),
]

CONTINENT_RECTS = {
    "Continent 1": pygame.Rect(70, 65, 390, 380),
    "Continent 2": pygame.Rect(610, 65, 545, 380),
    "Continent 3": pygame.Rect(80, 500, 160, 305),
    "Continent 4": pygame.Rect(610, 500, 380, 305),
}


def draw_arrow(screen, start, end, color):
    pygame.draw.line(screen, color, start, end, 7)
    pygame.draw.circle(screen, color, end, 11)


def decode_action(env, action):
    action = np.array(action, dtype=int)

    reinforce_target = int(action[0])
    attack_choice = int(action[1])

    reinforce_desc = f"Reinforce {env.territory_names[reinforce_target]}"

    if attack_choice == 0:
        attack_desc = "No attack"
        attack_pair = None
    else:
        attacker, defender = env.attack_actions[attack_choice - 1]
        attack_desc = (
            f"Attack {env.territory_names[attacker]} -> "
            f"{env.territory_names[defender]}"
        )
        attack_pair = (attacker, defender)

    return reinforce_desc, attack_desc, attack_pair


def draw_continents(screen, font):
    for name, rect in CONTINENT_RECTS.items():
        pygame.draw.rect(screen, WHITE, rect)
        pygame.draw.rect(screen, CONTINENT_BORDER, rect, 6)

        label = font.render(name, True, BLACK)

        if name == "Continent 3":
            screen.blit(label, (rect.right + 25, rect.top + 230))
        elif name == "Continent 4":
            screen.blit(label, (rect.right + 25, rect.top + 90))
        else:
            screen.blit(label, (rect.left, rect.top - 55))


def draw_board(screen, env, title_font, font, small_font, action=None, reward=None, info=None):
    screen.fill(WHITE)

    draw_continents(screen, title_font)

    for start, end in EDGES:
        pygame.draw.line(screen, BLACK, POSITIONS[start], POSITIONS[end], 4)

    reinforce_desc = "No action yet"
    attack_desc = ""
    attack_pair = None

    if action is not None:
        reinforce_desc, attack_desc, attack_pair = decode_action(env, action)

    if attack_pair is not None:
        attacker, defender = attack_pair
        draw_arrow(screen, POSITIONS[attacker], POSITIONS[defender], GREEN)

    for i in range(14):
        x, y = POSITIONS[i]
        color = BLUE if env.owners[i] == 0 else RED

        pygame.draw.circle(screen, color, (x, y), 52)
        pygame.draw.circle(screen, BLACK, (x, y), 52, 3)

        name_text = font.render(NAMES[i], True, WHITE)
        troop_text = small_font.render(str(env.troops[i]), True, WHITE)

        screen.blit(name_text, name_text.get_rect(center=(x, y - 12)))
        screen.blit(troop_text, troop_text.get_rect(center=(x, y + 20)))

    panel_x = 1030
    panel_y = 470

    screen.blit(title_font.render("MiniRisk Maskable PPO", True, BLACK), (panel_x, panel_y))
    screen.blit(font.render(f"Turn: {env.turn}", True, BLACK), (panel_x, panel_y + 60))

    screen.blit(font.render("PPO Turn:", True, BLACK), (panel_x, panel_y + 110))
    screen.blit(small_font.render(reinforce_desc, True, BLACK), (panel_x, panel_y + 150))
    screen.blit(small_font.render(attack_desc, True, BLACK), (panel_x, panel_y + 185))

    if reward is not None:
        screen.blit(font.render(f"Reward: {reward:.2f}", True, BLACK), (panel_x, panel_y + 245))

    player_count = int(np.sum(env.owners == 0))
    enemy_count = int(np.sum(env.owners == 1))

    screen.blit(
        small_font.render(
            f"Player territories: {player_count} | Enemy: {enemy_count}",
            True,
            BLACK,
        ),
        (panel_x, panel_y + 300),
    )

    player_reinforcements = env._calculate_reinforcements(0)
    enemy_reinforcements = env._calculate_reinforcements(1)

    screen.blit(
        small_font.render(
            f"Reinforcements - PPO: {player_reinforcements} | Enemy: {enemy_reinforcements}",
            True,
            BLACK,
        ),
        (panel_x, panel_y + 335),
    )

    if info is not None:
        screen.blit(
            small_font.render(
                f"Valid attacks: {info.get('successful_attacks', 0)} | "
                f"Invalid attacks: {info.get('invalid_attacks', 0)}",
                True,
                BLACK,
            ),
            (panel_x, panel_y + 370),
        )

    screen.blit(small_font.render("Blue = PPO | Red = Enemy", True, BLACK), (panel_x, panel_y + 415))
    screen.blit(small_font.render("Green = Masked Legal Attack", True, BLACK), (panel_x, panel_y + 450))

    pygame.display.flip()


def main():
    pygame.init()

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("MiniRisk Maskable PPO Visualization")

    title_font = pygame.font.SysFont(None, 44)
    font = pygame.font.SysFont(None, 34)
    small_font = pygame.font.SysFont(None, 27)

    env = MiniRiskEnv()
    model = MaskablePPO.load("ppo_minirisk")

    obs, _ = env.reset()
    done = False
    info = None

    draw_board(screen, env, title_font, font, small_font)
    time.sleep(1)

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        action_masks = get_action_masks(env)
        action, _ = model.predict(
            obs,
            deterministic=True,
            action_masks=action_masks,
        )

        obs, reward, terminated, truncated, info = env.step(action)

        draw_board(
            screen,
            env,
            title_font,
            font,
            small_font,
            action=action,
            reward=reward,
            info=info,
        )

        done = terminated or truncated
        time.sleep(1.5)

    time.sleep(3)
    pygame.quit()


if __name__ == "__main__":
    main()