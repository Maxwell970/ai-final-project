import os
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


def draw_centered_text(screen, text, font, y, color=BLACK):
    rendered = font.render(text, True, color)
    screen.blit(rendered, rendered.get_rect(center=(WIDTH // 2, y)))


def choose_model(screen, title_font, font, small_font):
    screen.fill(WHITE)

    draw_centered_text(screen, "Choose PPO Model", title_font, 220)
    draw_centered_text(screen, "Press 1: Original PPO Model", font, 320)
    draw_centered_text(screen, "Press 2: Mixed-Training PPO Model", font, 370)
    draw_centered_text(
        screen,
        "Original = trained against scripted enemy. Mixed = trained against mixed opponent strategies.",
        small_font,
        460,
    )

    pygame.display.flip()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return None, None

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    return "ppo_minirisk", "Original PPO"
                if event.key == pygame.K_2:
                    if os.path.exists("ppo_minirisk_mixed.zip"):
                        return "ppo_minirisk_mixed", "Mixed PPO"
                    else:
                        screen.fill(WHITE)
                        draw_centered_text(screen, "Mixed model not found.", title_font, 260, RED)
                        draw_centered_text(screen, "Run python train_mixed.py first.", font, 340)
                        draw_centered_text(screen, "Press 1 for Original PPO or close window.", small_font, 420)
                        pygame.display.flip()


def choose_enemy_policy(screen, title_font, font, small_font):
    screen.fill(WHITE)

    draw_centered_text(screen, "Choose Opponent", title_font, 220)
    draw_centered_text(screen, "Press 1: Scripted Enemy", font, 320)
    draw_centered_text(screen, "Press 2: Greedy Enemy", font, 370)
    draw_centered_text(
        screen,
        "Scripted attacks probabilistically. Greedy always takes its strongest legal attack.",
        small_font,
        460,
    )

    pygame.display.flip()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return None, None

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    return "scripted", "Scripted Enemy"
                if event.key == pygame.K_2:
                    return "greedy", "Greedy Enemy"


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


def draw_board(
    screen,
    env,
    title_font,
    font,
    small_font,
    action=None,
    reward=None,
    info=None,
    enemy_label="Scripted Enemy",
    model_label="Original PPO",
):
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
    panel_y = 420

    screen.blit(title_font.render("MiniRisk PPO", True, BLACK), (panel_x, panel_y))
    screen.blit(small_font.render(f"Model: {model_label}", True, BLACK), (panel_x, panel_y + 50))
    screen.blit(small_font.render(f"Opponent: {enemy_label}", True, BLACK), (panel_x, panel_y + 85))
    screen.blit(font.render(f"Turn: {env.turn}", True, BLACK), (panel_x, panel_y + 125))

    screen.blit(font.render("PPO Turn:", True, BLACK), (panel_x, panel_y + 175))
    screen.blit(small_font.render(reinforce_desc, True, BLACK), (panel_x, panel_y + 215))
    screen.blit(small_font.render(attack_desc, True, BLACK), (panel_x, panel_y + 250))

    if reward is not None:
        screen.blit(font.render(f"Reward: {reward:.2f}", True, BLACK), (panel_x, panel_y + 305))

    player_count = int(np.sum(env.owners == 0))
    enemy_count = int(np.sum(env.owners == 1))

    screen.blit(
        small_font.render(
            f"Player territories: {player_count} | Enemy: {enemy_count}",
            True,
            BLACK,
        ),
        (panel_x, panel_y + 360),
    )

    player_reinforcements = env._calculate_reinforcements(0)
    enemy_reinforcements = env._calculate_reinforcements(1)

    screen.blit(
        small_font.render(
            f"Reinforcements - PPO: {player_reinforcements} | Enemy: {enemy_reinforcements}",
            True,
            BLACK,
        ),
        (panel_x, panel_y + 395),
    )

    if info is not None:
        screen.blit(
            small_font.render(
                f"Valid attacks: {info.get('successful_attacks', 0)} | "
                f"Invalid attacks: {info.get('invalid_attacks', 0)}",
                True,
                BLACK,
            ),
            (panel_x, panel_y + 430),
        )

    screen.blit(small_font.render("Blue = PPO | Red = Enemy", True, BLACK), (panel_x, panel_y + 475))
    screen.blit(small_font.render("Green = PPO Attack", True, BLACK), (panel_x, panel_y + 510))

    pygame.display.flip()


def main():
    pygame.init()

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("MiniRisk PPO Visualization")

    title_font = pygame.font.SysFont(None, 44)
    font = pygame.font.SysFont(None, 34)
    small_font = pygame.font.SysFont(None, 27)

    model_path, model_label = choose_model(screen, title_font, font, small_font)
    if model_path is None:
        return

    enemy_policy, enemy_label = choose_enemy_policy(screen, title_font, font, small_font)
    if enemy_policy is None:
        return

    env = MiniRiskEnv(enemy_policy=enemy_policy)
    model = MaskablePPO.load(model_path)

    obs, _ = env.reset()
    done = False
    info = None

    draw_board(
        screen,
        env,
        title_font,
        font,
        small_font,
        enemy_label=enemy_label,
        model_label=model_label,
    )
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
            enemy_label=enemy_label,
            model_label=model_label,
        )

        done = terminated or truncated
        time.sleep(1.5)

    time.sleep(3)
    pygame.quit()


if __name__ == "__main__":
    main()