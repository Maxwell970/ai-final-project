import pygame
import time
import numpy as np
from stable_baselines3 import PPO
from env import MiniRiskEnv

WIDTH, HEIGHT = 1200, 800

BLUE = (70, 130, 255)
RED = (230, 80, 80)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (180, 180, 180)
GREEN = (70, 200, 120)
PURPLE = (160, 90, 220)

NAMES = list("ABCDEFGHIJKLMN")

POSITIONS = {
    0: (170, 170),   # A
    1: (330, 170),   # B
    2: (170, 320),   # C
    3: (330, 320),   # D

    4: (610, 170),   # E
    5: (770, 170),   # F

    6: (170, 560),   # G
    7: (330, 560),   # H
    8: (490, 560),   # I
    9: (170, 705),   # J
    10: (330, 705),  # K

    11: (750, 520),  # L
    12: (920, 520),  # M
    13: (750, 680),  # N
}

EDGES = [
    (0, 1), (0, 2),
    (1, 3), (1, 6),
    (2, 3), (2, 4),
    (3, 11),
    (4, 5),
    (6, 7), (6, 9),
    (7, 8), (7, 10),
    (9, 10), (9, 11),
    (10, 12),
    (11, 12), (11, 13),
]

CONTINENTS = {
    "C1": [0, 1, 2, 3],
    "C2": [6, 7, 8, 9, 10],
    "C3": [4, 5],
    "C4": [11, 12, 13],
}

CONTINENT_COLORS = {
    "C1": (220, 255, 220),
    "C2": (220, 240, 255),
    "C3": (255, 240, 210),
    "C4": (245, 225, 255),
}


def build_attack_actions():
    adjacency = {
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

    attack_actions = []
    for attacker in range(14):
        for defender in adjacency[attacker]:
            attack_actions.append((attacker, defender))

    return attack_actions


ATTACK_ACTIONS = build_attack_actions()


def draw_arrow(screen, start, end, color):
    pygame.draw.line(screen, color, start, end, 7)
    pygame.draw.circle(screen, color, end, 12)


def decode_action(action):
    action = np.array(action, dtype=int)

    reinforce_target = int(action[0])
    attack_choice = int(action[1])
    fortify_source = int(action[2])
    fortify_dest = int(action[3])
    fortify_amount = int(action[4])

    reinforce_desc = f"Reinforce {NAMES[reinforce_target]}"

    if attack_choice == 0:
        attack_desc = "No attack"
        attack_pair = None
    else:
        attack_index = attack_choice - 1

        if 0 <= attack_index < len(ATTACK_ACTIONS):
            attacker, defender = ATTACK_ACTIONS[attack_index]
            attack_desc = f"Attack {NAMES[attacker]} -> {NAMES[defender]}"
            attack_pair = (attacker, defender)
        else:
            attack_desc = "Invalid attack"
            attack_pair = None

    if fortify_amount == 0 or fortify_source == fortify_dest:
        fortify_desc = "No fortify"
        fortify_pair = None
    else:
        amount_text = "all movable" if fortify_amount == 4 else str(fortify_amount)
        fortify_desc = (
            f"Fortify {NAMES[fortify_source]} -> "
            f"{NAMES[fortify_dest]} ({amount_text})"
        )
        fortify_pair = (fortify_source, fortify_dest)

    return reinforce_desc, attack_desc, attack_pair, fortify_desc, fortify_pair


def draw_continent_backgrounds(screen, font):
    for name, territories in CONTINENTS.items():
        xs = [POSITIONS[t][0] for t in territories]
        ys = [POSITIONS[t][1] for t in territories]

        min_x, max_x = min(xs) - 85, max(xs) + 85
        min_y, max_y = min(ys) - 85, max(ys) + 85

        rect = pygame.Rect(min_x, min_y, max_x - min_x, max_y - min_y)
        pygame.draw.rect(screen, CONTINENT_COLORS[name], rect, border_radius=25)
        pygame.draw.rect(screen, GRAY, rect, 3, border_radius=25)

        label = font.render(name, True, BLACK)
        screen.blit(label, (min_x + 15, min_y + 12))


def draw_board(screen, env, font, small_font, action=None, reward=None):
    screen.fill(WHITE)

    draw_continent_backgrounds(screen, small_font)

    for start, end in EDGES:
        pygame.draw.line(screen, GRAY, POSITIONS[start], POSITIONS[end], 5)

    reinforce_desc = "No action yet"
    attack_desc = ""
    fortify_desc = ""
    attack_pair = None
    fortify_pair = None

    if action is not None:
        reinforce_desc, attack_desc, attack_pair, fortify_desc, fortify_pair = decode_action(action)

    if attack_pair is not None:
        attacker, defender = attack_pair
        draw_arrow(screen, POSITIONS[attacker], POSITIONS[defender], GREEN)

    if fortify_pair is not None:
        source, dest = fortify_pair
        draw_arrow(screen, POSITIONS[source], POSITIONS[dest], PURPLE)

    for i in range(14):
        x, y = POSITIONS[i]
        color = BLUE if env.owners[i] == 0 else RED

        pygame.draw.circle(screen, color, (x, y), 48)
        pygame.draw.circle(screen, BLACK, (x, y), 48, 3)

        name_text = font.render(NAMES[i], True, WHITE)
        troops_text = small_font.render(str(env.troops[i]), True, WHITE)

        screen.blit(name_text, name_text.get_rect(center=(x, y - 12)))
        screen.blit(troops_text, troops_text.get_rect(center=(x, y + 18)))

    panel_x = 860
    panel_y = 40

    title = font.render("MiniRisk PPO", True, BLACK)
    screen.blit(title, (panel_x, panel_y))

    turn_text = small_font.render(f"Turn: {env.turn}", True, BLACK)
    screen.blit(turn_text, (panel_x, panel_y + 45))

    action_header = small_font.render("PPO Turn:", True, BLACK)
    screen.blit(action_header, (panel_x, panel_y + 85))

    screen.blit(small_font.render(reinforce_desc, True, BLACK), (panel_x, panel_y + 120))
    screen.blit(small_font.render(attack_desc, True, BLACK), (panel_x, panel_y + 150))
    screen.blit(small_font.render(fortify_desc, True, BLACK), (panel_x, panel_y + 180))

    if reward is not None:
        reward_text = small_font.render(f"Reward: {reward:.2f}", True, BLACK)
        screen.blit(reward_text, (panel_x, panel_y + 225))

    player_count = int(np.sum(env.owners == 0))
    enemy_count = int(np.sum(env.owners == 1))

    counts = small_font.render(
        f"Player territories: {player_count} | Enemy: {enemy_count}",
        True,
        BLACK,
    )
    screen.blit(counts, (panel_x, panel_y + 265))

    legend1 = small_font.render("Blue = PPO | Red = Enemy", True, BLACK)
    legend2 = small_font.render("Green = Attack | Purple = Fortify", True, BLACK)
    screen.blit(legend1, (panel_x, panel_y + 315))
    screen.blit(legend2, (panel_x, panel_y + 345))

    pygame.display.flip()


def main():
    pygame.init()

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("MiniRisk v5 PPO Visualization")

    font = pygame.font.SysFont(None, 34)
    small_font = pygame.font.SysFont(None, 24)

    env = MiniRiskEnv()
    model = PPO.load("ppo_minirisk")

    obs, _ = env.reset()
    done = False

    draw_board(screen, env, font, small_font)
    time.sleep(1)

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        draw_board(screen, env, font, small_font, action=action, reward=reward)

        done = terminated or truncated
        time.sleep(1.5)

    time.sleep(3)
    pygame.quit()


if __name__ == "__main__":
    main()