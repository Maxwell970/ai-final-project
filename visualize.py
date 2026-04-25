import pygame
import time
import numpy as np
from stable_baselines3 import PPO
from env import MiniRiskEnv

WIDTH, HEIGHT = 1000, 650

BLUE = (70, 130, 255)
RED = (230, 80, 80)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (180, 180, 180)
GREEN = (70, 200, 120)
PURPLE = (160, 90, 220)

POSITIONS = {
    0: (180, 180),  # A
    1: (500, 180),  # B
    2: (820, 180),  # C
    3: (180, 440),  # D
    4: (500, 440),  # E
}

EDGES = [
    (0, 1),
    (1, 2),
    (0, 3),
    (1, 4),
    (3, 4),
]

NAMES = ["A", "B", "C", "D", "E"]

ATTACK_ACTIONS = [
    (0, 1),
    (0, 3),
    (1, 0),
    (1, 2),
    (1, 4),
    (2, 1),
    (3, 0),
    (3, 4),
    (4, 1),
    (4, 3),
]


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
        attacker, defender = ATTACK_ACTIONS[attack_choice - 1]
        attack_desc = f"Attack {NAMES[attacker]} -> {NAMES[defender]}"
        attack_pair = (attacker, defender)

    if fortify_amount == 0 or fortify_source == fortify_dest:
        fortify_desc = "No fortify"
        fortify_pair = None
    else:
        if fortify_amount == 4:
            amount_text = "all movable"
        else:
            amount_text = str(fortify_amount)

        fortify_desc = (
            f"Fortify {NAMES[fortify_source]} -> "
            f"{NAMES[fortify_dest]} ({amount_text})"
        )
        fortify_pair = (fortify_source, fortify_dest)

    return reinforce_desc, attack_desc, attack_pair, fortify_desc, fortify_pair


def draw_board(screen, env, font, small_font, action=None, reward=None):
    screen.fill(WHITE)

    # Draw map edges
    for start, end in EDGES:
        pygame.draw.line(screen, GRAY, POSITIONS[start], POSITIONS[end], 6)

    reinforce_desc = "No action yet"
    attack_desc = ""
    fortify_desc = ""
    attack_pair = None
    fortify_pair = None

    if action is not None:
        reinforce_desc, attack_desc, attack_pair, fortify_desc, fortify_pair = decode_action(action)

    # Draw attack arrow
    if attack_pair is not None:
        attacker, defender = attack_pair
        draw_arrow(screen, POSITIONS[attacker], POSITIONS[defender], GREEN)

    # Draw fortify arrow
    if fortify_pair is not None:
        source, dest = fortify_pair
        draw_arrow(screen, POSITIONS[source], POSITIONS[dest], PURPLE)

    # Draw territories
    for i in range(5):
        x, y = POSITIONS[i]
        color = BLUE if env.owners[i] == 0 else RED

        pygame.draw.circle(screen, color, (x, y), 70)
        pygame.draw.circle(screen, BLACK, (x, y), 70, 3)

        name_text = font.render(NAMES[i], True, WHITE)
        troops_text = font.render(str(env.troops[i]), True, WHITE)
        label_text = small_font.render("troops", True, WHITE)

        screen.blit(name_text, name_text.get_rect(center=(x, y - 28)))
        screen.blit(troops_text, troops_text.get_rect(center=(x, y + 5)))
        screen.blit(label_text, label_text.get_rect(center=(x, y + 35)))

    # Info panel
    turn_text = font.render(f"Turn: {env.turn}", True, BLACK)
    screen.blit(turn_text, (30, 25))

    action_header = small_font.render("PPO Turn:", True, BLACK)
    screen.blit(action_header, (30, 70))

    reinforce_text = small_font.render(reinforce_desc, True, BLACK)
    screen.blit(reinforce_text, (30, 105))

    attack_text = small_font.render(attack_desc, True, BLACK)
    screen.blit(attack_text, (30, 135))

    fortify_text = small_font.render(fortify_desc, True, BLACK)
    screen.blit(fortify_text, (30, 165))

    if reward is not None:
        reward_text = small_font.render(f"Reward: {reward:.2f}", True, BLACK)
        screen.blit(reward_text, (30, 205))

    legend1 = small_font.render("Blue = PPO Agent | Red = Enemy", True, BLACK)
    legend2 = small_font.render("Green arrow = Attack | Purple arrow = Fortify", True, BLACK)

    screen.blit(legend1, (300, 585))
    screen.blit(legend2, (300, 615))

    pygame.display.flip()


def main():
    pygame.init()

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("MiniRisk v4 PPO Visualization")

    font = pygame.font.SysFont(None, 38)
    small_font = pygame.font.SysFont(None, 26)

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