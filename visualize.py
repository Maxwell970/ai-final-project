import pygame
import time
from stable_baselines3 import PPO
from env import MiniRiskEnv

WIDTH, HEIGHT = 900, 600

BLUE = (70, 130, 255)
RED = (230, 80, 80)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (180, 180, 180)
GREEN = (70, 200, 120)

POSITIONS = {
    0: (180, 180),  # A
    1: (450, 180),  # B
    2: (720, 180),  # C
    3: (180, 420),  # D
    4: (450, 420),  # E
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


def draw_arrow(screen, start, end):
    pygame.draw.line(screen, GREEN, start, end, 7)
    pygame.draw.circle(screen, GREEN, end, 12)


def draw_board(screen, env, font, small_font, action=None, reward=None):
    screen.fill(WHITE)

    # Draw connections
    for start, end in EDGES:
        pygame.draw.line(screen, GRAY, POSITIONS[start], POSITIONS[end], 6)

    # Draw attack arrow if action is attack
    if action is not None and action >= 5:
        attack_index = action - 5
        if 0 <= attack_index < len(ATTACK_ACTIONS):
            attacker, defender = ATTACK_ACTIONS[attack_index]
            draw_arrow(screen, POSITIONS[attacker], POSITIONS[defender])

    # Draw territories
    for i in range(5):
        x, y = POSITIONS[i]
        color = BLUE if env.owners[i] == 0 else RED

        pygame.draw.circle(screen, color, (x, y), 65)
        pygame.draw.circle(screen, BLACK, (x, y), 65, 3)

        name_text = font.render(NAMES[i], True, WHITE)
        troop_text = font.render(f"{env.troops[i]} troops", True, WHITE)

        screen.blit(name_text, name_text.get_rect(center=(x, y - 18)))
        screen.blit(troop_text, troop_text.get_rect(center=(x, y + 22)))

    # Sidebar text
    turn_text = font.render(f"Turn: {env.turn}", True, BLACK)
    screen.blit(turn_text, (30, 25))

    if action is not None:
        if action <= 4:
            action_desc = f"Reinforce {NAMES[action]}"
        else:
            attacker, defender = ATTACK_ACTIONS[action - 5]
            action_desc = f"Attack {NAMES[attacker]} -> {NAMES[defender]}"

        action_text = small_font.render(f"PPO Action: {action_desc}", True, BLACK)
        screen.blit(action_text, (30, 65))

    if reward is not None:
        reward_text = small_font.render(f"Reward: {reward:.2f}", True, BLACK)
        screen.blit(reward_text, (30, 95))

    legend = small_font.render(
        "Blue = PPO Agent | Red = Enemy | Green = PPO Attack",
        True,
        BLACK,
    )
    screen.blit(legend, (250, 540))

    pygame.display.flip()


def main():
    pygame.init()

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("MiniRisk v3 AI Visualization")

    font = pygame.font.SysFont(None, 36)
    small_font = pygame.font.SysFont(None, 28)

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
        action = int(action)

        obs, reward, terminated, truncated, info = env.step(action)

        draw_board(screen, env, font, small_font, action=action, reward=reward)

        done = terminated or truncated
        time.sleep(1.5)

    time.sleep(3)
    pygame.quit()


if __name__ == "__main__":
    main()