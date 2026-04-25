import pygame
import time
from stable_baselines3 import PPO
from env import MiniRiskEnv

WIDTH, HEIGHT = 800, 400

BLUE = (70, 130, 255)
RED = (230, 80, 80)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (180, 180, 180)
GREEN = (70, 200, 120)

POSITIONS = {
    0: (150, 200),  # A
    1: (400, 200),  # B
    2: (650, 200),  # C
}

NAMES = ["A", "B", "C"]


def draw_arrow(screen, start, end):
    pygame.draw.line(screen, GREEN, start, end, 6)
    pygame.draw.circle(screen, GREEN, end, 10)


def draw_board(screen, env, font, action=None, reward=None):
    screen.fill(WHITE)

    # Draw connections
    pygame.draw.line(screen, GRAY, POSITIONS[0], POSITIONS[1], 5)
    pygame.draw.line(screen, GRAY, POSITIONS[1], POSITIONS[2], 5)

    # Draw attack arrow
    if action == 0:
        draw_arrow(screen, POSITIONS[0], POSITIONS[1])
    elif action == 1:
        draw_arrow(screen, POSITIONS[1], POSITIONS[2])

    # Draw territories
    for i in range(3):
        x, y = POSITIONS[i]
        color = BLUE if env.owners[i] == 0 else RED

        pygame.draw.circle(screen, color, (x, y), 60)
        pygame.draw.circle(screen, BLACK, (x, y), 60, 3)

        name_text = font.render(NAMES[i], True, WHITE)
        troop_text = font.render(str(env.troops[i]), True, WHITE)

        screen.blit(name_text, name_text.get_rect(center=(x, y - 15)))
        screen.blit(troop_text, troop_text.get_rect(center=(x, y + 20)))

    # Info text
    turn_text = font.render(f"Turn: {env.turn}", True, BLACK)
    screen.blit(turn_text, (30, 30))

    if action is not None:
        action_text = font.render(f"Action: {action}", True, BLACK)
        screen.blit(action_text, (30, 70))

    if reward is not None:
        reward_text = font.render(f"Reward: {reward}", True, BLACK)
        screen.blit(reward_text, (30, 110))

    legend = font.render("Blue = PPO Agent | Red = Enemy | Green = Attack", True, BLACK)
    screen.blit(legend, (180, 340))

    pygame.display.flip()


def main():
    pygame.init()

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("MiniRisk AI Visualization")
    font = pygame.font.SysFont(None, 32)

    env = MiniRiskEnv()
    model = PPO.load("ppo_minirisk")

    obs, _ = env.reset()
    done = False

    draw_board(screen, env, font)
    time.sleep(1)

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))

        draw_board(screen, env, font, action=int(action), reward=reward)

        done = terminated or truncated
        time.sleep(1.5)

    time.sleep(3)
    pygame.quit()


if __name__ == "__main__":
    main()