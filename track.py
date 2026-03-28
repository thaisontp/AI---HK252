import pygame
import numpy as np

pygame.init()

# =========================
# SCREEN
# =========================
# (width, height) = (500, 300)
SCREEN_WIDTH = 500
SCREEN_HEIGHT = 500
CELL_SIZE = 25

COLS = SCREEN_WIDTH // CELL_SIZE
ROWS = SCREEN_HEIGHT // CELL_SIZE

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

# =========================
# tên
pygame.display.set_caption("Grid Agent Movement")

# =========================
# FONT
font = pygame.font.SysFont(None, 30)

# =========================
# FPS
# =========================
clock = pygame.time.Clock()
FPS = 140

# =========================
# GAME VARIABLES
# =========================
step = 50       # số bước tối đa cho agent để hoàn thành nhiệm vụ
step_fixed = 50 # lưu giá trị gốc để reset sau khi hết bước
cost_step = -1   # điểm trừ mỗi bước đi

total_reward = 0
terminal = 0
point = 0 


# =========================
# COLOR
# =========================
WHITE = (255,255,255)
BLACK = (0,0,0)
BLUE = (0,0,255)
GREEN = (0,255,0)
RED = (255,0,0)
YELLOW = (255,255,0)
AQUA = (0,255,255)

# =========================
# DRAW GRID
# =========================
def draw_grid():
    x_start, x_end = 0, 500
    y_start, y_end = 0, 300
    for x in range(x_start, x_end + 1, CELL_SIZE):
        pygame.draw.line(screen, (200,200,200), (x, y_start), (x, y_end))
    for y in range(y_start, y_end + 1, CELL_SIZE):
        pygame.draw.line(screen, (200,200,200), (x_start, y), (x_end, y))
# =========================
# HELPER: GRID -> PIXEL
# =========================
def draw_cell(color, col, row):
    pygame.draw.rect(
        screen,
        color,
        (col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE)
    )

# =========================
# MAP (GRID-BASED)
# =========================
walls = [
    (2,0),(2,1),
    (4,4),(4,5),(4,6),(4,7),
    (0,11),(1,11),(2,11),(3,11),(4,11),
    (9,4),(9,5),(9,6),(9,7),
    (5,8),(5,9),
    (13,1),(13,2),
    (16,2),(17,2),(18,2),(19,2),
    (16,5),(16,6),(16,7),(16,8),(16,9),(16,10),
    (12,10),(13,10),(14,10),(15,10)
]

rewards = {(3,3): 100, (0,10): 50, (14,8): 75}
initial_rewards = rewards.copy()
booms = [(8,0), (12,1), (16,11)]
goal = (19,11)
start = (0,0)

# =========================
# AGENT
# =========================
agent_row, agent_col = start

# actions: up, down, left, right
actions = [
    (-1, 0),  # up
    (1, 0),   # down
    (0, -1),  # left
    (0, 1)    # right
]

def move(row, col, action):
    global point, step
    dr, dc = actions[action]
    new_r = row + dr
    new_c = col + dc

    step += cost_step  # trừ điểm mỗi bước đi
    print("Steps remaining:", step)
    if step <=0:
        print("Out of steps! Game over!")
        step = step_fixed
        point = 0
        rewards.clear()
        rewards.update(initial_rewards)
        return 0, 0  # reset to start

    # check boundary
    if not (0 <= new_r < ROWS and 0 <= new_c < COLS):
        return row, col

    # check wall
    if (new_c, new_r) in walls:
        return row, col
    
    # check reward
    if (new_c, new_r) in rewards:
        value = rewards.pop((new_c, new_r))
        print(value)
        point += value
        print("Reward collected! Total reward:", point)
    
    # check boom
    if (new_c, new_r) in booms:
        print("Boom! Game over!")
        point = 0
        return 0, 0  # reset to start
    
    # check goal
    if (new_c, new_r) == goal:
        print("Goal reached! You win!")
        point = 0
        step = step_fixed
        rewards.clear()
        rewards.update(initial_rewards)
        return 0, 0  # reset to start

    return new_r, new_c

# =========================
# mảng chứa Q-values (state-action values)
# =========================

goal_state = (19, 11)

# =========================
# GAME LOOP
# =========================
running = True
while running:
    clock.tick(FPS)  # chậm lại để thấy rõ "nhảy"

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        # =========================
        # CONTROL (simulate agent action)
        # =========================
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                agent_row, agent_col = move(agent_row, agent_col, 0)
            if event.key == pygame.K_DOWN:
                agent_row, agent_col = move(agent_row, agent_col, 1)
            if event.key == pygame.K_LEFT:
                agent_row, agent_col = move(agent_row, agent_col, 2)
            if event.key == pygame.K_RIGHT:
                agent_row, agent_col = move(agent_row, agent_col, 3)

    # =========================
    # DRAW
    # =========================
    screen.fill(WHITE)
    draw_grid()

    # walls
    for (c, r) in walls:
        draw_cell(BLACK, c, r)

    # rewards
    for (c, r) in rewards:
        draw_cell(GREEN, c, r)

    # booms
    for (c, r) in booms:
        draw_cell(RED, c, r)

    # goalx             
    draw_cell(AQUA, goal[0], goal[1])

    # start
    draw_cell(YELLOW, start[0], start[1])

    # agent
    draw_cell(BLUE, agent_col, agent_row)

    # display STEPS còn lại và POINT hiện tại
    if step >= 0:
        text_surface = font.render(f"Steps: {step}", True, (BLACK))
        screen.blit(text_surface, (0, 300))
        text_surface = font.render(f"Point: {point}", True, (BLACK))
        screen.blit(text_surface, (0, 325))

    pygame.display.flip()

    


pygame.quit()