import random

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

G = 5
goal = (4, 4)
num_actions = 4

Q = np.zeros((G, G, num_actions))

alpha = 0.1
gamma = 0.9
epsilon = 0.2
num_episodes = 300
max_steps = 50
window_size = 5


def choose_action(state):
    if random.uniform(0, 1) < epsilon:
        return random.randint(0, num_actions - 1)
    x, y = state
    return np.argmax(Q[x, y])


def take_action(state, action):
    x, y = state
    if action == 0:
        x_new, y_new = max(x - 1, 0), y
    elif action == 1:
        x_new, y_new = min(x + 1, G - 1), y
    elif action == 2:
        x_new, y_new = x, max(y - 1, 0)
    else:
        x_new, y_new = x, min(y + 1, G - 1)

    new_state = (x_new, y_new)
    reward = 10 if new_state == goal else -1
    return new_state, reward


def update_q(state, action, reward, new_state):
    x, y = state
    x_new, y_new = new_state
    max_q_new = np.max(Q[x_new, y_new])
    Q[x, y, action] = Q[x, y, action] + alpha * (
        reward + gamma * max_q_new - Q[x, y, action]
    )


episode_paths = []
episode_steps = []

for _ in range(num_episodes):
    state = (0, 0)
    path = [state]

    for _ in range(max_steps):
        action = choose_action(state)
        new_state, reward = take_action(state, action)
        update_q(state, action, reward, new_state)
        state = new_state
        path.append(state)

        if state == goal:
            break

    episode_paths.append(path)
    episode_steps.append(len(path) - 1)


def find_best_episode(steps, window=5):
    candidate_indices = []

    for i in range(window, len(steps) - window):
        current_step = steps[i]
        prev_steps = steps[i - window:i]
        next_steps = steps[i + 1:i + window + 1]

        if all(step > current_step for step in prev_steps + next_steps):
            candidate_indices.append(i)

    if candidate_indices:
        return min(candidate_indices, key=lambda index: steps[index])

    return min(range(len(steps)), key=lambda index: steps[index])


best_episode_index = find_best_episode(episode_steps, window_size)
print("Steps tung episode:", episode_steps)
print(
    f"Episode toi uu: {best_episode_index + 1}, steps = {episode_steps[best_episode_index]}"
)


fig, ax = plt.subplots(figsize=(6, 6))


def draw_episode(path, episode_index):
    ax.clear()
    ax.set_xlim(-0.5, G - 0.5)
    ax.set_ylim(G - 0.5, -0.5)
    ax.set_xticks(range(G))
    ax.set_yticks(range(G))
    ax.set_xticks(np.arange(-0.5, G, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, G, 1), minor=True)
    ax.grid(which="minor", color="black", linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)

    ax.scatter(0, 0, s=220, c="gold", edgecolors="black", label="Start")
    ax.scatter(goal[1], goal[0], s=220, c="limegreen", edgecolors="black", label="Goal")

    rows = [state[0] for state in path]
    cols = [state[1] for state in path]
    ax.plot(cols, rows, color="blue", marker="o", linewidth=2)

    for step_idx, (row, col) in enumerate(path):
        ax.text(col, row, str(step_idx), ha="center", va="center", color="white", fontsize=8)

    title = f"Episode {episode_index + 1} - Steps: {len(path) - 1}"
    if episode_index == best_episode_index:
        title += " (best)"
    ax.set_title(title)
    ax.legend(loc="upper left")


def update(frame):
    draw_episode(episode_paths[frame], frame)
    return []


ani = animation.FuncAnimation(
    fig,
    update,
    frames=best_episode_index + 1,
    interval=700,
    repeat=False,
)

plt.show()
