# pip install "gymnasium[toy-text]"
# sudo apt-get install ffmpeg

import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

import os
from pathlib import Path
import random
import numpy as np
import cv2
from datetime import datetime
from collections import deque

from PIL import Image, ImageDraw, ImageFont
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patheffects as path_effects
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.gridspec import GridSpec
import matplotlib.transforms as transforms
import matplotlib.colors as mcolors
import datetime


def count_holes(env):
    """Count holes in the environment"""
    desc = env.unwrapped.desc
    hole_count = 0
    hole_positions = []
    for i, row in enumerate(desc):
        for j, cell in enumerate(row):
            if cell == b'H':
                hole_count += 1
                hole_positions.append((i, j))
    return hole_count, hole_positions


def apply_fog(frame, agent_pos, visibility_range, grid_size):
    """Apply fog effect to the frame"""
    foggy_frame = frame.copy()
    fog_color = np.array([1, 1, 1])  # Gray color for fog
    cell_size = frame.shape[0] // grid_size
    agent_row, agent_col = agent_pos
    
    for row in range(grid_size):
        for col in range(grid_size):
            distance = abs(row - agent_row) + abs(col - agent_col)
            if distance > visibility_range:
                r_start, r_end = row * cell_size, (row + 1) * cell_size
                c_start, c_end = col * cell_size, (col + 1) * cell_size
                alpha = 0.95
                foggy_frame[r_start:r_end, c_start:c_end] = (
                    alpha * fog_color + (1 - alpha) * foggy_frame[r_start:r_end, c_start:c_end]
                ).astype(np.uint8)
    
    return foggy_frame


class FoggyFrozenLake(gym.Wrapper):
    """A wrapper for FrozenLake that limits the agent's view"""
    def __init__(self, env, visibility_range=1):
        super().__init__(env)
        self.visibility_range = visibility_range
        self.desc = np.array(self.unwrapped.desc)
        self.size = self.desc.shape[0]
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, info
    
    def render(self):
        full_frame = self.env.render()
        agent_pos = self.unwrapped.s // self.size, self.unwrapped.s % self.size
        foggy_frame = apply_fog(full_frame, agent_pos, self.visibility_range, self.size)
        return foggy_frame


def generate_exploration_path(env, max_steps=1000):
    """Generate a path that explores the entire grid efficiently"""
    desc = np.array(env.unwrapped.desc)
    size = desc.shape[0]
    actions = [0, 1, 2, 3]
    
    env.reset()
    visited = set()
    path = []
    current_pos = env.unwrapped.s
    visited.add(current_pos)
    
    holes = set()
    for i in range(size):
        for j in range(size):
            if desc[i, j] == b'H':
                holes.add(i * size + j)
    
    step_count = 0
    
    def get_next_state(state, action):
        row, col = state // size, state % size
        if action == 0:  # LEFT
            col = max(0, col - 1)
        elif action == 1:  # DOWN
            row = min(size - 1, row + 1)
        elif action == 2:  # RIGHT
            col = min(size - 1, col + 1)
        elif action == 3:  # UP
            row = max(0, row - 1)
        return row * size + col
    
    while step_count < max_steps:
        unvisited_neighbors = []
        for action in actions:
            next_state = get_next_state(current_pos, action)
            if next_state not in visited and next_state not in holes:
                unvisited_neighbors.append((action, next_state))
        
        if unvisited_neighbors:
            action, next_state = random.choice(unvisited_neighbors)
            path.append(action)
            current_pos = next_state
            visited.add(current_pos)
        else:
            queue = deque([(current_pos, [])])
            bfs_visited = {current_pos}
            found_path = False
            
            while queue and not found_path:
                state, state_path = queue.popleft()
                for action in actions:
                    next_state = get_next_state(state, action)
                    if next_state in holes:
                        continue
                    new_path = state_path + [action]
                    if next_state not in visited:
                        path.extend(new_path)
                        current_pos = next_state
                        visited.add(current_pos)
                        found_path = True
                        break
                    if next_state not in bfs_visited:
                        bfs_visited.add(next_state)
                        queue.append((next_state, new_path))
            
            if not found_path:
                break
        
        step_count += 1
        if len(visited) + len(holes) >= size * size:
            break
    
    return path


def visualize_exploration(env_to_explore, max_steps=1000, visibility_range=1, use_fog=True):
    """Generate and visualize a path that explores the entire grid"""
    env_to_explore.reset()
    path = generate_exploration_path(env_to_explore, max_steps)
    frames = [env_to_explore.render()]
    
    for action in path:
        obs, reward, done, truncated, info = env_to_explore.step(action)
        frames.append(env_to_explore.render())
        if done:
            break
    
    print(f"Exploration completed in {len(path)} steps")
    return frames, path


def get_grid_from_env(env):
    """Extract the grid layout from the environment"""
    desc = env.unwrapped.desc
    grid = np.zeros((len(desc), len(desc[0])))
    
    for i, row in enumerate(desc):
        for j, cell in enumerate(row):
            if cell == b'H':
                grid[i][j] = 1
    
    return grid


def find_shortest_paths(grid):
    """Find all shortest paths from start to goal without falling into holes"""
    n_rows, n_cols = grid.shape
    directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # Left, Down, Right, Up
    start_pos = (0, 0)
    goal_pos = (n_rows - 1, n_cols - 1)
    
    queue = deque([(start_pos, [])])
    visited = {start_pos: 0}
    shortest_paths = []
    shortest_length = float('inf')
    
    while queue:
        pos, path = queue.popleft()
        curr_dist = len(path)
        
        if curr_dist > shortest_length:
            break
        
        if pos == goal_pos:
            shortest_length = curr_dist
            shortest_paths.append(path)
            continue
        
        for action_idx, (di, dj) in enumerate(directions):
            ni, nj = pos[0] + di, pos[1] + dj
            new_pos = (ni, nj)
            
            if (0 <= ni < n_rows and 0 <= nj < n_cols and grid[ni][nj] == 0):
                new_dist = curr_dist + 1
                if new_pos not in visited or visited[new_pos] >= new_dist:
                    visited[new_pos] = new_dist
                    new_path = path + [action_idx]
                    queue.append((new_pos, new_path))
    
    return shortest_paths


def generate_random_non_goal_paths(grid, shortest_path_length, n):
    """Generate n random paths that don't reach the goal"""
    n_rows, n_cols = grid.shape
    directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    start_pos = (0, 0)
    goal_pos = (n_rows - 1, n_cols - 1)
    non_goal_paths = []
    
    def reaches_goal(path):
        pos = start_pos
        for action in path:
            di, dj = directions[action]
            new_pos = (pos[0] + di, pos[1] + dj)
            if not (0 <= new_pos[0] < n_rows and 0 <= new_pos[1] < n_cols) or grid[new_pos] == 1:
                return False
            pos = new_pos
            if pos == goal_pos:
                return True
        return False
    
    while len(non_goal_paths) < n:
        path = [random.randint(1, 2) for _ in range(shortest_path_length)]
        if not reaches_goal(path):
            non_goal_paths.append(path)
    
    return non_goal_paths


def create_multiple_choice_question(env, shortest_paths, non_goal_paths, save_path=None):
    """Create a multiple-choice question with path visualizations"""
    goal_path = random.choice(shortest_paths)
    all_paths = non_goal_paths + [goal_path]
    correct_index = len(all_paths) - 1
    
    shuffled_indices = list(range(len(all_paths)))
    random.shuffle(shuffled_indices)
    
    shuffled_paths = [all_paths[i] for i in shuffled_indices]
    correct_position = shuffled_indices.index(correct_index)
    correct_letter = chr(65 + correct_position)
    
    action_to_icon = ['⬅️', '⬇️', '➡️', '⬆️']
    question_text = "Which path leads to the goal without falling into holes?"
    
    n_paths = len(shuffled_paths)
    fig, axes = plt.subplots(1, n_paths, figsize=(5*n_paths, 5))
    if n_paths == 1:
        axes = [axes]
    
    option_text = ""
    for i, path in enumerate(shuffled_paths):
        letter = chr(65 + i)
        path_icons = " ".join([action_to_icon[action] for action in path])
        option_text += f"({letter}) {path_icons}\n"
    
    return question_text, option_text, correct_letter, fig


def create_simplified_animation(maze_frames, output_path, fps=4, figsize=(8, 8),
                               title="Maze Solver", subtitle="Pathfinding Visualization",
                               description_lines=None, credits=None, dpi=200):
    """Create a simplified maze animation without intro"""
    if description_lines is None:
        description_lines = ["This visualization demonstrates pathfinding algorithms"]
    
    if credits is None:
        credits = ["Please return a single letter (e.g. A)"]
    
    # Create figure with maze aspect ratio
    fig = plt.figure(figsize=figsize, facecolor='#1a1a2e')
    ax = fig.add_subplot(111)
    ax.set_facecolor('#0f3460')
    
    # Calculate frames
    n_maze_frames = len(maze_frames)
    n_outro_frames = int(fps * 6)  # 6 seconds outro
    total_frames = n_maze_frames + n_outro_frames
    
    def update(frame_num):
        ax.clear()
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # MAZE FRAMES
        if frame_num < n_maze_frames:
            ax.imshow(maze_frames[frame_num])
            
            # # Add step counter
            # step_text = ax.text(0.02, 0.98, f"Step {frame_num+1}/{n_maze_frames}", 
            #                   transform=ax.transAxes, fontsize=16,
            #                   verticalalignment='top', color='white',
            #                   bbox=dict(boxstyle='round,pad=0.5', facecolor='#16213e', 
            #                           alpha=0.8, edgecolor='#e94560'))
            # step_text.set_path_effects([path_effects.withStroke(linewidth=2, foreground='#0f3460')])
        
        # OUTRO SECTION
        else:
            outro_progress = (frame_num - n_maze_frames) / n_outro_frames
            
            # Dark background
            ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, 
                                 color='#1a1a2e', alpha=1.0))
            
            # Show title
            if outro_progress > 0.1:
                title_text = ax.text(0.5, 0.85, "Question", 
                                   transform=ax.transAxes, fontsize=28,
                                   horizontalalignment='center', color='white',
                                   weight='bold', alpha=min(1.0, (outro_progress - 0.1) * 3))
                
                # Add underline
                underline_width = min(0.3, (outro_progress - 0.1) * 0.6)
                ax.plot([0.5 - underline_width/2, 0.5 + underline_width/2], [0.83, 0.83], 
                       transform=ax.transAxes, color='#e94560', linewidth=3, 
                       alpha=min(1.0, (outro_progress - 0.1) * 3))
                
                title_text.set_path_effects([
                    path_effects.withStroke(linewidth=3, foreground='#0f3460'),
                ])
            
            # Show description lines
            if outro_progress > 0.3:
                for i, line in enumerate(description_lines):
                    line_progress = min(1.0, (outro_progress - 0.3 - i*0.1) * 3)
                    
                    if line_progress > 0:
                        alpha = line_progress
                        y_pos = 0.7 - i * 0.07
                        
                        desc_text = ax.text(0.5, y_pos, line, 
                                          transform=ax.transAxes, fontsize=16,
                                          horizontalalignment='center', color='white',
                                          alpha=alpha)
                        
                        desc_text.set_path_effects([
                            path_effects.withStroke(linewidth=2, foreground='#0f3460')
                        ])
            
            # Show credits
            if outro_progress > 0.7:
                for i, credit in enumerate(credits):
                    credit_progress = min(1.0, (outro_progress - 0.7 - i*0.05) * 4)
                    
                    if credit_progress > 0:
                        alpha = min(1.0, credit_progress * 1.5)
                        target_y = 0.2 - i * 0.05
                        current_y = target_y - 0.1 * (1 - min(1.0, credit_progress * 2))
                        
                        fontsize = 14 if i == 0 else 12
                        fontweight = 'bold' if i == 0 else 'normal'
                        color = '#e94560' if i == 0 else 'white'
                        
                        credit_text = ax.text(0.5, current_y, credit, 
                                            transform=ax.transAxes, fontsize=fontsize,
                                            horizontalalignment='center', color=color,
                                            alpha=alpha, weight=fontweight)
                        
                        credit_text.set_path_effects([
                            path_effects.withSimplePatchShadow(shadow_rgbFace='black', alpha=0.6)
                        ])
        
        return [ax]
    
    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=total_frames)
    
    # Save animation
    try:
        writer = animation.FFMpegWriter(fps=fps*2, bitrate=5000)
        ani.save(output_path, writer=writer, dpi=dpi)
        print(f"Animation saved to {output_path}")
    except Exception as e:
        print(f"Error saving with FFmpeg: {e}")
        try:
            gif_path = output_path.replace('.mp4', '.gif')
            writer = animation.PillowWriter(fps=fps)
            ani.save(gif_path, writer=writer, dpi=dpi)
            print(f"Saved as GIF instead: {gif_path}")
        except Exception as e2:
            print(f"Error saving with Pillow: {e2}")
    
    plt.close(fig)


def chunk_text(text, max_width=40, min_width=20):
    """Split text into chunks with roughly similar width"""
    words = text.split()
    chunks = []
    current_chunk = []
    current_width = 0
    
    for word in words:
        word_width = len(word)
        new_width = current_width + word_width + (1 if current_width > 0 else 0)
        
        if new_width <= max_width:
            current_chunk.append(word)
            current_width = new_width
        else:
            if current_width < min_width and len(current_chunk) > 0 and word != words[-1]:
                current_chunk.append(word)
                chunks.append(' '.join(current_chunk))
            else:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_width = word_width
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run Frozen Lake analysis with configurable parameters.')
    parser.add_argument('--size', type=int, default=10, help='Size of the random map (default: 10)')
    parser.add_argument('--question_name', type=str, default='count', choices=['count', 'min_length', 'agent_steps'], help='Name of the question for output files (default: script filename)')
    parser.add_argument('--use_fog', action='store_true', help='Enable fog effect (default: False)')
    parser.add_argument('--visibility_range', type=int, default=3, help='Visibility range when fog is enabled (default: 3)')
    parser.add_argument('--n_options', type=int, default=6, help='Number of multiple choice options (default: 6)')
    args = parser.parse_args()

    # Generate random map and setup environment
    random_map = generate_random_map(size=args.size)
    env = gym.make("FrozenLake-v1", desc=random_map, render_mode="rgb_array", is_slippery=False)
    
    # Create environment based on fog setting
    if args.use_fog:
        env_to_use = FoggyFrozenLake(env, visibility_range=args.visibility_range)
        title = "Foggy Frozen Lake"
        print(f"Using foggy environment with visibility range: {args.visibility_range}")
    else:
        env_to_use = env
        title = "Frozen Lake"
        print("Using regular environment without fog")
    
    # Generate exploration frames
    frames, path = visualize_exploration(env_to_use, max_steps=200, 
                                       visibility_range=args.visibility_range, 
                                       use_fog=args.use_fog)
    
    grid = get_grid_from_env(env)    
    # Get shortest paths and generate non-goal paths
    shortest_paths = find_shortest_paths(grid)
    shortest_path_length = len(shortest_paths[0]) if shortest_paths else 0
    non_goal_paths = generate_random_non_goal_paths(grid, shortest_path_length, args.n_options - 1)

    # Create a multiple choice question
    q_text, options, answer, fig = create_multiple_choice_question(
        env, shortest_paths, non_goal_paths, "mcq_question.png"
    )
    
    q_text = "Which of the listed move sequences carries the agent from Start to Goal without stepping onto a hole?"
    print(q_text)
    print("\n" + options)
    
    correct_answer = answer
    print(f"Answer: {correct_answer}")

    # Setup output paths
    script_path = __file__
    script_filename = os.path.basename(script_path)
    question_name = f"{script_filename.split('.')[0]}_sz{args.size}"
    if args.use_fog:
        question_name += f"_fog_vis{args.visibility_range}"
    else:
        question_name += "_nofog"
    
    question_dir = Path('questions')
    question_dir.mkdir(exist_ok=True)
    output_video = f"questions/{question_name}.mp4"

    subtitle = question_name
    description_lines = chunk_text(q_text) + [" "]*3 + [options]
    credits = ["Please return a single letter (e.g. A)", 
               "Nothing preceding or following it."]

    # Use simplified animation function
    create_simplified_animation(
        frames,
        output_video,
        fps=5,
        figsize=(8, 8),
        title=title,
        subtitle=subtitle,
        description_lines=description_lines,
        credits=credits,
        dpi=200
    )

    # Save solution files
    solution_dir = Path('solutions')
    solution_dir.mkdir(exist_ok=True)
    with open(f"solutions/{question_name}.txt", "w") as f:
        f.write(f"{correct_answer}")

    Image.fromarray(env.render()).save(f"solutions/{question_name}.png")

    question_text = f"{q_text}\n{options}\n{credits[0]},{credits[1]}"
    question_text_dir = Path('question_text')
    question_text_dir.mkdir(exist_ok=True)
    with open(f"question_text/{question_name}.txt", "w") as f:
        f.write(f"{question_text}")