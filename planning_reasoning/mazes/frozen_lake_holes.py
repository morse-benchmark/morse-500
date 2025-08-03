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
    # Get the description of the environment
    desc = env.unwrapped.desc
    
    # Count the number of 'H' characters (holes)
    hole_count = 0
    hole_positions = []
    for i, row in enumerate(desc):
        for j, cell in enumerate(row):
            if cell == b'H':  # In some versions, it's stored as bytes
                hole_count += 1
                hole_positions.append((i, j))
    
    return hole_count, hole_positions


def get_connected_holes(env, n_neighbors=8):
    # Get the description of the environment
    desc = np.array(env.unwrapped.desc)
    size = desc.shape[0]
    
    # Create a matrix to mark visited holes
    visited = np.zeros((size, size), dtype=bool)

    if n_neighbors == 8:
        # 8-connectivity directions (horizontal, vertical, and diagonal neighbors)
        directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]
    else:
        # 4-connectivity directions (horizontal, vertical neighbors)
        directions = [
                      (-1, 0),
            (0, -1),           (0, 1),
                      (1, 0), 
        ]
    
    # Function to check if a cell is valid and has a hole
    def is_valid(r, c):
        return (0 <= r < size and 
                0 <= c < size and 
                desc[r][c] == b'H' and 
                not visited[r][c])
    
    # BFS to find connected holes
    def bfs(start_r, start_c):
        queue = deque([(start_r, start_c)])
        visited[start_r][start_c] = True
        cells = [(start_r, start_c)]
        
        while queue:
            r, c = queue.popleft()
            
            for dr, dc in directions:
                new_r, new_c = r + dr, c + dc
                if is_valid(new_r, new_c):
                    visited[new_r][new_c] = True
                    queue.append((new_r, new_c))
                    cells.append((new_r, new_c))
        
        return cells
    
    # Find all connected components
    connected_components = []
    for r in range(size):
        for c in range(size):
            if desc[r][c] == b'H' and not visited[r][c]:
                component = bfs(r, c)
                connected_components.append(component)
    
    return connected_components


def visualize_path(env, path, visibility_range=1):
    """
    Visualize a path in the environment with limited visibility (fog)
    
    Args:
        env: The Frozen Lake environment
        path: List of actions to visualize
        visibility_range: How many cells the agent can see in each direction (default: 1)
    """
    env.reset()
    frames = []
    
    # Get the description array and size
    desc = np.array(env.unwrapped.desc)
    size = desc.shape[0]
    
    # Get the agent's position from the environment state
    agent_pos = env.unwrapped.s // size, env.unwrapped.s % size
    
    # Get the rendered initial state
    full_frame = env.render()
    foggy_frame = apply_fog(full_frame, agent_pos, visibility_range, size)
    frames.append(foggy_frame)
    
    for action in path:
        obs, reward, done, truncated, info = env.step(action)
        
        # Get the agent's position from the environment state after the action
        agent_pos = env.unwrapped.s // size, env.unwrapped.s % size
        
        # Get the full frame
        full_frame = env.render()
        
        # Apply fog and save the foggy frame
        foggy_frame = apply_fog(full_frame, agent_pos, visibility_range, size)
        frames.append(foggy_frame)
        
        if done:
            break
    
    # Display the frames as an animation
    import matplotlib.pyplot as plt
    from IPython.display import clear_output
    
    for i, frame in enumerate(frames):
        plt.figure(figsize=(5, 5))
        plt.imshow(frame)
        # plt.title(f"Step {i}: {action_names[path[i-1]] if i>0 else 'Start'}")
        plt.axis('off')
        plt.show()
        clear_output(wait=True)
        plt.pause(0.1)


def apply_fog(frame, agent_pos, visibility_range, grid_size):
    """
    Apply fog effect to the frame, only showing cells within visibility_range of the agent
    
    Args:
        frame: The full rendered frame
        agent_pos: (row, col) position of the agent in grid coordinates
        visibility_range: How many cells the agent can see in each direction
        grid_size: Size of the grid (e.g., 8 for an 8x8 grid)
        
    Returns:
        np.array: The foggy frame where areas outside visibility range are grayed out
    """
    import numpy as np
    
    # Create a copy of the frame to modify
    foggy_frame = frame.copy()
    
    # Create a fog mask (gray color)
    fog_color = np.array([1, 1, 1])  # Gray color for fog
    
    # Calculate the cell size based on the frame dimensions and the grid size
    cell_size = frame.shape[0] // grid_size
    
    # Get agent grid position
    agent_row, agent_col = agent_pos
    
    # Apply fog to all cells outside the visibility range
    for row in range(grid_size):
        for col in range(grid_size):
            # Calculate Manhattan distance from agent
            distance = abs(row - agent_row) + abs(col - agent_col)
            
            # If outside visibility range, apply fog
            if distance > visibility_range:
                # Calculate pixel coordinates for this cell
                r_start, r_end = row * cell_size, (row + 1) * cell_size
                c_start, c_end = col * cell_size, (col + 1) * cell_size
                
                # Apply a semi-transparent fog (blend with original)
                alpha = 0.95  # Fog opacity
                foggy_frame[r_start:r_end, c_start:c_end] = (
                    alpha * fog_color + (1 - alpha) * foggy_frame[r_start:r_end, c_start:c_end]
                ).astype(np.uint8)
    
    return foggy_frame


class FoggyFrozenLake(gym.Wrapper):
    """
    A wrapper for FrozenLake that limits the agent's view to a certain range.
    This creates a partially observable environment.
    """
    def __init__(self, env, visibility_range=1):
        super().__init__(env)
        self.visibility_range = visibility_range
        
        # Get the description array and size
        self.desc = np.array(self.unwrapped.desc)
        self.size = self.desc.shape[0]
        
        # Modify observation space if needed
        # If you want the agent to truly only see nearby cells:
        # self.observation_space = spaces.Box(...)
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # For a truly fog-limited agent, you'd return only visible cells
        # For now, we'll just return the standard observation
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # For a truly fog-limited agent, modify obs here to only include visible cells
        return obs, reward, terminated, truncated, info
    
    def render(self):
        # Get the full render from the base environment
        full_frame = self.env.render()
        
        # Get the agent's position
        agent_pos = self.unwrapped.s // self.size, self.unwrapped.s % self.size
        
        # Apply fog effect
        foggy_frame = apply_fog(full_frame, agent_pos, self.visibility_range, self.size)
        
        return foggy_frame
    

def generate_exploration_path(env, max_steps=1000):
    """
    Generate a path that explores the entire grid efficiently
    
    Args:
        env: The Frozen Lake environment
        max_steps: Maximum number of steps to prevent infinite loops
        
    Returns:
        list: A sequence of actions that covers the grid
    """
    # Get environment information
    desc = np.array(env.unwrapped.desc)
    size = desc.shape[0]
    
    # Define actions (0: LEFT, 1: DOWN, 2: RIGHT, 3: UP)
    actions = [0, 1, 2, 3]
    
    # Initialize exploration
    env.reset()
    
    # Track visited states and path
    visited = set()
    path = []
    current_pos = env.unwrapped.s
    visited.add(current_pos)
    
    # Keep track of holes to avoid
    holes = set()
    for i in range(size):
        for j in range(size):
            if desc[i, j] == b'H':
                holes.add(i * size + j)
    
    step_count = 0
    
    # Helper function to get next state given current state and action
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
    
    # Use a mix of DFS exploration and random exploration
    while step_count < max_steps:
        # Find unvisited neighbors
        unvisited_neighbors = []
        for action in actions:
            next_state = get_next_state(current_pos, action)
            if next_state not in visited and next_state not in holes:
                unvisited_neighbors.append((action, next_state))
        
        # If we have unvisited neighbors, go to one of them
        if unvisited_neighbors:
            # Prioritize unvisited states
            action, next_state = random.choice(unvisited_neighbors)
            path.append(action)
            current_pos = next_state
            visited.add(current_pos)
        else:
            # If all neighbors are visited, use BFS to find the shortest path to an unvisited state
            queue = deque([(current_pos, [])])
            bfs_visited = {current_pos}
            
            found_path = False
            while queue and not found_path:
                state, state_path = queue.popleft()
                
                for action in actions:
                    next_state = get_next_state(state, action)
                    
                    # Skip holes
                    if next_state in holes:
                        continue
                        
                    new_path = state_path + [action]
                    
                    # Check if we found an unvisited state
                    if next_state not in visited:
                        # Add the path to reach this unvisited state
                        path.extend(new_path)
                        current_pos = next_state
                        visited.add(current_pos)
                        found_path = True
                        break
                    
                    # Continue BFS if this state hasn't been visited in this BFS
                    if next_state not in bfs_visited:
                        bfs_visited.add(next_state)
                        queue.append((next_state, new_path))
            
            # If we couldn't find a path to an unvisited state, we've covered all accessible cells
            if not found_path:
                break
        
        step_count += 1
        
        # If we've visited all non-hole cells, we're done
        if len(visited) + len(holes) >= size * size:
            break
    
    return path

def visualize_exploration(foggy_env, max_steps=1000, visibility_range=1):
    """
    Generate and visualize a path that explores the entire grid
    
    Args:
        foggy_env: The Foggy Frozen Lake environment
        max_steps: Maximum number of steps to prevent infinite loops
        visibility_range: Visibility range for the foggy environment
    """
    # Reset the environment
    foggy_env.reset()
    
    # Generate exploration path
    path = generate_exploration_path(foggy_env, max_steps)
    
    # Render the initial state
    frames = [foggy_env.render()]
    
    # Follow the exploration path
    for action in path:
        obs, reward, done, truncated, info = foggy_env.step(action)
        frames.append(foggy_env.render())
        if done:
            break
    
    # Display the frames as an animation
    import matplotlib.pyplot as plt
    from IPython.display import clear_output
    
    # Define action names for visualization
    action_names = ["LEFT", "DOWN", "RIGHT", "UP"]
    
    for i, frame in enumerate(frames):
        plt.figure(figsize=(6, 6))
        plt.imshow(frame)
        plt.title(f"Step {i}: {action_names[path[i-1]] if i>0 else 'Start'}")
        plt.axis('off')
        plt.show()
        clear_output(wait=True)
        plt.pause(0.1)
    
    print(f"Exploration completed in {len(path)} steps")
    print(f"Visited {len(set([foggy_env.unwrapped.s // foggy_env.size + foggy_env.unwrapped.s % foggy_env.size for _ in range(1)]))} cells")
    
    return frames, path


def create_professional_plt_animation(maze_frames, output_path, distance_list = None, fps=4, figsize=(12, 12),
                                     title="Maze Solver", subtitle="Pathfinding Visualization",
                                     description_lines=None, credits=None,
                                     dpi=200):
    """
    Create a professional maze animation with advanced text effects using matplotlib.
    
    Parameters:
    maze_frames (list): List of numpy arrays containing maze images
    output_path (str): Path to save the output video
    fps (int): Frames per second
    figsize (tuple): Figure size in inches
    title (str): Main title
    subtitle (str): Subtitle text
    description_lines (list): List of description lines for ending credits
    credits (list): List of credit lines
    dpi (int): Resolution (dots per inch)
    """
    if description_lines is None:
        description_lines = [
            "This visualization demonstrates pathfinding algorithms",
            "finding optimal routes through complex mazes.",
            "Each step represents the algorithm's progression."
        ]
    
    if credits is None:
        credits = [
            "Created with matplotlib",
            f"Date: {datetime.datetime.now().strftime('%Y-%m-%d')}",
            # "© Your Name/Company"
        ]
    
    # Create figure with nice background
    fig = plt.figure(figsize=figsize, facecolor='#1a1a2e')
    
    # Use GridSpec for more control over layout
    gs = GridSpec(1, 1, figure=fig)
    ax = fig.add_subplot(gs[0, 0])
    
    # Set the axes background
    ax.set_facecolor('#0f3460')
    
    # Calculate frames
    n_maze_frames = len(maze_frames)
    n_intro_frames = int(fps * 3)  # 3 seconds intro
    n_outro_frames = int(fps * 6)  # 6 seconds outro
    n_transition_frames = int(fps * 0.5)  # 0.5 seconds transition between maze frames
    
    # Calculate total frames - intro + maze frames with transitions + outro
    total_maze_with_transitions = n_maze_frames + (n_maze_frames - 1) * n_transition_frames
    total_frames = n_intro_frames + total_maze_with_transitions + n_outro_frames
    
    # Track which maze frame we're on
    maze_frame_indices = []
    is_transition = []
    
    # Build frame mapping - which original maze frame to show at each animation frame
    current_idx = 0
    for i in range(n_maze_frames):
        # Add the main frame
        maze_frame_indices.append(i)
        is_transition.append(False)
        current_idx += 1
        
        # Add transition frames except after the last maze frame
        if i < n_maze_frames - 1:
            for _ in range(n_transition_frames):
                maze_frame_indices.append(i)  # Transition from current frame
                is_transition.append(True)
                current_idx += 1
    
    # Function to update the frame for animation
    def update(frame_num):
        ax.clear()
        
        # Remove all ticks and spines
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # INTRO SECTION
        if frame_num < n_intro_frames:
            # Intro animation
            progress = frame_num / n_intro_frames
            
            # Fancy background
            ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, 
                                  color='#1a1a2e', alpha=1.0))
            
            # Draw grid lines for visual effect
            grid_alpha = min(1.0, progress * 2)
            for i in range(1, 20):
                # Horizontal and vertical lines
                ax.axhline(y=i/20, color='#e94560', alpha=grid_alpha*0.2, linestyle='-', linewidth=1)
                ax.axvline(x=i/20, color='#e94560', alpha=grid_alpha*0.2, linestyle='-', linewidth=1)
            
            # Title appears with typing effect
            if progress > 0.2:
                title_progress = min(1.0, (progress - 0.2) / 0.4)
                visible_title = title[:int(len(title) * title_progress)]
                
                title_text = ax.text(0.5, 0.6, visible_title, 
                                   transform=ax.transAxes, fontsize=36,
                                   horizontalalignment='center', color='white',
                                   weight='bold')
                
                # Add fancy text effects
                title_text.set_path_effects([
                    path_effects.withStroke(linewidth=4, foreground='#e94560'),
                    path_effects.withSimplePatchShadow(shadow_rgbFace='black', alpha=0.8)
                ])
            
            # Subtitle fades in after title
            if progress > 0.6:
                subtitle_alpha = min(1.0, (progress - 0.6) / 0.3)
                subtitle_text = ax.text(0.5, 0.45, subtitle, 
                                      transform=ax.transAxes, fontsize=24,
                                      horizontalalignment='center', color='white',
                                      alpha=subtitle_alpha)
                
                subtitle_text.set_path_effects([
                    path_effects.withStroke(linewidth=2, foreground='#16213e')
                ])
                
            # "Loading" animation at the bottom
            if progress > 0.8:
                dots = "." * (1 + int((frame_num % 15) / 5))
                loading_text = ax.text(0.5, 0.3, f"Loading{dots}", 
                                     transform=ax.transAxes, fontsize=18,
                                     horizontalalignment='center', color='#e94560',
                                     alpha=0.8)
                
                # Progress bar
                bar_width = 0.4
                bar_height = 0.02
                bar_x = 0.5 - bar_width/2
                bar_y = 0.25
                
                # Background of progress bar
                ax.add_patch(Rectangle((bar_x, bar_y), bar_width, bar_height, 
                                      transform=ax.transAxes, color='#16213e', alpha=0.8,
                                      linewidth=1, edgecolor='#e94560'))
                
                # Filled portion of progress bar
                fill_width = bar_width * min(1.0, (progress - 0.8) * 5)
                ax.add_patch(Rectangle((bar_x, bar_y), fill_width, bar_height, 
                                     transform=ax.transAxes, color='#e94560', alpha=0.8))
            
        # MAZE FRAMES & TRANSITIONS
        elif frame_num < n_intro_frames + total_maze_with_transitions:
            # Calculate which frame in the sequence we're showing
            sequence_index = frame_num - n_intro_frames
            
            if sequence_index < len(maze_frame_indices):
                maze_idx = maze_frame_indices[sequence_index]
                transition = is_transition[sequence_index]
                
                if transition and maze_idx < n_maze_frames - 1:
                    # We're in a transition between maze_idx and maze_idx+1
                    frame1 = maze_frames[maze_idx]
                    frame2 = maze_frames[maze_idx + 1]
                    
                    # Calculate transition progress
                    local_progress = (sequence_index - maze_idx - (maze_idx * n_transition_frames)) / n_transition_frames
                    
                    # Create crossfade effect
                    if isinstance(frame1, np.ndarray) and isinstance(frame2, np.ndarray):
                        # Ensure both frames are properly formatted
                        if frame1.dtype != np.uint8 and frame1.max() <= 1.0:
                            frame1 = (frame1 * 255).astype(np.uint8)
                        if frame2.dtype != np.uint8 and frame2.max() <= 1.0:
                            frame2 = (frame2 * 255).astype(np.uint8)
                        
                        # Create blended frame
                        if frame1.shape == frame2.shape:
                            blended = cv2.addWeighted(
                                frame1, 1 - local_progress, 
                                frame2, local_progress, 0
                            ) if 'cv2' in globals() else frame1  # Fallback if cv2 not available
                            ax.imshow(blended)
                        else:
                            # If shapes don't match, just show current frame
                            ax.imshow(frame1)
                    else:
                        # If frames aren't numpy arrays, just show current frame
                        ax.imshow(maze_frames[maze_idx])
                else:
                    # Regular frame, not a transition
                    ax.imshow(maze_frames[maze_idx])
                
                
                if "solution" in output_path:    
                    # Add step counter with fancy styling
                    step_text = ax.text(0.02, 0.98, f"Step {maze_idx+1}/{n_maze_frames}", 
                                    transform=ax.transAxes, fontsize=16,
                                    verticalalignment='top', color='white',
                                    bbox=dict(boxstyle='round,pad=0.5', facecolor='#16213e', 
                                            alpha=0.8, edgecolor='#e94560'))
                    # Add title at the bottom
                    bottom_title = ax.text(0.5, 0.03, f"distance: {distance_list[maze_idx]}", 
                                        transform=ax.transAxes, fontsize=18,
                                        horizontalalignment='center', color='white',
                                        bbox=dict(boxstyle='round,pad=0.5', facecolor='#16213e', 
                                                alpha=0.8, edgecolor='#e94560'))
                    
                    # Add effects
                    step_text.set_path_effects([path_effects.withStroke(linewidth=2, foreground='#0f3460')])
                    bottom_title.set_path_effects([path_effects.withStroke(linewidth=2, foreground='#0f3460')])
            
        # OUTRO SECTION
        else:
            # Outro animation with credits
            outro_progress = (frame_num - (n_intro_frames + total_maze_with_transitions)) / n_outro_frames
            
            # Dark gradient background
            ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, 
                                 color='#1a1a2e', alpha=1.0))
            
            # Add some decorative elements
            # Fancy border
            border_alpha = min(1.0, outro_progress * 2)
            border = FancyBboxPatch((0.1, 0.1), 0.8, 0.8, 
                                   boxstyle=f"round,pad=0.02,rounding_size={0.05}", 
                                   transform=ax.transAxes, 
                                   linewidth=3, alpha=border_alpha, 
                                   edgecolor='#e94560', facecolor='none')
            ax.add_patch(border)
            
            # Show title again
            if outro_progress > 0.1:
                title_text = ax.text(0.5, 0.85, "Question", 
                                   transform=ax.transAxes, fontsize=28,
                                   horizontalalignment='center', color='white',
                                   weight='bold', alpha=min(1.0, (outro_progress - 0.1) * 3))
                
                # Add fancy underline
                underline_width = min(0.3, (outro_progress - 0.1) * 0.6)
                ax.plot([0.5 - underline_width/2, 0.5 + underline_width/2], [0.83, 0.83], 
                       transform=ax.transAxes, color='#e94560', linewidth=3, 
                       alpha=min(1.0, (outro_progress - 0.1) * 3))
                
                title_text.set_path_effects([
                    path_effects.withStroke(linewidth=3, foreground='#0f3460'),
                ])
            
            # Show description lines with staggered animation
            if outro_progress > 0.3:
                for i, line in enumerate(description_lines):
                    # Staggered appearance
                    line_progress = min(1.0, (outro_progress - 0.3 - i*0.1) * 3)
                    
                    if line_progress > 0:
                        # Different animation effects based on index
                        if i % 3 == 0:
                            # Fade in
                            alpha = line_progress
                            y_pos = 0.7 - i * 0.07
                            x_pos = 0.5
                        elif i % 3 == 1:
                            # Slide in from left
                            alpha = min(1.0, line_progress * 1.5)
                            y_pos = 0.7 - i * 0.07
                            x_pos = 0.5 - 0.2 * (1 - min(1.0, line_progress * 2))
                        else:
                            # Slide in from right
                            alpha = min(1.0, line_progress * 1.5)
                            y_pos = 0.7 - i * 0.07
                            x_pos = 0.5 + 0.2 * (1 - min(1.0, line_progress * 2))
                        
                        # Draw line with fancy styling
                        desc_text = ax.text(x_pos, y_pos, line, 
                                          transform=ax.transAxes, fontsize=16,
                                          horizontalalignment='center', color='white',
                                          alpha=alpha)
                        
                        # Add subtle text effect
                        desc_text.set_path_effects([
                            path_effects.withStroke(linewidth=2, foreground='#0f3460')
                        ])
            
            # Show credits at the bottom
            if outro_progress > 0.7:
                for i, credit in enumerate(credits):
                    # Credits fade in and slide up
                    credit_progress = min(1.0, (outro_progress - 0.7 - i*0.05) * 4)
                    
                    if credit_progress > 0:
                        # Animation effect: slide up while fading in
                        alpha = min(1.0, credit_progress * 1.5)
                        target_y = 0.2 - i * 0.05
                        current_y = target_y - 0.1 * (1 - min(1.0, credit_progress * 2))
                        
                        # Style based on index
                        if i == 0:
                            # Main credit
                            fontsize = 14
                            fontweight = 'bold'
                            color = '#e94560'
                        else:
                            # Secondary credits
                            fontsize = 12
                            fontweight = 'normal'
                            color = 'white'
                        
                        # Draw credit
                        credit_text = ax.text(0.5, current_y, credit, 
                                            transform=ax.transAxes, fontsize=fontsize,
                                            horizontalalignment='center', color=color,
                                            alpha=alpha, weight=fontweight)
                        
                        # Add subtle shadow
                        credit_text.set_path_effects([
                            path_effects.withSimplePatchShadow(shadow_rgbFace='black', alpha=0.6)
                        ])
        
        return [ax]
    
    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=total_frames)
    
    # Save animation
    try:
        # Try FFmpeg writer with high quality
        writer = animation.FFMpegWriter(fps=fps*2, bitrate=5000)
        ani.save(output_path, writer=writer, dpi=dpi)
        print(f"Animation saved to {output_path}")
    except Exception as e:
        print(f"Error saving with FFmpeg: {e}")
        try:
            # Try with PillowWriter (GIF) as fallback
            gif_path = output_path.replace('.mp4', '.gif')
            writer = animation.PillowWriter(fps=fps)
            ani.save(gif_path, writer=writer, dpi=dpi)
            print(f"Saved as GIF instead: {gif_path}")
        except Exception as e2:
            print(f"Error saving with Pillow: {e2}")
            print("Consider installing FFmpeg or trying a different approach")
    
    plt.close(fig)


def chunk_text(text, max_width=40, min_width=20):
    """
    Split text into chunks with roughly similar width, breaking at whitespace.
    
    Parameters:
    text (str): The input text to chunk
    max_width (int): Maximum preferred width for each line
    min_width (int): Minimum preferred width for each line (except last line)
    
    Returns:
    list: A list of text chunks
    """
    words = text.split()
    chunks = []
    current_chunk = []
    current_width = 0
    
    for word in words:
        # Calculate width if we add this word (plus a space)
        word_width = len(word)
        new_width = current_width + word_width + (1 if current_width > 0 else 0)
        
        if new_width <= max_width:
            # Word fits in current chunk, add it
            current_chunk.append(word)
            current_width = new_width
        else:
            # Word doesn't fit, start a new chunk
            
            # If current chunk is too short, and we can afford to go over max_width,
            # and this isn't the last word, put the word in the current chunk
            if current_width < min_width and len(current_chunk) > 0 and word != words[-1]:
                current_chunk.append(word)
                chunks.append(' '.join(current_chunk))
            else:
                # Otherwise, start a new chunk with this word
                if current_chunk:  # Only add chunk if it's not empty
                    chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_width = word_width
    
    # Add the last chunk if there's anything left
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run Frozen Lake analysis with configurable parameters.')
    parser.add_argument('--size', type=int, default=10, help='Size of the random map (default: 10)')
    parser.add_argument('--n_neighbors', type=int, default=4, choices=[4, 8], help='Number of neighbors for connectivity: 4 or 8 (default: 4)')
    parser.add_argument('--question_name', type=str, default='count', choices=['count', 'connected_area', 'connected_count'], help='Name of the question for output files (default: script filename)')
    args = parser.parse_args()

    # Generate random map and setup environment
    random_map = generate_random_map(size=args.size)
    env = gym.make("FrozenLake-v1", desc=random_map, render_mode="rgb_array", is_slippery=False)
    foggy_env = FoggyFrozenLake(env, visibility_range=3)
    frames, path = visualize_exploration(foggy_env, max_steps=200)
    

    n_neighbors = args.n_neighbors
    hole_count, hole_positions = count_holes(env)
    connected_components = get_connected_holes(env, n_neighbors)

    # Create the question text
    if args.question_name == 'count':
        question_text = (
            f"What is the number of holes on the frozen lake? "
        )
        correct_answer = hole_count
    elif args.question_name == 'connected_area': 
        question_text = (
            f"What is the size—i.e., the number of cells—of the largest connected group of holes, "
            f"where connectivity is defined by {n_neighbors}-neighbour adjacency? "
            f"(diagonal connection {'' if n_neighbors == 8 else 'does not'} count)"
        )
        # Find the largest connected component
        largest_component_size = 0
        largest_component_index = -1
        
        for i, component in enumerate(connected_components):
            if len(component) > largest_component_size:
                largest_component_size = len(component)
                largest_component_index = i

        correct_answer = largest_component_size
    else:
        question_text = (
            f"What is the number of connected group of holes, "
            f"where connectivity is defined by {n_neighbors}-neighbour adjacency? "
            f"(diagonal connection {'' if n_neighbors == 8 else 'does not'} count)"
        )
        correct_answer = len(connected_components)

    print(question_text)
    print(f"Answer: {correct_answer}")


    # Determine question name if not provided
    if args.question_name is None:
        script_path = __file__
        script_filename = os.path.basename(script_path)
        question_name = script_filename.split('.')[0]
    else:
        question_name = f"frozen_lake_holes_{args.question_name}_n{n_neighbors}_sz{args.size}"

    
    question_dir = Path('questions')
    question_dir.mkdir(exist_ok=True)
    output_video = f"questions/{question_name}.mp4"

    title = "Foggy Frozen Lake"
    subtitle = question_name
    description_lines = chunk_text(question_text)
    credits = ["Please return a single number (e.g. 3)", 
               "Nothing preceding or following it."]

    create_professional_plt_animation(
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

    solution_dir = Path('solutions')
    solution_dir.mkdir(exist_ok=True)
    with open(f"solutions/{question_name}.txt", "w") as f:
        f.write(f"{correct_answer}")
    Image.fromarray(env.render()).save(f"solutions/{question_name}.png")

    question_text = f"{question_text}\n{credits[0]}. {credits[1]}"
    question_text_dir = Path('question_text')
    question_text_dir.mkdir(exist_ok=True)
    with open(f"question_text/{question_name}.txt", "w") as f:
        f.write(f"{question_text}")
