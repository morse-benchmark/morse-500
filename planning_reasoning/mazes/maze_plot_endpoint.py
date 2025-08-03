# !pip install maze-dataset
# sudo apt-get install ffmpeg
from maze_dataset.generation import LatticeMazeGenerators
from maze_dataset.maze import SolvedMaze, TargetedLatticeMaze
from maze_dataset.plotting import MazePlot

import os
from pathlib import Path
import random
import numpy as np
import cv2

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patheffects as path_effects
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.gridspec import GridSpec
import matplotlib.transforms as transforms
import matplotlib.colors as mcolors
import datetime



def select_random_points(maze, N=0):
    """
    Select random points from the maze without repetition.
    
    Args:
        maze: The maze object with generation_meta containing grid_shape
        N: Number of additional points to select beyond start and end
    
    Returns:
        A list of points: [start, end, point1, point2, ..., pointN]
    """
    # Get the grid shape from the maze
    grid_shape = maze.generation_meta['grid_shape']
    rows, cols = grid_shape[0], grid_shape[1]
    
    # Create a list of all possible coordinates
    all_coordinates = [(r, c) for r in range(rows) for c in range(cols)]
    
    # Ensure we're not requesting more points than available
    total_points = 2 + N  # start + end + N additional points
    if total_points > rows * cols:
        raise ValueError(f"Cannot select {total_points} unique points from a {rows}x{cols} maze")
    
    # Randomly sample from all coordinates without replacement
    selected_points = random.sample(all_coordinates, total_points)
    
    # First point is start, second is end, rest are additional points
    start = selected_points[0]
    end = selected_points[1]
    additional_points = selected_points[2:] if N > 0 else []
    
    # Return as a list with start and end followed by additional points
    return start, end, additional_points


def solution_to_directions(solution):
    directions = []
    
    for i in range(1, len(solution)):
        current = solution[i-1]
        next_point = solution[i]
        
        # Calculate the difference between current and next position
        dy = next_point[0] - current[0]
        dx = next_point[1] - current[1]
        
        # Determine direction based on the difference
        if dy == 1 and dx == 0:
            directions.append("Go down")
        elif dy == -1 and dx == 0:
            directions.append("Go up")
        elif dy == 0 and dx == 1:
            directions.append("Go right")
        elif dy == 0 and dx == -1:
            directions.append("Go left")
        else:
            # Handle diagonal movements or jumps if they exist
            directions.append(f"move from {current} to {next_point}")
    
    return directions


def solution_to_grouped_directions(solution):
    if len(solution) <= 1:
        return []
    
    directions = []
    current_direction = None
    count = 0
    
    for i in range(1, len(solution)):
        current = solution[i-1]
        next_point = solution[i]
        
        # Calculate the difference between current and next position
        dy = next_point[0] - current[0]
        dx = next_point[1] - current[1]
        
        # Determine direction based on the difference
        icon_lists = [
            # ['👈', '👇', '👉', '👆'],
            # ['⏪', '⏬', '⏩', '⏫'],
            ['⬅️', '⬇️', '➡️', '⬆️'],
            # ['⬅', '⬇', '⮕', '⬆']
        ]
        icons = random.choice(icon_lists)

        if dy == 1 and dx == 0:
            direction = icons[1]
        elif dy == -1 and dx == 0:
            direction = icons[3]
        elif dy == 0 and dx == 1:
            direction = icons[2]
        elif dy == 0 and dx == -1:
            direction = icons[0]
        else:
            # Handle diagonal movements or jumps if they exist
            direction = f"move from {current} to {next_point}"
        
        # Check if this is the same direction as before
        if direction == current_direction:
            count += 1
        else:
            # Add the previous direction to our list if there was one
            if current_direction is not None:
                if count == 1:
                    directions.append(f"{current_direction}")
                else:
                    directions.append(f"{current_direction} {count} steps")
            
            # Start counting the new direction
            current_direction = direction
            count = 1
    
    # Don't forget to add the last direction
    if current_direction is not None:
        if count == 1:
            directions.append(f"{current_direction}")
        else:
            directions.append(f"{current_direction} {count} steps")
    
    return directions


def place_letter_at_pos(maze, letter, pos, ax):
    # letter: "A", "B", ...
    # pos: (row, col), 0 indexed

    # Get the axis limits
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    
    # Get the grid shape
    grid_shape = maze.generation_meta['grid_shape']
    rows, cols = grid_shape[0], grid_shape[1]
    
    # Calculate the cell size in plot coordinates
    cell_width = (x_max - x_min) / cols
    cell_height = (y_max - y_min) / rows

    # Calculate the center of the end cell in plot coordinates
    # Note: Depending on how your maze is plotted, you might need to flip the y-coordinate
    # If (0,0) is the top-left in the maze but bottom-left in the plot:
    end_x = x_min + (pos[1] + 0.5) * cell_width
    end_y = y_min + (rows - pos[0] - 0.5) * cell_height  # Flipped y
    
    # If (0,0) is the top-left in both the maze and the plot:
    # end_x = x_min + (end[1] + 0.5) * cell_width
    # end_y = y_min + (end[0] + 0.5) * cell_height
    
    # Add the letter "A" at the calculated position
    ax.text(end_x, end_y, letter, fontsize=14, color='white', 
            fontweight='bold', ha='center', va='center',
            bbox=dict(boxstyle="circle", fc="blue", ec="blue", alpha=0.8))


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
        writer = animation.FFMpegWriter(fps=fps, bitrate=5000)
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
    parser.add_argument('--question_name', type=str, default='count', choices=['count', 'min_length', 'agent_steps'], help='Name of the question for output files (default: script filename)')
    args = parser.parse_args()
    
    n_row = n_col = args.size
    print(f"Maze size {n_row}x{n_col}")
    sample_lattice_maze = LatticeMazeGenerators.gen_dfs(
        grid_shape=(n_row, n_col),
        lattice_dim=2,
        accessible_cells=None,
        max_tree_depth=None,
        start_coord=None,
    )

    # Select random start and end points
    # start, end = select_random_points(sample_lattice_maze)
    start, end, additional_points = select_random_points(sample_lattice_maze, 5)
    print(f"Start: {start}, End: {end}, Others: {additional_points}")
    letter_list = [chr(65 + i) for i in range(len(additional_points)+1)]
    random.shuffle(letter_list)
    correct_answer = letter_list[0]


    tgt_maze = TargetedLatticeMaze.from_lattice_maze(
        sample_lattice_maze,
        start_pos=start,
        end_pos=end,
    )

    solved_maze: SolvedMaze = SolvedMaze.from_targeted_lattice_maze(tgt_maze)
    solution = solved_maze.solution

    # Convert to directions
    # directions = solution_to_directions(solution)
    directions = solution_to_grouped_directions(solution)
    print("\nQuestion\n")
    instructions = ", ".join(directions)
    question_text = f"Starting at the green square, follow these steps: {instructions}. Where do you end at?"
    print(question_text)


    maze_frames = []
    for idx in range(len(letter_list)):
        # plot the question
        mp = MazePlot(sample_lattice_maze)
        mp.mark_coords([start], color="green", marker="s")
        mp.plot()
        
        # Get the current axis and turn it off
        ax = plt.gca()
        
        ax.set_axis_off()
        if idx == 0:
            place_letter_at_pos(sample_lattice_maze, correct_answer, end, ax)
        else:
            c = letter_list[idx]
            place_letter_at_pos(sample_lattice_maze, c, additional_points[idx-1], ax)
        
        # After your plotting code (mp.plot(), setting axis off, etc.)
        fig = plt.gcf()  # Get current figure
        
        # Make sure matplotlib is using a high-quality renderer
        plt.gcf().set_dpi(300)
        
        # Draw the figure to make sure it's rendered
        fig.canvas.draw()
        
        # Get the RGBA buffer from the figure
        w, h = fig.canvas.get_width_height()
        buffer = fig.canvas.buffer_rgba()
        image_array = np.asarray(buffer)
        
        # Convert RGBA to RGB if needed
        rgb_array = image_array[:, :, :3]
        # rgb_array = image_array[:, :, :4]
        print(f"rgb_array: {rgb_array.shape}")
        # Image.fromarray(rgb_array)
        maze_frames.append(rgb_array)


    # Example usage:
    script_path = __file__
    script_filename = os.path.basename(script_path)
    print(f"script_filename: {script_filename}")
    question_name = script_filename.split('.')[0]+f'_sz{args.size}'
    question_dir = Path('questions')
    question_dir.mkdir(exist_ok=True)
    output_video = f"questions/{question_name}.mp4"

    title = "Maze Runner"
    subtitle = "Planning Reasoning Test"
    # question_text = "By how many steps does the maximum shortest-path distance exceed the minimum shortest-path distance observed in the animation?"
    # instruction_text = "Return a single letter (e.g. A); nothing preceding or following it."
    description_lines = chunk_text(question_text)
    credits = ["Please return a single letter (e.g. A)", 
               "Nothing preceding or following it."]

    create_professional_plt_animation(
        maze_frames,
        output_video,
        fps=5,
        figsize=(12, 12),
        title=title,
        subtitle=subtitle,
        description_lines=description_lines,
        credits=credits,
        dpi=200
    )

    # output_video = f"{question_short_name}_solution.mp4"
    # create_professional_plt_animation(
    #     maze_frames_solution,
    #     output_video,
    #     distance_list = distance_list,
    #     fps=5,
    #     figsize=(10, 10),
    #     title=title,
    #     subtitle=subtitle,
    #     description_lines=description_lines,
    #     credits=credits,
    #     dpi=200
    # )

    solution_dir = Path('solutions')
    solution_dir.mkdir(exist_ok=True)
    with open(f"solutions/{question_name}.txt", "w") as f:
        f.write(f"{correct_answer}")


    question_text = f"{question_text}\n{credits[0]}. {credits[1]}"
    question_text_dir = Path('question_text')
    question_text_dir.mkdir(exist_ok=True)
    with open(question_text_dir/f"{question_name}.txt", "w") as f:
        f.write(f"{question_text}")