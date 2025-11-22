from manim import *
import random
import math
import os
import shutil
from pathlib import Path

# Setup directories
Path("questions").mkdir(exist_ok=True)
Path("solutions").mkdir(exist_ok=True)
Path("question_text").mkdir(exist_ok=True)
Path("reasoning_traces").mkdir(exist_ok=True)

config.media_dir = "manim_output"
config.verbosity = "WARNING"
config.pixel_height = 1080
config.pixel_width = 1920
config.frame_rate = 30
config.preview = False

# NUM_TRANSFORMS=[3-8] python3 color_change.py
class ColorChange2(Scene):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Set random seed for reproducibility
        self.seed = random.randint(1000, 9999)
        random.seed(self.seed)

        # Parameters - difficulty controls number of transforms
        self.num_transforms = int(os.getenv("NUM_TRANSFORMS", 4))

        # Initialize reasoning trace
        self.reasoning_trace = []
        self.reasoning_trace.append(f"Problem: Color change sequence")
        self.reasoning_trace.append(f"Number of transforms: {self.num_transforms}")
        self.reasoning_trace.append(f"Random Seed: {self.seed}")
        self.reasoning_trace.append("")
        self.reasoning_trace.append("=== CHRONOLOGICAL SCENE DESCRIPTION ===")
        self.reasoning_trace.append("")

    def construct(self):
        # Constrain count to reasonable bounds
        count = max(2, min(self.num_transforms, 8))
        
        # Full library of valid shapes with names and matching colors
        all_shapes_with_names = [
            (Circle(), "circle"),
            (Square(), "square"),
            (Triangle(), "triangle"),
            (RegularPolygon(5), "pentagon"),
            (Square().rotate(PI/4), "diamond"),
            (RegularPolygon(6), "hexagon"),
            (RegularPolygon(8), "octagon"),
            (Star(5), "star"),
            (Ellipse(width=2, height=1), "oval")
        ]
        all_colors = [
            YELLOW, WHITE, BLUE, GREEN, RED,
            PURPLE, ORANGE, TEAL, PINK
        ]
        color_names = [
            "yellow", "white", "blue", "green", "red",
            "purple", "orange", "teal", "pink"
        ]

        # Predefined pool of positions
        positions_pool = [
            LEFT * 2, RIGHT * 2, UP * 2, DOWN * 2,
            LEFT * 3, RIGHT * 3, UP * 1.5, DOWN * 1.5,
            LEFT * 1, RIGHT * 1
        ]

        # Randomly select shapes, colors, and positions for the transform chain
        chosen_shape_indices = random.sample(range(len(all_shapes_with_names)), count)
        chosen_color_indices = random.sample(range(len(all_colors)), count)
        chosen_positions = random.sample(positions_pool, count)

        # Create the initial shape
        initial_shape, initial_shape_name = all_shapes_with_names[chosen_shape_indices[0]]
        initial_shape = initial_shape.copy()  # Make a copy to avoid reference issues
        initial_color = all_colors[chosen_color_indices[0]]
        initial_shape.set_fill(initial_color, opacity=0.7)
        initial_shape.set_stroke(initial_color)
        initial_shape.move_to(chosen_positions[0])

        # Store the sequence for answer calculation
        shape_sequence = [(initial_shape_name, color_names[chosen_color_indices[0]])]

        # Helper function to describe position
        def describe_position(pos):
            if np.allclose(pos, LEFT * 2):
                return "the left side"
            elif np.allclose(pos, RIGHT * 2):
                return "the right side"
            elif np.allclose(pos, UP * 2):
                return "the upper area"
            elif np.allclose(pos, DOWN * 2):
                return "the lower area"
            elif np.allclose(pos, LEFT * 3):
                return "the far left"
            elif np.allclose(pos, RIGHT * 3):
                return "the far right"
            elif np.allclose(pos, UP * 1.5):
                return "the upper-middle area"
            elif np.allclose(pos, DOWN * 1.5):
                return "the lower-middle area"
            elif np.allclose(pos, LEFT * 1):
                return "the center-left"
            elif np.allclose(pos, RIGHT * 1):
                return "the center-right"
            else:
                return "the screen"

        # Add initial scene description
        position_desc = describe_position(chosen_positions[0])
        self.reasoning_trace.append(f"At the start, a {color_names[chosen_color_indices[0]]} {initial_shape_name} appears on screen at {position_desc}.")

        self.play(Create(initial_shape))
        self.wait(0.5)

        # Transform through the remaining shapes
        cumulative_time = 0.5  # Initial wait time
        for i in range(1, count):
            next_shape, next_shape_name = all_shapes_with_names[chosen_shape_indices[i]]
            next_shape = next_shape.copy()  # Make a copy to avoid reference issues
            next_color = all_colors[chosen_color_indices[i]]
            next_shape.set_fill(next_color, opacity=0.7)
            next_shape.set_stroke(next_color)
            next_shape.move_to(chosen_positions[i])

            # Store in sequence
            shape_sequence.append((next_shape_name, color_names[chosen_color_indices[i]]))

            # Add chronological transformation description
            # Estimate timing: transform animation is typically 1 second, plus 0.5 second wait
            cumulative_time += 1.0
            position_desc = describe_position(chosen_positions[i])
            self.reasoning_trace.append(f"At approximately {cumulative_time:.1f}s, the shape transforms into a {color_names[chosen_color_indices[i]]} {next_shape_name} at {position_desc}.")

            self.play(Transform(initial_shape, next_shape))
            self.wait(0.5)
            cumulative_time += 0.5
        
        # Fade out the final shape
        self.play(FadeOut(initial_shape))
        self.wait(0.5)

        # Generate question and answer
        # Ask about the color of the shape that appeared N turns before the final shape
        turns_back = random.randint(2, min(count-1, 4))  # At least 2 turns back, max 4
        target_index = count - 1 - turns_back  # Index of the target shape
        answer = shape_sequence[target_index][1]  # Color name

        # Add reasoning section
        self.reasoning_trace.append("")
        self.reasoning_trace.append("=== REASONING THROUGH THE SEQUENCE ===")
        self.reasoning_trace.append("")
        self.reasoning_trace.append(f"After observing the complete sequence of {count} transformations, we need to identify which shape appeared {turns_back} turns before the final {shape_sequence[-1][0]}.")
        self.reasoning_trace.append("")
        self.reasoning_trace.append("The complete sequence in order:")
        for idx, (shape_name, color_name) in enumerate(shape_sequence, 1):
            self.reasoning_trace.append(f"  {idx}. {color_name} {shape_name}")
        self.reasoning_trace.append("")
        self.reasoning_trace.append(f"The final shape is the {shape_sequence[-1][1]} {shape_sequence[-1][0]} at position {count}.")
        self.reasoning_trace.append(f"Counting back {turns_back} turns from the end (position {count}), we need to find the shape at position {target_index + 1}.")
        self.reasoning_trace.append(f"Position {count} - {turns_back} turns = Position {target_index + 1}")
        self.reasoning_trace.append("")
        self.reasoning_trace.append(f"At position {target_index + 1} in the sequence, we find a {shape_sequence[target_index][1]} {shape_sequence[target_index][0]}.")
        self.reasoning_trace.append(f"Therefore, the color of the shape that appeared {turns_back} turns before the {shape_sequence[-1][0]} is: {answer}")
        self.reasoning_trace.append("")
        self.reasoning_trace.append("=== TECHNICAL DETAILS ===")
        self.reasoning_trace.append("")
        self.reasoning_trace.append("Question generation:")
        self.reasoning_trace.append(f"  Final shape: {shape_sequence[-1][0]}")
        self.reasoning_trace.append(f"  Turns back: {turns_back}")
        self.reasoning_trace.append(f"  Target index: {target_index}")
        self.reasoning_trace.append(f"  Target shape: {shape_sequence[target_index][0]}")
        self.reasoning_trace.append(f"  Answer (color): {answer}")
        
        # Create the question
        final_shape_name = shape_sequence[-1][0]
        question_lines = [
            f"What was the color of the shape that appeared {turns_back} turns before the {final_shape_name}?",
            "",
            "Output in lower case"
        ]
        
        # Create question text objects
        question_texts = []
        line_height = 0.8
        start_y = 0.5
        
        for i, line in enumerate(question_lines):
            if line:  # Skip empty lines for spacing
                text = Text(line, font_size=28, weight=BOLD if i == 0 else NORMAL)
                text.move_to(UP * (start_y - i * line_height))
                question_texts.append(text)
        
        # Show question
        self.play(*[Write(text) for text in question_texts])
        self.wait(3)
        
        # Save solution and question text
        with open(f"solutions/colorchange2_n{self.num_transforms}_seed{self.seed}.txt", "w") as f:
            f.write(answer)
        
        question_text_content = (
            f"What was the color of the shape that appeared {turns_back} turns before the {final_shape_name}?\n"
            "Output in lower case"
        )
        with open(f"question_text/colorchange2_n{self.num_transforms}_seed{self.seed}.txt", "w") as f:
            f.write(question_text_content)

        # Save detailed reasoning trace
        self.reasoning_trace.append("")
        self.reasoning_trace.append(f"Final Answer: {answer}")
        with open(f"reasoning_traces/colorchange2_n{self.num_transforms}_seed{self.seed}.txt", "w") as f:
            f.write("\n".join(self.reasoning_trace))


if __name__ == "__main__":
    # Generate the color change video
    scene = ColorChange2()
    scene.render()

    # Move the output file with descriptive name
    output = Path("manim_output/videos/1080p30/ColorChange2.mp4")
    if output.exists():
        filename = f"colorchange2_n{scene.num_transforms}_seed{scene.seed}.mp4"
        shutil.move(str(output), f"questions/{filename}")
    else:
        # Debug: Print what files actually exist
        videos_dir = Path("manim_output/videos")
        if videos_dir.exists():
            print(f"Available folders in videos/: {list(videos_dir.iterdir())}")
            for folder in videos_dir.iterdir():
                if folder.is_dir():
                    subfolder = folder / "1080p30"
                    if subfolder.exists():
                        print(f"Files in {subfolder}: {list(subfolder.iterdir())}")
        else:
            print("manim_output/videos directory doesn't exist")

    # Final cleanup
    if os.path.exists("manim_output"):
        shutil.rmtree("manim_output")