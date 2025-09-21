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

config.media_dir = "manim_output"
config.verbosity = "WARNING"
config.pixel_height = 1080
config.pixel_width = 1920
config.frame_rate = 30
config.preview = False

# NUM_TRANSFORMS=[2-8] python3 color_change.py
class ColorChange2(Scene):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Set random seed for reproducibility
        self.seed = random.randint(1000, 9999)
        random.seed(self.seed)
        
        # Parameters - difficulty controls number of transforms
        self.num_transforms = int(os.getenv("NUM_TRANSFORMS", 4))

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
        
        self.play(Create(initial_shape))
        self.wait(0.5)

        # Store the sequence for answer calculation
        shape_sequence = [(initial_shape_name, color_names[chosen_color_indices[0]])]

        # Transform through the remaining shapes
        for i in range(1, count):
            next_shape, next_shape_name = all_shapes_with_names[chosen_shape_indices[i]]
            next_shape = next_shape.copy()  # Make a copy to avoid reference issues
            next_color = all_colors[chosen_color_indices[i]]
            next_shape.set_fill(next_color, opacity=0.7)
            next_shape.set_stroke(next_color)
            next_shape.move_to(chosen_positions[i])
            
            # Store in sequence
            shape_sequence.append((next_shape_name, color_names[chosen_color_indices[i]]))
            
            self.play(Transform(initial_shape, next_shape))
            self.wait(0.5)
        
        # Fade out the final shape
        self.play(FadeOut(initial_shape))
        self.wait(0.5)

        # Generate question and answer
        # Ask about the color of the shape that appeared N turns before the final shape
        turns_back = random.randint(2, min(count-1, 4))  # At least 2 turns back, max 4
        target_index = count - 1 - turns_back  # Index of the target shape
        answer = shape_sequence[target_index][1]  # Color name
        
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