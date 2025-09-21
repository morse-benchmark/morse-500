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

# NUM_SHAPES [2-8] range for difficulty adjustment. 
# NUM_SHAPES=2 python duration2d_modified.py

class duration2D(ThreeDScene):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Set random seed for reproducibility
        self.seed = random.randint(1000, 9999)
        random.seed(self.seed)
        
        # Parameters - difficulty controls number of shapes
        self.num_shapes = int(os.getenv("NUM_SHAPES", 5))

    def construct(self):
        # Camera setup
        self.set_camera_orientation(phi=75 * DEGREES, theta=45 * DEGREES)
        
        # Constrain count to reasonable bounds
        count = max(2, min(self.num_shapes, 8))
        
        # Full library of valid shapes with names and matching colors
        all_shapes_with_names = [
            (Circle(radius=1), "circle"),
            (Square(side_length=2), "square"),
            (Triangle().scale(1.5), "triangle"),
            (RegularPolygon(5).scale(1.2), "pentagon"),
            (Square(side_length=2).rotate(PI/4), "diamond"),
            (RegularPolygon(6).scale(1.1), "hexagon"),
            (RegularPolygon(8).scale(1.1), "octagon"),
            (Star(5).scale(1.2), "star")
        ]
        
        all_colors = [
            WHITE, DARK_BROWN, RED, GREEN, PINK,
            BLUE, YELLOW, PURPLE, ORANGE
        ]
        
        # Predefined pool of positions
        positions_pool = [
            LEFT * 3 + DOWN, RIGHT * 4 + DOWN,
            LEFT * 2 + UP, RIGHT * 2 + UP,
            ORIGIN, LEFT * 4 + UP,
            RIGHT * 4 + UP, RIGHT * 2 + DOWN,
            LEFT * 2 + DOWN, DOWN * 3,
            UP * 3, LEFT * 4
        ]

        # Randomly pick which shapes, colors, and positions to use
        chosen_shape_indices = random.sample(range(len(all_shapes_with_names)), count)
        chosen_color_indices = random.sample(range(len(all_colors)), count)
        chosen_positions = random.sample(positions_pool, count)

        # Generate random durations for each shape
        durations = [round(random.uniform(0.5, 3.0), 1) for _ in range(count)]
        
        # Track shapes and their durations for the answer
        shape_duration_pairs = []

        # Show shapes one by one
        for idx in range(count):
            shape_original, shape_name = all_shapes_with_names[chosen_shape_indices[idx]]
            shape = shape_original.copy()  # Make a copy to avoid reference issues
            color = all_colors[chosen_color_indices[idx]]
            duration = durations[idx]
            position = chosen_positions[idx]
            
            # Apply color and position
            shape.set_fill(color, opacity=0.6)
            shape.set_stroke(color)
            shape.move_to(position)
            
            # Store for answer
            shape_duration_pairs.append((shape_name, duration))
            
            # Animate the shape creation
            self.play(Create(shape), run_time=duration)
        
        # Wait a moment before showing question
        self.wait(0.5)

        # Create question text
        question_lines = [
            "List the duration of each of the shapes from the beginning",
            "(from drawing the object to disappearing).",
            "",
            "Answer to 1 decimal point and list them with comma separated values:",
            "e.g., 3.2s, 1.5s, 1.0s"
        ]
        
        # Create question text objects
        question_texts = []
        line_height = 0.6
        start_y = 2.5
        
        for i, line in enumerate(question_lines):
            if line:  # Skip empty lines for spacing
                text = Text(line, font_size=24, weight=BOLD if i == 0 else NORMAL)
                text.move_to(UP * (start_y - i * line_height))
                question_texts.append(text)
        
        # Add all question text as fixed elements
        self.add_fixed_in_frame_mobjects(*question_texts)
        
        # Animate question appearance
        self.play(*[FadeIn(text) for text in question_texts], run_time=1.0)
        self.wait(1.0)

        # Show instruction
        instruction_text = Text(
            "List the durations in the order the shapes appeared.",
            font_size=22,
            color=YELLOW
        ).move_to(DOWN * 2.8)
        
        self.add_fixed_in_frame_mobjects(instruction_text)
        self.play(FadeIn(instruction_text, shift=UP*0.3), run_time=0.8)
        self.wait(3)
        
        # Generate answer: durations in order of appearance
        duration_strings = [f"{duration}s" for _, duration in shape_duration_pairs]
        answer_string = ", ".join(duration_strings)
        
        # Save solution and question text
        with open(f"solutions/duration2d_n{self.num_shapes}_seed{self.seed}.txt", "w") as f:
            f.write(answer_string)
        
        question_text_content = (
            "List the duration of each of the shapes from the beginning (from drawing the object to disappearing).\n"
            "Answer to 1 decimal point and list them with comma separated values: e.g., 3.2s, 1.5s, 1.0s\n"
            "List the durations in the order the shapes appeared."
        )
        with open(f"question_text/duration2d_n{self.num_shapes}_seed{self.seed}.txt", "w") as f:
            f.write(question_text_content)


if __name__ == "__main__":
    # Generate the duration 2D video
    scene = duration2D()
    scene.render()

    # Move the output file with descriptive name
    output = Path("manim_output/videos/1080p30/duration2D.mp4")
    if output.exists():
        filename = f"duration2d_n{scene.num_shapes}_seed{scene.seed}.mp4"
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