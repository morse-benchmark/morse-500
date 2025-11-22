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

        # Initialize reasoning trace
        self.reasoning_trace = []
        self.reasoning_trace.append("=== CHRONOLOGICAL REASONING TRACE ===")
        self.reasoning_trace.append("")
        self.reasoning_trace.append("INITIAL SETUP:")
        self.reasoning_trace.append(f"- This video tests 2D shape duration tracking ability")
        self.reasoning_trace.append(f"- The scene will display {self.num_shapes} different shapes, one at a time")
        self.reasoning_trace.append(f"- Each shape will be drawn for a specific duration")
        self.reasoning_trace.append(f"- Random seed: {self.seed}")
        self.reasoning_trace.append("")

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

        # Add scene beginning to reasoning trace
        self.reasoning_trace.append("SCENE BEGINS:")
        self.reasoning_trace.append("The video starts with an empty scene. The camera is positioned at an angle.")
        self.reasoning_trace.append("")

        # Show shapes one by one
        cumulative_time = 0.0
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

            # Add chronological reasoning for this shape
            self.reasoning_trace.append(f"EVENT {idx+1} (Time: {cumulative_time:.1f}s - {cumulative_time + duration:.1f}s):")
            self.reasoning_trace.append(f"A {shape_name} shape appears on the screen. It is being drawn/created.")
            self.reasoning_trace.append(f"The {shape_name} is colored and positioned in the scene.")
            self.reasoning_trace.append(f"I observe that the {shape_name} takes time to fully appear as it's being drawn.")
            self.reasoning_trace.append(f"The drawing animation for this {shape_name} completes.")
            self.reasoning_trace.append(f"Duration observed: {duration:.1f} seconds")
            self.reasoning_trace.append("")

            cumulative_time += duration

            # Animate the shape creation
            self.play(Create(shape), run_time=duration)
        
        # Wait a moment before showing question
        self.clear()
        self.wait(0.5)

        # Add reasoning about end of animations
        self.reasoning_trace.append(f"ANIMATIONS COMPLETE (Time: {cumulative_time:.1f}s):")
        self.reasoning_trace.append("All shapes have been drawn. The scene clears and transitions to the question.")
        self.reasoning_trace.append("")

        # Create question text
        question_lines = [
            "List the duration of each of the shapes from the beginning",
            "(from the start of drawing).",
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
            "List the duration of each of the shapes from the beginning (from the start of drawing).\n"
            "Answer to 1 decimal point and list them with comma separated values: e.g., 3.2s, 1.5s, 1.0s\n"
            "List the durations in the order the shapes appeared."
        )
        with open(f"question_text/duration2d_n{self.num_shapes}_seed{self.seed}.txt", "w") as f:
            f.write(question_text_content)

        # Save detailed reasoning trace with final reasoning
        self.reasoning_trace.append("=== REASONING TO ANSWER ===")
        self.reasoning_trace.append("")
        self.reasoning_trace.append("Now I need to list the duration of each shape in the order they appeared.")
        self.reasoning_trace.append("")
        self.reasoning_trace.append("Let me recall each shape and its duration:")
        for i, (shape_name, duration) in enumerate(shape_duration_pairs):
            self.reasoning_trace.append(f"  {i+1}. The {shape_name} was drawn for {duration:.1f} seconds")
        self.reasoning_trace.append("")
        self.reasoning_trace.append("The question asks for the durations in order, formatted with comma-separated values.")
        self.reasoning_trace.append("Each duration should be listed to 1 decimal point with 's' suffix.")
        self.reasoning_trace.append("")
        self.reasoning_trace.append("Compiling the answer in the requested format:")
        for i, (_, duration) in enumerate(shape_duration_pairs):
            if i < len(shape_duration_pairs) - 1:
                self.reasoning_trace.append(f"  Shape {i+1}: {duration:.1f}s, (followed by comma)")
            else:
                self.reasoning_trace.append(f"  Shape {i+1}: {duration:.1f}s (last one, no comma)")
        self.reasoning_trace.append("")
        self.reasoning_trace.append(f"FINAL ANSWER: {answer_string}")

        with open(f"reasoning_traces/duration2d_n{self.num_shapes}_seed{self.seed}.txt", "w") as f:
            f.write("\n".join(self.reasoning_trace))


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