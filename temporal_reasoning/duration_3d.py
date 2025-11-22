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

class duration_3d(ThreeDScene):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Set random seed for reproducibility
        self.seed = random.randint(1000, 9999)
        random.seed(self.seed)

        # Parameters
        self.num_shapes = int(os.getenv("NUM_SHAPES", 5))

        # Initialize reasoning trace
        self.reasoning_trace = []
        self.reasoning_trace.append(f"Problem: 3D shape duration tracking")
        self.reasoning_trace.append(f"Number of shapes: {self.num_shapes}")
        self.reasoning_trace.append(f"Random Seed: {self.seed}")
        self.reasoning_trace.append("")

    def construct(self):
        # Camera setup
        self.set_camera_orientation(phi=75 * DEGREES, theta=45 * DEGREES)

        # Constrain count to reasonable bounds
        count = max(1, min(self.num_shapes, 10))

        # Full library of 3D shapes and matching colors
        all_shapes = [
            Sphere, Cube, Cylinder, Cone, Torus,
            Tetrahedron, Octahedron, Dodecahedron, Icosahedron, Star
        ]
        all_colors = [
            BLUE, GREEN, RED, YELLOW, PURPLE,
            ORANGE, TEAL, PINK, MAROON, GOLD
        ]

        # Color names for reasoning trace
        color_names = {
            BLUE: "blue", GREEN: "green", RED: "red", YELLOW: "yellow", PURPLE: "purple",
            ORANGE: "orange", TEAL: "teal", PINK: "pink", MAROON: "maroon", GOLD: "gold"
        }

        # Predefined pool of 10 candidate positions
        positions_pool = [
            LEFT * 4 + DOWN, RIGHT * 4 + DOWN,
            LEFT * 2 + UP, RIGHT * 2 + UP,
            ORIGIN, LEFT * 4 + UP,
            RIGHT * 4 + UP, RIGHT * 2 + DOWN,
            LEFT * 2 + DOWN, DOWN * 3
        ]

        # Position descriptions for reasoning trace
        position_names = [
            "far left lower area", "far right lower area",
            "mid-left upper area", "mid-right upper area",
            "center of the scene", "far left upper area",
            "far right upper area", "mid-right lower area",
            "mid-left lower area", "bottom center"
        ]

        # Randomly pick which shapes and positions to use
        chosen_indices = random.sample(range(len(all_shapes)), count)
        chosen_positions = random.sample(positions_pool, count)

        # Generate random durations
        durations = [random.uniform(0.5, 4.0) for _ in range(count)]

        # Derive answer: shape names ordered by duration descending
        shape_names = [all_shapes[i].__name__.lower() for i in chosen_indices]
        paired = list(zip(shape_names, durations))
        sorted_by_duration = sorted(paired, key=lambda x: x[1], reverse=True)
        answer_list = [name for name, _ in sorted_by_duration]

        # Add initial scene description
        self.reasoning_trace.append("=== CHRONOLOGICAL SCENE DESCRIPTION ===")
        self.reasoning_trace.append("")
        self.reasoning_trace.append("The video begins with a 3D scene view from an elevated angle. The scene is empty at first, with a dark background. Over the course of the video, different 3D shapes will appear one at a time, remain visible for varying durations, and then disappear. My task is to track how long each shape remains on screen to determine which shapes had the longest display times.")
        self.reasoning_trace.append("")

        # Track cumulative time for chronological description
        cumulative_time = 0.0

        # Describe each shape's appearance chronologically
        self.reasoning_trace.append("=== FRAME-BY-FRAME EVENTS ===")
        self.reasoning_trace.append("")

        for idx, shape_idx in enumerate(chosen_indices):
            ShapeClass = all_shapes[shape_idx]
            color = all_colors[shape_idx]
            shape_name = ShapeClass.__name__.lower()
            color_name = color_names[color]

            # Find position description
            pos_idx = positions_pool.index(chosen_positions[idx])
            pos_name = position_names[pos_idx]

            # Describe the appearance
            self.reasoning_trace.append(f"Time {cumulative_time:.2f}s: A {color_name} {shape_name} appears in the {pos_name}. The shape materializes with a creation animation that takes about 0.1 seconds. Once fully formed, the {shape_name} remains stationary on screen.")
            self.reasoning_trace.append("")

            cumulative_time += 0.1  # Creation animation time

            # Describe the duration
            self.reasoning_trace.append(f"Time {cumulative_time:.2f}s - {cumulative_time + durations[idx]:.2f}s: The {color_name} {shape_name} stays visible on screen for {durations[idx]:.2f} seconds. During this time, I carefully observe and mentally note the duration.")
            self.reasoning_trace.append("")

            cumulative_time += durations[idx]

            # Describe the disappearance
            self.reasoning_trace.append(f"Time {cumulative_time:.2f}s: The {color_name} {shape_name} begins to fade out. The fade-out animation takes approximately 0.2 seconds, and then the shape completely disappears from the scene, leaving the screen empty again.")
            self.reasoning_trace.append("")

            cumulative_time += 0.2  # Fade-out animation time

        # Add reasoning section
        self.reasoning_trace.append("=== REASONING TO DERIVE THE ANSWER ===")
        self.reasoning_trace.append("")
        self.reasoning_trace.append("Now that I have observed all the shapes and their respective display durations, I need to organize them from longest to shortest duration. Let me review the duration data I collected:")
        self.reasoning_trace.append("")

        self.reasoning_trace.append("Shape durations recorded:")
        for idx, shape_idx in enumerate(chosen_indices):
            shape_name = all_shapes[shape_idx].__name__.lower()
            self.reasoning_trace.append(f"  - {shape_name}: {durations[idx]:.2f} seconds")
        self.reasoning_trace.append("")

        self.reasoning_trace.append("Sorting these shapes by duration from longest to shortest:")
        for name, duration in sorted_by_duration:
            self.reasoning_trace.append(f"  {name}: {duration:.2f}s")
        self.reasoning_trace.append("")

        self.reasoning_trace.append(f"Therefore, the order of shapes from longest to shortest display time is: {', '.join(answer_list)}")

        # Animate each chosen shape
        for idx, shape_idx in enumerate(chosen_indices):
            ShapeClass = all_shapes[shape_idx]
            color = all_colors[shape_idx]
            shape = ShapeClass()
            shape.set_fill(color, opacity=0.6)
            shape.move_to(chosen_positions[idx])

            # Show, wait, then remove
            self.play(Create(shape), run_time=0.1)
            self.wait(durations[idx])
            self.play(FadeOut(shape), run_time=0.2)

        # Wait a moment before showing question
        self.wait(0.5)

        # Create multi-line question text
        question_lines = [
            "List the order of shapes that appeared longest to shortest",
            "with comma-separated values.",
            "",
            "The shape names are: Sphere, Cube, Cylinder, Cone, Torus,",
            "Tetrahedron, Octahedron, Dodecahedron, Icosahedron, Star."
        ]
        
        # Create question text objects for each line
        question_texts = []
        line_height = 0.6
        start_y = 2.0
        
        for i, line in enumerate(question_lines):
            if line:  # Skip empty lines for spacing
                text = Text(line, font_size=28, weight=BOLD if i == 0 else NORMAL)
                text.move_to(UP * (start_y - i * line_height))
                question_texts.append(text)
        
        # Add all question text as fixed elements
        self.add_fixed_in_frame_mobjects(*question_texts)
        
        # Animate question appearance
        self.play(*[FadeIn(text) for text in question_texts], run_time=1.0)
        self.wait(1.0)

        # Show instruction
        instruction_text = Text(
            "Return the answer as comma-separated values.",
            font_size=24,
            color=YELLOW
        ).move_to(DOWN * 2.5)
        
        self.add_fixed_in_frame_mobjects(instruction_text)
        self.play(FadeIn(instruction_text, shift=UP*0.3), run_time=0.8)
        self.wait(3)
        
        # Save solution and question text
        answer_string = ", ".join(answer_list)
        with open(f"solutions/duration_3d_n{self.num_shapes}_seed{self.seed}.txt", "w") as f:
            f.write(answer_string)
        
        question_text_content = (
            "List the order of shapes that appeared longest to shortest with comma-separated values.\n"
            "The shape names are: Sphere, Cube, Cylinder, Cone, Torus, Tetrahedron, Octahedron, Dodecahedron, Icosahedron, Star.\n"
            "Return the answer as comma-separated values."
        )
        with open(f"question_text/duration_3d_n{self.num_shapes}_seed{self.seed}.txt", "w") as f:
            f.write(question_text_content)

        # Save detailed reasoning trace
        self.reasoning_trace.append("")
        self.reasoning_trace.append(f"Final Answer: {answer_string}")
        with open(f"reasoning_traces/duration_3d_n{self.num_shapes}_seed{self.seed}.txt", "w") as f:
            f.write("\n".join(self.reasoning_trace))


if __name__ == "__main__":
    # Generate the duration video
    scene = duration_3d()
    scene.render()

    # Move the output file with descriptive name
    # Manim creates a folder with the class name
    output = Path("manim_output/videos/1080p30/duration_3d.mp4")
    if output.exists():
        filename = f"duration_3d_n{scene.num_shapes}_seed{scene.seed}.mp4"
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