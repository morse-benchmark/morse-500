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

class duration_3d(ThreeDScene):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Set random seed for reproducibility
        self.seed = random.randint(1000, 9999)
        random.seed(self.seed)
        
        # Parameters
        self.num_shapes = int(os.getenv("NUM_SHAPES", 5))

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

        # Predefined pool of 10 candidate positions
        positions_pool = [
            LEFT * 4 + DOWN, RIGHT * 4 + DOWN,
            LEFT * 2 + UP, RIGHT * 2 + UP,
            ORIGIN, LEFT * 4 + UP,
            RIGHT * 4 + UP, RIGHT * 2 + DOWN,
            LEFT * 2 + DOWN, DOWN * 3
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