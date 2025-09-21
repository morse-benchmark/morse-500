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

# NUM_SHAPES [3-8] range difficulty
# NUM_SHAPES=5 python3 numShape.py
class numShape(ThreeDScene):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Set random seed for reproducibility
        self.seed = random.randint(1000, 9999)
        random.seed(self.seed)
        
        # Parameters - difficulty controls number of shapes
        self.num_shapes = int(os.getenv("NUM_SHAPES", 5))

    def construct(self):
        # Set camera orientation
        self.set_camera_orientation(phi=75 * DEGREES, theta=45 * DEGREES)
        
        # Constrain count to reasonable bounds
        count = max(3, min(self.num_shapes, 8))
        
        # Full library of 3D shapes with names
        all_shapes_with_names = [
            (Cube(side_length=1), "cube"),
            (Sphere(radius=0.6), "sphere"),
            (Cone(base_radius=0.5, height=1), "cone"),
            (Prism(dimensions=[1, 2, 3]).rotate(PI / 2), "rectangular prism"),
            (Cylinder(radius=0.4, height=1.2), "cylinder"),
            (Tetrahedron(edge_length=1), "tetrahedron"),
            (Octahedron(edge_length=1), "octahedron"),
            (Torus(major_radius=0.8, minor_radius=0.3), "torus")
        ]
        
        # Predefined pool of object positions
        object_positions_pool = [
            LEFT * 5 + DOWN * 1, RIGHT * 5 + DOWN * 1,
            LEFT * 1 + UP, LEFT * 5 + UP * 1,
            RIGHT * 2 + UP, RIGHT * 4 + DOWN,
            LEFT * 2 + DOWN, UP * 2,
            DOWN * 2.5, RIGHT * 5 + UP
        ]
        
        # Predefined pool of number positions
        number_positions_pool = [
            LEFT * 3, RIGHT * 4 + DOWN * 2,
            DOWN * 2, RIGHT * 2 + UP,
            LEFT * 5, RIGHT * 3,
            LEFT * 1, UP * 2.5,
            DOWN * 3, LEFT * 4 + UP
        ]

        # Randomly select shapes and positions
        chosen_shape_indices = random.sample(range(len(all_shapes_with_names)), count)
        chosen_object_positions = random.sample(object_positions_pool, count)
        chosen_number_positions = random.sample(number_positions_pool, count)

        # Generate random numbers for each shape
        numbers = [round(random.uniform(100, 1000), 2) for _ in range(count)]
        
        # Create the sequence of shapes and numbers
        shape_sequence = []
        number_sequence = []
        
        shown_objects = []
        
        for i in range(count):
            shape_original, shape_name = all_shapes_with_names[chosen_shape_indices[i]]
            shape = shape_original.copy()  # Make a copy to avoid reference issues
            obj_pos = chosen_object_positions[i]
            num_pos = chosen_number_positions[i]
            number = numbers[i]
            
            # Store sequence information
            shape_sequence.append(shape_name)
            number_sequence.append(number)
            
            # Show the shape first
            shape.set_fill(RED, opacity=0.6)
            shape.move_to(obj_pos)
            self.play(Create(shape), run_time=1)
            shown_objects.append(shape)
            self.wait(0.5)
            self.play(FadeOut(shape), run_time=0.1)
            
            # Then show the number after the shape
            txt = Text(str(number), font_size=48).move_to(num_pos)
            self.play(Write(txt, run_time=1))
            self.play(FadeOut(txt, run_time=0.1))

        self.wait(0.5)

        # Generate question and answer
        # Pick two different shapes 
        idx1, idx2 = random.sample(range(count), 2)
        if idx1 > idx2:
            idx1, idx2 = idx2, idx1
            
        shape1, shape2 = shape_sequence[idx1], shape_sequence[idx2]
        
        # Randomly choose question type
        question_types = []
        
        # Type 1: Numbers between two shapes (always valid)
        question_types.append("between")
        
        # Type 2: Numbers before shape1 and after shape2 
        # Valid if idx1 > 0 (numbers exist before shape1) and idx2 < count (shape2 has a number after it)
        if idx1 > 0:
            question_types.append("before_and_after")
        
        # Type 3: Numbers before shape1 and right after shape2
        # Valid if idx1 > 0 (numbers exist before shape1) and idx2 < count (shape2 has a number after it)  
        if idx1 > 0:
            question_types.append("before_and_right_after")
        
        # Select random question type
        question_type = random.choice(question_types)
        
        if question_type == "between":
            # Numbers between shape1 and shape2: from idx1 to idx2-1 (inclusive)
            if idx2 - idx1 <= 0:
                numbers_to_sum = []
            else:
                numbers_to_sum = number_sequence[idx1:idx2]
            question_text = f"Sum up all the numbers between {shape1} and {shape2}."
            
        elif question_type == "before_and_after":
            # Numbers before shape1 (numbers 0 to idx1-1) and after shape2 (number idx2)
            numbers_before = number_sequence[:idx1] if idx1 > 0 else []
            numbers_after = number_sequence[idx2:] if idx2 < count else []
            numbers_to_sum = numbers_before + numbers_after
            question_text = f"Sum up the numbers before {shape1} and after {shape2}."
            
        elif question_type == "before_and_right_after":
            # Numbers before shape1 (numbers 0 to idx1-1) and right after shape2 (number idx2)
            # This is actually the same as "before_and_after" since there's only one number after each shape
            numbers_before = number_sequence[:idx1] if idx1 > 0 else []
            number_right_after = [number_sequence[idx2]] if idx2 < count else []
            numbers_to_sum = numbers_before + number_right_after
            question_text = f"Sum up the numbers before {shape1} and right after {shape2}."
        
        total_sum = sum(numbers_to_sum) if numbers_to_sum else 0
        
        # Create question text
        question_lines = [
            question_text,
            "",
            "Round to 2 decimal places."
        ]
        
        # Create question text objects
        question_texts = []
        line_height = 0.7
        start_y = 2.5
        
        for i, line in enumerate(question_lines):
            if line:  # Skip empty lines for spacing
                text = Text(line, font_size=28, weight=BOLD if i == 0 else NORMAL)
                text.move_to(UP * (start_y - i * line_height))
                question_texts.append(text)
        
        # Add all question text as fixed elements
        self.add_fixed_in_frame_mobjects(*question_texts)
        
        # Animate question appearance
        self.play(*[FadeIn(text) for text in question_texts], run_time=1.0)
        self.wait(3)
        
        # Format answer to 2 decimal places
        formatted_answer = f"{total_sum:.2f}"
        
        # Save solution and question text
        with open(f"solutions/numshape_n{self.num_shapes}_seed{self.seed}.txt", "w") as f:
            f.write(formatted_answer)
        
        question_text_content = (
            f"{question_text}\n"
            "Round to 2 decimal places."
        )
        with open(f"question_text/numshape_n{self.num_shapes}_seed{self.seed}.txt", "w") as f:
            f.write(question_text_content)


if __name__ == "__main__":
    # Generate the number shape video
    scene = numShape()
    scene.render()

    # Move the output file with descriptive name
    output = Path("manim_output/videos/1080p30/numShape.mp4")
    if output.exists():
        filename = f"numshape_n{scene.num_shapes}_seed{scene.seed}.mp4"
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