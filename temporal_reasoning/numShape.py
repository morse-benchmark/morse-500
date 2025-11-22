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

        # Initialize reasoning trace
        self.reasoning_trace = []
        self.reasoning_trace.append(f"Problem: Number and shape sequence")
        self.reasoning_trace.append(f"Number of shapes: {self.num_shapes}")
        self.reasoning_trace.append(f"Random Seed: {self.seed}")
        self.reasoning_trace.append("")
        self.reasoning_trace.append("=== CHRONOLOGICAL SCENE DESCRIPTION ===")
        self.reasoning_trace.append("")

    def get_position_description(self, position):
        """Helper method to describe a position in natural language"""
        x, y, z = position[0], position[1], position[2]

        # Describe horizontal position
        if x < -3:
            horizontal = "on the far left side of the screen"
        elif x < -1.5:
            horizontal = "on the left side of the screen"
        elif x < -0.5:
            horizontal = "slightly left of center"
        elif x < 0.5:
            horizontal = "in the center"
        elif x < 1.5:
            horizontal = "slightly right of center"
        elif x < 3.5:
            horizontal = "on the right side of the screen"
        else:
            horizontal = "on the far right side of the screen"

        # Describe vertical position
        if y < -2:
            vertical = " near the bottom"
        elif y < -0.5:
            vertical = " in the lower part"
        elif y < 0.5:
            vertical = ""
        elif y < 1.5:
            vertical = " in the upper part"
        else:
            vertical = " near the top"

        return horizontal + vertical

    def construct(self):
        # Set camera orientation
        self.set_camera_orientation(phi=75 * DEGREES, theta=45 * DEGREES)

        # Constrain count to reasonable bounds
        count = max(3, min(self.num_shapes, 8))

        # Add initial scene description
        self.reasoning_trace.append(f"The video begins with an empty 3D space viewed from an angled perspective. The scene is set up to display a sequence of {count} different 3D shapes, each followed by a number. As I watch, shapes will appear one at a time in various locations across the screen, with each shape being followed by its associated number.")
        
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

        # Add chronological event tracking header
        self.reasoning_trace.append("")
        self.reasoning_trace.append("Now, let me describe each event as it happens chronologically:")
        self.reasoning_trace.append("")
        
        for i in range(count):
            shape_original, shape_name = all_shapes_with_names[chosen_shape_indices[i]]
            shape = shape_original.copy()  # Make a copy to avoid reference issues
            obj_pos = chosen_object_positions[i]
            num_pos = chosen_number_positions[i]
            number = numbers[i]

            # Store sequence information
            shape_sequence.append(shape_name)
            number_sequence.append(number)

            # Add chronological description for shape appearance
            position_desc = self.get_position_description(obj_pos)
            self.reasoning_trace.append(f"Event {i+1}a: A red {shape_name} appears {position_desc}. It takes about 1 second to fully form, stays visible for a brief moment (about 0.5 seconds), then fades away.")

            # Show the shape first
            shape.set_fill(RED, opacity=0.6)
            shape.move_to(obj_pos)
            self.play(Create(shape), run_time=1)
            shown_objects.append(shape)
            self.wait(0.5)
            self.play(FadeOut(shape), run_time=0.1)

            # Add chronological description for number appearance
            num_position_desc = self.get_position_description(num_pos)
            self.reasoning_trace.append(f"Event {i+1}b: Immediately after the {shape_name} disappears, the number {number} appears {num_position_desc}. It takes about 1 second to write out, then quickly fades away.")

            # Then show the number after the shape
            txt = Text(str(number), font_size=48).move_to(num_pos)
            self.play(Write(txt, run_time=1))
            self.play(FadeOut(txt, run_time=0.1))

        # Add summary after all events
        self.reasoning_trace.append("")
        self.reasoning_trace.append("=== SEQUENCE SUMMARY ===")
        self.reasoning_trace.append("")
        self.reasoning_trace.append("Now that I've watched the entire sequence, let me list what appeared in order:")
        self.reasoning_trace.append("")
        for i, (shape_name, number) in enumerate(zip(shape_sequence, number_sequence)):
            self.reasoning_trace.append(f"Position {i+1}: {shape_name} followed by {number}")

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

        # Add detailed reasoning for the answer
        self.reasoning_trace.append("")
        self.reasoning_trace.append("=== ANSWERING THE QUESTION ===")
        self.reasoning_trace.append("")
        self.reasoning_trace.append(f"The question asks: {question_text}")
        self.reasoning_trace.append("")

        if question_type == "between":
            self.reasoning_trace.append(f"To answer this, I need to identify the positions of {shape1} and {shape2} in the sequence.")
            self.reasoning_trace.append(f"Looking back at my sequence, {shape1} appeared at position {idx1+1}.")
            self.reasoning_trace.append(f"And {shape2} appeared at position {idx2+1}.")
            self.reasoning_trace.append("")
            self.reasoning_trace.append(f"The numbers 'between' these two shapes means I need to include:")
            if idx2 - idx1 <= 1:
                self.reasoning_trace.append(f"- Actually, there are no numbers strictly between them since they are adjacent or the same.")
                self.reasoning_trace.append(f"- However, 'between' typically includes the number with the first shape up to (and including) the number with the second shape.")
            if numbers_to_sum:
                self.reasoning_trace.append(f"- The number at position {idx1+1} (with {shape1}): {number_sequence[idx1]}")
                for j in range(idx1+1, idx2):
                    self.reasoning_trace.append(f"- The number at position {j+1}: {number_sequence[j]}")
                self.reasoning_trace.append(f"- The number at position {idx2+1} (with {shape2}): {number_sequence[idx2]}")

        elif question_type == "before_and_after":
            self.reasoning_trace.append(f"To answer this, I need to identify which numbers came before {shape1} and which came after {shape2}.")
            self.reasoning_trace.append(f"Looking back at my sequence, {shape1} appeared at position {idx1+1}.")
            self.reasoning_trace.append(f"And {shape2} appeared at position {idx2+1}.")
            self.reasoning_trace.append("")
            self.reasoning_trace.append("Numbers BEFORE the first shape mentioned:")
            if idx1 == 0:
                self.reasoning_trace.append("- There are no numbers before the first position.")
            else:
                for j in range(idx1):
                    self.reasoning_trace.append(f"- Position {j+1}: {number_sequence[j]}")
            self.reasoning_trace.append("")
            self.reasoning_trace.append("Numbers AFTER the second shape mentioned:")
            if idx2 >= count - 1:
                self.reasoning_trace.append("- There are no numbers after this position.")
            else:
                for j in range(idx2, count):
                    self.reasoning_trace.append(f"- Position {j+1}: {number_sequence[j]}")

        elif question_type == "before_and_right_after":
            self.reasoning_trace.append(f"To answer this, I need to identify which numbers came before {shape1} and which number came right after {shape2}.")
            self.reasoning_trace.append(f"Looking back at my sequence, {shape1} appeared at position {idx1+1}.")
            self.reasoning_trace.append(f"And {shape2} appeared at position {idx2+1}.")
            self.reasoning_trace.append("")
            self.reasoning_trace.append("Numbers BEFORE the first shape mentioned:")
            if idx1 == 0:
                self.reasoning_trace.append("- There are no numbers before the first position.")
            else:
                for j in range(idx1):
                    self.reasoning_trace.append(f"- Position {j+1}: {number_sequence[j]}")
            self.reasoning_trace.append("")
            self.reasoning_trace.append("Number RIGHT AFTER the second shape mentioned:")
            if idx2 >= count - 1:
                self.reasoning_trace.append("- There is no number after this position.")
            else:
                self.reasoning_trace.append(f"- Position {idx2+1}: {number_sequence[idx2]}")

        self.reasoning_trace.append("")
        self.reasoning_trace.append("Now I'll add up all the relevant numbers:")
        if numbers_to_sum:
            calculation_steps = " + ".join([str(n) for n in numbers_to_sum])
            self.reasoning_trace.append(f"{calculation_steps} = {total_sum:.2f}")
        else:
            self.reasoning_trace.append("There are no numbers to sum, so the answer is 0.00")
        self.reasoning_trace.append("")
        self.reasoning_trace.append(f"Therefore, the answer is: {total_sum:.2f}")
        
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

        # Save detailed reasoning trace
        self.reasoning_trace.append("")
        self.reasoning_trace.append(f"Final Answer: {formatted_answer}")
        with open(f"reasoning_traces/numshape_n{self.num_shapes}_seed{self.seed}.txt", "w") as f:
            f.write("\n".join(self.reasoning_trace))


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