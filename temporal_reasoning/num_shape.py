from manim import *
import random
import math
import os
import shutil
from pathlib import Path

# ============================================================================
# Setup directories for output files
# ============================================================================
Path("questions").mkdir(exist_ok=True)          # Video files
Path("solutions").mkdir(exist_ok=True)          # Answer text files
Path("question_text").mkdir(exist_ok=True)      # Question text files
Path("reasoning_traces").mkdir(exist_ok=True)   # Step-by-step reasoning

# ============================================================================
# Manim configuration
# ============================================================================
config.media_dir = "manim_output"
config.verbosity = "WARNING"
config.pixel_height = 1080
config.pixel_width = 1920
config.frame_rate = 30
config.preview = False

# NUM_SHAPES [3-8] range difficulty
# NUM_SHAPES=5 python3 num_shape.py
class num_shape(ThreeDScene):
    """
    A 3D scene that generates a number and shape sequence puzzle:
    - Shows multiple 3D shapes appearing one at a time, each followed by a number
    - User must track the sequence and sum specific numbers based on shape positions
    - Generates question video, solution, and comprehensive reasoning trace
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # ====================================================================
        # Set random seed for reproducibility of sequence generation
        # ====================================================================
        self.seed = random.randint(1000, 9999)
        random.seed(self.seed)

        # ====================================================================
        # Parameters from environment variables (with defaults)
        # ====================================================================
        # Difficulty is controlled by number of shapes in sequence
        self.num_shapes = int(os.getenv("NUM_SHAPES", 5))

        # ====================================================================
        # Initialize reasoning trace and event logging
        # ====================================================================
        self.reasoning_trace = []

        # Track timing for reasoning trace using Manim's internal video time
        # This will be populated as the scene renders
        self.scene_events = []

    def log_event(self, description):
        """
        Log a scene event with video timestamp.
        Uses Manim's internal renderer.time which tracks actual video playback time,
        not wall-clock execution time.

        Args:
            description: Human-readable description of the event
        """
        # Get current video time from Manim's renderer
        # self.renderer.time tracks the cumulative duration of all animations/waits
        current_time = self.renderer.time

        self.scene_events.append({
            'time': current_time,
            'description': description
        })

    def format_time(self, seconds):
        """
        Format seconds as M:SS for display in reasoning trace.

        Args:
            seconds: Time in seconds (float)

        Returns:
            String formatted as "M:SS" (e.g., "2:37")
        """
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}:{secs:02d}"

    def get_position_description(self, position):
        """
        Helper method to describe a 3D position in natural language.
        This helps create human-readable descriptions of where objects appear.

        Args:
            position: 3D position vector [x, y, z]

        Returns:
            String describing the position (e.g., "on the left side of the screen in the upper part")
        """
        x, y, z = position[0], position[1], position[2]

        # Describe horizontal position (X-axis)
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

        # Describe vertical position (Y-axis)
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
        """
        Main scene construction method.
        This is called by Manim to build and render the entire scene.
        """
        # ====================================================================
        # Setup 3D camera view
        # ====================================================================
        # phi: angle from the z-axis (75° = looking down at ~15° from horizontal)
        # theta: rotation around z-axis (45° = viewing from front-right)
        self.set_camera_orientation(phi=75 * DEGREES, theta=45 * DEGREES)

        # ====================================================================
        # Constrain count to reasonable bounds for video length
        # ====================================================================
        # Ensures the sequence is neither too short (trivial) nor too long (tedious)
        count = max(3, min(self.num_shapes, 8))

        # ====================================================================
        # Log initial scene setup
        # ====================================================================
        self.log_event(f"Video begins with empty 3D space. Preparing to show sequence of {count} shapes")

        # ====================================================================
        # Full library of 3D shapes with descriptive names
        # ====================================================================
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

        # ====================================================================
        # Predefined pool of object positions
        # ====================================================================
        # These positions are spread across the visible area to ensure good visual separation
        object_positions_pool = [
            LEFT * 5 + DOWN * 1, RIGHT * 5 + DOWN * 1,
            LEFT * 1 + UP, LEFT * 5 + UP * 1,
            RIGHT * 2 + UP, RIGHT * 4 + DOWN,
            LEFT * 2 + DOWN, UP * 2,
            DOWN * 2.5, RIGHT * 5 + UP
        ]

        # ====================================================================
        # Predefined pool of number positions
        # ====================================================================
        # Numbers appear in different locations than their associated shapes
        # This prevents users from using spatial proximity as a cue
        number_positions_pool = [
            LEFT * 3, RIGHT * 4 + DOWN * 2,
            DOWN * 2, RIGHT * 2 + UP,
            LEFT * 5, RIGHT * 3,
            LEFT * 1, UP * 2.5,
            DOWN * 3, LEFT * 4 + UP
        ]

        # ====================================================================
        # Randomly select shapes and positions for this video
        # ====================================================================
        chosen_shape_indices = random.sample(range(len(all_shapes_with_names)), count)
        chosen_object_positions = random.sample(object_positions_pool, count)
        chosen_number_positions = random.sample(number_positions_pool, count)

        # ====================================================================
        # Generate random numbers for each shape
        # ====================================================================
        # Numbers range from 100 to 1000 with 2 decimal places
        numbers = [round(random.uniform(100, 1000), 2) for _ in range(count)]

        # ====================================================================
        # Create the sequence storage
        # ====================================================================
        shape_sequence = []   # Will store shape names in order
        number_sequence = []  # Will store numbers in order

        shown_objects = []  # Track displayed objects for potential cleanup

        # ====================================================================
        # Display shapes and numbers one at a time
        # ====================================================================
        for i in range(count):
            shape_original, shape_name = all_shapes_with_names[chosen_shape_indices[i]]
            shape = shape_original.copy()  # Make a copy to avoid reference issues
            obj_pos = chosen_object_positions[i]
            num_pos = chosen_number_positions[i]
            number = numbers[i]

            # Store sequence information for later question generation
            shape_sequence.append(shape_name)
            number_sequence.append(number)

            # Describe where the shape will appear
            position_desc = self.get_position_description(obj_pos)

            # ================================================================
            # Show the shape first
            # ================================================================
            self.log_event(f"Shape {i+1}/{count}: {shape_name} begins appearing {position_desc}")

            shape.set_fill(RED, opacity=0.6)
            shape.move_to(obj_pos)
            self.play(Create(shape), run_time=1)
            shown_objects.append(shape)

            self.log_event(f"Shape {i+1}: {shape_name} fully visible")
            self.wait(0.5)

            self.play(FadeOut(shape), run_time=0.1)
            self.log_event(f"Shape {i+1}: {shape_name} fades out")

            # ================================================================
            # Then show the number after the shape
            # ================================================================
            num_position_desc = self.get_position_description(num_pos)
            self.log_event(f"Number {i+1}/{count}: {number} begins appearing {num_position_desc}")

            txt = Text(str(number), font_size=48).move_to(num_pos)
            self.play(Write(txt, run_time=1))

            self.log_event(f"Number {i+1}: {number} fully visible")
            self.play(FadeOut(txt, run_time=0.1))
            self.log_event(f"Number {i+1}: {number} fades out")

        # ====================================================================
        # Brief pause after sequence completes
        # ====================================================================
        self.log_event("All shapes and numbers have been displayed. Sequence complete.")
        self.wait(0.5)

        # ====================================================================
        # Generate question and answer
        # ====================================================================
        # Pick two different shapes from the sequence
        idx1, idx2 = random.sample(range(count), 2)
        # Ensure idx1 comes before idx2 in the sequence
        if idx1 > idx2:
            idx1, idx2 = idx2, idx1

        shape1, shape2 = shape_sequence[idx1], shape_sequence[idx2]

        # ====================================================================
        # Determine valid question types based on the selected shapes
        # ====================================================================
        question_types = []

        # Type 1: Numbers between two shapes (always valid)
        question_types.append("between")

        # Type 2: Numbers before shape1 and after shape2
        # Valid only if idx1 > 0 (there are numbers before shape1)
        if idx1 > 0:
            question_types.append("before_and_after")

        # Type 3: Numbers before shape1 and right after shape2
        # Valid only if idx1 > 0 (there are numbers before shape1)
        if idx1 > 0:
            question_types.append("before_and_right_after")

        # Select random question type from valid options
        question_type = random.choice(question_types)
        # ====================================================================
        # Determine which numbers to sum based on question type
        # ====================================================================
        if question_type == "between":
            # Numbers between shape1 and shape2: from idx1 to idx2 (inclusive)
            # "Between" includes the number with the first shape up to the number with the second shape
            if idx2 - idx1 <= 0:
                numbers_to_sum = []
            else:
                numbers_to_sum = number_sequence[idx1:idx2]
            question_text = f"Sum up all the numbers between {shape1} and {shape2}."

        elif question_type == "before_and_after":
            # Numbers before shape1 (indices 0 to idx1-1) and after shape2 (index idx2 onward)
            numbers_before = number_sequence[:idx1] if idx1 > 0 else []
            numbers_after = number_sequence[idx2:] if idx2 < count else []
            numbers_to_sum = numbers_before + numbers_after
            question_text = f"Sum up the numbers before {shape1} and after {shape2}."

        elif question_type == "before_and_right_after":
            # Numbers before shape1 (indices 0 to idx1-1) and the number right after shape2 (index idx2)
            numbers_before = number_sequence[:idx1] if idx1 > 0 else []
            number_right_after = [number_sequence[idx2]] if idx2 < count else []
            numbers_to_sum = numbers_before + number_right_after
            question_text = f"Sum up the numbers before {shape1} and right after {shape2}."

        # Calculate the answer
        total_sum = sum(numbers_to_sum) if numbers_to_sum else 0

        # Store information needed for reasoning trace generation
        self.shape_sequence = shape_sequence
        self.number_sequence = number_sequence
        self.question_text = question_text
        self.question_type = question_type
        self.shape1 = shape1
        self.shape2 = shape2
        self.idx1 = idx1
        self.idx2 = idx2
        self.numbers_to_sum = numbers_to_sum
        self.total_sum = total_sum
        self.count = count
        # ====================================================================
        # Display the question
        # ====================================================================
        question_lines = [
            question_text,
            "",
            "Round to 2 decimal places."
        ]

        # Create question text objects with appropriate styling
        question_texts = []
        line_height = 0.7
        start_y = 2.5

        for i, line in enumerate(question_lines):
            if line:  # Skip empty lines for spacing
                text = Text(line, font_size=28, weight=BOLD if i == 0 else NORMAL)
                text.move_to(UP * (start_y - i * line_height))
                question_texts.append(text)

        # Add all question text as fixed elements (stays visible during camera movement)
        self.add_fixed_in_frame_mobjects(*question_texts)

        # Animate question appearance
        self.log_event("Question begins appearing on screen")
        self.play(*[FadeIn(text) for text in question_texts], run_time=1.0)
        self.log_event("Question fully visible")
        self.wait(3)
        self.log_event("Question display complete")
        # ====================================================================
        # Generate comprehensive reasoning trace
        # ====================================================================
        # This must be called AFTER all animations are complete
        # so that all timestamps are captured
        self.build_reasoning_trace()

        # ====================================================================
        # Format answer and save output files
        # ====================================================================
        formatted_answer = f"{self.total_sum:.2f}"

        # Save solution file (just the answer)
        with open(f"solutions/numshape_n{self.num_shapes}_seed{self.seed}.txt", "w") as f:
            f.write(formatted_answer)

        # Save question text file
        question_text_content = (
            f"{self.question_text}\n"
            "Round to 2 decimal places."
        )
        with open(f"question_text/numshape_n{self.num_shapes}_seed{self.seed}.txt", "w") as f:
            f.write(question_text_content)

        # Save reasoning trace file (comprehensive step-by-step solution)
        with open(f"reasoning_traces/numshape_n{self.num_shapes}_seed{self.seed}.txt", "w") as f:
            f.write("\n".join(self.reasoning_trace))

    def build_reasoning_trace(self):
        """
        Build a comprehensive, step-by-step reasoning trace.
        This explains how to solve the puzzle systematically with timestamps.

        The reasoning trace follows the format:
        1. Question statement
        2. Scene description with timestamps
        3. Step-by-step solution reasoning
        4. Final answer
        """
        self.reasoning_trace = []

        # ====================================================================
        # Introduction
        # ====================================================================
        self.reasoning_trace.append(f"**Question:** {self.question_text} Round to 2 decimal places.")
        self.reasoning_trace.append("")
        self.reasoning_trace.append("Let's solve this step by step.")
        self.reasoning_trace.append("")

        # ====================================================================
        # Scene description with timestamps
        # ====================================================================
        self.reasoning_trace.append("### Scene Description")
        self.reasoning_trace.append("")
        self.reasoning_trace.append(f"The video shows a sequence of {self.count} 3D shapes, each followed by a number. Here's what happens chronologically:")
        self.reasoning_trace.append("")

        for event in self.scene_events:
            time_str = self.format_time(event['time'])
            self.reasoning_trace.append(f"At {time_str}, {event['description']}")

        self.reasoning_trace.append("")

        # ====================================================================
        # Step 1: List the complete sequence
        # ====================================================================
        self.reasoning_trace.append("### Step 1: Identify the complete sequence")
        self.reasoning_trace.append("")
        self.reasoning_trace.append("After watching the entire video, I can list all shapes and their associated numbers in order:")
        self.reasoning_trace.append("")

        for i, (shape_name, number) in enumerate(zip(self.shape_sequence, self.number_sequence)):
            self.reasoning_trace.append(f"Position {i+1}: {shape_name} → {number}")

        self.reasoning_trace.append("")

        # ====================================================================
        # Step 2: Identify the shapes mentioned in the question
        # ====================================================================
        self.reasoning_trace.append("### Step 2: Locate the shapes mentioned in the question")
        self.reasoning_trace.append("")
        self.reasoning_trace.append(f"The question asks about **{self.shape1}** and **{self.shape2}**.")
        self.reasoning_trace.append("")
        self.reasoning_trace.append(f"- **{self.shape1}** appears at position {self.idx1+1}")
        self.reasoning_trace.append(f"- **{self.shape2}** appears at position {self.idx2+1}")
        self.reasoning_trace.append("")

        # ====================================================================
        # Step 3: Determine which numbers to sum
        # ====================================================================
        self.reasoning_trace.append("### Step 3: Determine which numbers to sum")
        self.reasoning_trace.append("")

        if self.question_type == "between":
            self.reasoning_trace.append(f"The question asks for numbers **between** {self.shape1} and {self.shape2}.")
            self.reasoning_trace.append("")
            self.reasoning_trace.append(f"'Between' includes the number with the first shape ({self.shape1} at position {self.idx1+1}) up to but not including the number with the second shape ({self.shape2} at position {self.idx2+1}).")
            self.reasoning_trace.append("")
            if self.idx2 - self.idx1 <= 1:
                self.reasoning_trace.append("Since these shapes are adjacent or the same, there are no numbers strictly between them.")
                self.reasoning_trace.append("However, 'between' typically includes at least the number with the first shape:")
                self.reasoning_trace.append("")
            if self.numbers_to_sum:
                self.reasoning_trace.append("The numbers to include are:")
                for j, num in enumerate(self.numbers_to_sum):
                    actual_position = self.idx1 + j + 1
                    self.reasoning_trace.append(f"  - Position {actual_position}: {num}")
            else:
                self.reasoning_trace.append("There are no numbers to sum in this case.")

        elif self.question_type == "before_and_after":
            self.reasoning_trace.append(f"The question asks for numbers **before** {self.shape1} and **after** {self.shape2}.")
            self.reasoning_trace.append("")

            self.reasoning_trace.append(f"**Numbers BEFORE {self.shape1}** (position {self.idx1+1}):")
            if self.idx1 == 0:
                self.reasoning_trace.append("  - None (it's the first shape)")
            else:
                for j in range(self.idx1):
                    self.reasoning_trace.append(f"  - Position {j+1}: {self.number_sequence[j]}")

            self.reasoning_trace.append("")
            self.reasoning_trace.append(f"**Numbers AFTER {self.shape2}** (position {self.idx2+1}):")
            if self.idx2 >= self.count - 1:
                self.reasoning_trace.append("  - None (it's the last shape)")
            else:
                for j in range(self.idx2, self.count):
                    self.reasoning_trace.append(f"  - Position {j+1}: {self.number_sequence[j]}")

        elif self.question_type == "before_and_right_after":
            self.reasoning_trace.append(f"The question asks for numbers **before** {self.shape1} and the number **right after** {self.shape2}.")
            self.reasoning_trace.append("")

            self.reasoning_trace.append(f"**Numbers BEFORE {self.shape1}** (position {self.idx1+1}):")
            if self.idx1 == 0:
                self.reasoning_trace.append("  - None (it's the first shape)")
            else:
                for j in range(self.idx1):
                    self.reasoning_trace.append(f"  - Position {j+1}: {self.number_sequence[j]}")

            self.reasoning_trace.append("")
            self.reasoning_trace.append(f"**Number RIGHT AFTER {self.shape2}** (position {self.idx2+1}):")
            if self.idx2 >= self.count - 1:
                self.reasoning_trace.append("  - None (it's the last shape)")
            else:
                self.reasoning_trace.append(f"  - Position {self.idx2+1}: {self.number_sequence[self.idx2]}")

        self.reasoning_trace.append("")

        # ====================================================================
        # Step 4: Calculate the sum
        # ====================================================================
        self.reasoning_trace.append("### Step 4: Calculate the sum")
        self.reasoning_trace.append("")

        if self.numbers_to_sum:
            calculation_steps = " + ".join([str(n) for n in self.numbers_to_sum])
            self.reasoning_trace.append(f"Sum = {calculation_steps}")
            self.reasoning_trace.append(f"Sum = {self.total_sum:.2f}")
        else:
            self.reasoning_trace.append("There are no numbers to sum, so the answer is 0.00")

        self.reasoning_trace.append("")

        # ====================================================================
        # Final answer
        # ====================================================================
        self.reasoning_trace.append("### Final Answer")
        self.reasoning_trace.append(f"Rounded to 2 decimal places: **{self.total_sum:.2f}**")
        self.reasoning_trace.append("")
        self.reasoning_trace.append(f"\\boxed{{{self.total_sum:.2f}}}")

# ============================================================================
# Main execution
# ============================================================================
if __name__ == "__main__":
    # Generate the number shape video
    scene = num_shape()
    scene.render()

    # ========================================================================
    # Move output file to questions directory with descriptive name
    # ========================================================================
    output = Path("manim_output/videos/1080p30/num_shape.mp4")
    if output.exists():
        filename = f"numshape_n{scene.num_shapes}_seed{scene.seed}.mp4"
        shutil.move(str(output), f"questions/{filename}")
        print(f"✓ Video saved: questions/{filename}")
    else:
        # Debug: Print what files actually exist if output not found
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

    # ========================================================================
    # Final cleanup
    # ========================================================================
    if os.path.exists("manim_output"):
        shutil.rmtree("manim_output")