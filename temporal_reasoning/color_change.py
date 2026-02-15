from manim import *
import random
import math
import os
import shutil
from pathlib import Path

# ============================================================================
# Setup directories for output files
# ============================================================================
Path("questions").mkdir(exist_ok=True)  # Video files
Path("solutions").mkdir(exist_ok=True)  # Answer text files
Path("question_text").mkdir(exist_ok=True)  # Question text files
Path("reasoning_traces").mkdir(exist_ok=True)  # Step-by-step reasoning

# ============================================================================
# Manim configuration
# ============================================================================
config.media_dir = "manim_output"
config.verbosity = "WARNING"
config.pixel_height = 1080
config.pixel_width = 1920
config.frame_rate = 30
config.preview = False


# NUM_TRANSFORMS=[3-8] python3 color_change.py
class color_change(Scene):
    """
    A scene that generates a color change puzzle:
    - Shows a shape transforming through multiple colors and forms
    - User must track the sequence and answer questions about past states
    - Generates question video, solution, and detailed reasoning trace
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # ====================================================================
        # Random seed for reproducibility of puzzle generation
        # ====================================================================
        self.seed = random.randint(1000, 9999)
        random.seed(self.seed)

        # ====================================================================
        # Configuration parameters
        # ====================================================================
        # Difficulty controls number of transforms (can be set via environment variable)
        # NUM_TRANSFORMS=5 python3 color_change.py
        self.num_transforms = int(os.getenv("NUM_TRANSFORMS", 4))

        # ====================================================================
        # Initialize data structures for reasoning trace
        # ====================================================================
        # Store the final reasoning trace as a list of strings
        self.reasoning_trace = []

        # Track timing for reasoning trace using Manim's internal video time
        # This will be populated as the scene renders with timestamped events
        self.scene_events = []

    def log_event(self, description):
        """
        Log a scene event with video timestamp.
        Uses Manim's internal renderer.time which tracks actual video playback time,
        not wall-clock execution time.
        """
        # Get current video time from Manim's renderer
        # self.renderer.time tracks the cumulative duration of all animations/waits
        current_time = self.renderer.time

        self.scene_events.append({"time": current_time, "description": description})

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

    def construct(self):
        """
        Main scene construction method.
        This is called by Manim to build and render the entire scene.
        """
        # ====================================================================
        # Setup: Validate and constrain parameters
        # ====================================================================

        # ====================================================================
        # Shape and color libraries
        # ====================================================================
        # Full library of valid shapes with their human-readable names
        # Each tuple contains (Manim shape object, descriptive name)
        all_shapes_with_names = [
            (Circle(), "circle"),
            (Square(), "square"),
            (Triangle(), "triangle"),
            (RegularPolygon(5), "pentagon"),
            (Square().rotate(PI / 4), "diamond"),
            (RegularPolygon(6), "hexagon"),
            (RegularPolygon(8), "octagon"),
            (Star(5), "star"),
            (Ellipse(width=2, height=1), "oval"),
        ]

        count = max(1, min(self.num_transforms, len(all_shapes_with_names)))

        # Color palette and their corresponding names
        # These must be in matching order for correct name lookup
        all_colors = [YELLOW, WHITE, BLUE, GREEN, RED, PURPLE, ORANGE, TEAL, PINK]
        color_names = [
            "yellow",
            "white",
            "blue",
            "green",
            "red",
            "purple",
            "orange",
            "teal",
            "pink",
        ]

        # ====================================================================
        # Position library for spatial variety
        # ====================================================================
        # Predefined pool of positions creates visual variety and spatial context
        # This helps make each transformation more distinct and memorable
        positions_pool = [
            LEFT * 2,
            RIGHT * 2,
            UP * 2,
            DOWN * 2,
            LEFT * 3,
            RIGHT * 3,
            UP * 1.5,
            DOWN * 1.5,
            LEFT * 1,
            RIGHT * 1,
        ]

        # ====================================================================
        # Generate random puzzle configuration
        # ====================================================================
        # Randomly select shapes, colors, and positions for the transform chain
        # Using random.sample ensures no duplicates within each category
        # This creates a unique puzzle each time with different seed
        chosen_shape_indices = []
        last_shape = None

        for _ in range(count):
            # Filter out the shape used in the previous step
            choices = [
                i
                for i in range(len(all_shapes_with_names))
                if not last_shape or i != last_shape
            ]

            current_shape = random.choice(choices)
            chosen_shape_indices.append(current_shape)

            # Update the tracker for the next iteration
            last_shape = current_shape

        chosen_color_indices = random.choices(range(len(all_colors)), k=count)
        chosen_positions = random.sample(positions_pool, count)

        # ====================================================================
        # Create the initial shape
        # ====================================================================
        # Extract the first shape from our random selection
        initial_shape, initial_shape_name = all_shapes_with_names[
            chosen_shape_indices[0]
        ]
        initial_shape = (
            initial_shape.copy()
        )  # Make a copy to avoid reference issues with Manim objects
        initial_color = all_colors[chosen_color_indices[0]]

        # Style the shape with fill and stroke
        initial_shape.set_fill(initial_color, opacity=0.7)
        initial_shape.set_stroke(initial_color)
        initial_shape.move_to(chosen_positions[0])

        # ====================================================================
        # Initialize sequence tracking
        # ====================================================================
        # Store the complete sequence for answer calculation and reasoning trace
        # Each entry is a tuple: (shape_name, color_name, position_vector)
        shape_sequence = [
            (
                initial_shape_name,
                color_names[chosen_color_indices[0]],
                chosen_positions[0],
            )
        ]

        # ====================================================================
        # Helper function to describe position in natural language
        # ====================================================================
        def describe_position(pos):
            """
            Convert position vector to human-readable description.

            This makes the reasoning trace more natural and easier to follow
            by using spatial language instead of coordinates.

            Args:
                pos: NumPy array representing position [x, y, z]

            Returns:
                String description of the position (e.g., "the left side")
            """
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

        # ====================================================================
        # Show initial shape
        # ====================================================================
        # Log and animate the appearance of the first shape
        # This sets the baseline for the transformation sequence
        position_desc = describe_position(chosen_positions[0])

        # Log BEFORE animation starts
        self.log_event(
            f"A {color_names[chosen_color_indices[0]]} {initial_shape_name} appears at {position_desc}"
        )
        self.play(Create(initial_shape))
        self.wait(0.5)
        # Log AFTER animation completes
        self.log_event(f"Initial {initial_shape_name} is fully visible")

        # ====================================================================
        # Transform through the remaining shapes
        # ====================================================================
        # Iterate through each subsequent transformation in the sequence
        # Each transform changes both the shape AND the color simultaneously
        for i in range(1, count):
            # Prepare the next shape in the sequence
            next_shape, next_shape_name = all_shapes_with_names[chosen_shape_indices[i]]
            next_shape = (
                next_shape.copy()
            )  # Make a copy to avoid reference issues with Manim objects
            next_color = all_colors[chosen_color_indices[i]]

            # Apply visual styling to the new shape
            next_shape.set_fill(next_color, opacity=0.7)
            next_shape.set_stroke(next_color)
            next_shape.move_to(chosen_positions[i])

            # ================================================================
            # Record this transformation in our sequence
            # ================================================================
            # Store in sequence with position for detailed reasoning trace
            shape_sequence.append(
                (
                    next_shape_name,
                    color_names[chosen_color_indices[i]],
                    chosen_positions[i],
                )
            )

            # ================================================================
            # Animate the transformation
            # ================================================================
            # Log transformation event BEFORE animation starts
            position_desc = describe_position(chosen_positions[i])
            self.log_event(
                f"Shape begins transforming into a {color_names[chosen_color_indices[i]]} {next_shape_name} at {position_desc}"
            )

            # Execute the transform animation (morphs current shape into next shape)
            self.play(Transform(initial_shape, next_shape))
            self.wait(0.5)

            # Log transformation event AFTER animation completes
            self.log_event(
                f"Transformation to {color_names[chosen_color_indices[i]]} {next_shape_name} is complete"
            )

        # ====================================================================
        # Fade out the final shape
        # ====================================================================
        # Remove the final shape from view before presenting the question
        # Log BEFORE animation starts
        self.log_event("Final shape begins fading out")
        self.play(FadeOut(initial_shape))
        self.wait(0.5)
        # Log AFTER animation completes
        self.log_event("All shapes have disappeared from screen")

        # ====================================================================
        # Generate question and determine answer
        # ====================================================================
        # Ask about the color of the shape that appeared N turns before the final shape
        # This tests memory of the temporal sequence
        turns_back = random.randint(0, count - 1)
        target_index = count - 1 - turns_back  # Index of the target shape (0-indexed)
        answer = shape_sequence[target_index][1]  # Color name (index 1 in tuple)

        # ====================================================================
        # Store data for reasoning trace generation
        # ====================================================================
        # These will be used by build_reasoning_trace() after rendering completes
        self.shape_sequence = shape_sequence
        self.turns_back = turns_back
        self.target_index = target_index
        self.answer = answer
        self.count = count

        # ====================================================================
        # Display the question
        # ====================================================================
        # Build the question text based on the final shape and selected offset
        question_lines = [
            f"What was the color of the shape that ",
            f"appeared {turns_back} turns before the final shape?",
            "",
            "Output in lower case",
        ]

        # ====================================================================
        # Create and position question text objects
        # ====================================================================
        question_texts = []
        line_height = 0.8
        start_y = 0.5

        for i, line in enumerate(question_lines):
            if line:  # Skip empty lines for spacing
                # First line is bold (the question), others are normal weight
                text = Text(line, font_size=28, weight=BOLD if i == 0 else NORMAL)
                text.move_to(UP * (start_y - i * line_height))
                question_texts.append(text)

        # ====================================================================
        # Animate question appearance
        # ====================================================================
        # Log BEFORE animation starts
        self.log_event("Question text appears on screen")
        self.play(*[Write(text) for text in question_texts])
        self.wait(3)
        # Log AFTER animation completes and wait finishes
        self.log_event("Question remains visible for viewer to read")

        # ====================================================================
        # Generate comprehensive reasoning trace
        # ====================================================================
        # This creates a detailed step-by-step explanation of how to solve the puzzle
        # Following the same structure as CubeRollScene for consistency
        self.build_reasoning_trace()

        # ====================================================================
        # Save output files
        # ====================================================================
        # Create three output files with consistent naming:
        # 1. Solution file (just the answer for automated grading)
        # 2. Question text file (for reference and dataset documentation)
        # 3. Reasoning trace file (detailed solution walkthrough)

        # Solution file (just the answer)
        with open(
            f"solutions/color_change_n{self.num_transforms}_seed{self.seed}.txt", "w"
        ) as f:
            f.write(answer)

        # Question text file
        question_text_content = (
            f"What was the color of the shape that appeared {turns_back} turns before the final shape?\n"
            "Output in lower case"
        )
        with open(
            f"question_text/color_change_n{self.num_transforms}_seed{self.seed}.txt",
            "w",
        ) as f:
            f.write(question_text_content)

        # Reasoning trace file
        with open(
            f"reasoning_traces/color_change_n{self.num_transforms}_seed{self.seed}.txt",
            "w",
        ) as f:
            f.write("\n".join(self.reasoning_trace))

    def build_reasoning_trace(self):
        """
        Build a comprehensive, step-by-step reasoning trace.
        This explains how to solve the puzzle systematically, following
        the same structure as CubeRollScene for consistency.
        """
        self.reasoning_trace = []

        # ====================================================================
        # Introduction
        # ====================================================================
        self.reasoning_trace.append(
            "**Question:** What was the color of the shape that appeared {} turns before the {}?".format(
                self.turns_back, self.shape_sequence[-1][0]
            )
        )
        self.reasoning_trace.append("")
        self.reasoning_trace.append("Let's solve this step by step.")
        self.reasoning_trace.append("")

        # ====================================================================
        # Scene description with timestamps
        # ====================================================================
        self.reasoning_trace.append("### Scene Description")
        self.reasoning_trace.append("")
        self.reasoning_trace.append(f"**Problem Type:** Color change sequence tracking")
        self.reasoning_trace.append(f"**Number of transformations:** {self.count}")
        self.reasoning_trace.append(f"**Random Seed:** {self.seed}")
        self.reasoning_trace.append("")
        self.reasoning_trace.append("**What happens in the video:**")
        self.reasoning_trace.append(
            f"A shape appears on screen and goes through {self.count} different forms, changing both its shape and color with each transformation. After all transformations are complete, the shape disappears and a question is presented asking about the color of a specific shape in the sequence."
        )
        self.reasoning_trace.append("")
        self.reasoning_trace.append("**Timeline of Events:**")
        self.reasoning_trace.append("")

        for event in self.scene_events:
            time_str = self.format_time(event["time"])
            self.reasoning_trace.append(f"At {time_str}, {event['description']}")

        self.reasoning_trace.append("")

        # ====================================================================
        # Step 1: Understanding the sequence
        # ====================================================================
        self.reasoning_trace.append("### Step 1: Understand the complete sequence")
        self.reasoning_trace.append("")
        self.reasoning_trace.append(
            f"The video shows a series of {self.count} transformations. Each transformation changes both the shape and color."
        )
        self.reasoning_trace.append(
            "To solve this problem, we need to track the entire sequence from beginning to end."
        )
        self.reasoning_trace.append("")
        self.reasoning_trace.append("**Complete sequence in chronological order:**")
        self.reasoning_trace.append("")

        for idx, (shape_name, color_name, position) in enumerate(
            self.shape_sequence, 1
        ):
            self.reasoning_trace.append(
                f"{idx}. **{color_name.capitalize()} {shape_name}**"
            )

        self.reasoning_trace.append("")

        # ====================================================================
        # Step 2: Count backwards to find target
        # ====================================================================
        self.reasoning_trace.append(
            "### Step 2: Count backwards to find the target shape"
        )
        self.reasoning_trace.append("")
        self.reasoning_trace.append(
            f'The question asks: "What was the color of the shape that appeared **{self.turns_back} turns before** the final_shape?"'
        )
        self.reasoning_trace.append("")
        self.reasoning_trace.append(
            "To find this, we need to count backwards from the final shape:"
        )
        self.reasoning_trace.append("")
        self.reasoning_trace.append(f"- Final shape position: **{self.count}**")
        self.reasoning_trace.append(f"- Count back: **{self.turns_back} turns**")
        self.reasoning_trace.append(
            f"- Calculation: {self.count} - {self.turns_back} = **{self.target_index + 1}**"
        )
        self.reasoning_trace.append("")
        self.reasoning_trace.append(
            f"So we need to find the shape at position **{self.target_index + 1}** in our sequence."
        )
        self.reasoning_trace.append("")

        # ====================================================================
        # Step 3: Determine the answer
        # ====================================================================
        self.reasoning_trace.append("### Step 3: Determine the answer")
        self.reasoning_trace.append("")
        target_shape_name = self.shape_sequence[self.target_index][0]
        target_color_name = self.shape_sequence[self.target_index][1]

        self.reasoning_trace.append(
            f"Looking at position {self.target_index + 1} in our sequence:"
        )
        self.reasoning_trace.append("")

        # Show context (shapes before and after)
        context_start = max(0, self.target_index - 1)
        context_end = min(len(self.shape_sequence), self.target_index + 2)

        for idx in range(context_start, context_end):
            shape_name, color_name, _ = self.shape_sequence[idx]
            if idx == self.target_index:
                self.reasoning_trace.append(
                    f"{idx + 1}. **{color_name.capitalize()} {shape_name}** ← This is our target!"
                )
            else:
                self.reasoning_trace.append(
                    f"{idx + 1}. {color_name.capitalize()} {shape_name}"
                )

        self.reasoning_trace.append("")
        self.reasoning_trace.append(
            f"At position {self.target_index + 1}, we find a **{target_color_name} {target_shape_name}**."
        )
        self.reasoning_trace.append("")

        # ====================================================================
        # Step 4: Verification
        # ====================================================================
        self.reasoning_trace.append("### Step 4: Verify the answer")
        self.reasoning_trace.append("")
        self.reasoning_trace.append(
            "Let's verify by counting forward from our target to the final shape:"
        )
        self.reasoning_trace.append("")

        verification_steps = []
        for i in range(self.target_index, len(self.shape_sequence)):
            shape_name, color_name, _ = self.shape_sequence[i]
            turns_from_target = i - self.target_index
            if i == self.target_index:
                verification_steps.append(
                    f"- Position {i + 1}: {color_name} {shape_name} (our target, 0 turns ahead)"
                )
            elif i == len(self.shape_sequence) - 1:
                verification_steps.append(
                    f"- Position {i + 1}: {color_name} {shape_name} (final shape, {turns_from_target} turns ahead)"
                )
            else:
                verification_steps.append(
                    f"- Position {i + 1}: {color_name} {shape_name} ({turns_from_target} turn{'s' if turns_from_target > 1 else ''} ahead)"
                )

        for step in verification_steps:
            self.reasoning_trace.append(step)

        self.reasoning_trace.append("")
        self.reasoning_trace.append(
            f"Counting from position {self.target_index + 1} to position {self.count}, we have exactly **{self.turns_back} turns**, confirming our answer."
        )
        self.reasoning_trace.append("")

        # ====================================================================
        # Final answer
        # ====================================================================
        self.reasoning_trace.append("### Final Answer")
        self.reasoning_trace.append("")
        self.reasoning_trace.append(
            f"The color of the shape that appeared {self.turns_back} turns before the final shape is **{self.answer}**."
        )
        self.reasoning_trace.append("")
        self.reasoning_trace.append(f"\\boxed{{{self.answer}}}")


# ============================================================================
# Main execution
# ============================================================================
if __name__ == "__main__":
    # Generate the color change video
    scene = color_change()
    scene.render()

    # ========================================================================
    # Move output file to questions directory with descriptive name
    # ========================================================================
    output = Path("manim_output/videos/1080p30/color_change.mp4")
    if output.exists():
        filename = f"color_change_n{scene.num_transforms}_seed{scene.seed}.mp4"
        shutil.move(str(output), f"questions/{filename}")
        print(f"✓ Video saved: questions/{filename}")
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
