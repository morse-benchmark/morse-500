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

class color_objects(Scene):
    """
    A scene that generates a color matching puzzle:
    - Shows a sequence of colored shapes appearing and disappearing
    - Shows a final shape that transitions through multiple colors
    - User must identify which initial shape had the same color as a specific color of the final shape
    - Generates question video, solution, and detailed reasoning trace
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Set random seed for reproducibility of the puzzle
        self.seed = random.randint(1000, 9999)
        random.seed(self.seed)

        # Parameters from environment variables (with defaults)
        self.difficulty = int(os.getenv("DIFFICULTY", 1))

        # Initialize reasoning trace storage
        self.reasoning_trace = []

        # Track timing for reasoning trace using Manim's internal video time
        # This will be populated with events as the scene is constructed
        self.scene_events = []

    def log_event(self, description):
        """
        Log a scene event with video timestamp.
        Uses Manim's internal renderer.time which tracks actual video playback time,
        not wall-clock execution time.

        Args:
            description: Human-readable description of what's happening in the scene
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
            String formatted as "M:SS" (e.g., "0:37")
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
        # Define all possible shapes and colors
        # ====================================================================
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

        # Define all possible colors with consistent mapping
        all_colors = [RED, BLUE, GREEN, YELLOW, PURPLE, ORANGE, PINK, TEAL, MAROON, GRAY]

        # ====================================================================
        # Set difficulty-based parameters
        # ====================================================================
        # Difficulty controls the number of initial shapes to remember
        if self.difficulty == 1:
            num_shapes = 4  # Easy: 4 shapes
        elif self.difficulty == 2:
            num_shapes = 6  # Medium: 6 shapes
        elif self.difficulty == 3:
            num_shapes = 8  # Hard: 8 shapes
        else:
            num_shapes = 4  # Default to easy
        
        # ====================================================================
        # Generate puzzle configuration
        # ====================================================================
        # Randomly select shapes and colors for initial sequence
        selected_shape_data = random.sample(all_shapes_with_names, num_shapes)
        selected_shapes = [shape for shape, name in selected_shape_data]
        shape_names = [name for shape, name in selected_shape_data]
        selected_colors = random.sample(all_colors, num_shapes)

        # Choose which transition color position will be the target (1-5)
        # This is the ordinal position (first, second, third, etc.) that the user will be asked about
        target_color_index = random.randint(1, 5)

        # Generate 5 DISTINCT transition colors for the final shape
        # The target position must use a color from the initial shapes (to ensure a valid answer)
        target_color = random.choice(selected_colors)

        # Build list of 5 distinct transition colors
        transition_colors = [None] * 5
        transition_colors[target_color_index - 1] = target_color  # Place target color at specified position

        # Fill remaining positions with distinct colors (different from target)
        available_colors = all_colors.copy()
        available_colors.remove(target_color)  # Ensure all 5 colors are distinct

        for i in range(5):
            if transition_colors[i] is None:  # If position not filled yet
                chosen_color = random.choice(available_colors)
                transition_colors[i] = chosen_color
                available_colors.remove(chosen_color)  # Maintain distinctness

        # ====================================================================
        # Create color name mapping for reasoning trace
        # ====================================================================
        color_name_map = {
            RED: "red",
            BLUE: "blue",
            GREEN: "green",
            YELLOW: "yellow",
            PURPLE: "purple",
            ORANGE: "orange",
            PINK: "pink",
            TEAL: "teal",
            MAROON: "maroon",
            GRAY: "gray"
        }

        # ====================================================================
        # Display initial sequence of colored shapes
        # ====================================================================
        # Each shape appears briefly, then fades out
        # This creates the memory challenge for the user
        shape_color_pairs = []  # Store for later reference

        for i, (shape, color) in enumerate(zip(selected_shapes, selected_colors)):
            # Configure shape appearance
            shape.set_fill(color, opacity=0.8)
            shape_color_pairs.append((shape, color, shape_names[i]))

            # Log event BEFORE animation
            color_name = color_name_map[color]
            self.log_event(f"A {color_name} {shape_names[i]} begins to appear")

            # Animate shape creation
            self.play(Create(shape), run_time=0.4)

            # Log event AFTER creation completes
            self.log_event(f"The {color_name} {shape_names[i]} is fully visible")

            # Fade out the shape
            self.play(FadeOut(shape), run_time=0.2)

            # Log event AFTER fadeout completes
            self.log_event(f"The {color_name} {shape_names[i]} has faded out")

        # ====================================================================
        # Create the final shape (distinct from initial shapes)
        # ====================================================================
        # Final shape should be different from all initial shapes to avoid confusion
        remaining_shapes_with_names = [item for item in all_shapes_with_names if item not in selected_shape_data]

        # Select a shape that wasn't in the initial sequence
        if remaining_shapes_with_names:
            final_shape_data = random.choice(remaining_shapes_with_names)
            final_shape = final_shape_data[0].copy()
            final_shape_name = final_shape_data[1]
        else:
            # Fallback (shouldn't happen with current setup - we have 9 shapes and use max 8)
            final_shape = Circle()
            final_shape_name = "circle"

        # ====================================================================
        # Display final shape with initial color
        # ====================================================================
        # The final shape starts with the first transition color
        final_shape.set_fill(transition_colors[0], opacity=0.8)

        # Log BEFORE animation
        self.log_event(f"A {final_shape_name} appears with {color_name_map[transition_colors[0]]} color")

        # Animate creation
        self.play(Create(final_shape), run_time=0.4)

        # Log AFTER creation
        self.log_event(f"The {final_shape_name} is fully visible in {color_name_map[transition_colors[0]]}")

        # Hold for a moment
        self.wait(0.4)

        # ====================================================================
        # Animate color transitions on final shape
        # ====================================================================
        # The final shape will cycle through all 5 colors (starting color + 4 transitions)
        for i, col in enumerate(transition_colors[1:], start=1):
            prev_color = color_name_map[transition_colors[i-1]]
            curr_color = color_name_map[col]

            # Log BEFORE transition
            self.log_event(f"The {final_shape_name} begins transitioning from {prev_color} to {curr_color}")

            # Animate color change
            self.play(final_shape.animate.set_fill(col, opacity=0.8), run_time=0.4)

            # Log AFTER transition completes
            self.log_event(f"The {final_shape_name} is now {curr_color}")

        # ====================================================================
        # Remove final shape before showing question
        # ====================================================================
        self.wait(0.5)

        # Log BEFORE fadeout
        self.log_event(f"The {final_shape_name} begins to fade out")

        self.play(FadeOut(final_shape), run_time=0.5)

        # Log AFTER fadeout
        self.log_event(f"The {final_shape_name} has disappeared")

        # ====================================================================
        # Determine the answer
        # ====================================================================
        # Create color map for display names (capitalized for UI)
        color_map = {
            RED: "Red",
            BLUE: "Blue",
            GREEN: "Green",
            YELLOW: "Yellow",
            PURPLE: "Purple",
            ORANGE: "Orange",
            PINK: "Pink",
            TEAL: "Teal",
            MAROON: "Maroon",
            GRAY: "Gray"
        }

        # Find which original shape had the target color
        answer_shape = None
        for i, (shape, color, shape_name) in enumerate(shape_color_pairs):
            if color == target_color:
                answer_shape = shape_name
                break

        # Since we guaranteed the target color exists in initial shapes, answer should always be found
        if answer_shape is None:
            answer_shape = "error"  # This should never happen with current logic

        # Define ordinal names for the question
        ordinal_numbers = ["first", "second", "third", "fourth", "fifth"]
        ordinal = ordinal_numbers[target_color_index - 1]

        # ====================================================================
        # Display the question
        # ====================================================================
        question_text = f"Which shape had the same color as \n the {ordinal} color of the final shape?"
        question = Text(question_text, font_size=30, weight=BOLD).move_to(UP * 2.5)

        # ====================================================================
        # Create color palette for reference
        # ====================================================================
        # Show all colors that appeared in the scene (no highlighting to avoid giving away answer)
        palette_squares = []
        palette_labels = []
        palette_y = 0.5
        palette_spacing = 1.0

        # Combine all colors and remove duplicates while preserving order
        scene_colors = selected_colors + transition_colors
        unique_colors = list(dict.fromkeys(scene_colors))

        # Calculate starting position to center the palette
        total_palette_width = (len(unique_colors) - 1) * palette_spacing
        start_x = -total_palette_width / 2

        for i, color in enumerate(unique_colors):
            # Create colored square
            square = Square(
                side_length=0.4,
                fill_opacity=1,
                color=color,
                stroke_width=2,
                stroke_color=WHITE
            ).move_to([start_x + i * palette_spacing, palette_y, 0])

            # Create label below square
            label = Text(
                color_map[color],
                font_size=20,
                color=WHITE
            ).move_to([start_x + i * palette_spacing, palette_y - 0.5, 0])

            # Add to palette list
            palette_squares.append(square)
            palette_labels.append(label)

        # ====================================================================
        # Animate question and palette appearance
        # ====================================================================
        # Log BEFORE question appears
        self.log_event("Question text begins to appear")

        self.play(FadeIn(question), run_time=0.8)

        # Log AFTER question appears
        self.log_event(f"Question displayed: 'Which shape had the same color as the {ordinal} color of the final shape?'")

        self.wait(0.3)

        # Log BEFORE palette appears
        self.log_event("Color palette begins to appear")

        # Animate palette squares appearing one by one
        for square, label in zip(palette_squares, palette_labels):
            self.play(
                FadeIn(square, scale=0.8),
                FadeIn(label, shift=UP*0.2),
                run_time=0.15
            )

        # Log AFTER palette appears
        self.log_event(f"Color palette fully displayed with {len(unique_colors)} colors")

        # Show the answer instruction after a pause
        self.wait(2)

        note_text = Text(
            f"Return the answer as a shape name in lower case.",
            font_size=32,
            color=WHITE
        ).move_to(DOWN * 2)

        # Log BEFORE instruction appears
        self.log_event("Answer instruction begins to appear")

        self.play(FadeIn(note_text, shift=UP*0.3), run_time=0.8)

        # Log AFTER instruction appears
        self.log_event("Answer instruction displayed: 'Return the answer as a shape name in lower case'")

        self.wait(3)

        # Log final wait
        self.log_event("Video ends")

        # ====================================================================
        # Store data for reasoning trace generation
        # ====================================================================
        # Save these for the reasoning trace builder method
        self.final_shape_name = final_shape_name
        self.transition_colors = transition_colors
        self.target_color_index = target_color_index
        self.target_color = target_color
        self.ordinal = ordinal
        self.shape_color_pairs = shape_color_pairs
        self.answer_shape = answer_shape
        self.color_name_map = color_name_map
        self.ordinal_numbers = ordinal_numbers

        # ====================================================================
        # Generate comprehensive reasoning trace
        # ====================================================================
        self.build_reasoning_trace()

        # ====================================================================
        # Save output files
        # ====================================================================
        # Question text for file
        question_for_file = f"Which shape had the same color as the {ordinal} color of the final shape?\nReturn the answer as a shape name in lower case."

        # Solution file (just the answer)
        with open(f"solutions/color_objects_d{self.difficulty}_seed{self.seed}.txt", "w") as f:
            f.write(str(answer_shape))

        # Question text file
        with open(f"question_text/color_objects_d{self.difficulty}_seed{self.seed}.txt", "w") as f:
            f.write(question_for_file)

        # Reasoning trace file (generated by build_reasoning_trace)
        with open(f"reasoning_traces/color_objects_d{self.difficulty}_seed{self.seed}.txt", "w") as f:
            f.write("\n".join(self.reasoning_trace))

    def build_reasoning_trace(self):
        """
        Build a comprehensive, step-by-step reasoning trace.
        This explains how to solve the puzzle systematically following the CubeRollScene pattern.
        """
        self.reasoning_trace = []

        # ====================================================================
        # Introduction and Question
        # ====================================================================
        self.reasoning_trace.append(f"**Question:** Which shape had the same color as the {self.ordinal} color of the final shape?")
        self.reasoning_trace.append("")
        self.reasoning_trace.append("Let's solve this step by step.")
        self.reasoning_trace.append("")

        # ====================================================================
        # Scene Description with Timestamps
        # ====================================================================
        self.reasoning_trace.append("### Scene Description")
        self.reasoning_trace.append("")
        self.reasoning_trace.append("Here is a chronological description of everything that happened in the video:")
        self.reasoning_trace.append("")

        for event in self.scene_events:
            time_str = self.format_time(event['time'])
            self.reasoning_trace.append(f"At {time_str}, {event['description']}")

        self.reasoning_trace.append("")

        # ====================================================================
        # Step 1: Understand the initial sequence
        # ====================================================================
        self.reasoning_trace.append("### Step 1: Recall the initial sequence of shapes")
        self.reasoning_trace.append("")
        self.reasoning_trace.append(f"At the beginning of the video, {len(self.shape_color_pairs)} shapes appeared one by one, each with a specific color:")
        self.reasoning_trace.append("")

        for i, (shape, color, shape_name) in enumerate(self.shape_color_pairs, 1):
            self.reasoning_trace.append(f"{i}. **{shape_name.capitalize()}**: {self.color_name_map[color]}")

        self.reasoning_trace.append("")
        self.reasoning_trace.append("These shapes appeared briefly and then faded out. We need to remember these color-shape pairings.")
        self.reasoning_trace.append("")

        # ====================================================================
        # Step 2: Identify the final shape's color sequence
        # ====================================================================
        self.reasoning_trace.append("### Step 2: Identify the colors of the final shape")
        self.reasoning_trace.append("")
        self.reasoning_trace.append(f"After the initial sequence, a **{self.final_shape_name}** appeared and cycled through {len(self.transition_colors)} different colors:")
        self.reasoning_trace.append("")

        for idx, col in enumerate(self.transition_colors, start=1):
            self.reasoning_trace.append(f"- **{self.ordinal_numbers[idx-1].capitalize()} color**: {self.color_name_map[col]}")

        self.reasoning_trace.append("")

        # ====================================================================
        # Step 3: Identify the target color
        # ====================================================================
        self.reasoning_trace.append("### Step 3: Identify the target color")
        self.reasoning_trace.append("")
        self.reasoning_trace.append(f"The question asks about the **{self.ordinal} color** of the final {self.final_shape_name}.")
        self.reasoning_trace.append("")
        self.reasoning_trace.append(f"Looking at our list above, the {self.ordinal} color was: **{self.color_name_map[self.target_color]}**")
        self.reasoning_trace.append("")

        # ====================================================================
        # Step 4: Match with initial shapes
        # ====================================================================
        self.reasoning_trace.append("### Step 4: Find which initial shape had this color")
        self.reasoning_trace.append("")
        self.reasoning_trace.append(f"Now we need to look back at the initial sequence and find which shape was **{self.color_name_map[self.target_color]}**.")
        self.reasoning_trace.append("")
        self.reasoning_trace.append("Checking each shape from the initial sequence:")
        self.reasoning_trace.append("")

        for i, (shape, color, shape_name) in enumerate(self.shape_color_pairs):
            if color == self.target_color:
                self.reasoning_trace.append(f"- {shape_name.capitalize()}: {self.color_name_map[color]} ← **This matches!**")
            else:
                self.reasoning_trace.append(f"- {shape_name.capitalize()}: {self.color_name_map[color]}")

        self.reasoning_trace.append("")

        # ====================================================================
        # Final Answer
        # ====================================================================
        self.reasoning_trace.append("### Final Answer")
        self.reasoning_trace.append("")
        self.reasoning_trace.append(f"The shape that had {self.color_name_map[self.target_color]} color in the initial sequence was the **{self.answer_shape}**.")
        self.reasoning_trace.append("")
        self.reasoning_trace.append(f"\\boxed{{{self.answer_shape}}}")

# ============================================================================
# Main execution
# ============================================================================
# Generate the color objects video
scene = color_objects()
scene.render()

# ============================================================================
# Move output file to questions directory with descriptive name
# ============================================================================
# Try multiple possible output paths (Manim's output structure can vary)
possible_paths = [
    Path("manim_output/videos/color_objects/1080p30/color_objects.mp4"),
    Path("manim_output/videos/1080p30/color_objects.mp4"),
    Path("manim_output/videos/1080p30/1080p30/color_objects.mp4")
]

output_found = False
for output_path in possible_paths:
    if output_path.exists():
        filename = f"color_objects_d{scene.difficulty}_seed{scene.seed}.mp4"
        shutil.move(str(output_path), f"questions/{filename}")
        output_found = True
        print(f"✓ Video saved: questions/{filename}")
        break

if not output_found:
    # Debug: Print what files actually exist
    videos_dir = Path("manim_output/videos")
    if videos_dir.exists():
        print(f"Available folders in videos/: {list(videos_dir.iterdir())}")
        for folder in videos_dir.iterdir():
            if folder.is_dir():
                print(f"Contents of {folder}: {list(folder.iterdir())}")
                for subfolder in folder.iterdir():
                    if subfolder.is_dir():
                        print(f"Files in {subfolder}: {list(subfolder.iterdir())}")
    else:
        print("manim_output/videos directory doesn't exist")

# ============================================================================
# Final cleanup
# ============================================================================
if os.path.exists("manim_output"):
    shutil.rmtree("manim_output")

