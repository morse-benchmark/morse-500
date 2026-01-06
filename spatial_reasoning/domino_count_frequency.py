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

class domino_count_frequency(Scene):
    """
    A FREQUENCY-parameterized scene that generates a domino counting puzzle:
    - Shows dominoes appearing one by one in different colors
    - Dominoes fall in a chain reaction
    - User must count how many dominoes were a specific color
    - FREQUENCY parameter (0.0-1.0) controls packing density (number of dominoes in fixed area):
      * FREQUENCY=0.0 → 5 dominoes (sparse, very easy)
      * FREQUENCY=0.5 → 38 dominoes (medium density)
      * FREQUENCY=1.0 → 70 dominoes (dense, very hard)
    - Generates question video, solution, and detailed reasoning trace
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Set random seed for reproducibility
        self.seed = random.randint(1000, 9999)
        random.seed(self.seed)

        # ====================================================================
        # FREQUENCY Parameter Implementation
        # ====================================================================
        # Read FREQUENCY from environment variable (0.0 to 1.0)
        # Default to 0.5 (medium difficulty) if not specified
        frequency_input = float(os.getenv("FREQUENCY", 0.5))

        # Clamp FREQUENCY to valid range [0.0, 1.0]
        self.frequency_param = max(0.0, min(1.0, frequency_input))

        # FREQUENCY controls spacing pattern complexity (NOT domino count)
        # Number of dominoes is FIXED, spacing pattern frequency varies
        # - FREQUENCY=0.0 → Uniform spacing (simple, regular pattern)
        # - FREQUENCY=0.5 → Medium spacing variation
        # - FREQUENCY=1.0 → High frequency spacing oscillation (complex pattern)

        # Fixed number of dominoes for FREQUENCY variant (same as DENSITY=0.5)
        self.num_dominoes = 38  # Fixed at medium count

        # Initialize reasoning trace storage
        self.reasoning_trace = []

        # Track scene events with timestamps for reasoning trace
        # This will store tuples of (time, description)
        self.scene_events = []

        # Track color distribution for reasoning
        self.color_distribution = {}

        # Store domino appearance details for reasoning
        self.domino_details = []

    def log_event(self, description):
        """
        Log a scene event with video timestamp.
        Uses Manim's internal renderer.time which tracks actual video playback time,
        not wall-clock execution time.

        Args:
            description: String describing what happened in the scene
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

    def construct(self):
        """
        Main scene construction method.
        This is called by Manim to build and render the entire scene.
        """
        # ====================================================================
        # Setup: Domino layout parameters - FIXED DOMINO COUNT
        # ====================================================================
        # FREQUENCY controls spacing pattern complexity (NOT domino count)
        # Number of dominoes is FIXED, spacing pattern varies by frequency
        domino_width = 0.2
        domino_height = 0.6

        # Base spacing values (used for uniform grid at low frequency)
        base_spacing_x = 0.3
        base_spacing_y = 0.8

        # Generate spacing pattern based on FREQUENCY
        # Low frequency = uniform spacing (regular grid)
        # High frequency = irregular spacing (complex pattern)

        # Fixed usable screen area
        screen_width = 12  # Fixed physical screen width for domino layout
        max_per_row = int(screen_width / base_spacing_x)
        num_rows = math.ceil(self.num_dominoes / max_per_row)
        dominoes_per_row = math.ceil(self.num_dominoes / num_rows)

        # Create position jitter based on FREQUENCY
        # Low freq: no jitter (regular grid)
        # High freq: high jitter (irregular pattern)
        max_jitter_x = self.frequency_param * 0.25  # 0 to 0.25 units
        max_jitter_y = self.frequency_param * 0.5   # 0 to 0.5 units

        # Pre-generate random jitter for each domino position
        position_jitters = []
        for i in range(self.num_dominoes):
            jitter_x = random.uniform(-max_jitter_x, max_jitter_x)
            jitter_y = random.uniform(-max_jitter_y, max_jitter_y)
            position_jitters.append((jitter_x, jitter_y))

        # ====================================================================
        # Setup: Choose target color and positions
        # ====================================================================
        # Randomly select how many dominoes will be the target color
        # Scale with SIZE: at low SIZE, keep count very low; at high SIZE, allow more
        # Formula ensures answer is reasonable for counting
        max_target = max(1, min(int(self.num_dominoes * 0.4), 20))
        answer = random.randint(1, max_target)

        # Define available colors
        colors = [RED, BLUE, GREEN, YELLOW, PURPLE, ORANGE, TEAL, PINK]

        # Randomly choose which color to ask about
        color_choice = random.choice(colors)

        # Randomly select which positions will have the target color
        # This ensures they're distributed throughout the sequence
        choice_positions = set(random.sample(range(self.num_dominoes), answer))

        self.log_event(f"Scene setup complete: {self.num_dominoes} dominoes total (SIZE={self.frequency_param:.2f}), {answer} will be target color")

        # Create a color list without our target color for the other dominoes
        other_colors = colors.copy()
        other_colors.remove(color_choice)

        # Initialize color distribution tracking
        # This will help us verify our answer in the reasoning trace
        self.color_distribution = {color: 0 for color in colors}

        # ====================================================================
        # Animation: Display dominoes one by one
        # ====================================================================
        self.log_event("Starting domino appearance sequence")

        dominoes = []
        for i in range(self.num_dominoes):
            # ------------------------------------------------------------
            # Calculate position for this domino
            # ------------------------------------------------------------
            # Determine which row and column this domino belongs to
            row = i // dominoes_per_row
            col = i % dominoes_per_row

            # Center the dominoes in each row (handles partial last row)
            actual_dominoes_in_row = min(dominoes_per_row, self.num_dominoes - row * dominoes_per_row)
            total_width = (actual_dominoes_in_row - 1) * base_spacing_x
            start_x = -total_width / 2

            # Calculate final position on screen (base grid position)
            x_base = start_x + col * base_spacing_x
            y_base = (num_rows - 1) * base_spacing_y / 2 - row * base_spacing_y

            # Apply FREQUENCY-based jitter to create irregular spacing
            jitter_x, jitter_y = position_jitters[i]
            x = x_base + jitter_x
            y = y_base + jitter_y

            # ------------------------------------------------------------
            # Assign color to this domino
            # ------------------------------------------------------------
            if i in choice_positions:
                # This is one of our target color dominoes
                color = color_choice
            else:
                # Choose a random non-target color
                color = random.choice(other_colors)

            # Track color distribution for reasoning trace
            self.color_distribution[color] += 1

            # Store details for reasoning trace
            self.domino_details.append({
                'index': i + 1,  # 1-indexed for human readability
                'color': color,
                'position': (row, col),
                'is_target': i in choice_positions
            })

            # ------------------------------------------------------------
            # Create and animate the domino
            # ------------------------------------------------------------
            domino = Rectangle(
                width=domino_width,
                height=domino_height,
                fill_opacity=1,
                color=color,
                stroke_width=1,
                stroke_color=WHITE
            ).move_to([x, y, 0])
            dominoes.append(domino)

            # Animate its appearance with a quick fade-in
            # Using very short run_time to keep video concise
            self.play(FadeIn(domino), run_time=0.05)

            # Log milestone events (every 10th domino to avoid log spam)
            if (i + 1) % 10 == 0 or i == self.num_dominoes - 1:
                self.log_event(f"Domino {i + 1} of {self.num_dominoes} appears")

        # Brief pause to let viewer see all dominoes
        self.wait(0.5)
        self.log_event("All dominoes have appeared on screen")

        # ====================================================================
        # Animation: Domino chain reaction (falling sequence)
        # ====================================================================
        self.log_event("Starting domino fall sequence")

        for i, domino in enumerate(dominoes):
            # ------------------------------------------------------------
            # Calculate fall direction and pivot point
            # ------------------------------------------------------------
            # Determine which row and column for visual variety
            row = i // dominoes_per_row
            col = i % dominoes_per_row

            # Alternate falling direction for more interesting visual effect
            # Even columns fall right, odd columns fall left
            angle = -PI/3 if (col % 2 == 0) else PI/3
            pivot = domino.get_bottom()  # Rotate around bottom edge

            # Animate the fall with quick rotation
            self.play(Rotate(domino, angle=angle, about_point=pivot), run_time=0.08)

            # Log milestone events during falling
            if (i + 1) % 10 == 0 or i == self.num_dominoes - 1:
                self.log_event(f"Domino {i + 1} has fallen")

        # Pause to appreciate the fallen dominoes
        self.wait(1)
        self.log_event("Domino fall sequence complete")

        # ====================================================================
        # Animation: Clear the screen
        # ====================================================================
        self.log_event("Starting domino removal animation")

        # Make all dominoes disappear upward with a nice effect
        self.play(
            *[FadeOut(domino, shift=UP*0.5) for domino in dominoes],
            run_time=1
        )

        self.wait(0.5)
        self.log_event("All dominoes have been removed from screen")

        # ====================================================================
        # Setup: Color palette and names
        # ====================================================================
        # Create mapping from Manim color objects to human-readable names
        color_map = {
            RED: "Red",
            BLUE: "Blue",
            GREEN: "Green",
            YELLOW: "Yellow",
            PURPLE: "Purple",
            ORANGE: "Orange",
            TEAL: "Teal",
            PINK: "Pink",
        }

        # ====================================================================
        # Display: Question text
        # ====================================================================
        question_text = f"How many dominoes were {color_map[color_choice]}?"
        question = Text(question_text, font_size=36, weight=BOLD).move_to(UP * 2.5)

        self.log_event("Question text appears on screen")
        self.play(FadeIn(question), run_time=0.8)
        self.wait(0.3)

        # ====================================================================
        # Display: Color palette with labels
        # ====================================================================
        palette_squares = []
        palette_labels = []
        palette_y = 0.5
        palette_spacing = 1.2

        # Calculate starting position to center the palette horizontally
        total_palette_width = (len(colors) - 1) * palette_spacing
        start_x = -total_palette_width / 2

        self.log_event("Color palette starts appearing")

        for i, color in enumerate(colors):
            # ------------------------------------------------------------
            # Create colored square
            # ------------------------------------------------------------
            square = Square(
                side_length=0.4,
                fill_opacity=1,
                color=color,
                stroke_width=2,
                stroke_color=WHITE
            ).move_to([start_x + i * palette_spacing, palette_y, 0])

            # ------------------------------------------------------------
            # Create label below square
            # ------------------------------------------------------------
            label = Text(
                color_map[color],
                font_size=20,
                color=WHITE
            ).move_to([start_x + i * palette_spacing, palette_y - 0.5, 0])

            # ------------------------------------------------------------
            # Highlight the target color with yellow border
            # ------------------------------------------------------------
            if color == color_choice:
                highlight = Square(
                    side_length=0.5,
                    fill_opacity=0,
                    stroke_width=4,
                    stroke_color=YELLOW
                ).move_to(square.get_center())
                # Group square and highlight together
                palette_squares.append(VGroup(square, highlight))
            else:
                palette_squares.append(square)

            palette_labels.append(label)

        # Animate palette squares and labels appearing one by one
        for square, label in zip(palette_squares, palette_labels):
            self.play(
                FadeIn(square, scale=0.8),  # Squares scale up as they appear
                FadeIn(label, shift=UP*0.2),  # Labels slide up slightly
                run_time=0.2
            )

        self.log_event(f"Color palette fully displayed with {color_map[color_choice]} highlighted")

        # ====================================================================
        # Display: Answer instruction
        # ====================================================================
        # Pause before showing instruction
        self.wait(2)

        note_text = Text(
            f"Return the answer as a number.",
            font_size=32,
            color=color_choice
        ).move_to(DOWN * 2)

        self.log_event("Answer instruction appears")
        self.play(FadeIn(note_text, shift=UP*0.3), run_time=0.8)
        self.wait(3)
        self.log_event("Scene ends with question and instruction visible")

        # ====================================================================
        # Generate reasoning trace
        # ====================================================================
        self.color_choice = color_choice
        self.color_choice_name = color_map[color_choice]
        self.answer = answer
        self.build_reasoning_trace(color_map)

        # ====================================================================
        # Save output files
        # ====================================================================
        # Solution file (just the answer number)
        with open(f"solutions/domino_count_n{self.num_dominoes}_density{self.frequency_param:.2f}_seed{self.seed}.txt", "w") as f:
            f.write(str(answer))

        # Question text file (plain text version)
        with open(f"question_text/domino_count_n{self.num_dominoes}_density{self.frequency_param:.2f}_seed{self.seed}.txt", "w") as f:
            f.write(f"How many dominoes were {color_map[color_choice].lower()}?\nReturn the answer as a number.")

        # Reasoning trace file (detailed step-by-step solution)
        with open(f"reasoning_traces/domino_count_n{self.num_dominoes}_density{self.frequency_param:.2f}_seed{self.seed}.txt", "w") as f:
            f.write("\n".join(self.reasoning_trace))

    def build_reasoning_trace(self, color_map):
        """
        Build a comprehensive, step-by-step reasoning trace.
        This explains how to solve the puzzle systematically.

        Args:
            color_map: Dictionary mapping Manim color objects to names
        """
        self.reasoning_trace = []

        # ====================================================================
        # Introduction
        # ====================================================================
        self.reasoning_trace.append(f"**Question:** How many dominoes were {self.color_choice_name}?")
        self.reasoning_trace.append("")
        self.reasoning_trace.append("Let's solve this step by step by carefully observing the video.")
        self.reasoning_trace.append("")

        # ====================================================================
        # SIZE Parameter Info
        # ====================================================================
        self.reasoning_trace.append("### FREQUENCY Parameter")
        self.reasoning_trace.append("")
        self.reasoning_trace.append(f"- FREQUENCY: {self.frequency_param:.2f} (spacing pattern complexity)")
        self.reasoning_trace.append(f"- Number of dominoes: {self.num_dominoes} (fixed count)")
        self.reasoning_trace.append(f"- Difficulty: {'Easy' if self.frequency_param < 0.3 else 'Medium' if self.frequency_param < 0.7 else 'Hard'}")
        self.reasoning_trace.append("")

        # ====================================================================
        # Scene description with timestamps
        # ====================================================================
        self.reasoning_trace.append("### Scene Description")
        self.reasoning_trace.append("")
        self.reasoning_trace.append("The video shows the following sequence of events:")
        self.reasoning_trace.append("")

        for event in self.scene_events:
            time_str = self.format_time(event['time'])
            self.reasoning_trace.append(f"- At {time_str}: {event['description']}")

        self.reasoning_trace.append("")

        # ====================================================================
        # Step 1: Understanding the problem
        # ====================================================================
        self.reasoning_trace.append("### Step 1: Understand the problem")
        self.reasoning_trace.append("")
        self.reasoning_trace.append(f"We need to count how many dominoes are **{self.color_choice_name}** out of **{self.num_dominoes} total dominoes**.")
        self.reasoning_trace.append("")
        self.reasoning_trace.append("The dominoes appear one at a time, each with a specific color. After they all appear, they fall in sequence. Our task is to remember and count how many had the target color.")
        self.reasoning_trace.append("")

        # ====================================================================
        # Step 2: Color distribution overview
        # ====================================================================
        self.reasoning_trace.append("### Step 2: Observe the color distribution")
        self.reasoning_trace.append("")
        self.reasoning_trace.append("The dominoes appear in the following colors:")
        self.reasoning_trace.append("")

        # Sort colors by count for clearer presentation
        sorted_colors = sorted(self.color_distribution.items(),
                              key=lambda x: x[1],
                              reverse=True)

        for color, count in sorted_colors:
            if count > 0:
                color_name = color_map[color]
                marker = " ← **Target color**" if color == self.color_choice else ""
                self.reasoning_trace.append(f"- **{color_name}**: {count} dominoes{marker}")

        self.reasoning_trace.append("")

        # ====================================================================
        # Step 3: Detailed domino-by-domino tracking
        # ====================================================================
        self.reasoning_trace.append("### Step 3: Track each domino as it appears")
        self.reasoning_trace.append("")

        # Count target color dominoes as we go
        target_count = 0

        # Show detailed tracking in groups of 10 for readability
        for i in range(0, self.num_dominoes, 10):
            # Get dominoes in this group (up to 10)
            group = self.domino_details[i:min(i+10, self.num_dominoes)]

            self.reasoning_trace.append(f"**Dominoes {i+1} to {min(i+10, self.num_dominoes)}:**")

            group_summary = []
            for detail in group:
                color_name = color_map[detail['color']]

                if detail['is_target']:
                    target_count += 1
                    marker = f" ✓ ({self.color_choice_name} #{target_count})"
                else:
                    marker = ""

                group_summary.append(f"  {detail['index']}. {color_name}{marker}")

            self.reasoning_trace.extend(group_summary)
            self.reasoning_trace.append("")

        # ====================================================================
        # Step 4: Verification by counting
        # ====================================================================
        self.reasoning_trace.append("### Step 4: Count the target color dominoes")
        self.reasoning_trace.append("")

        # Find all target dominoes
        target_indices = [d['index'] for d in self.domino_details if d['is_target']]

        self.reasoning_trace.append(f"The dominoes that are **{self.color_choice_name}** appear at positions:")
        self.reasoning_trace.append("")

        # Format indices nicely (e.g., "1, 3, 7, 12, ...")
        # Break into lines if too many
        if len(target_indices) <= 20:
            self.reasoning_trace.append(f"  {', '.join(map(str, target_indices))}")
        else:
            # Show in groups of 20
            for i in range(0, len(target_indices), 20):
                group = target_indices[i:i+20]
                self.reasoning_trace.append(f"  {', '.join(map(str, group))}")

        self.reasoning_trace.append("")
        self.reasoning_trace.append(f"Counting these positions: **{len(target_indices)} dominoes** are {self.color_choice_name}.")
        self.reasoning_trace.append("")

        # ====================================================================
        # Step 5: Alternative counting method (verification)
        # ====================================================================
        self.reasoning_trace.append("### Step 5: Verify using color distribution")
        self.reasoning_trace.append("")
        self.reasoning_trace.append("We can verify by checking the color distribution table:")
        self.reasoning_trace.append("")

        total_check = 0
        for color, count in sorted_colors:
            if count > 0:
                color_name = color_map[color]
                total_check += count
                if color == self.color_choice:
                    self.reasoning_trace.append(f"- {color_name}: **{count}** ← This is our answer")
                else:
                    self.reasoning_trace.append(f"- {color_name}: {count}")

        self.reasoning_trace.append("")
        self.reasoning_trace.append(f"Total dominoes: {total_check} (should equal {self.num_dominoes}) ✓")
        self.reasoning_trace.append("")

        # ====================================================================
        # Final answer
        # ====================================================================
        self.reasoning_trace.append("### Final Answer")
        self.reasoning_trace.append("")
        self.reasoning_trace.append(f"By carefully tracking each domino as it appears and counting those that are {self.color_choice_name}, we find:")
        self.reasoning_trace.append("")
        self.reasoning_trace.append(f"**{self.answer}** dominoes were {self.color_choice_name}.")
        self.reasoning_trace.append("")
        self.reasoning_trace.append(f"\\boxed{{{self.answer}}}")

# ============================================================================
# Main execution
# ============================================================================
if __name__ == "__main__":
    # Generate the domino counting video
    scene = domino_count_frequency()
    scene.render()

    # ========================================================================
    # Move the output file to questions directory with descriptive name
    # ========================================================================
    # Manim creates a folder structure under media_dir
    output = Path("manim_output/videos/1080p30/domino_count_frequency.mp4")

    if output.exists():
        filename = f"domino_count_n{scene.num_dominoes}_frequency{scene.frequency_param:.2f}_seed{scene.seed}.mp4"
        shutil.move(str(output), f"questions/{filename}")
        print(f"✓ Video saved: questions/{filename}")
        jitter_max = scene.frequency_param * 0.25
        print(f"✓ FREQUENCY parameter: {scene.frequency_param:.2f} (position jitter: ±{jitter_max:.3f} units, {scene.num_dominoes} dominoes)")
        print(f"✓ Solution saved: solutions/domino_count_n{scene.num_dominoes}_frequency{scene.frequency_param:.2f}_seed{scene.seed}.txt")
        print(f"✓ Question saved: question_text/domino_count_n{scene.num_dominoes}_frequency{scene.frequency_param:.2f}_seed{scene.seed}.txt")
        print(f"✓ Reasoning saved: reasoning_traces/domino_count_n{scene.num_dominoes}_frequency{scene.frequency_param:.2f}_seed{scene.seed}.txt")
    else:
        # Debug: Print what files actually exist if video not found
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
    # Final cleanup: Remove temporary Manim output directory
    # ========================================================================
    if os.path.exists("manim_output"):
        shutil.rmtree("manim_output")
        print("✓ Cleaned up temporary files")
