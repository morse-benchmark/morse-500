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


class duration_3d(ThreeDScene):
    """
    A 3D scene that generates a duration tracking puzzle:
    - Shows different 3D shapes appearing one at a time
    - Each shape remains visible for a random duration
    - User must determine which shapes appeared longest to shortest
    - Generates question video, solution, and reasoning trace
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Set random seed for reproducibility
        self.seed = random.randint(1000, 9999)
        random.seed(self.seed)

        # Parameters from environment variables (with defaults)
        self.num_shapes = int(os.getenv("NUM_SHAPES", 5))

        # Initialize reasoning trace storage
        self.reasoning_trace = []

        # Track timing for reasoning trace using Manim's internal video time
        # This will be set when construct() is called and renderer is available
        self.scene_events = []

    def log_event(self, description):
        """
        Log a scene event with video timestamp.
        Uses Manim's internal renderer.time which tracks actual video playback time,
        not wall-clock execution time.

        WHY: Using renderer.time instead of wall-clock time ensures timestamps match
        the actual video playback time, making the reasoning trace accurate for viewers.
        Wall-clock time would include rendering overhead and be inconsistent.
        """
        # Get current video time from Manim's renderer
        # WHY: self.renderer.time tracks the cumulative duration of all animations/waits,
        # which is exactly what appears in the final video
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
        # Setup 3D camera view
        # ====================================================================
        # WHY: Elevated view angle allows viewers to see the spatial layout clearly
        # phi: angle from the z-axis (75° = looking down at ~15° from horizontal)
        # theta: rotation around z-axis (45° = viewing from front-right)
        # This combination provides depth perception while keeping shapes recognizable
        self.set_camera_orientation(phi=75 * DEGREES, theta=45 * DEGREES)

        # ====================================================================
        # Initialize scene parameters
        # ====================================================================
        # WHY: Use diverse, easily distinguishable 3D shapes
        # Each shape has a unique silhouette to minimize confusion
        all_shapes = [
            Sphere,
            Cube,
            Cylinder,
            Cone,
            Torus,
            Tetrahedron,
            Octahedron,
            Dodecahedron,
            Icosahedron,
            Star,
        ]
        # WHY: Each shape gets a unique color for additional differentiation
        # Colors are chosen to be visually distinct even in different lighting
        all_colors = [
            BLUE,
            GREEN,
            RED,
            YELLOW,
            PURPLE,
            ORANGE,
            TEAL,
            PINK,
            MAROON,
            GOLD,
        ]

        # Color names for reasoning trace and event logging
        color_names = {
            BLUE: "blue",
            GREEN: "green",
            RED: "red",
            YELLOW: "yellow",
            PURPLE: "purple",
            ORANGE: "orange",
            TEAL: "teal",
            PINK: "pink",
            MAROON: "maroon",
            GOLD: "gold",
        }

        # Predefined pool of 10 candidate positions in 3D space
        # These are distributed across the visible frame
        positions_pool = [
            LEFT * 4 + DOWN,
            RIGHT * 4 + DOWN,
            LEFT * 2 + UP,
            RIGHT * 2 + UP,
            ORIGIN,
            LEFT * 4 + UP,
            RIGHT * 4 + UP,
            RIGHT * 2 + DOWN,
            LEFT * 2 + DOWN,
            DOWN * 3,
        ]

        # Human-readable position descriptions for reasoning trace
        position_names = [
            "far left lower area",
            "far right lower area",
            "mid-left upper area",
            "mid-right upper area",
            "center of the scene",
            "far left upper area",
            "far right upper area",
            "mid-right lower area",
            "mid-left lower area",
            "bottom center",
        ]

        # WHY: Constrain count to 1-10 to keep puzzle manageable and video length reasonable
        # More than 10 shapes would make duration tracking too difficult for viewers
        count = max(1, min(self.num_shapes, 10))

        # ====================================================================
        # Generate puzzle data
        # ====================================================================
        # WHY: Random sampling ensures each puzzle instance is unique and unpredictable
        # Randomly select which shapes and positions to use
        chosen_indices = random.sample(range(len(all_shapes)), count)
        chosen_positions = random.sample(positions_pool, count)

        # WHY: Duration range (0.5-4.0s) is long enough to perceive differences
        # but short enough to keep total video length reasonable
        # Generate random durations for each shape (0.5 to 4.0 seconds)
        durations = [random.uniform(0.5, 4.0) for _ in range(count)]

        # ====================================================================
        # Derive the correct answer
        # ====================================================================
        # WHY: Pre-compute the answer before animation so we can save it to files
        # Answer: shape names ordered by duration from longest to shortest
        shape_names = [all_shapes[i].__name__.lower() for i in chosen_indices]
        paired = list(zip(shape_names, durations))
        sorted_by_duration = sorted(paired, key=lambda x: x[1], reverse=True)
        answer_list = [name for name, _ in sorted_by_duration]

        # WHY: Store shape details for comprehensive reasoning trace generation
        # This allows us to reference shape properties when building the trace later
        self.shape_details = []
        for idx, shape_idx in enumerate(chosen_indices):
            ShapeClass = all_shapes[shape_idx]
            color = all_colors[shape_idx]
            shape_name = ShapeClass.__name__.lower()
            color_name = color_names[color]
            # Use numpy array comparison instead of list.index() to avoid ambiguous truth value error
            pos_idx = next(
                i
                for i, pos in enumerate(positions_pool)
                if (pos == chosen_positions[idx]).all()
            )
            pos_name = position_names[pos_idx]

            self.shape_details.append(
                {
                    "index": idx,
                    "shape_name": shape_name,
                    "color_name": color_name,
                    "position_name": pos_name,
                    "duration": durations[idx],
                }
            )

        # ====================================================================
        # Animate the scene
        # ====================================================================
        # WHY: Log the initial state so viewers know what they're looking at
        self.log_event("Scene begins with empty 3D view from elevated angle")

        # WHY: Process shapes sequentially so viewers can focus on one at a time
        # This makes duration comparison easier than showing all shapes simultaneously
        for idx, shape_idx in enumerate(chosen_indices):
            ShapeClass = all_shapes[shape_idx]
            color = all_colors[shape_idx]
            shape_name = ShapeClass.__name__.lower()
            color_name = color_names[color]
            # Use numpy array comparison instead of list.index() to avoid ambiguous truth value error
            pos_idx = next(
                i
                for i, pos in enumerate(positions_pool)
                if (pos == chosen_positions[idx]).all()
            )
            pos_name = position_names[pos_idx]

            # Create the shape with appropriate styling
            # WHY: Semi-transparent fill (0.6 opacity) maintains 3D depth perception
            shape = ShapeClass()
            shape.set_fill(color, opacity=0.6)
            shape.move_to(chosen_positions[idx])

            # WHY: Log BEFORE animation starts to mark the exact beginning timestamp
            self.log_event(
                f"Shape {idx + 1}/{count}: {color_name.capitalize()} {shape_name} begins to appear in {pos_name}"
            )

            # WHY: Brief creation animation (0.1s) makes appearance smooth without adding significant time
            self.play(Create(shape), run_time=0.1)

            # WHY: Log AFTER animation completes - this marks when timing measurement starts
            self.log_event(
                f"{color_name.capitalize()} {shape_name} is now fully visible and stable"
            )

            # WHY: This wait() is the actual duration we're measuring!
            # It's the time between "fully visible" and "begins to fade"
            self.wait(durations[idx])

            # WHY: Log BEFORE fade out begins - this marks when timing measurement ends
            self.log_event(
                f"{color_name.capitalize()} {shape_name} begins to fade out (was visible for {durations[idx]:.2f}s)"
            )

            # WHY: Fade out animation (0.2s) provides smooth exit
            self.play(FadeOut(shape), run_time=0.2)

            # WHY: Log AFTER fade completes to mark the shape's complete lifecycle
            self.log_event(
                f"{color_name.capitalize()} {shape_name} has completely disappeared from view"
            )

        # ====================================================================
        # Display the question
        # ====================================================================
        # WHY: Mark the transition from observation to question phase
        self.log_event("All shapes have been displayed")
        self.wait(0.5)

        # WHY: Multi-line format improves readability on screen
        # Break long text into digestible chunks
        question_lines = [
            "List the order of shapes that appeared longest to shortest",
            "with comma-separated values.",
            "",
            "The shape names are: Sphere, Cube, Cylinder, Cone, Torus,",
            "Tetrahedron, Octahedron, Dodecahedron, Icosahedron, Star.",
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

        # Add all question text as fixed elements (stays in place during camera movement)
        self.add_fixed_in_frame_mobjects(*question_texts)

        # Animate question appearance
        self.play(*[FadeIn(text) for text in question_texts], run_time=1.0)
        self.log_event("Question text appears on screen")
        self.wait(1.0)

        # Show instruction text
        instruction_text = Text(
            "Return the answer as comma-separated values.", font_size=24, color=YELLOW
        ).move_to(DOWN * 2.5)

        self.add_fixed_in_frame_mobjects(instruction_text)
        self.play(FadeIn(instruction_text, shift=UP * 0.3), run_time=0.8)
        self.log_event("Instruction text displayed")
        self.wait(3)

        # ====================================================================
        # Store results and generate output files
        # ====================================================================
        # Store answer and sorted data for reasoning trace
        self.answer_string = ", ".join(answer_list)
        self.sorted_by_duration = sorted_by_duration

        # Generate comprehensive reasoning trace
        self.build_reasoning_trace()

        # ====================================================================
        # Save output files
        # ====================================================================
        # Solution file (just the answer)
        with open(
            f"solutions/duration_3d_n{self.num_shapes}_seed{self.seed}.txt", "w"
        ) as f:
            f.write(self.answer_string)

        # Question text file
        question_text_content = (
            "List the order of shapes that appeared longest to shortest with comma-separated values.\n"
            "The shape names are: Sphere, Cube, Cylinder, Cone, Torus, Tetrahedron, Octahedron, Dodecahedron, Icosahedron, Star.\n"
            "Return the answer as comma-separated values."
        )
        with open(
            f"question_text/duration_3d_n{self.num_shapes}_seed{self.seed}.txt", "w"
        ) as f:
            f.write(question_text_content)

        # Reasoning trace file
        with open(
            f"reasoning_traces/duration_3d_n{self.num_shapes}_seed{self.seed}.txt", "w"
        ) as f:
            f.write("\n".join(self.reasoning_trace))

    def build_reasoning_trace(self):
        """
        Build a comprehensive, step-by-step reasoning trace.
        This explains how to solve the puzzle systematically with timestamps.

        Following the structure from cube_path.py:
        1. Question statement
        2. Scene description with timestamps
        3. Step-by-step reasoning process
        4. Final answer

        WHY: A detailed reasoning trace helps users understand the solution methodology
        and provides a reference for how to approach similar duration-tracking problems.
        """
        self.reasoning_trace = []

        # ====================================================================
        # Introduction
        # ====================================================================
        self.reasoning_trace.append(
            "**Question:** List the order of shapes that appeared longest to shortest with comma-separated values."
        )
        self.reasoning_trace.append("")
        self.reasoning_trace.append(
            "Let's solve this step by step by carefully tracking the duration each shape appears on screen."
        )
        self.reasoning_trace.append("")

        # ====================================================================
        # Scene description with timestamps
        # ====================================================================
        self.reasoning_trace.append("### Scene Description")
        self.reasoning_trace.append("")
        self.reasoning_trace.append(
            "The video shows a 3D scene viewed from an elevated angle. Different colored shapes appear one at a time, remain visible for varying durations, then disappear. Here's the chronological timeline:"
        )
        self.reasoning_trace.append("")

        for event in self.scene_events:
            time_str = self.format_time(event["time"])
            self.reasoning_trace.append(f"At {time_str}, {event['description']}")

        self.reasoning_trace.append("")

        # ====================================================================
        # Step 1: Understanding the task
        # ====================================================================
        self.reasoning_trace.append("### Step 1: Understand the task")
        self.reasoning_trace.append("")
        self.reasoning_trace.append(
            f"The video displays **{len(self.shape_details)} different 3D shapes**, each appearing sequentially. Each shape:"
        )
        self.reasoning_trace.append(
            "- Appears with a brief creation animation (~0.1 seconds)"
        )
        self.reasoning_trace.append(
            "- Remains visible for a specific duration (this is what we need to track)"
        )
        self.reasoning_trace.append(
            "- Disappears with a fade-out animation (~0.2 seconds)"
        )
        self.reasoning_trace.append("")
        self.reasoning_trace.append(
            "**Key insight:** The critical measurement is the time between when the shape becomes fully visible and when it begins to fade out. The creation and fade-out animations are constant across all shapes, so they don't affect the relative ordering."
        )
        self.reasoning_trace.append("")
        self.reasoning_trace.append(
            "My task is to determine which shapes appeared for the longest durations and order them from longest to shortest."
        )
        self.reasoning_trace.append("")

        # ====================================================================
        # Step 2: Track each shape's duration
        # ====================================================================
        self.reasoning_trace.append(
            "### Step 2: Track each shape's appearance duration"
        )
        self.reasoning_trace.append("")
        self.reasoning_trace.append(
            "Let me carefully observe and record how long each shape remains fully visible on screen. I'll note both the shape's visual characteristics and its display duration:"
        )
        self.reasoning_trace.append("")

        for idx, detail in enumerate(self.shape_details, 1):
            self.reasoning_trace.append(
                f"**Shape {idx}: {detail['shape_name'].capitalize()}**"
            )
            self.reasoning_trace.append(f"  - **Color**: {detail['color_name']}")
            self.reasoning_trace.append(f"  - **Location**: {detail['position_name']}")
            self.reasoning_trace.append(
                f"  - **Duration visible**: {detail['duration']:.2f} seconds"
            )
            self.reasoning_trace.append("")

        # ====================================================================
        # Step 3: Compare and sort durations
        # ====================================================================
        self.reasoning_trace.append("### Step 3: Compare and sort by duration")
        self.reasoning_trace.append("")
        self.reasoning_trace.append(
            "Now I'll organize the shapes by their display durations from longest to shortest. This ranking determines the final answer:"
        )
        self.reasoning_trace.append("")

        # Create a summary table
        self.reasoning_trace.append("| Rank | Shape | Color | Duration |")
        self.reasoning_trace.append("|------|-------|-------|----------|")
        for rank, (name, duration) in enumerate(self.sorted_by_duration, 1):
            # Find the color for this shape
            color_name = next(
                d["color_name"] for d in self.shape_details if d["shape_name"] == name
            )
            self.reasoning_trace.append(
                f"| {rank} | {name.capitalize()} | {color_name} | {duration:.2f}s |"
            )
        self.reasoning_trace.append("")

        # ====================================================================
        # Step 4: Verify the ordering
        # ====================================================================
        self.reasoning_trace.append("### Step 4: Verify the ordering")
        self.reasoning_trace.append("")
        self.reasoning_trace.append(
            "Let me verify this ordering by comparing adjacent ranks:"
        )
        self.reasoning_trace.append("")

        for i in range(len(self.sorted_by_duration) - 1):
            curr_name, curr_dur = self.sorted_by_duration[i]
            next_name, next_dur = self.sorted_by_duration[i + 1]
            diff = curr_dur - next_dur
            self.reasoning_trace.append(
                f"- {curr_name.capitalize()} ({curr_dur:.2f}s) appeared {diff:.2f}s longer than {next_name.capitalize()} ({next_dur:.2f}s)"
            )
        self.reasoning_trace.append("")
        self.reasoning_trace.append(
            "The ordering is confirmed: each shape in the list appeared longer than the next."
        )
        self.reasoning_trace.append("")

        # ====================================================================
        # Step 5: Derive final answer
        # ====================================================================
        self.reasoning_trace.append("### Step 5: Construct the final answer")
        self.reasoning_trace.append("")
        self.reasoning_trace.append(
            "Based on the duration tracking and verification, the shapes ordered from longest to shortest display time are:"
        )
        self.reasoning_trace.append("")

        for rank, (name, duration) in enumerate(self.sorted_by_duration, 1):
            self.reasoning_trace.append(
                f"{rank}. {name.capitalize()} ({duration:.2f}s)"
            )
        self.reasoning_trace.append("")

        # ====================================================================
        # Final answer
        # ====================================================================
        self.reasoning_trace.append("### Final Answer")
        self.reasoning_trace.append("")
        self.reasoning_trace.append(
            f"The order of shapes from longest to shortest display time is: **{self.answer_string}**"
        )
        self.reasoning_trace.append("")
        self.reasoning_trace.append(f"\\boxed{{{self.answer_string}}}")


# ============================================================================
# Main execution
# ============================================================================
if __name__ == "__main__":
    # Generate the duration tracking video
    scene = duration_3d()
    scene.render()

    # ========================================================================
    # Move output file to questions directory with descriptive name
    # ========================================================================
    # Manim creates output at: manim_output/videos/1080p30/<ClassName>.mp4
    output = Path("manim_output/videos/1080p30/duration_3d.mp4")
    if output.exists():
        # Create descriptive filename: duration_3d_n<shapes>_seed<seed>.mp4
        # This allows multiple variations to coexist and be identified
        filename = f"duration_3d_n{scene.num_shapes}_seed{scene.seed}.mp4"
        shutil.move(str(output), f"questions/{filename}")
        print(f"✓ Video saved: questions/{filename}")
        print(
            f"✓ Solution saved: solutions/duration_3d_n{scene.num_shapes}_seed{scene.seed}.txt"
        )
        print(
            f"✓ Reasoning trace saved: reasoning_traces/duration_3d_n{scene.num_shapes}_seed{scene.seed}.txt"
        )
    else:
        # Debug: Print directory structure to diagnose rendering issues
        print("❌ Error: Expected output video not found")
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
    # Cleanup: Remove temporary Manim output directory
    # ========================================================================
    # We've already moved the video to the questions directory,
    # so we can safely remove the temporary rendering files
    if os.path.exists("manim_output"):
        shutil.rmtree("manim_output")
