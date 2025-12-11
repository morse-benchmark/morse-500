from manim import *
import random
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
config.pixel_height = 1080
config.pixel_width = 1920
config.frame_rate = 30
config.verbosity = "WARNING"
config.preview = False

class color_sequence(Scene):
    """
    A scene that generates a color sequence tracking puzzle:
    - Shows a circle that changes colors over time
    - Displays interference objects as distractors
    - User must track the sequence of colors in the first object
    - Generates question video, solution, and detailed reasoning trace
    """

    def __init__(self, difficulty=2, **kwargs):
        super().__init__(**kwargs)

        # Difficulty determines complexity: color count and interference level
        self.difficulty = int(os.getenv("DIFFICULTY", 2))

        # Set random seed for reproducibility of color sequence generation
        self.seed = random.randint(1000, 9999)
        random.seed(self.seed)

        # Will store the sequence of colors shown in the video
        self.color_sequence = []

        # Mapping from Manim color objects to single-letter abbreviations
        self.color_letters = {
            RED: "R", BLUE: "B", GREEN: "G", YELLOW: "Y",
            ORANGE: "O", PURPLE: "P", TEAL: "T"
        }

        # Full color names for natural language in reasoning trace
        self.color_names = {
            RED: "Red", BLUE: "Blue", GREEN: "Green", YELLOW: "Yellow",
            ORANGE: "Orange", PURPLE: "Purple", TEAL: "Teal"
        }

        # Initialize reasoning trace storage - will be built during scene construction
        self.reasoning_trace = []

        # Track scene events with timestamps using Manim's internal video time
        # This will be populated during construct() when renderer is available
        self.scene_events = []

    def log_event(self, description):
        """
        Log a scene event with video timestamp.
        Uses Manim's internal renderer.time which tracks actual video playback time,
        not wall-clock execution time.

        Args:
            description: String describing the event that occurred
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
        # Setup: Determine difficulty parameters
        # ====================================================================
        colors_available = list(self.color_letters.keys())

        # Difficulty affects: number of color changes and distractor frequency
        color_count = {1: 5, 2: 8, 3: 11}[self.difficulty]
        interference_count = {1: 1, 2: 3, 3: 6}[self.difficulty]  # Max distractors per event
        interference_chance = {1: 0.1, 2: 0.4, 3: 0.8}[self.difficulty]  # Probability of distractors

        # ====================================================================
        # Generate the color sequence for the target object
        # ====================================================================
        self.color_sequence = random.choices(colors_available, k=color_count)

        # Track interference events for reasoning trace
        self.interference_events = []

        # ====================================================================
        # Show the first object (circle) with initial color
        # ====================================================================
        target_shape = Circle(fill_opacity=0.9).set_color(self.color_sequence[0])

        # Log the initial appearance
        self.log_event("Video begins with a blank screen")

        self.play(Create(target_shape), run_time=0.3)
        self.log_event(f"A {self.color_names[self.color_sequence[0]].lower()} circle appears at the center of the screen")

        self.wait(0.3)

        # ====================================================================
        # Animate color changes with occasional interference
        # ====================================================================
        for i, color in enumerate(self.color_sequence[1:], start=1):
            # Randomly add interference objects (distractors)
            add_interference_now = random.random() < interference_chance
            if add_interference_now:
                num_interference = random.randint(1, interference_count)

                # Log before interference appears
                self.log_event(f"{num_interference} distractor object(s) begin to appear")

                # Record this interference for reasoning trace
                self.interference_events.append({
                    'position': i,  # Where in sequence this occurs
                    'count': num_interference
                })

                self.add_interference(num_interference)

                # Log after interference completes
                self.log_event(f"Distractor object(s) fade away (these should be ignored)")

            # Change the target circle's color
            prev_color = self.color_sequence[i - 1]
            self.log_event(f"Circle begins changing color from {self.color_names[prev_color].lower()} to {self.color_names[color].lower()}")

            self.play(target_shape.animate.set_color(color), run_time=0.5)

            self.log_event(f"Circle is now {self.color_names[color].lower()}")

            self.wait(0.5)

        # ====================================================================
        # Remove the target object
        # ====================================================================
        self.log_event("Circle begins to fade out")

        self.play(FadeOut(target_shape), run_time=0.5)

        self.log_event("Circle has disappeared from screen")

        self.wait(0.5)

        # ====================================================================
        # Display the question
        # ====================================================================
        question_text = Text(
            "What was the sequence of colors of the first object that appeared?\n"
            "Use the first letter of each color (e.g., RGBY).",
            font_size=30
        ).to_edge(UP)

        self.log_event("Question text appears on screen")

        self.play(Write(question_text))

        self.log_event("Question is fully displayed")

        self.wait(1.5)

        # ====================================================================
        # Display color palette reference
        # ====================================================================
        # This helps viewers remember which letter corresponds to which color
        palette_squares = []
        palette_labels = []
        palette_y = -2.5
        palette_spacing = 1.2

        colors = list(self.color_letters.keys())
        letter_map = self.color_letters

        total_palette_width = (len(colors) - 1) * palette_spacing
        start_x = -total_palette_width / 2

        # Create color swatches with labels
        for i, color in enumerate(colors):
            x_pos = start_x + i * palette_spacing

            square = Square(
                side_length=0.4,
                fill_opacity=1,
                color=color,
                stroke_width=2,
                stroke_color=WHITE
            ).move_to([x_pos, palette_y, 0])

            label = Text(
                letter_map[color],
                font_size=20,
                color=WHITE
            ).next_to(square, DOWN, buff=0.15)

            palette_squares.append(square)
            palette_labels.append(label)

        self.log_event("Color palette reference begins to appear")

        # Animate palette appearance
        for square, label in zip(palette_squares, palette_labels):
            self.play(
                FadeIn(square, scale=0.8),
                FadeIn(label, shift=UP * 0.2),
                run_time=0.2
            )

        self.log_event("Color palette is fully displayed")

        self.wait(2)

        # ====================================================================
        # Generate reasoning trace
        # ====================================================================
        self.build_reasoning_trace()

        # ====================================================================
        # Save output files
        # ====================================================================
        answer = "".join([self.color_letters[c] for c in self.color_sequence])
        basename = f"color_sequence_d{self.difficulty}_seed{self.seed}"

        # Solution file (just the answer)
        with open(f"solutions/{basename}.txt", "w") as f:
            f.write(answer)

        # Question text file
        with open(f"question_text/{basename}.txt", "w") as f:
            f.write("What was the sequence of colors of the first object that appeared?\nUse the first letter of each color (e.g., RGBY).")

        # Detailed reasoning trace file
        with open(f"reasoning_traces/{basename}.txt", "w") as f:
            f.write("\n".join(self.reasoning_trace))

    def build_reasoning_trace(self):
        """
        Build a comprehensive, step-by-step reasoning trace.
        This explains how to solve the puzzle systematically.
        """
        self.reasoning_trace = []

        # ====================================================================
        # Question statement
        # ====================================================================
        self.reasoning_trace.append("**Question:** What was the sequence of colors of the first object that appeared?")
        self.reasoning_trace.append("Use the first letter of each color (e.g., RGBY).")
        self.reasoning_trace.append("")
        self.reasoning_trace.append("Let's solve this step by step.")
        self.reasoning_trace.append("")

        # ====================================================================
        # Scene description with timestamps
        # ====================================================================
        self.reasoning_trace.append("### Scene Description")
        self.reasoning_trace.append("")
        self.reasoning_trace.append("Here's what happens in the video, with precise timestamps:")
        self.reasoning_trace.append("")

        for event in self.scene_events:
            time_str = self.format_time(event['time'])
            self.reasoning_trace.append(f"At {time_str}, {event['description']}")

        self.reasoning_trace.append("")

        # ====================================================================
        # Step 1: Identify what to track
        # ====================================================================
        self.reasoning_trace.append("### Step 1: Identify what to track")
        self.reasoning_trace.append("")
        self.reasoning_trace.append("The question asks for the color sequence of the **first object that appeared**.")
        self.reasoning_trace.append(f"The first object was a circle that appeared at the beginning and was initially {self.color_names[self.color_sequence[0]].lower()}.")
        self.reasoning_trace.append("")

        if self.interference_events:
            self.reasoning_trace.append(f"**Important:** During the video, there were {len(self.interference_events)} interference event(s) where distractor objects briefly appeared.")
            self.reasoning_trace.append("These distractors should be **ignored** - we only track the original circle.")
        else:
            self.reasoning_trace.append("**Note:** No distractor objects appeared in this video, so tracking is straightforward.")

        self.reasoning_trace.append("")

        # ====================================================================
        # Step 2: Track the color changes
        # ====================================================================
        self.reasoning_trace.append("### Step 2: Track all color changes of the first object")
        self.reasoning_trace.append("")
        self.reasoning_trace.append("The circle went through the following color sequence:")
        self.reasoning_trace.append("")

        for idx, color in enumerate(self.color_sequence):
            color_name = self.color_names[color]
            color_letter = self.color_letters[color]
            if idx == 0:
                self.reasoning_trace.append(f"{idx + 1}. **Started as {color_name}** → First letter: **{color_letter}**")
            else:
                self.reasoning_trace.append(f"{idx + 1}. **Changed to {color_name}** → First letter: **{color_letter}**")

        self.reasoning_trace.append("")

        # ====================================================================
        # Step 3: Build the answer
        # ====================================================================
        self.reasoning_trace.append("### Step 3: Construct the answer")
        self.reasoning_trace.append("")
        self.reasoning_trace.append("We need to use the first letter of each color name, in the order they appeared:")
        self.reasoning_trace.append("")

        answer_letters = [self.color_letters[c] for c in self.color_sequence]
        color_names_list = [self.color_names[c] for c in self.color_sequence]

        for idx, (name, letter) in enumerate(zip(color_names_list, answer_letters)):
            self.reasoning_trace.append(f"- {name} → {letter}")

        self.reasoning_trace.append("")
        self.reasoning_trace.append(f"Concatenating these letters: {' + '.join(answer_letters)} = **{''.join(answer_letters)}**")
        self.reasoning_trace.append("")

        # ====================================================================
        # Final answer
        # ====================================================================
        self.reasoning_trace.append("### Final Answer")
        self.reasoning_trace.append("")
        answer = "".join(answer_letters)
        self.reasoning_trace.append(f"The sequence of colors is: **{answer}**")
        self.reasoning_trace.append("")
        self.reasoning_trace.append(f"\\boxed{{{answer}}}")

    def add_interference(self, count=1):
        """
        Add distractor shapes that briefly appear and disappear.
        These serve as interference to make the task more challenging.

        Args:
            count: Number of distractor objects to create
        """
        # Various shapes to use as distractors
        distract_shapes = [Square, Triangle, Star, Arrow, RegularPolygon]

        for _ in range(count):
            # Randomly select shape type and size
            shape_cls = random.choice(distract_shapes)
            shape = shape_cls().scale(random.uniform(0.2, 0.5))

            # Give it a random color from our palette
            shape.set_fill(random.choice(list(self.color_letters.keys())), opacity=0.8)
            shape.set_stroke(WHITE, width=1)

            # Place at random position on screen
            shape.move_to([
                random.uniform(-6, 6),
                random.uniform(-3.5, 3.5),
                0
            ])

            # Animate: spin and fade out simultaneously
            spin = Rotate(shape, angle=random.uniform(-PI, PI), run_time=0.3)
            fade = FadeOut(shape, run_time=0.3)
            self.add(shape)
            self.play(spin, fade, lag_ratio=0.2)


# ============================================================================
# Main execution
# ============================================================================
if __name__ == "__main__":
    # Generate the color sequence video
    scene = color_sequence()
    scene.render()

    # ========================================================================
    # Move output file to questions directory with descriptive name
    # ========================================================================
    video_path = Path(f"manim_output/videos/1080p30/color_sequence.mp4")
    if video_path.exists():
        filename = f"color_sequence_d{scene.difficulty}_seed{scene.seed}.mp4"
        shutil.move(str(video_path), f"questions/{filename}")
        print(f"✓ Video saved: questions/{filename}")
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

    # Final cleanup - remove temporary manim output directory
    if os.path.exists("manim_output"):
        shutil.rmtree("manim_output")
