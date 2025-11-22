from manim import *
import random
import os
import shutil
from pathlib import Path

# Setup directories
Path("questions").mkdir(exist_ok=True)
Path("solutions").mkdir(exist_ok=True)
Path("question_text").mkdir(exist_ok=True)
Path("reasoning_traces").mkdir(exist_ok=True)

# Config
config.media_dir = "manim_output"
config.pixel_height = 1080
config.pixel_width = 1920
config.frame_rate = 30
config.verbosity = "WARNING"
config.preview = False

class color_sequence(Scene):
    def __init__(self, difficulty=2, **kwargs):
        super().__init__(**kwargs)

        self.difficulty = int(os.getenv("DIFFICULTY", 2))
        self.seed = random.randint(1000, 9999)
        random.seed(self.seed)

        self.color_sequence = []
        self.color_letters = {
            RED: "R", BLUE: "B", GREEN: "G", YELLOW: "Y",
            ORANGE: "O", PURPLE: "P", TEAL: "T"
        }

        # Initialize reasoning trace
        self.reasoning_trace = []
        self.reasoning_trace.append(f"Problem: Color sequence tracking")
        self.reasoning_trace.append(f"Difficulty: {self.difficulty}")
        self.reasoning_trace.append(f"Random Seed: {self.seed}")
        self.reasoning_trace.append("")

    def construct(self):
        colors_available = list(self.color_letters.keys())
        color_count = {1: 5, 2: 8, 3: 11}[self.difficulty]
        interference_count = {1: 1, 2: 3, 3: 6}[self.difficulty]
        interference_chance = {1: 0.1, 2: 0.4, 3: 0.8}[self.difficulty]

        # Generate target sequence
        self.color_sequence = random.choices(colors_available, k=color_count)

        # Add chronological scene description
        self.reasoning_trace.append("=== Scene Description ===")
        self.reasoning_trace.append("The video begins with a blank screen. We're tracking a sequence of color changes in the first object that appears.")
        self.reasoning_trace.append("")
        self.reasoning_trace.append("=== Chronological Events ===")

        # Track timing for narrative
        current_time = 0.0

        # First color appears
        target_shape = Circle(fill_opacity=0.9).set_color(self.color_sequence[0])
        first_color_name = self.color_letters[self.color_sequence[0]]
        self.reasoning_trace.append(f"At t={current_time:.1f}s: A circle appears on screen. It is {first_color_name} (first letter: {first_color_name}).")
        self.reasoning_trace.append(f"This is the first object to appear, so we need to track all color changes of this circle.")

        self.play(Create(target_shape), run_time=0.3)
        current_time += 0.3
        self.wait(0.3)
        current_time += 0.3

        # Track interference events
        interference_events = []

        for i, color in enumerate(self.color_sequence[1:], start=1):
            # Check for interference
            add_interference_now = random.random() < interference_chance
            if add_interference_now:
                num_interference = random.randint(1, interference_count)
                interference_events.append((current_time, num_interference))
                self.reasoning_trace.append(f"At t={current_time:.1f}s: {num_interference} distractor object(s) briefly appear and fade away. These are interference - we ignore them and focus on the original circle.")
                self.add_interference(num_interference)
                current_time += 0.6  # Approximate time for interference

            # Color change
            color_name = self.color_letters[color]
            self.reasoning_trace.append(f"At t={current_time:.1f}s: The circle changes color to {color_name} (first letter: {color_name}).")

            self.play(target_shape.animate.set_color(color), run_time=0.5)
            current_time += 0.5
            self.wait(0.5)
            current_time += 0.5

        self.reasoning_trace.append(f"At t={current_time:.1f}s: The circle fades out and disappears from the screen.")
        current_time += 0.5
        self.play(FadeOut(target_shape), run_time=0.5)
        self.wait(0.5)
        current_time += 0.5

        # Add reasoning section
        self.reasoning_trace.append("")
        self.reasoning_trace.append("=== Reasoning to Answer ===")
        self.reasoning_trace.append("The question asks for the sequence of colors of the FIRST object that appeared.")
        self.reasoning_trace.append("The first object was the circle that appeared at the beginning.")
        self.reasoning_trace.append("We need to list all the colors it went through, in order, using the first letter of each color.")
        self.reasoning_trace.append("")
        self.reasoning_trace.append("Let me trace through the color changes chronologically:")

        # Build the sequence description
        for idx, color in enumerate(self.color_sequence):
            color_letter = self.color_letters[color]
            if idx == 0:
                self.reasoning_trace.append(f"{idx + 1}. Started as {color_letter}")
            else:
                self.reasoning_trace.append(f"{idx + 1}. Changed to {color_letter}")

        self.reasoning_trace.append("")
        self.reasoning_trace.append("Note: Any other shapes that briefly appeared were distractors and should be ignored.")
        self.reasoning_trace.append("We only track the original circle that was the first object to appear.")
        self.reasoning_trace.append("")

        # Build the answer
        answer_letters = [self.color_letters[c] for c in self.color_sequence]
        self.reasoning_trace.append(f"Concatenating the first letters in order: {' -> '.join(answer_letters)}")
        self.reasoning_trace.append(f"Final sequence: {''.join(answer_letters)}")

        # Question prompt
        question_text = Text(
            "What was the sequence of colors of the first object that appeared?\n"
            "Use the first letter of each color (e.g., RGBY).",
            font_size=30
        ).to_edge(UP)
        self.play(Write(question_text))
        self.wait(1.5)

        # Color palette display
        palette_squares = []
        palette_labels = []
        palette_y = -2.5
        palette_spacing = 1.2

        colors = list(self.color_letters.keys())
        letter_map = self.color_letters

        total_palette_width = (len(colors) - 1) * palette_spacing
        start_x = -total_palette_width / 2

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

        for square, label in zip(palette_squares, palette_labels):
            self.play(
                FadeIn(square, scale=0.8),
                FadeIn(label, shift=UP * 0.2),
                run_time=0.2
            )

        self.wait(2)

        # Save solution and question
        answer = "".join([self.color_letters[c] for c in self.color_sequence])
        basename = f"colorsq_d{self.difficulty}_seed{self.seed}"
        with open(f"solutions/{basename}.txt", "w") as f:
            f.write(answer)
        with open(f"question_text/{basename}.txt", "w") as f:
            f.write("What was the sequence of colors of the first object that appeared?\nUse the first letter of each color (e.g., RGBY).")

        # Save detailed reasoning trace
        with open(f"reasoning_traces/{basename}.txt", "w") as f:
            f.write("\n".join(self.reasoning_trace))

    def add_interference(self, count=1):
        distract_shapes = [Square, Triangle, Star, Arrow, RegularPolygon]
        for _ in range(count):
            shape_cls = random.choice(distract_shapes)
            shape = shape_cls().scale(random.uniform(0.2, 0.5))
            shape.set_fill(random.choice(list(self.color_letters.keys())), opacity=0.8)
            shape.set_stroke(WHITE, width=1)
            shape.move_to([
                random.uniform(-6, 6),
                random.uniform(-3.5, 3.5),
                0
            ])
            spin = Rotate(shape, angle=random.uniform(-PI, PI), run_time=0.3)
            fade = FadeOut(shape, run_time=0.3)
            self.add(shape)
            self.play(spin, fade, lag_ratio=0.2)


if __name__ == "__main__":
    # Render & save output
    scene = color_sequence()
    scene.render()

    # Move video file
    video_path = Path(f"manim_output/videos/1080p30/color_sequence.mp4")
    if video_path.exists():
        filename = f"colorsq_d{scene.difficulty}_seed{scene.seed}.mp4"
        shutil.move(str(video_path), f"questions/{filename}")
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

    # Cleanup
    if os.path.exists("manim_output"):
        shutil.rmtree("manim_output")
