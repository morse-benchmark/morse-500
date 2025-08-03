from manim import *
import random
import os
import shutil
from pathlib import Path

# Setup directories
Path("questions").mkdir(exist_ok=True)
Path("solutions").mkdir(exist_ok=True)
Path("question_text").mkdir(exist_ok=True)

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

    def construct(self):
        colors_available = list(self.color_letters.keys())
        color_count = {1: 5, 2: 8, 3: 11}[self.difficulty]
        interference_count = {1: 1, 2: 3, 3: 6}[self.difficulty]
        interference_chance = {1: 0.1, 2: 0.4, 3: 0.8}[self.difficulty]

        # Generate target sequence
        self.color_sequence = random.choices(colors_available, k=color_count)
        target_shape = Circle(fill_opacity=0.9).set_color(self.color_sequence[0])
        self.play(Create(target_shape), run_time=0.3)
        self.wait(0.3)
        
        for color in self.color_sequence[1:]:
            if random.random() < interference_chance:
                self.add_interference(random.randint(1, interference_count))
            self.play(target_shape.animate.set_color(color), run_time=0.5)
            self.wait(0.5)

        self.play(FadeOut(target_shape), run_time=0.5)
        self.wait(0.5)

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
