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

class domino_count(Scene):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Set random seed for reproducibility
        self.seed = random.randint(1000, 9999)
        random.seed(self.seed)

        # Parameters
        self.num_dominoes = int(os.getenv("NUM_DOMINOES", 30))

        # Initialize reasoning trace
        self.reasoning_trace = []
        self.reasoning_trace.append(f"Problem: Domino counting")
        self.reasoning_trace.append(f"Number of dominoes: {self.num_dominoes}")
        self.reasoning_trace.append(f"Random Seed: {self.seed}")
        self.reasoning_trace.append("")
        self.reasoning_trace.append("=== CHRONOLOGICAL SCENE DESCRIPTION ===")
        self.reasoning_trace.append("")
        
    def construct(self):
        domino_width = 0.2
        domino_height = 0.6
        spacing_x = 0.3
        spacing_y = 0.8
        
        # Calculate optimal layout for multiple rows
        screen_width = 12  # Approximate usable screen width
        max_per_row = int(screen_width / spacing_x)
        num_rows = math.ceil(self.num_dominoes / max_per_row)
        dominoes_per_row = math.ceil(self.num_dominoes / num_rows)

        # Choose a particular color and count
        answer = random.randint(1, min(self.num_dominoes, 20))
        colors = [RED, BLUE, GREEN, YELLOW, PURPLE, ORANGE, TEAL, PINK]
        color_choice = random.choice(colors)
        choice_positions = set(random.sample(range(self.num_dominoes), answer))

        print(f"Chosen color: {color_choice}, Answer: {answer}, Positions: {choice_positions}")

        # Create a color map for text descriptions
        color_map = {
            RED: "red",
            BLUE: "blue",
            GREEN: "green",
            YELLOW: "yellow",
            PURPLE: "purple",
            ORANGE: "orange",
            TEAL: "teal",
            PINK: "pink",
        }

        # Add initial scene description
        self.reasoning_trace.append(f"The video begins with an empty scene.")
        self.reasoning_trace.append(f"Dominoes begin appearing one by one in a grid layout of {num_rows} row{'s' if num_rows > 1 else ''}.")
        self.reasoning_trace.append("")

        # Create a color list without our target color
        other_colors = colors.copy()
        other_colors.remove(color_choice)

        # Build and display dominoes one by one
        dominoes = []
        domino_colors = []  # Track colors for reasoning trace

        for i in range(self.num_dominoes):
            # Calculate row and column position
            row = i // dominoes_per_row
            col = i % dominoes_per_row

            # Center the dominoes in each row
            actual_dominoes_in_row = min(dominoes_per_row, self.num_dominoes - row * dominoes_per_row)
            total_width = (actual_dominoes_in_row - 1) * spacing_x
            start_x = -total_width / 2

            # Position on screen
            x = start_x + col * spacing_x
            y = (num_rows - 1) * spacing_y / 2 - row * spacing_y

            # Color of each domino
            if i in choice_positions:
                color = color_choice
            else:
                color = random.choice(other_colors)

            domino_colors.append(color)

            # Create domino
            domino = Rectangle(
                width=domino_width,
                height=domino_height,
                fill_opacity=1,
                color=color,
                stroke_width=1,
                stroke_color=WHITE
            ).move_to([x, y, 0])
            dominoes.append(domino)

            # Add reasoning trace for this domino appearing
            position_desc = f"position {i + 1}"
            if num_rows > 1:
                position_desc = f"row {row + 1}, column {col + 1} (position {i + 1})"

            self.reasoning_trace.append(f"At {position_desc}, a {color_map[color]} domino appears.")

            # Animate its appearance
            self.play(FadeIn(domino), run_time=0.05)

        # Wait a moment before the chain reaction
        self.wait(0.5)

        # Add description of the falling sequence
        self.reasoning_trace.append("")
        self.reasoning_trace.append(f"After all {self.num_dominoes} dominoes are placed, a chain reaction begins.")
        self.reasoning_trace.append("The dominoes fall in sequence:")
        self.reasoning_trace.append("")

        # Simulate the dominoes falling one by one in sequence
        for i, domino in enumerate(dominoes):
            # Calculate which direction to fall based on position
            row = i // dominoes_per_row
            col = i % dominoes_per_row

            # Alternate falling direction for visual variety
            angle = -PI/3 if (col % 2 == 0) else PI/3
            pivot = domino.get_bottom()

            # Add reasoning trace for each falling domino
            position_desc = f"position {i + 1}"
            if num_rows > 1:
                position_desc = f"row {row + 1}, column {col + 1} (position {i + 1})"

            direction = "to the left" if (col % 2 == 0) else "to the right"
            self.reasoning_trace.append(f"The {color_map[domino_colors[i]]} domino at {position_desc} falls {direction}.")

            # Each domino falls individually
            self.play(Rotate(domino, angle=angle, about_point=pivot), run_time=0.08)

        self.wait(1)

        # Add description of dominoes disappearing
        self.reasoning_trace.append("")
        self.reasoning_trace.append("After the falling sequence completes, all dominoes fade away from the scene.")
        self.reasoning_trace.append("")

        # Make dominoes disappear with a nice animation
        self.play(
            *[FadeOut(domino, shift=UP*0.5) for domino in dominoes],
            run_time=1
        )

        self.wait(0.5)

        # Create capitalized color palette with names for display
        color_map_display = {
            RED: "Red",
            BLUE: "Blue",
            GREEN: "Green",
            YELLOW: "Yellow",
            PURPLE: "Purple",
            ORANGE: "Orange",
            TEAL: "Teal",
            PINK: "Pink",
        }
        
        # Question text
        question_text = f"How many dominoes were {color_map_display[color_choice]}?"
        question = Text(question_text, font_size=36, weight=BOLD).move_to(UP * 2.5)

        # Add reasoning trace for question appearance
        self.reasoning_trace.append("=== REASONING AND COUNTING ===")
        self.reasoning_trace.append("")
        self.reasoning_trace.append(f"The question appears: '{question_text}'")
        self.reasoning_trace.append("")

        # Create color palette
        palette_squares = []
        palette_labels = []
        palette_y = 0.5
        palette_spacing = 1.2

        # Calculate starting position to center the palette
        total_palette_width = (len(colors) - 1) * palette_spacing
        start_x = -total_palette_width / 2

        for i, color in enumerate(colors):
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
                color_map_display[color],
                font_size=20,
                color=WHITE
            ).move_to([start_x + i * palette_spacing, palette_y - 0.5, 0])

            # Highlight the target color
            if color == color_choice:
                highlight = Square(
                    side_length=0.5,
                    fill_opacity=0,
                    stroke_width=4,
                    stroke_color=YELLOW
                ).move_to(square.get_center())
                palette_squares.append(VGroup(square, highlight))
            else:
                palette_squares.append(square)

            palette_labels.append(label)

        # Animate question and palette appearance
        self.play(FadeIn(question), run_time=0.8)
        self.wait(0.3)

        self.reasoning_trace.append("A color palette appears showing all available colors, with the target color highlighted.")
        self.reasoning_trace.append("")

        # Animate palette squares appearing one by one
        for square, label in zip(palette_squares, palette_labels):
            self.play(
                FadeIn(square, scale=0.8),
                FadeIn(label, shift=UP*0.2),
                run_time=0.2
            )

        # Show the answer after a pause
        self.wait(2)

        note_text = Text(
            f"Return the answer as a number.",
            font_size=32,
            color=color_choice
        ).move_to(DOWN * 2)

        self.play(FadeIn(note_text, shift=UP*0.3), run_time=0.8)
        self.wait(3)

        # Add detailed counting reasoning
        self.reasoning_trace.append(f"To answer how many dominoes were {color_map[color_choice]}, I need to review the sequence.")
        self.reasoning_trace.append("")

        # Identify positions with the target color
        target_positions = []
        for i, clr in enumerate(domino_colors):
            if clr == color_choice:
                target_positions.append(i + 1)

        # Create detailed counting trace
        self.reasoning_trace.append(f"Reviewing all {self.num_dominoes} dominoes that appeared:")
        for i, clr in enumerate(domino_colors):
            position_desc = f"position {i + 1}"
            if num_rows > 1:
                row = i // dominoes_per_row
                col = i % dominoes_per_row
                position_desc = f"row {row + 1}, column {col + 1} (position {i + 1})"

            if clr == color_choice:
                self.reasoning_trace.append(f"  - At {position_desc}: {color_map[clr]} domino ✓")
            else:
                self.reasoning_trace.append(f"  - At {position_desc}: {color_map[clr]} domino")

        self.reasoning_trace.append("")
        self.reasoning_trace.append(f"The {color_map[color_choice]} dominoes appeared at positions: {target_positions}")
        self.reasoning_trace.append(f"Counting these occurrences: {answer} dominoes were {color_map[color_choice]}.")

        # Save solution and question text
        with open(f"solutions/domino_count_n{scene.num_dominoes}_seed{scene.seed}.txt", "w") as f:
            f.write(str(answer))
        with open(f"question_text/domino_count_n{scene.num_dominoes}_seed{scene.seed}.txt", "w") as f:
            f.write(f"How many dominoes were {color_map[color_choice]}?\nReturn the answer as a number.")

        # Save detailed reasoning trace
        self.reasoning_trace.append("")
        self.reasoning_trace.append(f"Final Answer: {answer}")
        with open(f"reasoning_traces/domino_count_n{scene.num_dominoes}_seed{scene.seed}.txt", "w") as f:
            f.write("\n".join(scene.reasoning_trace))

if __name__ == "__main__":
    # Generate multiple domino counting videos
    scene = domino_count()
    scene.render()

    # Move the output file with descriptive name
    # Manim creates a folder with the class name
    output = Path("manim_output/videos/1080p30/domino_count.mp4")
    if output.exists():
        filename = f"domino_count_n{scene.num_dominoes}_seed{scene.seed}.mp4"
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