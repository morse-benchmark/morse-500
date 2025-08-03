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

        # Create a color list without our target color
        other_colors = colors.copy()
        other_colors.remove(color_choice)

        # Build and display dominoes one by one
        dominoes = []
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

            # Animate its appearance
            self.play(FadeIn(domino), run_time=0.05)

        # Wait a moment before the chain reaction
        self.wait(0.5)

        # Simulate the dominoes falling one by one in sequence
        for i, domino in enumerate(dominoes):
            # Calculate which direction to fall based on position
            row = i // dominoes_per_row
            col = i % dominoes_per_row
            
            # Alternate falling direction for visual variety
            angle = -PI/3 if (col % 2 == 0) else PI/3
            pivot = domino.get_bottom()
            
            # Each domino falls individually
            self.play(Rotate(domino, angle=angle, about_point=pivot), run_time=0.08)

        self.wait(1)
        
        # Make dominoes disappear with a nice animation
        self.play(
            *[FadeOut(domino, shift=UP*0.5) for domino in dominoes],
            run_time=1
        )
        
        self.wait(0.5)
        
        # Create color palette with names
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
        
        # Question text
        question_text = f"How many dominoes were {color_map[color_choice]}?"
        question = Text(question_text, font_size=36, weight=BOLD).move_to(UP * 2.5)
        
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
                color_map[color],
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
        
        # Save solution and question text
        with open(f"solutions/domino_count_n{scene.num_dominoes}_seed{scene.seed}.txt", "w") as f:
            f.write(str(answer))
        with open(f"question_text/domino_count_n{scene.num_dominoes}_seed{scene.seed}.txt", "w") as f:
            f.write(f"How many dominoes were {color_map[color_choice].lower()}?\nReturn the answer as a number.")

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