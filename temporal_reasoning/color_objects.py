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

class color_objects(Scene):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Set random seed for reproducibility
        self.seed = random.randint(1000, 9999)
        random.seed(self.seed)
        
        # Parameters
        self.difficulty = int(os.getenv("DIFFICULTY", 1))
        
    def construct(self):
        # Define all possible shapes with their names
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
        
        # Define all possible colors
        all_colors = [RED, BLUE, GREEN, YELLOW, PURPLE, ORANGE, PINK, TEAL, MAROON, GRAY]
        
        # Set number of shapes based on difficulty
        if self.difficulty == 1:
            num_shapes = 4
        elif self.difficulty == 2:
            num_shapes = 6
        elif self.difficulty == 3:
            num_shapes = 8
        else:
            num_shapes = 4
        
        # Randomly select shapes and colors for initial sequence
        selected_shape_data = random.sample(all_shapes_with_names, num_shapes)
        selected_shapes = [shape for shape, name in selected_shape_data]
        shape_names = [name for shape, name in selected_shape_data]
        selected_colors = random.sample(all_colors, num_shapes)
        
        # Choose which transition color position will be the target (1-5)
        target_color_index = random.randint(1, 5)
        
        # Generate 5 DISTINCT transition colors, ensuring the target position uses one of the initial shape colors
        remaining_colors = [c for c in all_colors if c not in selected_colors]
        
        # First, assign the target color (must be from initial shapes)
        target_color = random.choice(selected_colors)
        
        # Generate all 5 transition colors ensuring they are distinct
        transition_colors = [None] * 5
        transition_colors[target_color_index - 1] = target_color  # Place target color at specified position
        
        # Fill remaining positions with distinct colors
        available_colors = all_colors.copy()
        available_colors.remove(target_color)  # Remove target color to ensure distinctness
        
        for i in range(5):
            if transition_colors[i] is None:  # If position not filled yet
                chosen_color = random.choice(available_colors)
                transition_colors[i] = chosen_color
                available_colors.remove(chosen_color)  # Ensure distinctness
        
        # Quickly display each shape in turn
        shape_color_pairs = []
        for i, (shape, color) in enumerate(zip(selected_shapes, selected_colors)):
            shape.set_fill(color, opacity=0.8)
            shape_color_pairs.append((shape, color, shape_names[i]))
            self.play(Create(shape), run_time=0.4)
            self.play(FadeOut(shape), run_time=0.2)

        # Final shape should be different from all initial shapes to avoid confusion
        # Get the shapes that weren't selected for the initial sequence
        remaining_shapes_with_names = [item for item in all_shapes_with_names if item not in selected_shape_data]
        
        # If we have remaining shapes, pick one; otherwise create a simple circle as fallback
        if remaining_shapes_with_names:
            final_shape_data = random.choice(remaining_shapes_with_names)
            final_shape = final_shape_data[0].copy()
        else:
            # Fallback to a circle if somehow all shapes were used (shouldn't happen with current setup)
            final_shape = Circle()
        
        # Set initial color for final shape to be the first transition color
        final_shape.set_fill(transition_colors[0], opacity=0.8)
        self.play(Create(final_shape), run_time=0.4)
        self.wait(0.4)
        
        # Smoother color transitions with longer duration (4 more color changes)
        for i, col in enumerate(transition_colors[1:], start=2):
            self.play(final_shape.animate.set_fill(col, opacity=0.8), run_time=0.4)

        # Wait before showing question
        self.wait(0.5)
        
        # Remove final shape
        self.play(FadeOut(final_shape), run_time=0.5)
        
        # Create color map for names
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
        
        # Since we guaranteed the target color exists in initial shapes, answer_shape should never be None
        if answer_shape is None:
            answer_shape = "error"  # This should never happen with the new logic
        
        # Question text - clearer about color numbering (1-5)
        ordinal_numbers = ["first", "second", "third", "fourth", "fifth"]
        ordinal = ordinal_numbers[target_color_index - 1]
        
        question_text = f"Which shape had the same color as \n the {ordinal} color of the final shape?"
        question = Text(question_text, font_size=30, weight=BOLD).move_to(UP * 2.5)
        
        # Create color palette
        palette_squares = []
        palette_labels = []
        palette_y = 0.5
        palette_spacing = 1.0
        
        # Use all colors that appeared in the scene (initial colors + transition colors)
        scene_colors = selected_colors + transition_colors
        unique_colors = list(dict.fromkeys(scene_colors))  # Remove duplicates while preserving order
        
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
            
            # Add square to palette (no highlighting to avoid giving away the answer)
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
                run_time=0.15
            )
        
        # Show the answer instruction after a pause
        self.wait(2)

        note_text = Text(
            f"Return the answer as a shape name in lower case.",
            font_size=32,
            color=WHITE
        ).move_to(DOWN * 2)
        
        self.play(FadeIn(note_text, shift=UP*0.3), run_time=0.8)
        self.wait(3)
        
        # Save solution and question text
        question_for_file = f"Which shape had the same color as the {ordinal} color of the final shape?\nReturn the answer as a shape name in lower case."
        
        with open(f"solutions/color_objects_d{self.difficulty}_seed{self.seed}.txt", "w") as f:
            f.write(str(answer_shape))
        with open(f"question_text/color_objects_d{self.difficulty}_seed{self.seed}.txt", "w") as f:
            f.write(question_for_file)

# Generate color objects video
scene = color_objects()
scene.render()

# Move the output file with descriptive name
# Try multiple possible output paths
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
        print(f"Video moved from {output_path} to questions/{filename}")
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

# Final cleanup
if os.path.exists("manim_output"):
    shutil.rmtree("manim_output")

