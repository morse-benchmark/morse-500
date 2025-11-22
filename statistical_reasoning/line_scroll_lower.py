from manim import *
import numpy as np
import random
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


class LineScrollLower(Scene):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Set random seed for reproducibility
        self.seed = random.randint(1000, 9999)
        random.seed(self.seed)
        np.random.seed(self.seed)

        # Parameters
        self.difficulty = int(os.getenv("DIFFICULTY", 0))

        # Store for reasoning trace
        self.frame_events = []
        self.data_arrays = []
        self.groups = []
        self.group_low = None
        self.group_high = None
        self.answer = None

    def construct(self):
        # Set background and text colors based on difficulty
        background_color = BLACK
        text_color = WHITE
        if self.difficulty == 1:
            background_color = WHITE
            text_color = BLACK
        elif self.difficulty == 2:
            background_color = BLUE
            text_color = BLACK

        # Update config for this scene
        config.background_color = background_color

        # Generate data arrays
        np.random.seed(self.seed * self.difficulty if self.difficulty > 0 else self.seed)
        self.data_arrays = []
        for _ in range(2 + self.difficulty):
            self.data_arrays.append(np.random.uniform(30, 60, size=8 + self.difficulty * 2))

        # Generate group names
        np.random.seed(self.seed * self.difficulty if self.difficulty > 0 else self.seed)
        self.groups = [chr(k) for k in range(65, 65 + 26)]
        np.random.shuffle(self.groups)
        self.group_low = self.groups[0]
        self.group_high = self.groups[1]

        # Calculate answer: count points where group_high does NOT exceed group_low
        # This means counting where data_arrays[1] < data_arrays[0]
        self.answer = int(np.sum(self.data_arrays[1] < self.data_arrays[0]))

        # Create question text
        question_text = f"At how many points does Group {self.group_high} not exceed Group {self.group_low} in the following graph?\nPlease answer with just a number and nothing else"

        # Display title
        title = Text(question_text, font_size=28, color=text_color)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(0.5)

        # Set up colors for lines
        np.random.seed(self.seed * self.difficulty if self.difficulty > 0 else self.seed)
        COLORS = [RED, WHITE, LIGHT_BROWN, GREEN, PURPLE, YELLOW, ORANGE, TEAL, MAROON, PINK, DARK_BLUE]
        np.random.shuffle(COLORS)

        # Create axes
        ax = Axes(
            x_range=[1, 6 + self.difficulty * 2 + 1, 1],
            y_range=[np.floor(np.min(self.data_arrays)), np.ceil(np.max(self.data_arrays)), 5],
            x_length=5,
            y_length=5,
            axis_config={
                "include_numbers": True,
                "color": text_color,
                "label_constructor": lambda val: Text(str(int(val)), color=text_color, font_size=24)
            }
        )

        self.play(Create(ax.y_axis), run_time=0.3)
        self.play(Create(ax.x_axis), run_time=0.3)

        # Create legend
        legends = []
        np.random.seed(self.seed * self.difficulty if self.difficulty > 0 else self.seed)
        groups_copy = [chr(k) for k in range(65, 65 + 26)]
        np.random.shuffle(groups_copy)
        for k in range(len(self.data_arrays)):
            legends.append(VGroup(
                Square(side_length=0.3, color=COLORS[k], fill_opacity=1),
                Text(f"Group {groups_copy[k]}", color=COLORS[k], font_size=DEFAULT_FONT_SIZE - 30)
            ).arrange(RIGHT))
        np.random.shuffle(legends)
        legend = VGroup(*legends).arrange(DOWN, aligned_edge=LEFT).to_edge(DOWN + LEFT, buff=0.5)
        self.play(Write(legend), run_time=1)

        # Track cumulative time for reasoning trace
        cumulative_time = 0.3 + 0.3 + 1 + 0.5  # axes + legend + wait

        # Slide window over time
        num_frames = self.difficulty * 2 + 6
        for i in range(num_frames):
            frame_start_time = cumulative_time
            frame = VGroup()
            frame_points = {}  # Track which points are visible in this frame for each group

            for k, data in enumerate(self.data_arrays):
                color = COLORS[k]
                visible_data = data[i:i + 5]
                x_coords = list(range(i + 1, i + 1 + len(visible_data)))
                segment_points = [ax.c2p(x, y) for x, y in zip(x_coords, visible_data)]

                # Store frame data for reasoning trace
                frame_points[groups_copy[k]] = list(zip(x_coords, visible_data))

                for j in range(len(segment_points) - 1):
                    line = Line(segment_points[j], segment_points[j + 1], color=color)
                    frame.add(line)

            # Record this frame for reasoning trace
            self.frame_events.append({
                "frame_number": i + 1,
                "start_time": frame_start_time,
                "end_time": frame_start_time + 0.6,
                "x_range": (i + 1, i + 5),
                "points": frame_points
            })

            self.play(Create(frame), run_time=0.3)
            cumulative_time += 0.3
            self.play(FadeOut(frame), run_time=0.3)
            cumulative_time += 0.3

        self.wait(1)

        # Save solution
        with open(f"solutions/line_scroll_lower_d{self.difficulty}_seed{self.seed}.txt", "w") as f:
            f.write(str(self.answer))

        # Save question text
        with open(f"question_text/line_scroll_lower_d{self.difficulty}_seed{self.seed}.txt", "w") as f:
            f.write(question_text)

        # Generate and save reasoning trace
        reasoning_trace = self.generate_reasoning_trace()
        with open(f"reasoning_traces/line_scroll_lower_d{self.difficulty}_seed{self.seed}.txt", "w") as f:
            f.write(reasoning_trace)

    def generate_reasoning_trace(self):
        """Generate a detailed reasoning trace describing the scene chronologically"""

        trace = []
        trace.append("=== CHRONOLOGICAL OBSERVATION ===\n")

        trace.append(f"The video displays an animated line graph showing data for {len(self.data_arrays)} groups.")
        trace.append(f"The question asks: 'At how many points does Group {self.group_high} not exceed Group {self.group_low}?'\n")

        trace.append("The graph uses a sliding window approach, displaying 5 data points at a time.")
        trace.append(f"Total frames shown: {len(self.frame_events)}\n")

        # Describe the animation sequence
        trace.append("Animation Sequence:")
        for i, event in enumerate(self.frame_events[:3]):  # Show first 3 frames as examples
            trace.append(f"\nFrame {event['frame_number']} (Time: {event['start_time']:.2f}s - {event['end_time']:.2f}s):")
            trace.append(f"  - X-axis range visible: {event['x_range'][0]} to {event['x_range'][1]}")
            trace.append(f"  - Shows a window of data points for all groups")

        if len(self.frame_events) > 3:
            trace.append(f"\n... (continuing through {len(self.frame_events)} total frames)")

        trace.append("\n\n=== ANALYSIS ===\n")

        trace.append("To answer this question, I need to:")
        trace.append(f"1. Identify which lines correspond to Group {self.group_high} and Group {self.group_low}")
        trace.append(f"2. Compare their values at each data point")
        trace.append(f"3. Count how many points where Group {self.group_high} does NOT exceed Group {self.group_low}")
        trace.append(f"   (i.e., where Group {self.group_high} is less than or equal to Group {self.group_low})\n")

        # Reconstruct the full data comparison
        trace.append("Full Data Comparison:")
        trace.append(f"Group {self.group_low} data points: {[f'{x:.2f}' for x in self.data_arrays[0]]}")
        trace.append(f"Group {self.group_high} data points: {[f'{x:.2f}' for x in self.data_arrays[1]]}\n")

        trace.append("\n=== REASONING PROCESS ===\n")

        # Count and explain each comparison
        trace.append("Point-by-point comparison:")
        count = 0
        for i in range(len(self.data_arrays[0])):
            low_val = self.data_arrays[0][i]
            high_val = self.data_arrays[1][i]
            exceeds = high_val > low_val

            if not exceeds:
                count += 1
                trace.append(f"  Point {i + 1}: Group {self.group_low}={low_val:.2f}, Group {self.group_high}={high_val:.2f} -> Group {self.group_high} does NOT exceed ✓")
            else:
                trace.append(f"  Point {i + 1}: Group {self.group_low}={low_val:.2f}, Group {self.group_high}={high_val:.2f} -> Group {self.group_high} exceeds")

        trace.append(f"\nTotal points where Group {self.group_high} does NOT exceed Group {self.group_low}: {count}")

        trace.append("\n\n=== FINAL ANSWER ===\n")
        trace.append(f"Answer: {self.answer}\n")

        trace.append("\n=== REASONING SUMMARY ===\n")
        trace.append(f"The video showed an animated line graph with {len(self.data_arrays)} groups displaying data ")
        trace.append(f"through a sliding window of 5 points at a time. By analyzing the complete dataset for ")
        trace.append(f"Group {self.group_low} and Group {self.group_high}, I compared their values at each of the ")
        trace.append(f"{len(self.data_arrays[0])} data points. Group {self.group_high} did not exceed Group {self.group_low} ")
        trace.append(f"at {self.answer} points (where Group {self.group_high}'s value was less than or equal to Group {self.group_low}'s value).")

        return "\n".join(trace)


if __name__ == "__main__":
    # Generate the line scroll video
    scene = LineScrollLower()
    scene.render()

    # Move the output file with descriptive name
    output = Path("manim_output/videos/1080p30/LineScrollLower.mp4")
    if output.exists():
        filename = f"line_scroll_lower_d{scene.difficulty}_seed{scene.seed}.mp4"
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
