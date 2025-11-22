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


class LineScrollLeft(Scene):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Set random seed for reproducibility
        self.seed = random.randint(1000, 9999)
        random.seed(self.seed)

        # Parameters
        self.difficulty = int(os.getenv("DIFFICULTY", 1))

        # Background and text colors based on difficulty
        if self.difficulty == 1:
            self.background_color = WHITE
            self.text_color = BLACK
        elif self.difficulty == 2:
            self.background_color = BLACK
            self.text_color = YELLOW
        else:
            self.background_color = BLACK
            self.text_color = WHITE

        # Store for reasoning trace
        self.frame_events = []
        self.data_arrays = []
        self.group_names = []
        self.group_low = None
        self.group_high = None
        self.answer = 0

    def construct(self):
        # Set background color
        self.camera.background_color = self.background_color

        # Generate data
        np.random.seed(self.seed * self.difficulty)
        self.data_arrays = []
        for _ in range(2 + self.difficulty):
            self.data_arrays.append(
                np.random.uniform(30, 60, size=8 + self.difficulty * 2)
            )

        # Generate group names
        np.random.seed(self.seed * self.difficulty)
        groups = [chr(k) for k in range(65, 65 + 26)]
        np.random.shuffle(groups)
        self.group_names = groups[: len(self.data_arrays)]
        self.group_low = groups[0]
        self.group_high = groups[1]

        # Calculate answer
        self.answer = int(np.sum(self.data_arrays[1] < self.data_arrays[0]))

        # Create question text
        question_text = f"At how many points does Group {self.group_high} not exceed Group {self.group_low} in the following graph?"
        title = Text(question_text, font_size=28, color=self.text_color)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(0.5)

        # Setup colors
        np.random.seed(self.seed * self.difficulty)
        COLORS = [
            BLUE,
            RED,
            WHITE,
            LIGHT_BROWN,
            GREEN,
            PURPLE,
            YELLOW,
            ORANGE,
            TEAL,
            MAROON,
            PINK,
            DARK_BLUE,
        ]
        np.random.shuffle(COLORS)

        # Create axes
        ax = Axes(
            x_range=[1, 6 + self.difficulty * 2 + 1, 1],
            y_range=[
                np.floor(np.min(self.data_arrays)),
                np.ceil(np.max(self.data_arrays)),
                5,
            ],
            x_length=5,
            y_length=5,
            axis_config={
                "include_numbers": True,
                "color": self.text_color,
                "label_constructor": lambda val: Text(
                    str(val), color=self.text_color, font_size=24
                ),
            },
        )

        self.play(Create(ax.y_axis), run_time=0.3)
        self.play(Create(ax.x_axis), run_time=0.3)

        # Create legend
        legends = []
        np.random.seed(self.seed * self.difficulty)
        groups = [chr(k) for k in range(65, 65 + 26)]
        np.random.shuffle(groups)
        for k in range(len(self.data_arrays)):
            legend_item = VGroup(
                Square(side_length=0.3, color=COLORS[k], fill_opacity=1),
                Text(f"Group {groups[k]}", color=COLORS[k], font_size=18),
            ).arrange(RIGHT)
            legends.append(legend_item)

        np.random.shuffle(legends)
        legend = (
            VGroup(*legends).arrange(DOWN, aligned_edge=LEFT).to_edge(DOWN + LEFT, buff=0.5)
        )
        self.play(Write(legend), run_time=1)

        # Track cumulative time for reasoning trace
        cumulative_time = 0.0
        cumulative_time += 0.5 + 0.3 + 0.3 + 1  # title + axes + legend

        # Slide window over time (scrolling left)
        total_frames = self.difficulty * 2 + 6
        for i in range(total_frames, 0, -1):
            frame = VGroup()
            frame_data = {}

            for k, data in enumerate(self.data_arrays):
                color = COLORS[k]
                group_name = groups[k]
                # Extract visible portion of data
                visible_data = data[i : i + 5]
                x_coords = list(range(i + 1, i + 1 + len(visible_data)))

                # Store data points for this frame
                if group_name not in frame_data:
                    frame_data[group_name] = []
                frame_data[group_name] = list(zip(x_coords, visible_data))

                # Create line segments
                segment_points = [ax.c2p(x, y) for x, y in zip(x_coords, visible_data)]
                for j in range(len(segment_points) - 1):
                    line = Line(segment_points[j], segment_points[j + 1], color=color)
                    frame.add(line)

            # Record frame event for reasoning trace
            frame_number = total_frames - i + 1
            self.frame_events.append(
                {
                    "frame": frame_number,
                    "start_time": cumulative_time,
                    "data": frame_data,
                    "x_range": (i + 1, i + 5) if len(visible_data) > 0 else (0, 0),
                }
            )

            self.play(Create(frame), run_time=0.3)
            cumulative_time += 0.3
            self.play(FadeOut(frame), run_time=0.3)
            cumulative_time += 0.3

        self.wait(1)
        cumulative_time += 1

        # Save solution
        with open(
            f"solutions/line_scroll_left_d{self.difficulty}_seed{self.seed}.txt", "w"
        ) as f:
            f.write(str(self.answer))

        # Save question text
        question_text_content = (
            f"At how many points does Group {self.group_high} not exceed Group {self.group_low} in the following graph?\n"
            "Please answer with just a number and nothing else."
        )
        with open(
            f"question_text/line_scroll_left_d{self.difficulty}_seed{self.seed}.txt", "w"
        ) as f:
            f.write(question_text_content)

        # Generate and save reasoning trace
        reasoning_trace = self.generate_reasoning_trace()
        with open(
            f"reasoning_traces/line_scroll_left_d{self.difficulty}_seed{self.seed}.txt",
            "w",
        ) as f:
            f.write(reasoning_trace)

    def generate_reasoning_trace(self):
        """Generate a detailed reasoning trace describing the scene chronologically"""

        trace = []
        trace.append("=== CHRONOLOGICAL OBSERVATION ===\n")

        trace.append(
            f"The video displays an animated line graph showing {len(self.data_arrays)} different groups' data over time."
        )
        trace.append(f"The groups are labeled: {', '.join(['Group ' + g for g in self.group_names])}")
        trace.append(
            f"The question asks: At how many points does Group {self.group_high} not exceed Group {self.group_low}?\n"
        )

        trace.append("Animation sequence:")
        trace.append(
            f"- A title appears asking the question about Groups {self.group_high} and {self.group_low}"
        )
        trace.append("- X and Y axes are drawn")
        trace.append(
            f"- A legend appears showing {len(self.data_arrays)} groups with different colors"
        )
        trace.append(
            f"- The graph animates through {len(self.frame_events)} frames, each showing a 5-point window"
        )
        trace.append("- The window scrolls from right to left across the data")
        trace.append(
            f"- Each frame displays for 0.6 seconds (0.3s create + 0.3s fade out)\n"
        )

        trace.append("\n=== DATA ANALYSIS ===\n")

        trace.append("Complete data points for each group:")
        for k, data in enumerate(self.data_arrays):
            group_name = self.group_names[k]
            trace.append(f"\nGroup {group_name}:")
            data_str = ", ".join([f"{val:.2f}" for val in data])
            trace.append(f"  [{data_str}]")

        trace.append("\n\n=== COMPARISON ANALYSIS ===\n")

        trace.append(
            f"To answer the question, I need to compare Group {self.group_high} with Group {self.group_low} at each point."
        )
        trace.append(
            f"Specifically, I'm counting points where Group {self.group_high} does NOT exceed Group {self.group_low}."
        )
        trace.append(
            f"This means counting points where Group {self.group_high} <= Group {self.group_low}.\n"
        )

        # Get the data for the two groups being compared
        group_high_data = self.data_arrays[1]  # Second group
        group_low_data = self.data_arrays[0]  # First group

        trace.append("Point-by-point comparison:")
        count = 0
        for idx, (val_high, val_low) in enumerate(zip(group_high_data, group_low_data), 1):
            comparison = (
                "DOES NOT EXCEED" if val_high < val_low else "exceeds"
            )
            if val_high < val_low:
                count += 1
                trace.append(
                    f"  Point {idx}: Group {self.group_high} = {val_high:.2f}, Group {self.group_low} = {val_low:.2f} → {comparison} ✓"
                )
            else:
                trace.append(
                    f"  Point {idx}: Group {self.group_high} = {val_high:.2f}, Group {self.group_low} = {val_low:.2f} → {comparison}"
                )

        trace.append(f"\nTotal points where Group {self.group_high} does not exceed Group {self.group_low}: {count}")

        trace.append("\n=== REASONING PROCESS ===\n")

        trace.append(
            f"The graph scrolls through all data points by showing a sliding 5-point window."
        )
        trace.append(
            f"While the animation creates a dynamic visualization, the underlying data remains constant."
        )
        trace.append(
            f"To solve this problem, I need to examine all {len(group_high_data)} data points across both groups.\n"
        )

        trace.append("The question has a specific meaning:")
        trace.append(
            f'- "Group {self.group_high} not exceed Group {self.group_low}" means Group {self.group_high} <= Group {self.group_low}'
        )
        trace.append(
            f"- This occurs when Group {self.group_high}'s value is less than Group {self.group_low}'s value"
        )
        trace.append("- We count the total number of such points\n")

        trace.append("By comparing the values at each of the data points:")
        trace.append(f"- Found {self.answer} points where Group {self.group_high} < Group {self.group_low}")
        trace.append(
            f"- Found {len(group_high_data) - self.answer} points where Group {self.group_high} >= Group {self.group_low}"
        )

        trace.append("\n=== FINAL ANSWER ===\n")

        trace.append(
            f"Group {self.group_high} does not exceed Group {self.group_low} at {self.answer} points."
        )
        trace.append(f"\nAnswer: {self.answer}")

        return "\n".join(trace)


if __name__ == "__main__":
    # Generate the line scroll video
    scene = LineScrollLeft()
    scene.render()

    # Move the output file with descriptive name
    output = Path("manim_output/videos/1080p30/LineScrollLeft.mp4")
    if output.exists():
        filename = f"line_scroll_left_d{scene.difficulty}_seed{scene.seed}.mp4"
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
