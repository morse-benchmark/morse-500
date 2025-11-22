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


class LineScroll(Scene):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Set random seed for reproducibility
        self.seed = random.randint(1000, 9999)
        random.seed(self.seed)
        np.random.seed(self.seed)

        # Parameters - can be set via environment variables
        self.difficulty = int(os.getenv("DIFFICULTY", 1))

        # Set colors based on difficulty
        if self.difficulty == 1:
            self.background_color = WHITE
            self.text_color = BLACK
        elif self.difficulty == 2:
            self.background_color = YELLOW
            self.text_color = BLACK
        else:
            self.background_color = BLACK
            self.text_color = WHITE

        # Store for reasoning trace
        self.frame_events = []
        self.data_arrays = []
        self.groups = []
        self.group_colors = []
        self.group_low = None
        self.group_high = None
        self.answer = None

    def construct(self):
        # Set background color
        config.background_color = self.background_color

        # Generate data arrays based on difficulty
        np.random.seed(self.seed * self.difficulty)
        self.data_arrays = []
        num_groups = 2 + self.difficulty
        num_points = 8 + self.difficulty * 2

        for _ in range(num_groups):
            self.data_arrays.append(np.random.uniform(30, 60, size=num_points))

        # Generate group names
        np.random.seed(self.seed * self.difficulty)
        all_groups = [chr(k) for k in range(65, 65 + 26)]
        np.random.shuffle(all_groups)
        self.groups = all_groups[:num_groups]
        self.group_low = self.groups[0]
        self.group_high = self.groups[1]

        # Calculate answer: count how many points group_high exceeds group_low
        self.answer = int(np.sum(self.data_arrays[1] > self.data_arrays[0]))

        # Generate colors for groups
        np.random.seed(self.seed * self.difficulty)
        all_colors = [BLUE, RED, WHITE, LIGHT_BROWN, GREEN, PURPLE, ORANGE, TEAL, MAROON, PINK, DARK_BLUE]
        np.random.shuffle(all_colors)
        self.group_colors = all_colors[:num_groups]

        # Create title/question
        question_text = f"At how many points does Group {self.group_high} exceed Group {self.group_low} in the following graph?"
        title = Text(question_text, font_size=28, color=self.text_color)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(0.5)

        # Create axes
        ax = Axes(
            x_range=[1, num_points + 1, 1],
            y_range=[np.floor(np.min(self.data_arrays)), np.ceil(np.max(self.data_arrays)), 5],
            x_length=5,
            y_length=5,
            axis_config={
                "include_numbers": True,
                "color": self.text_color,
                "label_constructor": lambda val: Text(str(int(val)), color=self.text_color, font_size=24)
            }
        )

        self.play(Create(ax.y_axis), run_time=0.3)
        self.play(Create(ax.x_axis), run_time=0.3)

        # Create legend
        legends = []
        for k in range(len(self.data_arrays)):
            legends.append(VGroup(
                Square(side_length=0.3, color=self.group_colors[k], fill_opacity=1),
                Text(f"Group {self.groups[k]}", color=self.group_colors[k], font_size=DEFAULT_FONT_SIZE - 30)
            ).arrange(RIGHT))
        np.random.shuffle(legends)
        legend = VGroup(*legends).arrange(DOWN, aligned_edge=LEFT).to_edge(DOWN + LEFT, buff=0.5)
        self.play(Write(legend), run_time=1)

        # Track cumulative time for reasoning trace
        cumulative_time = 0.0
        cumulative_time += 0.5 + 0.3 + 0.3 + 1.0  # title + y_axis + x_axis + legend

        # Slide window over time - this creates the scrolling effect
        window_size = 5
        num_frames = self.difficulty * 2 + 6

        for i in range(num_frames):
            frame_start_time = cumulative_time

            # Build frame data: what values are visible in this window
            frame_data = {}
            for k, data in enumerate(self.data_arrays):
                visible_data = data[i:i + window_size]
                x_positions = list(range(i + 1, i + 1 + len(visible_data)))
                frame_data[self.groups[k]] = list(zip(x_positions, visible_data))

            # Check if group_high exceeds group_low at any visible points in this frame
            exceedances_in_frame = []
            if len(self.data_arrays[1][i:i + window_size]) > 0:
                for idx, (val_high, val_low) in enumerate(zip(self.data_arrays[1][i:i + window_size],
                                                                self.data_arrays[0][i:i + window_size])):
                    if val_high > val_low:
                        exceedances_in_frame.append(i + idx + 1)

            # Record event for reasoning trace
            self.frame_events.append({
                "frame_number": i + 1,
                "x_range": (i + 1, i + window_size),
                "start_time": frame_start_time,
                "duration": 0.6,  # 0.3 create + 0.3 fadeout
                "data": frame_data,
                "exceedances": exceedances_in_frame
            })

            # Create lines for this frame
            frame = VGroup()
            for k, data in enumerate(self.data_arrays):
                color = self.group_colors[k]
                visible_data = data[i:i + window_size]
                segment_points = [ax.c2p(x, y) for x, y in zip(range(i + 1, i + 1 + len(visible_data)), visible_data)]

                for j in range(len(segment_points) - 1):
                    line = Line(segment_points[j], segment_points[j + 1], color=color)
                    frame.add(line)

            self.play(Create(frame), run_time=0.3)
            cumulative_time += 0.3

            self.play(FadeOut(frame), run_time=0.3)
            cumulative_time += 0.3

        self.wait(1)

        # Save solution
        with open(f"solutions/line_scroll_d{self.difficulty}_seed{self.seed}.txt", "w") as f:
            f.write(str(self.answer))

        # Save question text
        question_text_content = (
            f"At how many points does Group {self.group_high} exceed Group {self.group_low} in the following graph?\n"
            "Please answer with just a number and nothing else."
        )
        with open(f"question_text/line_scroll_d{self.difficulty}_seed{self.seed}.txt", "w") as f:
            f.write(question_text_content)

        # Generate and save reasoning trace
        reasoning_trace = self.generate_reasoning_trace()
        with open(f"reasoning_traces/line_scroll_d{self.difficulty}_seed{self.seed}.txt", "w") as f:
            f.write(reasoning_trace)

    def generate_reasoning_trace(self):
        """Generate a detailed reasoning trace describing the scene chronologically"""

        trace = []
        trace.append("=== TASK OVERVIEW ===\n")
        trace.append(f"Question: At how many points does Group {self.group_high} exceed Group {self.group_low} in the following graph?")
        trace.append(f"Number of groups displayed: {len(self.data_arrays)}")
        trace.append(f"Total data points per group: {len(self.data_arrays[0])}")
        trace.append(f"Difficulty level: {self.difficulty}")
        trace.append("")

        trace.append("\n=== GROUP INFORMATION ===\n")
        for i, group_name in enumerate(self.groups):
            color_name = self.group_colors[i]
            if hasattr(color_name, 'name'):
                color_name = color_name.name
            trace.append(f"Group {group_name}: Color = {color_name}")
        trace.append("")

        trace.append("\n=== CHRONOLOGICAL OBSERVATION ===\n")
        trace.append("The video shows a scrolling line graph where a window of 5 consecutive points is displayed at a time.")
        trace.append("Each frame shows a different segment of the data as the window scrolls from left to right.\n")

        # Describe each frame
        for event in self.frame_events:
            trace.append(f"Frame {event['frame_number']} (Time: {event['start_time']:.2f}s - {event['start_time'] + event['duration']:.2f}s):")
            trace.append(f"  X-axis range visible: {event['x_range'][0]} to {event['x_range'][1]}")

            # Show data values for the two groups we're comparing
            if self.group_low in event['data'] and self.group_high in event['data']:
                trace.append(f"  Group {self.group_low} values: {[f'{y:.2f}' for x, y in event['data'][self.group_low]]}")
                trace.append(f"  Group {self.group_high} values: {[f'{y:.2f}' for x, y in event['data'][self.group_high]]}")

            if event['exceedances']:
                trace.append(f"  Points where Group {self.group_high} exceeds Group {self.group_low}: {event['exceedances']}")
            else:
                trace.append(f"  No exceedances in this frame")
            trace.append("")

        trace.append("\n=== DETAILED COMPARISON ANALYSIS ===\n")
        trace.append(f"Comparing Group {self.group_high} vs Group {self.group_low} at each point:\n")

        all_exceedances = []
        for i in range(len(self.data_arrays[0])):
            val_low = self.data_arrays[0][i]
            val_high = self.data_arrays[1][i]
            exceeds = val_high > val_low

            trace.append(f"Point {i + 1}:")
            trace.append(f"  Group {self.group_low}: {val_low:.2f}")
            trace.append(f"  Group {self.group_high}: {val_high:.2f}")
            trace.append(f"  Does Group {self.group_high} exceed Group {self.group_low}? {'YES' if exceeds else 'NO'}")

            if exceeds:
                all_exceedances.append(i + 1)
            trace.append("")

        trace.append("\n=== COUNTING PROCESS ===\n")
        trace.append(f"Points where Group {self.group_high} exceeds Group {self.group_low}:")
        if all_exceedances:
            trace.append(f"  {all_exceedances}")
            trace.append(f"\nTotal count: {len(all_exceedances)} points")
        else:
            trace.append("  None")
            trace.append("\nTotal count: 0 points")
        trace.append("")

        trace.append("\n=== FINAL ANSWER ===\n")
        trace.append(f"Group {self.group_high} exceeds Group {self.group_low} at {self.answer} points.")
        trace.append("")

        trace.append("\n=== REASONING SUMMARY ===\n")
        trace.append(
            f"I observed a scrolling line graph showing {len(self.data_arrays)} groups over {len(self.data_arrays[0])} data points. "
        )
        trace.append(
            f"The graph displayed 5 consecutive points at a time in a scrolling window. "
        )
        trace.append(
            f"To answer the question, I compared the values of Group {self.group_high} and Group {self.group_low} "
        )
        trace.append(
            f"at each of the {len(self.data_arrays[0])} points. "
        )
        trace.append(
            f"By counting how many times Group {self.group_high}'s value was greater than Group {self.group_low}'s value, "
        )
        trace.append(
            f"I determined that Group {self.group_high} exceeded Group {self.group_low} at {self.answer} points."
        )

        return "\n".join(trace)


if __name__ == "__main__":
    # Generate the line scroll video
    scene = LineScroll()
    scene.render()

    # Move the output file with descriptive name
    output = Path("manim_output/videos/1080p30/LineScroll.mp4")
    if output.exists():
        filename = f"line_scroll_d{scene.difficulty}_seed{scene.seed}.mp4"
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
