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


class LineScrollLeftUpper(Scene):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Set random seed for reproducibility
        self.seed = random.randint(1000, 9999)
        random.seed(self.seed)

        # Parameters
        self.difficulty = int(os.getenv("DIFFICULTY", 1))

        # Set colors based on difficulty
        if self.difficulty == 1:
            self.background_color = WHITE
            self.text_color = BLACK
            config.background_color = self.background_color
        elif self.difficulty == 2:
            self.background_color = WHITE
            self.text_color = RED
            config.background_color = self.background_color
        else:
            self.background_color = BLACK
            self.text_color = WHITE
            config.background_color = self.background_color

        # Store for reasoning trace
        self.frame_events = []
        self.data_arrays = []
        self.groups = []
        self.group_low = ""
        self.group_high = ""
        self.colors = []

    def construct(self):
        # Generate data
        np.random.seed(self.seed * self.difficulty)
        self.data_arrays = []
        for _ in range(2 + self.difficulty):
            self.data_arrays.append(np.random.uniform(30, 60, size=8 + self.difficulty * 2))

        # Generate group names
        np.random.seed(self.seed * self.difficulty)
        self.groups = [chr(k) for k in range(65, 65 + 26)]
        np.random.shuffle(self.groups)
        self.group_low = self.groups[0]
        self.group_high = self.groups[1]

        # Calculate answer
        answer = np.sum(self.data_arrays[1] > self.data_arrays[0])

        # Create question text
        question_text = f"At how many points does Group {self.group_high} exceed Group {self.group_low} in the following graph?"

        # Display title
        title = Text(question_text, font_size=28, color=self.text_color)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(0.5)

        # Setup colors
        np.random.seed(self.seed * self.difficulty)
        COLORS = [BLUE, RED, LIGHT_BROWN, GREEN, PURPLE, YELLOW, ORANGE, TEAL, MAROON, PINK, DARK_BLUE]
        np.random.shuffle(COLORS)
        self.colors = COLORS[:len(self.data_arrays)]

        # Create axes
        ax = Axes(
            x_range=[1, (6 + self.difficulty * 2) + 1, 1],
            y_range=[np.floor(np.min(self.data_arrays)), np.ceil(np.max(self.data_arrays)), 5],
            x_length=5,
            y_length=5,
            axis_config={
                "include_numbers": True,
                "color": self.text_color,
                "label_constructor": lambda val: Text(str(val), color=self.text_color, font_size=24)
            }
        )

        self.play(Create(ax.y_axis), run_time=0.3)
        self.play(Create(ax.x_axis), run_time=0.3)

        # Create legend
        legends = []
        np.random.seed(self.seed * self.difficulty)
        groups_copy = [chr(k) for k in range(65, 65 + 26)]
        np.random.shuffle(groups_copy)
        for k in range(len(self.data_arrays)):
            legends.append(VGroup(
                Square(side_length=0.3, color=self.colors[k], fill_opacity=1),
                Text(f"Group {groups_copy[k]}", color=self.colors[k], font_size=DEFAULT_FONT_SIZE - 30)
            ).arrange(RIGHT))
        np.random.shuffle(legends)
        legend = VGroup(*legends).arrange(DOWN, aligned_edge=LEFT).to_edge(DOWN + LEFT, buff=0.5)
        self.play(Write(legend), run_time=1)

        # Slide window over time (scrolling from right to left)
        cumulative_time = 0.0
        total_frames = self.difficulty * 2 + 6

        for i in range(total_frames, 0, -1):
            frame = VGroup()
            frame_data = {}

            for k, data in enumerate(self.data_arrays):
                color = self.colors[k]
                segment_points = [ax.c2p(x, y) for x, y in zip(range(i + 1, i + 1 + len(data[i:i + 5])), data[i:i + 5])]

                # Store segment data for reasoning trace
                if i not in frame_data:
                    frame_data[i] = []

                if len(segment_points) > 0:
                    frame_data[i].append({
                        'group': groups_copy[k],
                        'color': color,
                        'points': [(x, y) for x, y in zip(range(i + 1, i + 1 + len(data[i:i + 5])), data[i:i + 5])],
                        'values': data[i:i + 5].tolist()
                    })

                for j in range(len(segment_points) - 1):
                    line = Line(segment_points[j], segment_points[j + 1], color=color)
                    frame.add(line)

            # Record event for reasoning trace
            self.frame_events.append({
                'frame_number': total_frames - i + 1,
                'x_range_start': i + 1,
                'frame_data': frame_data.get(i, []),
                'start_time': cumulative_time,
                'display_duration': 0.6
            })

            self.play(Create(frame), run_time=0.3)
            cumulative_time += 0.3
            self.play(FadeOut(frame), run_time=0.3)
            cumulative_time += 0.3

        self.wait(1)

        # Save solution
        with open(f"solutions/line_scroll_left_upper_diff{self.difficulty}_seed{self.seed}.txt", "w") as f:
            f.write(str(answer))

        # Save question text
        question_text_content = (
            f"At how many points does Group {self.group_high} exceed Group {self.group_low} in the following graph?\n"
            "Please answer with just a number and nothing else."
        )
        with open(f"question_text/line_scroll_left_upper_diff{self.difficulty}_seed{self.seed}.txt", "w") as f:
            f.write(question_text_content)

        # Generate and save reasoning trace
        reasoning_trace = self.generate_reasoning_trace(answer)
        with open(f"reasoning_traces/line_scroll_left_upper_diff{self.difficulty}_seed{self.seed}.txt", "w") as f:
            f.write(reasoning_trace)

    def generate_reasoning_trace(self, answer):
        """Generate a detailed reasoning trace describing the scene chronologically"""

        trace = []
        trace.append("=== CHRONOLOGICAL OBSERVATION ===\n")

        # Describe the setup
        trace.append("Scene Setup:")
        trace.append(f"  - Question displayed: 'At how many points does Group {self.group_high} exceed Group {self.group_low}?'")
        trace.append(f"  - Number of groups displayed: {len(self.data_arrays)}")
        trace.append(f"  - Axes created with x-range and y-range for data visualization")
        trace.append(f"  - Legend created showing {len(self.data_arrays)} groups with different colors")
        trace.append("")

        # Describe the scrolling animation
        trace.append("Scrolling Animation:")
        trace.append(f"  - The graph displays a sliding window that scrolls from right to left")
        trace.append(f"  - Total frames displayed: {len(self.frame_events)}")
        trace.append(f"  - Each frame shows a 5-point window of the line graph")
        trace.append(f"  - The window moves leftward, revealing the full data progression")
        trace.append("")

        # Sample some frames for detailed observation
        trace.append("Key Frame Observations:")
        sample_frames = [0, len(self.frame_events) // 2, len(self.frame_events) - 1]
        for frame_idx in sample_frames:
            if frame_idx < len(self.frame_events):
                event = self.frame_events[frame_idx]
                trace.append(f"  Frame {event['frame_number']}:")
                trace.append(f"    - X-range starts at: {event['x_range_start']}")
                trace.append(f"    - Displayed at time: {event['start_time']:.2f}s")
        trace.append("")

        trace.append("\n=== ANALYSIS ===\n")

        # Analyze the data arrays
        trace.append("Data Analysis:")
        trace.append(f"  - Group {self.group_low} data points: {self.data_arrays[0].tolist()}")
        trace.append(f"  - Group {self.group_high} data points: {self.data_arrays[1].tolist()}")
        trace.append("")

        trace.append("Comparison at each point:")
        exceed_points = []
        for idx in range(len(self.data_arrays[0])):
            group_low_val = self.data_arrays[0][idx]
            group_high_val = self.data_arrays[1][idx]
            exceeds = group_high_val > group_low_val
            if exceeds:
                exceed_points.append(idx + 1)
            trace.append(f"  Point {idx + 1}: Group {self.group_low} = {group_low_val:.2f}, Group {self.group_high} = {group_high_val:.2f} -> {self.group_high} {'exceeds' if exceeds else 'does not exceed'} {self.group_low}")
        trace.append("")

        trace.append("\n=== REASONING PROCESS ===\n")

        trace.append("To answer the question, I need to count how many points Group {self.group_high} exceeds Group {self.group_low}.")
        trace.append("")
        trace.append(f"Points where Group {self.group_high} exceeds Group {self.group_low}:")
        if exceed_points:
            trace.append(f"  {', '.join(map(str, exceed_points))}")
        else:
            trace.append("  None")
        trace.append("")
        trace.append(f"Total count: {len(exceed_points)} points")
        trace.append("")

        trace.append("\n=== FINAL ANSWER ===\n")
        trace.append(f"Group {self.group_high} exceeds Group {self.group_low} at {answer} points in the graph.")
        trace.append("")

        trace.append("\n=== REASONING SUMMARY ===\n")
        trace.append(f"I observed a scrolling line graph showing {len(self.data_arrays)} different groups. ")
        trace.append(f"The animation displayed a sliding window that moved from right to left, ")
        trace.append(f"revealing the complete data progression over time. ")
        trace.append(f"By comparing the values of Group {self.group_high} and Group {self.group_low} at each x-axis point, ")
        trace.append(f"I counted {answer} points where Group {self.group_high}'s value exceeded Group {self.group_low}'s value. ")
        trace.append(f"This was determined by examining all {len(self.data_arrays[0])} data points across the x-axis.")

        return "\n".join(trace)


if __name__ == "__main__":
    # Generate the line scroll video
    scene = LineScrollLeftUpper()
    scene.render()

    # Move the output file with descriptive name
    output = Path("manim_output/videos/1080p30/LineScrollLeftUpper.mp4")
    if output.exists():
        filename = f"line_scroll_left_upper_diff{scene.difficulty}_seed{scene.seed}.mp4"
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
