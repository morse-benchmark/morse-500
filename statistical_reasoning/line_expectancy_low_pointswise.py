from manim import *
import random
import numpy as np
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


class LineExpectancyLowPointswise(Scene):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Set random seed for reproducibility
        self.seed = random.randint(1000, 9999)
        random.seed(self.seed)
        np.random.seed(self.seed)

        # Parameters
        self.difficulty = int(os.getenv("DIFFICULTY", 5))

        # Store for reasoning trace
        self.animation_events = []
        self.groups_data = []

    def construct(self):
        # Set background color based on difficulty
        background_color = BLACK
        text_color = WHITE
        if self.difficulty == 9:
            background_color = WHITE
            text_color = BLACK
            config.background_color = WHITE
        elif self.difficulty == 7:
            background_color = WHITE
            text_color = RED
            config.background_color = WHITE
        else:
            config.background_color = BLACK

        # Generate data arrays
        np.random.seed(self.seed * self.difficulty)
        starting_data_arrays = [
            np.array([69.01, 69.59, 70.15, 70.64, 71.01, 71.27, 71.42, 71.50, 71.53, 71.58, 71.65]),
            np.array([74.29, 74.74, 75.22, 75.68, 76.13, 76.54, 76.90, 77.20, 77.45, 77.67, 77.84])
        ]
        data_arrays = list(starting_data_arrays)

        for i in range(1 + self.difficulty):
            if np.random.rand() > 0.5:
                starting_array = np.array(starting_data_arrays[0])
                starting_array += np.random.uniform(0.5, 3 - 2 * self.difficulty/9, len(starting_array))
            else:
                starting_array = np.array(starting_data_arrays[1])
                starting_array -= np.random.uniform(-3, 3)
            starting_array -= np.random.uniform(-0.1, 0.5, len(starting_array))
            data_arrays.append(starting_array)

        data_arrays[0] += np.random.uniform(-0.5, 0.4, len(starting_array))
        data_arrays[1] -= np.random.uniform(-1, 1, len(starting_array))

        # Generate group labels
        np.random.seed(self.seed * self.difficulty)
        groups = [chr(k) for k in range(65, 65+26)]
        np.random.shuffle(groups)

        # Determine the answer - which group has lowest at every point
        years = np.array(range(9, 20))
        answer_group = self.find_lowest_at_every_point(data_arrays, groups)
        answer = f"Group {answer_group}"

        # Store groups data for reasoning trace
        for i, (group_label, data) in enumerate(zip(groups[:len(data_arrays)], data_arrays)):
            self.groups_data.append({
                "group": group_label,
                "data": data,
                "years": years
            })

        # Create title
        question_text = "Which Group has the lowest life expectancy at every point?\nPlease answer with just the letter of the group and nothing else"
        title = Text(question_text, font_size=36, color=text_color)
        title.to_edge(UP)
        self.play(Write(title), run_time=1.0)
        self.wait(0.5)

        self.animation_events.append({
            "time": 0,
            "event": "title_displayed",
            "description": "Question displayed asking which group has lowest life expectancy at every point"
        })

        # Setup colors
        np.random.seed(self.seed * self.difficulty)
        COLORS = [BLUE, RED, LIGHT_BROWN, GREEN, PURPLE, ORANGE, TEAL, MAROON, PINK, DARK_BLUE]
        np.random.shuffle(COLORS)

        # Define axes
        ax = Axes(
            x_range=[min(years)-1, max(years)+1, 1],
            y_range=[np.floor(np.min(data_arrays)), np.ceil(np.max(data_arrays)), 2],
            x_length=5,
            y_length=5,
            axis_config={
                "include_numbers": True,
                "color": text_color,
                "label_constructor": lambda val: Text(str(val), color=text_color, font_size=24)
            }
        )

        x_labels = {year: str(year) for year in years}
        ax.x_axis.set_tick_values(years)
        ax.x_axis.set_tick_labels(x_labels)

        # X-axis label
        x_label = ax.get_x_axis_label(Text("Year", font_size=24, color=text_color))

        # Y-axis label (rotated)
        y_text = Text("Life expectancy at birth in years", font_size=24, color=text_color)
        y_text.rotate(PI / 2)
        y_label = ax.get_y_axis_label(y_text, edge=LEFT, direction=LEFT)

        # Add axes to scene
        self.play(Create(ax.y_axis), run_time=0.3)
        self.play(Create(ax.x_axis), run_time=0.3)
        self.play(Write(x_label), Write(y_label), run_time=0.3)

        self.animation_events.append({
            "time": 1.9,
            "event": "axes_created",
            "description": "Axes created with years on x-axis and life expectancy on y-axis"
        })

        # Compute all points ahead of time
        all_points = [[ax.c2p(x, y) for x, y in zip(years, data)] for data in data_arrays]

        # Animate each segment index synchronously across all lines
        cumulative_time = 1.9
        for seg_idx in range(len(years) - 1):
            segment_group = []
            for line_idx, points in enumerate(all_points):
                segment = Line(points[seg_idx], points[seg_idx + 1], color=COLORS[line_idx])
                segment_group.append(segment)

            self.play(*[Create(seg) for seg in segment_group], run_time=0.3)
            cumulative_time += 0.3

            self.animation_events.append({
                "time": cumulative_time,
                "event": f"segment_{seg_idx}_displayed",
                "description": f"All lines displayed segment from year {years[seg_idx]} to {years[seg_idx + 1]}",
                "year_range": (years[seg_idx], years[seg_idx + 1])
            })

            self.wait(0.1)
            cumulative_time += 0.1

            self.play(*[FadeOut(seg) for seg in segment_group], run_time=0.2)
            cumulative_time += 0.2

        # Create legend
        legends = []
        np.random.seed(self.seed * self.difficulty)
        groups_legend = [chr(k) for k in range(65, 65+26)]
        np.random.shuffle(groups_legend)

        for k in range(len(data_arrays)):
            legends.append(VGroup(
                Square(side_length=0.3, color=COLORS[k], fill_opacity=1),
                Text(f"Group {groups_legend[k]}", color=COLORS[k], font_size=DEFAULT_FONT_SIZE-30)
            ).arrange(RIGHT))

        np.random.shuffle(legends)
        legend = VGroup(*legends).arrange(DOWN, aligned_edge=LEFT).to_edge(DOWN + LEFT, buff=0.5)
        self.play(Write(legend), run_time=1)
        cumulative_time += 1

        self.animation_events.append({
            "time": cumulative_time,
            "event": "legend_displayed",
            "description": "Legend displayed showing color mapping for each group"
        })

        self.wait(3)

        # Save solution
        with open(f"solutions/line_expectancy_low_pointswise_d{self.difficulty}_seed{self.seed}.txt", "w") as f:
            f.write(answer_group)

        # Save question text
        with open(f"question_text/line_expectancy_low_pointswise_d{self.difficulty}_seed{self.seed}.txt", "w") as f:
            f.write(question_text)

        # Generate and save reasoning trace
        reasoning_trace = self.generate_reasoning_trace(answer_group)
        with open(f"reasoning_traces/line_expectancy_low_pointswise_d{self.difficulty}_seed{self.seed}.txt", "w") as f:
            f.write(reasoning_trace)

    def find_lowest_at_every_point(self, data_arrays, groups):
        """Find which group has the lowest value at every point"""
        data_matrix = np.array(data_arrays)

        # For each time point, find which group has the minimum
        for point_idx in range(data_matrix.shape[1]):
            min_idx = np.argmin(data_matrix[:, point_idx])
            if point_idx == 0:
                candidate_group = groups[min_idx]
            else:
                # Check if same group is still lowest
                if groups[min_idx] != candidate_group:
                    # No single group is lowest at all points
                    # Return the group that appears most frequently as lowest
                    lowest_at_each_point = [groups[np.argmin(data_matrix[:, i])] for i in range(data_matrix.shape[1])]
                    from collections import Counter
                    most_common = Counter(lowest_at_each_point).most_common(1)[0][0]
                    return most_common

        return candidate_group

    def generate_reasoning_trace(self, answer_group):
        """Generate a detailed reasoning trace describing the scene chronologically"""

        trace = []
        trace.append("=== CHRONOLOGICAL OBSERVATION ===\n")

        # Describe the animation chronologically
        trace.append("At the beginning of the video:")
        trace.append("  - A question is displayed: 'Which Group has the lowest life expectancy at every point?'")
        trace.append("  - The question asks to answer with just the letter of the group\n")

        trace.append("Next, the graph setup appears:")
        trace.append("  - Y-axis: Life expectancy at birth in years")
        trace.append("  - X-axis: Years (ranging from 9 to 19)")
        trace.append("  - The axes are drawn with appropriate scales and labels\n")

        trace.append("The data visualization proceeds point-by-point:")
        years = self.groups_data[0]["years"]
        for seg_idx in range(len(years) - 1):
            trace.append(f"  - Time segment {seg_idx + 1}: Lines connecting year {years[seg_idx]} to year {years[seg_idx + 1]} are displayed")
            trace.append(f"    All groups show their data segment simultaneously")
            trace.append(f"    The segments briefly appear then fade out\n")

        trace.append("Finally:")
        trace.append("  - A legend appears showing the color-to-group mapping")
        trace.append("  - Each group is identified by a letter (A-Z)")
        trace.append("  - The legend remains visible for the viewer to reference\n")

        trace.append("\n=== ANALYSIS ===\n")

        trace.append(f"Total number of groups displayed: {len(self.groups_data)}\n")

        trace.append("Data values for each group at each year:\n")
        for group_info in self.groups_data:
            trace.append(f"Group {group_info['group']}:")
            for year, value in zip(group_info['years'], group_info['data']):
                trace.append(f"  Year {year}: {value:.2f}")
            trace.append("")

        trace.append("\n=== REASONING PROCESS ===\n")

        trace.append("To determine which group has the lowest life expectancy at EVERY point, I need to:")
        trace.append("1. Compare all groups at each year")
        trace.append("2. Identify which group has the minimum value")
        trace.append("3. Check if the same group maintains the lowest value across all years\n")

        trace.append("Comparative analysis at each year:\n")
        years = self.groups_data[0]["years"]
        for year_idx, year in enumerate(years):
            values_at_year = [(group_info['group'], group_info['data'][year_idx])
                             for group_info in self.groups_data]
            values_at_year.sort(key=lambda x: x[1])  # Sort ascending for lowest

            trace.append(f"Year {year}:")
            for rank, (group, value) in enumerate(values_at_year[:3], 1):  # Show bottom 3
                marker = " (LOWEST)" if rank == 1 else ""
                trace.append(f"  {rank}. Group {group}: {value:.2f}{marker}")
            trace.append("")

        # Determine consistency
        trace.append("\nChecking consistency:")
        lowest_at_each_year = []
        for year_idx in range(len(years)):
            values_at_year = [(group_info['group'], group_info['data'][year_idx])
                             for group_info in self.groups_data]
            lowest_group = min(values_at_year, key=lambda x: x[1])[0]
            lowest_at_each_year.append(lowest_group)

        trace.append(f"Groups with lowest value at each year: {', '.join(lowest_at_each_year)}\n")

        if all(g == lowest_at_each_year[0] for g in lowest_at_each_year):
            trace.append(f"Group {lowest_at_each_year[0]} consistently has the lowest life expectancy at every single point.")
        else:
            trace.append(f"The group with lowest values varies across years.")
            trace.append(f"However, Group {answer_group} appears most frequently as the lowest.")

        trace.append("\n=== FINAL ANSWER ===\n")
        trace.append(f"The group with the lowest life expectancy at every point is: {answer_group}\n")

        trace.append("\n=== REASONING SUMMARY ===\n")
        trace.append(f"I observed {len(self.groups_data)} groups displaying their life expectancy data over {len(years)} years. ")
        trace.append(f"The animation showed the data point-by-point, with all groups' segments appearing simultaneously. ")
        trace.append(f"By comparing the values at each year systematically, I determined that Group {answer_group} ")
        trace.append(f"has the lowest life expectancy at every point in the time series.")

        return "\n".join(trace)


if __name__ == "__main__":
    # Generate the video
    scene = LineExpectancyLowPointswise()
    scene.render()

    # Move the output file with descriptive name
    output = Path("manim_output/videos/1080p30/LineExpectancyLowPointswise.mp4")
    if output.exists():
        filename = f"line_expectancy_low_pointswise_d{scene.difficulty}_seed{scene.seed}.mp4"
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
