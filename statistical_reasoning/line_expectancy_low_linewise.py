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


class LifeExpectancyLowLinewise(Scene):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Set random seed for reproducibility
        self.seed = random.randint(1000, 9999)
        random.seed(self.seed)
        np.random.seed(self.seed)

        # Parameters
        self.difficulty = int(os.getenv("DIFFICULTY", 5))

        # Store for reasoning trace
        self.group_events = []
        self.data_arrays = []
        self.years = []
        self.groups = []
        self.colors = []

    def construct(self):
        # Generate data
        background_color = BLACK
        text_color = WHITE

        if self.difficulty == 9:
            background_color = WHITE
            text_color = BLACK
        if self.difficulty == 7:
            background_color = BLACK
            text_color = YELLOW

        config.background_color = background_color

        np.random.seed(self.seed * self.difficulty)

        starting_data_arrays = [
            np.array([69.01, 69.59, 70.15, 70.64, 71.01, 71.27, 71.42, 71.50, 71.53, 71.58, 71.65]),
            np.array([74.29, 74.74, 75.22, 75.68, 76.13, 76.54, 76.90, 77.20, 77.45, 77.67, 77.84])
        ]

        self.data_arrays = [np.array(arr) for arr in starting_data_arrays]

        for i in range(1 + self.difficulty):
            if np.random.rand() > 0.5:
                starting_array = np.array(starting_data_arrays[0])
                starting_array += np.random.uniform(0.5, 3 - 2 * self.difficulty/9, len(starting_array))
            else:
                starting_array = np.array(starting_data_arrays[1])
                starting_array -= np.random.uniform(-3, 3)
            starting_array -= np.random.uniform(-0.1, 0.5, len(starting_array))
            self.data_arrays.append(starting_array)

        self.data_arrays[0] += np.random.uniform(-0.5, 0.4, len(self.data_arrays[0]))
        self.data_arrays[1] -= np.random.uniform(-1, 1, len(self.data_arrays[1]))

        np.random.seed(self.seed * self.difficulty)
        self.groups = [chr(k) for k in range(65, 65+26)]
        np.random.shuffle(self.groups)

        # Determine answer: group with lowest life expectancy at every point
        min_values_per_year = np.min(self.data_arrays, axis=0)
        answer_group = None
        for idx, data in enumerate(self.data_arrays):
            if np.allclose(data, min_values_per_year):
                answer_group = self.groups[idx]
                break

        if answer_group is None:
            # Find the group that is lowest at most points
            lowest_counts = []
            for idx, data in enumerate(self.data_arrays):
                count = sum(1 for i in range(len(data)) if data[i] == min_values_per_year[i])
                lowest_counts.append((count, idx))
            lowest_counts.sort(reverse=True)
            answer_group = self.groups[lowest_counts[0][1]]

        self.answer = f"Group {answer_group}"
        self.question = "Which Group has the lowest life expectancy at every point?\nPlease answer with just the letter of the group and nothing else"

        # Display question
        title = Text(self.question, font_size=36, color=text_color)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(0.5)

        # Setup colors
        np.random.seed(self.seed * self.difficulty)
        COLORS = [BLUE, RED, WHITE, LIGHT_BROWN, GREEN, PURPLE, YELLOW, ORANGE, TEAL, MAROON, PINK, DARK_BLUE]
        np.random.shuffle(COLORS)
        self.colors = COLORS[:len(self.data_arrays)]

        # Setup years
        self.years = np.array(range(9, 20))

        # Define axes
        ax = Axes(
            x_range=[min(self.years)-1, max(self.years)+1, 1],
            y_range=[np.floor(np.min(self.data_arrays)), np.ceil(np.max(self.data_arrays)), 2],
            x_length=5,
            y_length=5,
            axis_config={
                "include_numbers": True,
                "color": text_color,
                "label_constructor": lambda val: Text(str(val), color=text_color, font_size=24)
            }
        )

        x_labels = {year: str(year) for year in self.years}
        ax.x_axis.set_tick_values(self.years)
        ax.x_axis.set_tick_labels(x_labels)

        # X-axis label (standard horizontal)
        x_label = ax.get_x_axis_label(Text("Year", font_size=24, color=text_color))

        # Y-axis label (rotate for vertical orientation)
        y_text = Text("Life expectancy at birth in years", font_size=24, color=text_color)
        y_text.rotate(PI / 2)  # Rotate 90 degrees counterclockwise
        y_label = ax.get_y_axis_label(y_text, edge=LEFT, direction=LEFT)

        # Add to scene
        self.play(Create(ax.y_axis), run_time=0.3)
        self.play(Create(ax.x_axis), run_time=0.3)
        self.play(Write(x_label), Write(y_label), run_time=0.3)

        # Track cumulative time for reasoning trace
        cumulative_time = 0.5 + 0.3 + 0.3 + 0.3  # title + axes + labels

        # Animate each data line one at a time
        for k, data in enumerate(self.data_arrays):
            points = [ax.c2p(x, y) for x, y in zip(self.years, data)]
            full_graph = VGroup()

            # Record event for reasoning trace
            start_time = cumulative_time
            group_name = self.groups[k]
            color_name = self.colors[k]

            # Create segments and add to full graph
            for i in range(len(points) - 1):
                line = Line(points[i], points[i + 1], color=self.colors[k])
                full_graph.add(line)

            # Show segments one-by-one
            segment_time = 0.0
            for line in full_graph:
                self.play(Create(line), run_time=0.05)
                segment_time += 0.05

            # Wait before fading out the full line
            self.play(FadeOut(full_graph), run_time=0.2)

            end_time = cumulative_time + segment_time + 0.2

            self.group_events.append({
                "group": group_name,
                "color": color_name,
                "data": data.tolist(),
                "start_time": start_time,
                "display_duration": segment_time,
                "end_time": end_time,
                "order": k + 1
            })

            cumulative_time = end_time

        # Create legend
        legends = []
        np.random.seed(self.seed * self.difficulty)
        groups_legend = [chr(k) for k in range(65, 65+26)]
        np.random.shuffle(groups_legend)

        for k in range(len(self.data_arrays)):
            legends.append(VGroup(
                Square(side_length=0.3, color=self.colors[k], fill_opacity=1),
                Text(f"Group {groups_legend[k]}", color=self.colors[k], font_size=DEFAULT_FONT_SIZE-30)
            ).arrange(RIGHT))

        np.random.shuffle(legends)
        legend = VGroup(*legends).arrange(DOWN, aligned_edge=LEFT).to_edge(DOWN + LEFT, buff=0.5)
        self.play(Write(legend), run_time=1)
        self.wait(3)

        # Save solution
        with open(f"solutions/line_expectancy_low_linewise_d{self.difficulty}_seed{self.seed}.txt", "w") as f:
            f.write(self.answer)

        # Save question text
        with open(f"question_text/line_expectancy_low_linewise_d{self.difficulty}_seed{self.seed}.txt", "w") as f:
            f.write(self.question)

        # Generate and save reasoning trace
        reasoning_trace = self.generate_reasoning_trace()
        with open(f"reasoning_traces/line_expectancy_low_linewise_d{self.difficulty}_seed{self.seed}.txt", "w") as f:
            f.write(reasoning_trace)

    def generate_reasoning_trace(self):
        """Generate a detailed reasoning trace describing the scene chronologically"""

        trace = []
        trace.append("=== CHRONOLOGICAL OBSERVATION ===\n")

        trace.append("The video begins with a question displayed at the top:")
        trace.append(f'"{self.question}"\n')

        trace.append("An axes system is created showing:")
        trace.append(f"  - X-axis: Years from {min(self.years)} to {max(self.years)}")
        trace.append(f"  - Y-axis: Life expectancy ranging from approximately {np.floor(np.min(self.data_arrays)):.1f} to {np.ceil(np.max(self.data_arrays)):.1f} years\n")

        trace.append("The following line graphs are drawn sequentially, each appearing briefly then fading:\n")

        # Describe each line appearance in order
        for i, event in enumerate(self.group_events, 1):
            trace.append(f"Line {i}: Group {event['group']}")
            trace.append(f"  - Appeared at time {event['start_time']:.2f}s")
            trace.append(f"  - Color: {event['color']}")
            trace.append(f"  - Data points (year, life expectancy):")
            for year, value in zip(self.years, event['data']):
                trace.append(f"    * Year {year}: {value:.2f} years")
            trace.append(f"  - Display duration: {event['display_duration']:.2f} seconds")
            trace.append(f"  - Faded out at time {event['end_time']:.2f}s")
            trace.append("")

        trace.append("After all lines are shown, a legend appears identifying each group by its color.\n")

        trace.append("\n=== ANALYSIS ===\n")

        trace.append("To determine which group has the lowest life expectancy at every point,")
        trace.append("I need to compare the values across all groups for each year.\n")

        # Analyze each year
        trace.append("Year-by-year comparison:\n")
        for year_idx, year in enumerate(self.years):
            values_at_year = []
            for event in self.group_events:
                values_at_year.append((event['group'], event['data'][year_idx]))

            # Sort by value to find minimum
            values_at_year.sort(key=lambda x: x[1])
            trace.append(f"Year {year}:")
            for group, value in values_at_year[:3]:  # Show top 3 lowest
                marker = " <- LOWEST" if group == values_at_year[0][0] else ""
                trace.append(f"  - Group {group}: {value:.2f}{marker}")
            if len(values_at_year) > 3:
                trace.append(f"  - ... ({len(values_at_year) - 3} more groups)")
            trace.append("")

        trace.append("\n=== REASONING PROCESS ===\n")

        # Find which group is consistently lowest
        trace.append("Identifying which group has the lowest value at each year:\n")

        min_counts = {}
        for year_idx, year in enumerate(self.years):
            min_value = min(event['data'][year_idx] for event in self.group_events)
            for event in self.group_events:
                if event['data'][year_idx] == min_value:
                    if event['group'] not in min_counts:
                        min_counts[event['group']] = []
                    min_counts[event['group']].append(year)

        for group, years_list in min_counts.items():
            trace.append(f"Group {group} has the lowest value in years: {', '.join(map(str, years_list))} ({len(years_list)} years)")

        trace.append("")

        # Determine the answer
        answer_group = self.answer.replace("Group ", "")
        trace.append(f"Group {answer_group} consistently has the lowest life expectancy across all time points.")
        trace.append(f"This makes it the group with the lowest life expectancy at every point.\n")

        trace.append("\n=== FINAL ANSWER ===\n")
        trace.append(f"{self.answer}")
        trace.append("")

        trace.append("\n=== REASONING SUMMARY ===\n")
        trace.append(f"I observed {len(self.group_events)} different groups plotted as line graphs showing life expectancy over time. ")
        trace.append(f"Each line appeared sequentially, displaying data from year {min(self.years)} to {max(self.years)}. ")
        trace.append(f"By comparing the life expectancy values across all groups at each year, I identified which group ")
        trace.append(f"consistently had the lowest values throughout the entire time period. ")
        trace.append(f"The answer is {self.answer}.")

        return "\n".join(trace)


if __name__ == "__main__":
    # Generate the line expectancy video
    scene = LifeExpectancyLowLinewise()
    scene.render()

    # Move the output file with descriptive name
    output = Path("manim_output/videos/1080p30/LifeExpectancyLowLinewise.mp4")
    if output.exists():
        filename = f"line_expectancy_low_linewise_d{scene.difficulty}_seed{scene.seed}.mp4"
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
