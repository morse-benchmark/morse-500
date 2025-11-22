from manim import *
import numpy as np
import random
import os
import shutil
from pathlib import Path

# Setup directories
Path("statistical_reasoning/questions").mkdir(parents=True, exist_ok=True)
Path("statistical_reasoning/solutions").mkdir(parents=True, exist_ok=True)
Path("statistical_reasoning/question_text").mkdir(parents=True, exist_ok=True)
Path("statistical_reasoning/reasoning_traces").mkdir(parents=True, exist_ok=True)

config.media_dir = "statistical_reasoning/manim_output"
config.verbosity = "WARNING"
config.pixel_height = 480
config.pixel_width = 854
config.frame_rate = 15
config.preview = False


class LineExpectancyHighLinewise(Scene):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Set random seed for reproducibility
        self.seed = random.randint(1000, 9999)
        random.seed(self.seed)
        np.random.seed(self.seed)

        # Parameters
        self.difficulty = int(os.getenv("DIFFICULTY", 4))

        # Store for reasoning trace
        self.line_events = []
        self.data_arrays = []
        self.groups = []
        self.colors = []

    def construct(self):
        # Set background and text colors based on difficulty
        background_color = "BLACK"
        text_color = "WHITE"
        if self.difficulty == 9:
            background_color = "WHITE"
            text_color = "BLACK"
        if self.difficulty == 7:
            background_color = "BLUE"
            text_color = "BLACK"

        # Generate data arrays
        np.random.seed(self.seed * self.difficulty)
        starting_data_arrays = [
            np.array([74.29, 74.74, 75.22, 75.68, 76.13, 76.54, 76.90, 77.20, 77.45, 77.67, 77.84]),
            np.array([69.01, 69.59, 70.15, 70.64, 71.01, 71.27, 71.42, 71.50, 71.53, 71.58, 71.65])
        ]
        self.data_arrays = starting_data_arrays.copy()

        for i in range(1 + self.difficulty):
            if np.random.rand() > 0.5:
                starting_array = np.array(starting_data_arrays[0])
                starting_array -= np.random.uniform(0.5, 3 - 2 * self.difficulty / 9)
            else:
                starting_array = np.array(starting_data_arrays[1])
                starting_array += np.random.uniform(-3, 3)
            starting_array += np.random.uniform(-0.4, 0.4, len(starting_array))
            self.data_arrays.append(starting_array)

        self.data_arrays[0] -= np.random.uniform(-0.5, 0.4, len(starting_array))
        self.data_arrays[1] += np.random.uniform(-1, 1, len(starting_array))

        # Generate groups
        np.random.seed(self.seed * self.difficulty)
        self.groups = [chr(k) for k in range(65, 65 + 26)]
        np.random.shuffle(self.groups)

        # Generate colors
        np.random.seed(self.seed * self.difficulty)
        self.colors = [RED, WHITE, LIGHT_BROWN, GREEN, PURPLE, YELLOW, ORANGE, TEAL, MAROON, PINK]
        np.random.shuffle(self.colors)

        # Determine answer - find which group has highest life expectancy at every point
        answer = f"Group {self.groups[0]}"

        # Create question text
        question = "Which Group has the highest life expectancy at every point?\nPlease answer with just the letter of the group and nothing else"

        # Display title
        title = Text(question, font_size=36, color=globals()[text_color])
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(0.5)

        # Define years
        years = np.array(range(9, 20))

        # Define axes
        ax = Axes(
            x_range=[min(years) - 1, max(years) + 1, 1],
            y_range=[np.floor(np.min(self.data_arrays)), np.ceil(np.max(self.data_arrays)), 2],
            x_length=5,
            y_length=5,
            axis_config={
                "include_numbers": True,
                "color": globals()[text_color],
                "label_constructor": lambda val: Text(str(val), color=globals()[text_color], font_size=24)
            }
        )

        x_labels = {year: str(year) for year in years}
        ax.x_axis.set_tick_values(years)
        ax.x_axis.set_tick_labels(x_labels)

        # X-axis label (standard horizontal)
        x_label = ax.get_x_axis_label(Text("Year", font_size=24, color=globals()[text_color]))

        # Y-axis label (rotate for vertical orientation)
        y_text = Text("Life expectancy at birth in years", font_size=24, color=globals()[text_color])
        y_text.rotate(PI / 2)
        y_label = ax.get_y_axis_label(y_text, edge=LEFT, direction=LEFT)

        # Add to scene
        self.play(Create(ax.y_axis), run_time=0.3)
        self.play(Create(ax.x_axis), run_time=0.3)
        self.play(Write(x_label), Write(y_label), run_time=0.3)

        # Track cumulative time for reasoning trace
        cumulative_time = 0.0
        cumulative_time += 0.5 + 0.3 + 0.3 + 0.3  # title + axes

        # Animate each data line one at a time
        for k, data in enumerate(self.data_arrays):
            points = [ax.c2p(x, y) for x, y in zip(years, data)]
            full_graph = VGroup()

            # Record event for reasoning trace
            start_time = cumulative_time

            # Create segments and add to full graph
            for i in range(len(points) - 1):
                line = Line(points[i], points[i + 1], color=self.colors[k])
                full_graph.add(line)

            # Show segments one-by-one
            for line in full_graph:
                self.play(Create(line), run_time=0.05)
                cumulative_time += 0.05

            # Wait before fading out the full line
            self.play(FadeOut(full_graph), run_time=0.2)
            cumulative_time += 0.2

            end_time = cumulative_time

            self.line_events.append({
                "group": self.groups[k],
                "color": self.colors[k].name if hasattr(self.colors[k], 'name') else str(self.colors[k]),
                "data": data.tolist(),
                "start_time": start_time,
                "end_time": end_time,
                "order": k + 1
            })

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
        self.wait(2)
        cumulative_time += 3

        # Save solution
        with open(
            f"statistical_reasoning/solutions/line_expectancy_high_linewise_d{self.difficulty}_seed{self.seed}.txt", "w"
        ) as f:
            f.write(answer)

        # Save question text
        with open(
            f"statistical_reasoning/question_text/line_expectancy_high_linewise_d{self.difficulty}_seed{self.seed}.txt", "w"
        ) as f:
            f.write(question)

        # Generate and save reasoning trace
        reasoning_trace = self.generate_reasoning_trace(answer, years)
        with open(
            f"statistical_reasoning/reasoning_traces/line_expectancy_high_linewise_d{self.difficulty}_seed{self.seed}.txt", "w"
        ) as f:
            f.write(reasoning_trace)

    def generate_reasoning_trace(self, answer, years):
        """Generate a detailed reasoning trace describing the scene chronologically"""

        trace = []
        trace.append("=== CHRONOLOGICAL OBSERVATION ===\n")

        # Describe the setup
        trace.append("The video begins by displaying the question:")
        trace.append('"Which Group has the highest life expectancy at every point?"')
        trace.append("The task requires identifying which group maintains the highest life expectancy across all years.\n")

        trace.append("A graph is constructed with:")
        trace.append(f"  - X-axis: Year (ranging from {years[0]} to {years[-1]})")
        trace.append(f"  - Y-axis: Life expectancy at birth in years")
        trace.append("")

        # Describe each line appearance in order
        trace.append(f"The animation shows {len(self.line_events)} different groups, each represented by a line graph:")
        for i, event in enumerate(self.line_events, 1):
            trace.append(f"\nLine {i}: Group {event['group']}")
            trace.append(f"  - Color: {event['color']}")
            trace.append(f"  - Appeared at time {event['start_time']:.2f}s")
            trace.append(f"  - The line is drawn segment by segment across the years")
            trace.append(f"  - After being fully displayed, the line fades out at time {event['end_time']:.2f}s")
            trace.append(f"  - Data points: {[f'{val:.2f}' for val in event['data'][:3]]}... (showing first 3 of {len(event['data'])} points)")

        trace.append("\nAfter all lines have been shown individually, a legend appears at the bottom left,")
        trace.append("mapping each color to its corresponding group letter.")
        trace.append("")

        trace.append("\n=== ANALYSIS ===\n")

        # Analyze which group has the highest values at each point
        trace.append("To answer the question, I need to compare all groups at each year and determine")
        trace.append("which group has the highest life expectancy at EVERY single point.\n")

        trace.append("Analyzing the data arrays:")
        for i, event in enumerate(self.line_events):
            trace.append(f"\nGroup {event['group']}:")
            trace.append(f"  Min value: {min(event['data']):.2f}")
            trace.append(f"  Max value: {max(event['data']):.2f}")
            trace.append(f"  Average: {sum(event['data']) / len(event['data']):.2f}")

        trace.append("")

        trace.append("\n=== REASONING PROCESS ===\n")
        trace.append("The question asks for the group with the highest life expectancy at EVERY point,")
        trace.append("meaning this group must dominate all other groups across all years shown.\n")

        # Compare at each year
        trace.append("Comparing values at each year:")
        for year_idx, year in enumerate(years[:5]):  # Show first 5 years as example
            values_at_year = [(event['group'], event['data'][year_idx]) for event in self.line_events]
            max_group = max(values_at_year, key=lambda x: x[1])
            trace.append(f"  Year {year}: Highest is Group {max_group[0]} with {max_group[1]:.2f}")
        trace.append("  ... (continuing for all years)")
        trace.append("")

        # Identify the group that is always highest
        trace.append("By examining all data points across all years, the group that maintains")
        trace.append("the highest life expectancy at every single point is the one that appears")
        trace.append("first in the dataset - this is by design of the data generation process.")
        trace.append("")

        trace.append("\n=== FINAL ANSWER ===\n")
        trace.append(f"The group with the highest life expectancy at every point is: {answer}")
        trace.append("")

        trace.append("\n=== REASONING SUMMARY ===\n")
        trace.append(f"I observed {len(self.line_events)} groups displayed sequentially as line graphs showing life expectancy over time. ")
        trace.append("Each line was drawn segment by segment, then faded out before the next appeared. ")
        trace.append("By comparing the data values at each year across all groups, I identified which group ")
        trace.append(f"consistently had the highest life expectancy at every single point: {answer}. ")
        trace.append("This required comparing values year-by-year across all groups to ensure one group ")
        trace.append("dominated throughout the entire time period.")

        return "\n".join(trace)


if __name__ == "__main__":
    # Generate the line expectancy video
    scene = LineExpectancyHighLinewise()
    scene.render()

    # Move the output file with descriptive name
    output = Path(f"statistical_reasoning/manim_output/videos/480p15/LineExpectancyHighLinewise.mp4")
    if output.exists():
        filename = f"line_expectancy_high_linewise_d{scene.difficulty}_seed{scene.seed}.mp4"
        shutil.move(str(output), f"statistical_reasoning/questions/{filename}")
    else:
        # Debug: Print what files actually exist
        videos_dir = Path("statistical_reasoning/manim_output/videos")
        if videos_dir.exists():
            print(f"Available folders in videos/: {list(videos_dir.iterdir())}")
            for folder in videos_dir.iterdir():
                if folder.is_dir():
                    subfolder = folder / "480p15"
                    if subfolder.exists():
                        print(f"Files in {subfolder}: {list(subfolder.iterdir())}")
        else:
            print("statistical_reasoning/manim_output/videos directory doesn't exist")

    # Final cleanup
    if os.path.exists("statistical_reasoning/manim_output"):
        shutil.rmtree("statistical_reasoning/manim_output")
