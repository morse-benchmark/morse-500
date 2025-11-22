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


class GroundingSegmentationTable(Scene):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Set random seed for reproducibility
        self.seed = random.randint(1000, 9999)
        random.seed(self.seed)
        np.random.seed(self.seed)

        # Parameters
        self.difficulty = int(os.getenv("DIFFICULTY", 5))

        # Store for reasoning trace
        self.table_events = []
        self.headers = []
        self.rows = []
        self.selected_header = ""
        self.threshold = 0.0
        self.answer = 0
        self.background_color = BLACK
        self.text_color = WHITE

    def construct(self):
        # Set colors based on difficulty
        if self.difficulty == 9:
            self.background_color = WHITE
            self.text_color = BLACK
        elif self.difficulty == 7:
            self.background_color = RED
            self.text_color = BLACK
        else:
            self.background_color = BLACK
            self.text_color = WHITE

        # Update scene background
        self.camera.background_color = self.background_color

        # Generate table data
        all_headers = [
            "ADE20K",
            "Cityscapes",
            "Mapillary Vistas",
            "A-847",
            "PC-459",
            "A-150",
            "PC-59",
        ]
        row_methods = [
            "ALIGN [38, 28]",
            "ALIGN w/ proposal [38, 28]",
            "LSeg+ [46, 28]",
            "OpenSeg [28]",
            "OpenSeg [28] w/ L. Narr",
            "FC-CLIP (ours)",
        ]

        # Choose methods and headers based on difficulty
        num_methods = min(6, 2 + self.difficulty // 2)
        num_headers = min(7, 2 + (self.difficulty + 1) // 2)

        chosen_methods = np.random.choice(row_methods, num_methods, replace=False)
        self.headers = list(np.random.choice(all_headers, num_headers, replace=False))

        # Generate table rows with scores
        self.rows = []
        for method in chosen_methods:
            row = [method] + list(
                np.round(
                    np.clip(
                        np.random.normal(
                            np.random.uniform(20, 80), 25 / (1 + self.difficulty), num_headers
                        ),
                        0,
                        100,
                    ),
                    2,
                )
            )
            self.rows.append(row)

        # Select a header and determine threshold
        self.selected_header = np.random.choice(self.headers)
        col_index = self.headers.index(self.selected_header)

        # Extract column values (skip method name)
        rows_array = np.array(self.rows, dtype=object)
        col_values = rows_array[:, col_index + 1].astype(float)

        # Set threshold
        self.threshold = np.round(np.random.choice(col_values) - 0.1, 2)
        self.answer = int(np.sum(col_values > self.threshold))

        # Create question
        question = f"How many methods in the table achieve an {self.selected_header} score higher than {self.threshold}?"

        # Display title
        title = Text(
            question + "\nPlease answer with just a number and nothing else",
            font_size=24,
            color=self.text_color,
        )
        title.to_edge(UP)
        self.play(Write(title), run_time=1.0)
        self.wait(0.5)

        # Record event
        self.table_events.append(
            {
                "event": "question_displayed",
                "time": 0.0,
                "question": question,
            }
        )

        # Prepare table data
        table_headers = ["Method"] + self.headers
        table_data = [table_headers] + self.rows

        # Create table
        table = Table(
            table_data,
            include_outer_lines=True,
            h_buff=0.5,
            v_buff=0.3,
        ).scale(0.55)

        # Set colors
        for entry in table.get_entries():
            entry.set_color(self.text_color)
        for line in table.get_horizontal_lines() + table.get_vertical_lines():
            line.set_color(self.text_color)

        # Animate table creation
        cumulative_time = 1.5
        self.play(Create(table.get_horizontal_lines()[0]), run_time=0.5)
        cumulative_time += 0.5
        self.table_events.append(
            {
                "event": "header_line_created",
                "time": cumulative_time,
            }
        )

        self.play(Create(table.get_vertical_lines()), run_time=0.5)
        cumulative_time += 0.5
        self.table_events.append(
            {
                "event": "vertical_lines_created",
                "time": cumulative_time,
            }
        )

        self.play(Create(table.get_horizontal_lines()[1:]), run_time=0.5)
        cumulative_time += 0.5
        self.table_events.append(
            {
                "event": "horizontal_lines_created",
                "time": cumulative_time,
            }
        )

        # Animate rows appearing and disappearing
        for i in range(len(table_data)):
            row_label = table_data[i][0] if i > 0 else "Headers"
            self.play(FadeIn(table.get_rows()[i]), run_time=0.5)
            cumulative_time += 0.5
            self.table_events.append(
                {
                    "event": "row_appeared",
                    "time": cumulative_time,
                    "row_index": i,
                    "row_label": row_label,
                    "row_data": table_data[i],
                }
            )

            self.play(FadeOut(table.get_rows()[i]), run_time=0.5)
            cumulative_time += 0.5
            self.table_events.append(
                {
                    "event": "row_disappeared",
                    "time": cumulative_time,
                    "row_index": i,
                    "row_label": row_label,
                }
            )

        self.wait(1.0)

        # Save outputs
        self.save_outputs()

    def save_outputs(self):
        """Save all outputs with seed-based filenames"""
        base_filename = f"table_methods_dissapearing_rows_colq_d{self.difficulty}_seed{self.seed}"

        # Save solution
        with open(f"solutions/{base_filename}.txt", "w") as f:
            f.write(str(self.answer))

        # Save question text
        question_text = (
            f"How many methods in the table achieve an {self.selected_header} score higher than {self.threshold}?\n"
            "Please answer with just a number and nothing else."
        )
        with open(f"question_text/{base_filename}.txt", "w") as f:
            f.write(question_text)

        # Generate and save reasoning trace
        reasoning_trace = self.generate_reasoning_trace()
        with open(f"reasoning_traces/{base_filename}.txt", "w") as f:
            f.write(reasoning_trace)

    def generate_reasoning_trace(self):
        """Generate a detailed reasoning trace describing the scene chronologically"""

        trace = []
        trace.append("=== CHRONOLOGICAL OBSERVATION ===\n")

        trace.append(
            f"At time {self.table_events[0]['time']:.2f}s: The question is displayed:"
        )
        trace.append(
            f'  "{self.table_events[0]["question"]} Please answer with just a number and nothing else."\n'
        )

        trace.append(
            f"At time {self.table_events[1]['time']:.2f}s: The header horizontal line is created."
        )
        trace.append(
            f"At time {self.table_events[2]['time']:.2f}s: The vertical lines are created."
        )
        trace.append(
            f"At time {self.table_events[3]['time']:.2f}s: The remaining horizontal lines are created.\n"
        )

        trace.append("The table rows then appear and disappear sequentially:")

        # Track rows and their data
        row_data_list = []
        for event in self.table_events[4:]:
            if event["event"] == "row_appeared":
                trace.append(
                    f"  At time {event['time']:.2f}s: Row {event['row_index']} ({event['row_label']}) appears."
                )
                row_data_list.append(
                    {
                        "index": event["row_index"],
                        "label": event["row_label"],
                        "data": event["row_data"],
                    }
                )
            elif event["event"] == "row_disappeared":
                trace.append(
                    f"  At time {event['time']:.2f}s: Row {event['row_index']} ({event['row_label']}) disappears."
                )

        trace.append("\n=== ANALYSIS ===\n")

        trace.append("From the briefly displayed rows, I need to:")
        trace.append(
            f"1. Identify the column for '{self.selected_header}' scores"
        )
        trace.append(
            f"2. Count how many methods have a score higher than {self.threshold}\n"
        )

        trace.append("Reconstructed table data from memory:")
        trace.append("Headers: " + str(["Method"] + self.headers))
        trace.append("")
        for row_info in row_data_list[1:]:  # Skip header row
            trace.append(f"  {row_info['label']}: {row_info['data']}")
        trace.append("")

        trace.append(f"The column for '{self.selected_header}' is at position {self.headers.index(self.selected_header) + 1}.\n")

        trace.append("\n=== REASONING PROCESS ===\n")

        trace.append(
            f"Examining the '{self.selected_header}' column, I need to count values > {self.threshold}:\n"
        )

        col_index = self.headers.index(self.selected_header)
        count = 0
        for row_info in row_data_list[1:]:  # Skip header row
            method = row_info["data"][0]
            score = row_info["data"][col_index + 1]
            is_higher = score > self.threshold
            if is_higher:
                count += 1
            trace.append(
                f"  {method}: {score} {'>' if is_higher else '<='} {self.threshold} -> {'COUNT' if is_higher else 'skip'}"
            )

        trace.append(f"\nTotal methods with {self.selected_header} score > {self.threshold}: {count}")

        trace.append("\n=== FINAL ANSWER ===\n")
        trace.append(f"{self.answer}")

        trace.append("\n=== REASONING SUMMARY ===\n")
        trace.append(
            f"I observed a table with {len(self.rows)} methods and {len(self.headers)} score columns. "
        )
        trace.append(
            f"Each row appeared briefly and then disappeared. By tracking the '{self.selected_header}' column "
        )
        trace.append(
            f"values and counting those exceeding the threshold of {self.threshold}, I determined that "
        )
        trace.append(
            f"{self.answer} method(s) meet the criteria."
        )

        return "\n".join(trace)


if __name__ == "__main__":
    # Generate the table video
    scene = GroundingSegmentationTable()
    scene.render()

    # Move the output file with descriptive name
    output = Path("manim_output/videos/1080p30/GroundingSegmentationTable.mp4")
    if output.exists():
        filename = f"table_methods_dissapearing_rows_colq_d{scene.difficulty}_seed{scene.seed}.mp4"
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
