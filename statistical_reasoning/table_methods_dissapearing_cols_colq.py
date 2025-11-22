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


class TableMethodsDisappearingColsColq(Scene):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Set random seed for reproducibility
        self.seed = random.randint(1000, 9999)
        random.seed(self.seed)
        np.random.seed(self.seed)

        # Parameters - can be set via environment variables
        self.difficulty = int(os.getenv("DIFFICULTY", 0))

        # Set colors based on difficulty
        if self.difficulty == 9:
            self.background_color = WHITE
            self.text_color = BLACK
        elif self.difficulty == 7:
            self.background_color = YELLOW
            self.text_color = BLACK
        else:
            self.background_color = BLACK
            self.text_color = WHITE

        # Data
        self.all_headers = [
            "ADE20K", "Cityscapes", "Mapillary Vistas", "A-847",
            "PC-459", "A-150", "PC-59"
        ]
        self.row_methods = [
            "ALIGN [38, 28]",
            "ALIGN w/ proposal [38, 28]",
            "LSeg+ [46, 28]",
            "OpenSeg [28]",
            "OpenSeg [28] w/ L. Narr",
            "FC-CLIP (ours)"
        ]

        # Store for reasoning trace
        self.column_events = []
        self.table_data = None
        self.headers = None
        self.rows = None
        self.chosen_methods = None
        self.selected_header = None
        self.threshold = None
        self.answer = None
        self.question = None

    def construct(self):
        # Set background color
        config.background_color = self.background_color

        # Generate table data
        self.chosen_methods = list(
            np.random.choice(
                self.row_methods,
                min(6, 2 + self.difficulty//2),
                replace=False
            )
        )
        self.headers = list(
            np.random.choice(
                self.all_headers,
                min(7, 2 + (self.difficulty+1)//2),
                replace=False
            )
        )

        # Generate score data
        self.rows = []
        for k in range(len(self.chosen_methods)):
            method_name = self.chosen_methods[k]
            scores = list(
                np.round(
                    np.clip(
                        np.random.normal(
                            np.random.uniform(20, 80),
                            25/(1+self.difficulty),
                            len(self.headers)
                        ),
                        0,
                        100
                    ),
                    2
                )
            )
            self.rows.append([method_name] + scores)

        # Convert to numpy array for easier column extraction
        rows_array = np.array(self.rows, dtype=object)

        # Select a random header (column) for the question
        self.selected_header = np.random.choice(self.headers)
        col_idx = self.headers.index(self.selected_header)

        # Get column values (skip the method name, so add 1 to index)
        col = rows_array[:, 1 + col_idx]
        col = np.float64(col)

        # Set threshold slightly below a random value from the column
        self.threshold = np.round(np.random.choice(col) - 0.1, 2)

        # Calculate answer: count how many methods have score > threshold
        self.answer = int(np.sum(col > self.threshold))

        # Generate question
        self.question = (
            f"Question: How many methods in the table achieve an {self.selected_header} "
            f"score higher than {self.threshold}?\nPlease answer with just a number and nothing else"
        )

        # Display question
        title = Text(self.question, font_size=20, color=self.text_color)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(0.5)

        # Create table
        table_headers = ["\\text{method}"] + ["\\text{" + h + "}" for h in self.headers]
        table_rows = []
        for row in self.rows:
            formatted_row = ["\\text{" + str(row[0]) + "}"] + [str(row[i]) for i in range(1, len(row))]
            table_rows.append(formatted_row)

        self.table_data = [table_headers] + table_rows

        table = MathTable(
            self.table_data,
            include_outer_lines=True,
            h_buff=0.5,
            v_buff=0.3,
        ).scale(0.55)

        for entry in table.get_entries():
            entry.set_color(self.text_color)
        for line in table.get_horizontal_lines() + table.get_vertical_lines():
            line.set_color(self.text_color)

        # Track cumulative time for reasoning trace
        cumulative_time = 0.0

        # Create table structure
        self.play(Create(table.get_horizontal_lines()[0]), run_time=0.5)
        cumulative_time += 0.5
        self.play(Create(table.get_vertical_lines()), run_time=0.5)
        cumulative_time += 0.5
        self.play(Create(table.get_horizontal_lines()[1:]), run_time=0.5)
        cumulative_time += 0.5

        # Display columns one by one (fade in and fade out)
        columns = table.get_columns()
        for col_idx, column in enumerate(columns):
            col_start_time = cumulative_time

            # Record event for reasoning trace
            column_data = []
            for row_idx in range(len(self.table_data)):
                if col_idx < len(self.table_data[row_idx]):
                    column_data.append(self.table_data[row_idx][col_idx])

            self.column_events.append({
                "column_number": col_idx + 1,
                "column_name": table_headers[col_idx] if col_idx < len(table_headers) else f"Column {col_idx + 1}",
                "start_time": col_start_time,
                "fade_in_duration": 0.5,
                "fade_out_duration": 0.5,
                "data": column_data
            })

            self.play(FadeIn(column), run_time=0.5)
            cumulative_time += 0.5
            self.play(FadeOut(column), run_time=0.5)
            cumulative_time += 0.5

        self.wait(1)

        # Save solution
        with open(f"solutions/table_methods_dissapearing_cols_colq_d{self.difficulty}_seed{self.seed}.txt", "w") as f:
            f.write(str(self.answer))

        # Save question text
        question_text_content = self.question
        with open(f"question_text/table_methods_dissapearing_cols_colq_d{self.difficulty}_seed{self.seed}.txt", "w") as f:
            f.write(question_text_content)

        # Generate and save reasoning trace
        reasoning_trace = self.generate_reasoning_trace()
        with open(f"reasoning_traces/table_methods_dissapearing_cols_colq_d{self.difficulty}_seed{self.seed}.txt", "w") as f:
            f.write(reasoning_trace)

    def generate_reasoning_trace(self):
        """Generate a detailed reasoning trace describing the scene chronologically"""

        trace = []
        trace.append("=== TASK OVERVIEW ===\n")
        trace.append(f"Question: {self.question}")
        trace.append(f"Difficulty level: {self.difficulty}")
        trace.append(f"Number of methods in table: {len(self.chosen_methods)}")
        trace.append(f"Number of dataset columns: {len(self.headers)}")
        trace.append(f"Target column: {self.selected_header}")
        trace.append(f"Threshold value: {self.threshold}")
        trace.append("")

        trace.append("\n=== CHRONOLOGICAL OBSERVATION ===\n")
        trace.append("The video shows a table comparing different segmentation methods across various datasets.")
        trace.append("Columns appear and disappear one at a time, each displaying briefly (fades in, then fades out).")
        trace.append("The first column shows method names, and subsequent columns show performance scores on different datasets.")
        trace.append("To answer the question, I need to remember the scores from the specific target column when it appears.\n")

        # Describe each column appearance
        for event in self.column_events:
            trace.append(f"Column {event['column_number']} - {event['column_name']}:")
            trace.append(f"  Time: {event['start_time']:.2f}s - {event['start_time'] + event['fade_in_duration'] + event['fade_out_duration']:.2f}s")
            trace.append(f"  Fade in: {event['fade_in_duration']:.2f}s, Fade out: {event['fade_out_duration']:.2f}s")
            trace.append(f"  Data displayed:")
            for i, data_point in enumerate(event['data']):
                trace.append(f"    Row {i}: {data_point}")
            trace.append("")

        trace.append("\n=== DATA EXTRACTION ===\n")
        trace.append(f"From the table, I need to extract the {self.selected_header} scores for all methods:\n")

        trace.append(f"Method - {self.selected_header} Score Mapping:")
        col_idx = self.headers.index(self.selected_header)
        for row in self.rows:
            method_name = row[0]
            score = row[1 + col_idx]
            trace.append(f"  {method_name}: {score}")
        trace.append("")

        trace.append("\n=== ANALYSIS ===\n")
        trace.append(f"The question asks: How many methods achieve an {self.selected_header} score higher than {self.threshold}?\n")

        trace.append("Comparing each method's score to the threshold:")
        count = 0
        for row in self.rows:
            method_name = row[0]
            score = float(row[1 + col_idx])
            is_above = score > self.threshold
            count += int(is_above)

            comparison = ">" if is_above else "<="
            trace.append(f"  {method_name}: {score} {comparison} {self.threshold} → {'ABOVE' if is_above else 'NOT above'}")
        trace.append("")

        trace.append("\n=== REASONING PROCESS ===\n")
        trace.append(f"Step 1: Identify the target column - {self.selected_header}")
        trace.append(f"Step 2: Extract all scores from this column")
        trace.append(f"Step 3: Compare each score to the threshold value {self.threshold}")
        trace.append(f"Step 4: Count how many scores are strictly greater than the threshold")
        trace.append("")

        trace.append("\n=== FINAL ANSWER ===\n")
        trace.append(f"Number of methods with {self.selected_header} score > {self.threshold}: {self.answer}")
        trace.append("")

        trace.append("\n=== REASONING SUMMARY ===\n")
        trace.append(
            f"I observed a table with {len(self.chosen_methods)} different segmentation methods "
            f"and their performance scores across {len(self.headers)} different datasets. "
        )
        trace.append(
            f"The table was displayed with columns appearing and disappearing sequentially, "
            f"requiring me to remember the information when each column was briefly visible. "
        )
        trace.append(
            f"The question focused on the {self.selected_header} column, asking how many methods "
            f"achieved a score higher than {self.threshold}. "
        )
        trace.append(
            f"By examining each method's {self.selected_header} score and comparing it to the threshold, "
            f"I counted {self.answer} method(s) that exceeded the threshold value."
        )

        return "\n".join(trace)


if __name__ == "__main__":
    # Generate the table video
    scene = TableMethodsDisappearingColsColq()
    scene.render()

    # Move the output file with descriptive name
    output = Path("manim_output/videos/1080p30/TableMethodsDisappearingColsColq.mp4")
    if output.exists():
        filename = f"table_methods_dissapearing_cols_colq_d{scene.difficulty}_seed{scene.seed}.mp4"
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
