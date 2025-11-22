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


class TableMethodsDisappearingColsRowQ(Scene):
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
            self.text_color = BLUE
        else:
            self.background_color = BLACK
            self.text_color = WHITE

        # Data - datasets and methods from the notebook
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
        self.selected_method = None
        self.answer = None
        self.question = None

    def construct(self):
        # Set background color
        config.background_color = self.background_color

        # Generate table data
        # Choose methods based on difficulty
        num_methods = min(6, 2 + self.difficulty // 2)
        self.chosen_methods = list(np.random.choice(self.row_methods, num_methods, replace=False))

        # Choose dataset columns based on difficulty
        num_datasets = min(7, 2 + (self.difficulty + 1) // 2)
        self.headers = list(np.random.choice(self.all_headers, num_datasets, replace=False))

        # Generate performance scores
        self.rows = []
        for method in self.chosen_methods:
            row = [method]
            # Generate random scores between 0 and 100
            scores = np.round(
                np.clip(
                    np.random.normal(
                        np.random.uniform(20, 80),
                        25 / (1 + self.difficulty),
                        len(self.headers)
                    ),
                    0,
                    100
                ),
                2
            )
            row.extend(scores)
            self.rows.append(row)

        # Select a random method to ask about
        self.selected_method = np.random.choice(self.chosen_methods)

        # Find which column has the highest score for this method
        method_idx = self.chosen_methods.index(self.selected_method)
        method_scores = np.array(self.rows[method_idx][1:], dtype=float)
        max_score_idx = np.argmax(method_scores)
        self.answer = self.headers[max_score_idx]
        max_score = method_scores[max_score_idx]

        # Generate question
        self.question = f"Question: On which column does the {self.selected_method} method achieve the highest score?\nPlease answer with just a column header and nothing else"

        # Display question
        title = Text(self.question, font_size=20, color=self.text_color)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(0.5)
        self.play(FadeOut(title))

        # Create table
        table_headers = ["\\text{method}"] + ["\\text{" + h + "}" for h in self.headers]
        table_rows = []
        for row in self.rows:
            # Format method name and scores
            formatted_row = ["\\text{" + str(row[0]) + "}"] + [str(score) for score in row[1:]]
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
        with open(f"solutions/table_methods_dissapearing_cols_rowq_d{self.difficulty}_seed{self.seed}.txt", "w") as f:
            f.write(str(self.answer))

        # Save question text
        question_text_content = self.question
        with open(f"question_text/table_methods_dissapearing_cols_rowq_d{self.difficulty}_seed{self.seed}.txt", "w") as f:
            f.write(question_text_content)

        # Generate and save reasoning trace
        reasoning_trace = self.generate_reasoning_trace(max_score, max_score_idx)
        with open(f"reasoning_traces/table_methods_dissapearing_cols_rowq_d{self.difficulty}_seed{self.seed}.txt", "w") as f:
            f.write(reasoning_trace)

    def generate_reasoning_trace(self, max_score, max_score_idx):
        """Generate a detailed reasoning trace describing the scene chronologically"""

        trace = []
        trace.append("=== TASK OVERVIEW ===\n")
        trace.append(f"Question: {self.question}")
        trace.append(f"Difficulty level: {self.difficulty}")
        trace.append(f"Number of methods in table: {len(self.chosen_methods)}")
        trace.append(f"Number of datasets (columns): {len(self.headers)}")
        trace.append("")

        trace.append("\n=== CHRONOLOGICAL OBSERVATION ===\n")
        trace.append("The video shows a table of method performance scores across different datasets.")
        trace.append("Each column (representing a dataset) appears and disappears one at a time.")
        trace.append("Each column is displayed briefly (fades in, then fades out) before the next column appears.")
        trace.append("This requires remembering the information from all columns to determine which dataset")
        trace.append("has the highest score for the specified method.\n")

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
        trace.append("From the table, I extract the complete performance scores for all methods:\n")

        # Show full table data
        trace.append("Complete Table:")
        trace.append(f"  Methods: {', '.join(self.chosen_methods)}")
        trace.append(f"  Datasets: {', '.join(self.headers)}")
        trace.append("")

        for row in self.rows:
            method_name = row[0]
            scores = row[1:]
            score_str = ", ".join([f"{self.headers[i]}: {scores[i]}" for i in range(len(self.headers))])
            trace.append(f"  {method_name}: {score_str}")
        trace.append("")

        trace.append("\n=== ANALYSIS ===\n")
        trace.append(f"The question asks about the '{self.selected_method}' method.")
        trace.append(f"I need to find which dataset (column) this method achieved the highest score on.\n")

        # Show the specific method's scores
        method_idx = self.chosen_methods.index(self.selected_method)
        method_scores = self.rows[method_idx][1:]

        trace.append(f"Scores for '{self.selected_method}':")
        for i, score in enumerate(method_scores):
            trace.append(f"  {self.headers[i]}: {score}")
        trace.append("")

        trace.append("\n=== REASONING PROCESS ===\n")
        trace.append("To find the highest score, I compare all values:\n")

        # Show comparison
        sorted_scores = sorted([(self.headers[i], method_scores[i]) for i in range(len(self.headers))],
                               key=lambda x: x[1], reverse=True)

        trace.append("Sorted scores (highest to lowest):")
        for i, (dataset, score) in enumerate(sorted_scores, 1):
            marker = " <- HIGHEST" if i == 1 else ""
            trace.append(f"  {i}. {dataset}: {score}{marker}")
        trace.append("")

        trace.append(f"The maximum score is {max_score} on the '{self.answer}' dataset.")
        trace.append("")

        trace.append("\n=== FINAL ANSWER ===\n")
        trace.append(f"The '{self.selected_method}' method achieves its highest score on: {self.answer}")
        trace.append("")

        trace.append("\n=== REASONING SUMMARY ===\n")
        trace.append(
            f"I observed a table with {len(self.chosen_methods)} methods and their performance scores "
            f"across {len(self.headers)} different datasets. "
        )
        trace.append(
            f"The table was displayed in a format where columns appeared and disappeared sequentially. "
        )
        trace.append(
            f"To answer the question about which dataset the '{self.selected_method}' method performed best on, "
        )
        trace.append(
            f"I needed to remember the scores from when each column was briefly visible. "
        )
        trace.append(
            f"By comparing all scores for the '{self.selected_method}' method across all datasets, "
        )
        trace.append(
            f"I determined that the highest score of {max_score} was achieved on the '{self.answer}' dataset."
        )

        return "\n".join(trace)


if __name__ == "__main__":
    # Generate the table video
    scene = TableMethodsDisappearingColsRowQ()
    scene.render()

    # Move the output file with descriptive name
    output = Path("manim_output/videos/1080p30/TableMethodsDisappearingColsRowQ.mp4")
    if output.exists():
        filename = f"table_methods_dissapearing_cols_rowq_d{scene.difficulty}_seed{scene.seed}.mp4"
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
