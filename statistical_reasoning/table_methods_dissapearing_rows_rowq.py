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


class TableMethodsDisappearingRowsRowQ(Scene):
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
            self.background_color = BLUE
            self.text_color = YELLOW
        else:
            self.background_color = BLACK
            self.text_color = WHITE

        # Data lists for methods and dataset headers
        self.all_headers = [
            "ADE20K", "Cityscapes", "Mapillary Vistas",
            "A-847", "PC-459", "A-150", "PC-59"
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
        self.table_events = []
        self.headers = []
        self.chosen_methods = []
        self.rows = []
        self.selected_method = None
        self.selected_method_row = None
        self.question_text = ""
        self.answer = None

    def construct(self):
        # Set background color
        config.background_color = self.background_color

        # Generate table data based on difficulty
        num_methods = min(6, max(2, 2 + self.difficulty // 2))
        num_headers = min(7, max(2, 2 + (self.difficulty + 1) // 2))

        # Choose methods and headers
        self.chosen_methods = list(np.random.choice(
            self.row_methods,
            num_methods,
            replace=False
        ))
        self.headers = list(np.random.choice(
            self.all_headers,
            num_headers,
            replace=False
        ))

        # Generate row data with scores
        self.rows = []
        for method in self.chosen_methods:
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
            ).tolist()
            self.rows.append([method, scores])

        # Select a random method to ask about
        selected_idx = np.random.choice(range(len(self.chosen_methods)))
        self.selected_method = self.chosen_methods[selected_idx]
        self.selected_method_row = self.rows[selected_idx][1]

        # Find the column with the highest score for the selected method
        max_val = np.max(self.selected_method_row)
        max_idx = list(self.selected_method_row).index(max_val)
        self.answer = self.headers[max_idx]

        # Generate question text
        self.question_text = (
            f"Question: On which column does the {self.selected_method} method achieve the highest score?\n"
            f"Please answer with just a column header and nothing else"
        )

        # Display question
        title = Text(self.question_text, font_size=24, color=self.text_color)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(0.5)

        # Record event
        self.table_events.append({
            "event": "question_displayed",
            "time": 0.0,
            "text": self.question_text
        })

        cumulative_time = 0.5

        # Prepare table data for MathTable
        table_headers = ["\\text{method}"] + [f"\\text{{{h}}}" for h in self.headers]
        table_rows = []

        for method, scores in self.rows:
            formatted_row = [f"\\text{{{method}}}"] + [str(score) for score in scores]
            table_rows.append(formatted_row)

        table_data = [table_headers] + table_rows

        # Create table
        table = MathTable(
            table_data,
            include_outer_lines=True,
            h_buff=0.5,
            v_buff=0.3,
        ).scale(0.55)

        # Set colors for table
        for entry in table.get_entries():
            entry.set_color(self.text_color)
        for line in table.get_horizontal_lines() + table.get_vertical_lines():
            line.set_color(self.text_color)

        # Animate table creation
        self.play(Create(table.get_horizontal_lines()[0]), run_time=0.5)
        cumulative_time += 0.5
        self.table_events.append({
            "event": "table_top_line_created",
            "time": cumulative_time
        })

        self.play(Create(table.get_vertical_lines()), run_time=0.5)
        cumulative_time += 0.5
        self.table_events.append({
            "event": "table_vertical_lines_created",
            "time": cumulative_time
        })

        self.play(Create(table.get_horizontal_lines()[1:]), run_time=0.5)
        cumulative_time += 0.5
        self.table_events.append({
            "event": "table_horizontal_lines_created",
            "time": cumulative_time
        })

        # Animate rows appearing and disappearing
        for i in range(len(table_data)):
            row_start_time = cumulative_time
            self.play(FadeIn(table.get_rows()[i]), run_time=0.5)
            cumulative_time += 0.5

            row_end_time = cumulative_time
            self.play(FadeOut(table.get_rows()[i]), run_time=0.5)
            cumulative_time += 0.5

            # Record event
            if i == 0:
                row_content = table_headers
                row_type = "header"
            else:
                row_content = table_rows[i - 1]
                row_type = "data"

            self.table_events.append({
                "event": "row_displayed",
                "row_index": i,
                "row_type": row_type,
                "content": row_content,
                "start_time": row_start_time,
                "end_time": row_end_time,
                "duration": 0.5
            })

        self.wait(1)

        # Save solution
        with open(
            f"solutions/table_methods_disappearing_rows_rowq_d{self.difficulty}_seed{self.seed}.txt", "w"
        ) as f:
            f.write(str(self.answer))

        # Save question text
        with open(
            f"question_text/table_methods_disappearing_rows_rowq_d{self.difficulty}_seed{self.seed}.txt", "w"
        ) as f:
            f.write(self.question_text)

        # Generate and save reasoning trace
        reasoning_trace = self.generate_reasoning_trace()
        with open(
            f"reasoning_traces/table_methods_disappearing_rows_rowq_d{self.difficulty}_seed{self.seed}.txt", "w"
        ) as f:
            f.write(reasoning_trace)

    def generate_reasoning_trace(self):
        """Generate a detailed reasoning trace describing the scene chronologically"""

        trace = []
        trace.append("=== CHRONOLOGICAL OBSERVATION ===\n")

        trace.append(f"Initial Question: {self.question_text}\n")

        trace.append("The video displays a table that is constructed step by step:")
        trace.append("1. First, the top horizontal line appears")
        trace.append("2. Then all vertical lines are drawn")
        trace.append("3. Then the remaining horizontal lines appear")
        trace.append("4. Finally, each row appears briefly and then disappears\n")

        trace.append("Table Structure:")
        trace.append(f"  Headers: method, {', '.join(self.headers)}\n")

        trace.append("Rows that appeared (in order):")
        for event in self.table_events:
            if event["event"] == "row_displayed":
                row_idx = event["row_index"]
                # Clean up LaTeX formatting from content
                cleaned_content = [str(c).replace('\\text{', '').replace('}', '') for c in event['content']]
                if event["row_type"] == "header":
                    trace.append(f"  Row {row_idx + 1} (Header): {', '.join(cleaned_content)}")
                else:
                    trace.append(f"  Row {row_idx + 1}: {', '.join(cleaned_content)}")
                trace.append(f"    - Appeared at {event['start_time']:.2f}s, disappeared at {event['end_time']:.2f}s")
        trace.append("")

        trace.append("\n=== ANALYSIS ===\n")

        trace.append("From the table, I need to extract the scores for each method across all dataset columns:")
        trace.append("\nMethod scores across datasets:")
        for method, scores in self.rows:
            trace.append(f"\n  {method}:")
            for i, header in enumerate(self.headers):
                trace.append(f"    - {header}: {scores[i]}")

        trace.append(f"\n\nThe question specifically asks about: {self.selected_method}")
        trace.append(f"Scores for {self.selected_method}:")
        for i, header in enumerate(self.headers):
            trace.append(f"  - {header}: {self.selected_method_row[i]}")
        trace.append("")

        trace.append("\n=== REASONING PROCESS ===\n")

        trace.append(f"To find which column the {self.selected_method} method achieves the highest score:")
        trace.append("1. Identify the row corresponding to the selected method")
        trace.append("2. Compare all the scores in that row across different columns")
        trace.append("3. Find the maximum score")
        trace.append("4. Determine which column header corresponds to that maximum score\n")

        trace.append(f"Comparing scores for {self.selected_method}:")
        max_score = max(self.selected_method_row)
        for i, (header, score) in enumerate(zip(self.headers, self.selected_method_row)):
            if score == max_score:
                trace.append(f"  {header}: {score} ← HIGHEST")
            else:
                trace.append(f"  {header}: {score}")

        trace.append(f"\nThe maximum score is {max_score}, which occurs in the column: {self.answer}")
        trace.append("")

        trace.append("\n=== FINAL ANSWER ===\n")
        trace.append(f"The column where {self.selected_method} achieves the highest score is: {self.answer}")
        trace.append("")

        trace.append("\n=== REASONING SUMMARY ===\n")
        trace.append(f"The video presented a table showing various methods and their performance scores across different datasets. ")
        trace.append(f"The question asked which column the {self.selected_method} method achieves its highest score. ")
        trace.append(f"By extracting the scores for {self.selected_method} from the table and comparing them across all columns, ")
        trace.append(f"I identified that the highest score of {max(self.selected_method_row)} occurs in the {self.answer} column.")

        return "\n".join(trace)


if __name__ == "__main__":
    # Generate the table video
    scene = TableMethodsDisappearingRowsRowQ()
    scene.render()

    # Move the output file with descriptive name
    output = Path("manim_output/videos/1080p30/TableMethodsDisappearingRowsRowQ.mp4")
    if output.exists():
        filename = f"table_methods_disappearing_rows_rowq_d{scene.difficulty}_seed{scene.seed}.mp4"
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
