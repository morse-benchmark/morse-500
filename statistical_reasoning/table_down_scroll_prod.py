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


class ScrollingTable(Scene):
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
            self.text_color = BLACK
        else:
            self.background_color = BLACK
            self.text_color = WHITE

        # Store for reasoning trace
        self.table_data = None
        self.cells_to_multiply = []
        self.answer = None
        self.rows = 0
        self.cols = 0
        self.scroll_events = []

    def construct(self):
        # Set background color
        config.background_color = self.background_color

        # Calculate table dimensions based on difficulty
        self.rows = min(20, 2 * self.difficulty + 4)
        self.cols = min(11, 2 * self.difficulty + 7)

        # Generate table data
        np.random.seed(self.seed * self.difficulty)
        self.table_data = np.zeros(shape=(self.rows, self.cols))

        for i in range(self.rows):
            if i == 0:
                continue
            for j in range(self.cols):
                value = np.random.randint(0, 10)
                if j != 0:
                    self.table_data[i, j] = value

        # Generate random cells to multiply
        np.random.seed(self.seed * self.difficulty)
        for _ in range(self.difficulty + 1):
            t_row = np.random.randint(1, self.rows)
            t_col = np.random.randint(1, self.cols)
            self.cells_to_multiply.append((t_row, t_col))

        # Format cells text for question
        cells_text = ""
        if self.difficulty == 0:
            cells_text = f"{self.cells_to_multiply[0]}"
        else:
            for i in range(len(self.cells_to_multiply) - 1):
                cells_text += f"{self.cells_to_multiply[i]},\n "
            if self.difficulty == 1:
                cells_text = cells_text[:-3] + "\n"
            cells_text += f"and {self.cells_to_multiply[-1]}"

        # Calculate answer
        self.answer = 1
        for c in self.cells_to_multiply:
            self.answer *= self.table_data[c]

        # Create and display title/question
        question_text = f"With (x, y) referring to the value in the cell at Row x, Column y, what is the product of {cells_text}?\nPlease answer with just a number and nothing else"
        title = Text(question_text, font_size=20, color=self.text_color)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(0.5)
        self.play(FadeOut(title))

        # Create table
        cell_width, cell_height = 1.25, 0.5
        table = VGroup()

        # Generate header row
        header_row = VGroup()
        for j in range(self.cols):
            cell = Rectangle(width=cell_width, height=cell_height, stroke_color=self.text_color)
            if j == 0:
                text = Text("", font_size=24, color=self.text_color).move_to(cell.get_center())
            else:
                text = Text(f"Col {j}", font_size=24, color=self.text_color).move_to(cell.get_center())
            cell_group = VGroup(cell, text)
            cell_group.move_to(RIGHT * j * cell_width)
            header_row.add(cell_group)
        table.add(header_row)

        # Generate data rows
        np.random.seed(self.seed * self.difficulty)
        for i in range(self.rows):
            if i == 0:
                continue
            row = VGroup()
            for j in range(self.cols):
                value = np.random.randint(0, 10)
                cell = Rectangle(width=cell_width, height=cell_height, stroke_color=self.text_color)
                if j == 0:
                    text = Text(f"Row {i}", font_size=24, color=self.text_color).move_to(cell.get_center())
                else:
                    text = Text(str(value), font_size=24, color=self.text_color).move_to(cell.get_center())
                cell_group = VGroup(cell, text)
                cell_group.move_to(RIGHT * j * cell_width + UP * i * cell_height)
                row.add(cell_group)
            table.add(row)

        # Position top of table at vertical center
        table_height = self.rows * cell_height
        table_width = self.cols * cell_width
        table.move_to(ORIGIN + UP * (table_height - cell_height))

        # Record initial state
        self.scroll_events.append({
            "event_type": "initial_display",
            "time": 0.0,
            "visible_rows": self._get_visible_rows(table, 0)
        })

        self.add(table)

        # Animate scrolling downwards
        scroll_distance = cell_height * 30  # Scroll 30 rows worth of distance
        scroll_time = 4.0

        # Record scroll event
        self.scroll_events.append({
            "event_type": "scroll_down",
            "time": 0.0,
            "scroll_distance": scroll_distance,
            "duration": scroll_time,
            "description": "Table scrolls downward, revealing lower rows"
        })

        self.play(table.animate.shift(DOWN * scroll_distance), run_time=scroll_time)

        # Record final state
        self.scroll_events.append({
            "event_type": "final_display",
            "time": scroll_time,
            "visible_rows": self._get_visible_rows(table, scroll_distance)
        })

        # Hold the final frame for a bit
        self.wait(1)

        # Save solution
        with open(f"solutions/table_down_scroll_prod_d{self.difficulty}_seed{self.seed}.txt", "w") as f:
            f.write(str(int(self.answer)))

        # Save question text
        question_text_content = (
            f"With (x, y) referring to the value in the cell at Row x, Column y, what is the product of {cells_text}?\n"
            "Please answer with just a number and nothing else."
        )
        with open(f"question_text/table_down_scroll_prod_d{self.difficulty}_seed{self.seed}.txt", "w") as f:
            f.write(question_text_content)

        # Generate and save reasoning trace
        reasoning_trace = self.generate_reasoning_trace()
        with open(f"reasoning_traces/table_down_scroll_prod_d{self.difficulty}_seed{self.seed}.txt", "w") as f:
            f.write(reasoning_trace)

    def _get_visible_rows(self, table, shift_amount):
        """Helper to estimate which rows are visible given a scroll amount"""
        # This is a simplified estimation
        cell_height = 0.5
        rows_shifted = int(shift_amount / cell_height)
        return f"Approximately rows shifted by {rows_shifted} positions"

    def generate_reasoning_trace(self):
        """Generate a detailed reasoning trace describing the scene chronologically"""

        trace = []
        trace.append("=== TASK OVERVIEW ===\n")

        # Format cells for display
        cells_display = ", ".join([f"({r}, {c})" for r, c in self.cells_to_multiply])
        trace.append(f"Question: What is the product of the values in cells: {cells_display}?")
        trace.append(f"Table dimensions: {self.rows} rows × {self.cols} columns")
        trace.append(f"Number of cells to multiply: {len(self.cells_to_multiply)}")
        trace.append(f"Difficulty level: {self.difficulty}")
        trace.append("")

        trace.append("\n=== CHRONOLOGICAL OBSERVATION ===\n")
        trace.append("The video begins by showing a question at the top of the screen.")
        trace.append(f"The question asks for the product of specific cell values using (row, column) notation.")
        trace.append("")

        trace.append("Initial Table Display:")
        trace.append(f"  - A table appears with {self.rows} rows and {self.cols} columns")
        trace.append("  - Row 0 contains column headers (Col 1, Col 2, ...)")
        trace.append("  - Column 0 contains row headers (Row 1, Row 2, ...)")
        trace.append("  - The table contains random single-digit integers (0-9)")
        trace.append("  - Initially, the top portion of the table is visible")
        trace.append("")

        trace.append("Scrolling Animation:")
        trace.append("  - The table scrolls downward for 4 seconds")
        trace.append("  - As it scrolls, earlier rows move off the top of the screen")
        trace.append("  - Later rows (with higher row numbers) become visible")
        trace.append("  - This scrolling motion allows viewing the entire table content")
        trace.append("")

        trace.append("Final State:")
        trace.append("  - The video holds on the final frame for 1 second")
        trace.append("  - The lower portion of the table is now visible")
        trace.append("")

        trace.append("\n=== TABLE DATA ANALYSIS ===\n")
        trace.append("Complete table contents (Row, Column: Value):\n")

        # Display table data in a structured way
        for i in range(1, self.rows):  # Skip header row
            row_data = []
            for j in range(1, self.cols):  # Skip header column
                value = int(self.table_data[i, j])
                row_data.append(f"({i},{j}):{value}")
            trace.append(f"  Row {i}: " + ", ".join(row_data))
        trace.append("")

        trace.append("\n=== CELL IDENTIFICATION ===\n")
        trace.append("The question asks for the product of the following cells:")
        for idx, (row, col) in enumerate(self.cells_to_multiply, 1):
            value = int(self.table_data[row, col])
            trace.append(f"  Cell {idx}: ({row}, {col}) = {value}")
        trace.append("")

        trace.append("\n=== REASONING PROCESS ===\n")
        trace.append("To find the product, I need to:")
        trace.append("1. Identify each cell specified in the question using (row, column) notation")
        trace.append("2. Locate each cell's value in the table")
        trace.append("3. Multiply all these values together")
        trace.append("")

        trace.append("Step-by-step multiplication:")
        product = 1
        calculation_steps = []
        for idx, (row, col) in enumerate(self.cells_to_multiply, 1):
            value = int(self.table_data[row, col])
            old_product = product
            product *= value
            if idx == 1:
                calculation_steps.append(f"  Step {idx}: Start with {value}")
            else:
                calculation_steps.append(f"  Step {idx}: {old_product} × {value} = {product}")

        trace.extend(calculation_steps)
        trace.append("")

        trace.append("\n=== FINAL ANSWER ===\n")
        trace.append(f"The product of all specified cell values is: {int(self.answer)}")
        trace.append("")

        trace.append("\n=== REASONING SUMMARY ===\n")
        trace.append(
            f"I observed a scrolling table with {self.rows} rows and {self.cols} columns containing random integers. "
        )
        trace.append(
            f"The question asked for the product of {len(self.cells_to_multiply)} specific cells identified by (row, column) coordinates. "
        )
        trace.append(
            "By locating each cell in the table, extracting its value, and multiplying all values together, "
        )
        trace.append(
            f"I calculated that the product equals {int(self.answer)}."
        )

        return "\n".join(trace)


if __name__ == "__main__":
    # Generate the table scroll video
    scene = ScrollingTable()
    scene.render()

    # Move the output file with descriptive name
    output = Path("manim_output/videos/1080p30/ScrollingTable.mp4")
    if output.exists():
        filename = f"table_down_scroll_prod_d{scene.difficulty}_seed{scene.seed}.mp4"
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
