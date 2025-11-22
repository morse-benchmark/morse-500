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


class ScrollingTable(Scene):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Set random seed for reproducibility
        self.seed = random.randint(1000, 9999)
        random.seed(self.seed)

        # Parameters
        self.difficulty = int(os.getenv("DIFFICULTY", 5))

        # Store for reasoning trace
        self.table_data = None
        self.target_cells = []
        self.cell_values = []
        self.rows = 0
        self.cols = 0
        self.background_color = "BLACK"
        self.text_color = "WHITE"

    def construct(self):
        # Configure colors based on difficulty
        if self.difficulty == 9:
            self.background_color = "WHITE"
            self.text_color = "BLACK"
            config.background_color = WHITE
        elif self.difficulty == 7:
            self.background_color = "WHITE"
            self.text_color = "RED"
            config.background_color = WHITE
        else:
            config.background_color = BLACK

        # Generate table dimensions
        self.rows = min(20, 2 * self.difficulty + 4)
        self.cols = min(11, 2 * self.difficulty + 7)

        # Generate table data
        self.table_data = np.zeros(shape=(self.rows, self.cols))
        np.random.seed(self.seed * self.difficulty)

        for i in range(self.rows):
            if i == 0:
                continue
            for j in range(self.cols):
                value = np.random.randint(0, 99)
                if j != 0:
                    self.table_data[i, j] = value

        # Generate target cells
        for _ in range(self.difficulty + 1):
            t_row = np.random.randint(1, self.rows)
            t_col = np.random.randint(1, self.cols)
            self.target_cells.append((t_row, t_col))
            self.cell_values.append(int(self.table_data[t_row, t_col]))

        # Format cells text for question
        cells_text = ""
        if self.difficulty == 0:
            cells_text = f"{self.target_cells[0]}"
        else:
            for i in range(len(self.target_cells) - 1):
                cells_text += f"{self.target_cells[i]},\n "
            if self.difficulty == 1:
                cells_text = cells_text[:-3] + "\n "
            cells_text += f"and {self.target_cells[-1]}"

        # Calculate answer
        answer = sum(self.cell_values)

        # Get text color for Manim
        text_color_obj = self._get_color_object(self.text_color)

        # Display question
        question = f"With (x, y) referring to the value in the cell at Row x, Column y,\nwhat is the sum of {cells_text}?\nPlease answer with just a number and nothing else"

        title = Text(question, font_size=20, color=text_color_obj)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(0.5)
        self.play(FadeOut(title))

        # Create table
        cell_width, cell_height = 1.25, 0.5
        table = VGroup()

        # Header row
        header_row = VGroup()
        for j in range(self.cols):
            cell = Rectangle(width=cell_width, height=cell_height, stroke_color=text_color_obj)
            if j == 0:
                text = Text(f"", font_size=24, color=text_color_obj).move_to(cell.get_center())
            else:
                text = Text(f"Col {j}", font_size=24, color=text_color_obj).move_to(cell.get_center())
            cell_group = VGroup(cell, text)
            cell_group.move_to(RIGHT * j * cell_width)
            header_row.add(cell_group)
        table.add(header_row)

        # Data rows
        np.random.seed(self.seed * self.difficulty)
        for i in range(self.rows):
            if i == 0:
                continue
            row = VGroup()
            for j in range(self.cols):
                value = np.random.randint(0, 99)
                cell = Rectangle(width=cell_width, height=cell_height, stroke_color=text_color_obj)
                if j == 0:
                    text = Text(f"Row {i}", font_size=24, color=text_color_obj).move_to(cell.get_center())
                else:
                    text = Text(str(value), font_size=24, color=text_color_obj).move_to(cell.get_center())
                cell_group = VGroup(cell, text)
                cell_group.move_to(RIGHT * j * cell_width + DOWN * i * cell_height)
                row.add(cell_group)
            table.add(row)

        # Position table - bottom of table at vertical center
        table_height = self.rows * cell_height
        table.move_to(ORIGIN + DOWN * (table_height - cell_height))

        self.add(table)

        # Animate scrolling upwards
        scroll_distance = cell_height * 30
        self.play(table.animate.shift(UP * scroll_distance), run_time=4)

        # Hold the final frame
        self.wait(1)

        # Save solution
        with open(
            f"solutions/table_upwards_scroll_sum_d{self.difficulty}_seed{self.seed}.txt", "w"
        ) as f:
            f.write(str(answer))

        # Save question text
        question_text_content = f"With (x, y) referring to the value in the cell at Row x, Column y, what is the sum of {cells_text}?\nPlease answer with just a number and nothing else"
        with open(
            f"question_text/table_upwards_scroll_sum_d{self.difficulty}_seed{self.seed}.txt", "w"
        ) as f:
            f.write(question_text_content)

        # Generate and save reasoning trace
        reasoning_trace = self.generate_reasoning_trace(answer)
        with open(
            f"reasoning_traces/table_upwards_scroll_sum_d{self.difficulty}_seed{self.seed}.txt", "w"
        ) as f:
            f.write(reasoning_trace)

    def _get_color_object(self, color_str):
        """Convert color string to Manim color object"""
        color_map = {
            "WHITE": WHITE,
            "BLACK": BLACK,
            "BLUE": BLUE,
            "RED": RED,
            "GREEN": GREEN,
            "YELLOW": YELLOW,
        }
        return color_map.get(color_str, WHITE)

    def generate_reasoning_trace(self, answer):
        """Generate a detailed reasoning trace describing the scene chronologically"""

        trace = []
        trace.append("=== CHRONOLOGICAL OBSERVATION ===\n")

        trace.append("1. Question Display (0.0s - 0.5s):")
        trace.append(f"   The video begins by displaying a question that asks for the sum of specific table cells.")
        trace.append(f"   The notation (x, y) refers to the value in the cell at Row x, Column y.")
        trace.append(f"   Target cells: {', '.join(str(cell) for cell in self.target_cells)}")
        trace.append("")

        trace.append("2. Initial Table Display (0.5s):")
        trace.append(f"   A table appears with {self.rows} rows and {self.cols} columns.")
        trace.append(f"   The table contains randomly generated integer values between 0 and 99.")
        trace.append(f"   The first row contains column headers (Col 1, Col 2, ...).")
        trace.append(f"   The first column contains row headers (Row 1, Row 2, ...).")
        trace.append(f"   The table is initially positioned with its bottom portion visible on screen.")
        trace.append("")

        trace.append("3. Table Scrolling Animation (0.5s - 4.5s):")
        trace.append(f"   The table scrolls vertically upward over 4 seconds.")
        trace.append(f"   This allows viewing of cells that were initially off-screen at the top.")
        trace.append(f"   The animation reveals the full extent of the table's height.")
        trace.append(f"   As the table scrolls up, rows that were at the bottom move upward and eventually off-screen.")
        trace.append(f"   Rows that were initially hidden at the top come into view.")
        trace.append("")

        trace.append("4. Final Frame Hold (4.5s - 5.5s):")
        trace.append(f"   The table remains stationary for 1 second after scrolling completes.")
        trace.append(f"   This provides time to observe the final position of the table.")
        trace.append("")

        trace.append("\n=== ANALYSIS ===\n")

        trace.append("Table Structure:")
        trace.append(f"  - Dimensions: {self.rows} rows × {self.cols} columns")
        trace.append(f"  - Background color: {self.background_color}")
        trace.append(f"  - Text color: {self.text_color}")
        trace.append(f"  - Cell values: Random integers from 0 to 99")
        trace.append(f"  - Scroll direction: Upward (vertical)")
        trace.append("")

        trace.append("Target Cells and Their Values:")
        for i, (cell, value) in enumerate(zip(self.target_cells, self.cell_values), 1):
            trace.append(f"  Cell {i}: {cell} → Value: {value}")
        trace.append("")

        trace.append("\n=== REASONING PROCESS ===\n")

        trace.append("To solve this problem, I need to:")
        trace.append("1. Identify the coordinates of all target cells mentioned in the question")
        trace.append("2. Locate each cell in the table using the (row, column) notation")
        trace.append("3. Track the cells as the table scrolls upward to ensure correct identification")
        trace.append("4. Extract the numeric value from each target cell")
        trace.append("5. Sum all the extracted values")
        trace.append("")

        trace.append("Step-by-step calculation:")
        running_sum = 0
        for i, (cell, value) in enumerate(zip(self.target_cells, self.cell_values), 1):
            if i == 1:
                trace.append(f"  Start: {value}")
                running_sum = value
            else:
                trace.append(f"  + {value} = {running_sum + value}")
                running_sum += value
        trace.append("")

        trace.append("\n=== FINAL ANSWER ===\n")
        trace.append(f"The sum of values at cells {', '.join(str(cell) for cell in self.target_cells)} is: {answer}")
        trace.append("")

        trace.append("\n=== REASONING SUMMARY ===\n")
        trace.append(
            f"The video presents a table with {self.rows} rows and {self.cols} columns containing random numeric values. "
        )
        trace.append(
            f"The table scrolls vertically upward to reveal cells that extend beyond the initial view. "
        )
        trace.append(
            f"By carefully tracking the table during the upward scroll and identifying the {len(self.target_cells)} target cells "
        )
        trace.append(
            f"specified in the question, I extracted their values ({', '.join(str(v) for v in self.cell_values)}). "
        )
        trace.append(
            f"Summing these values together yields the final answer: {answer}."
        )

        return "\n".join(trace)


if __name__ == "__main__":
    # Generate the scrolling table video
    scene = ScrollingTable()
    scene.render()

    # Move the output file with descriptive name
    output = Path("manim_output/videos/1080p30/ScrollingTable.mp4")
    if output.exists():
        filename = f"table_upwards_scroll_sum_d{scene.difficulty}_seed{scene.seed}.mp4"
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
