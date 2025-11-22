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


class TableFruitsDisappearingCols(Scene):
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
            self.background_color = WHITE
            self.text_color = RED
        else:
            self.background_color = BLACK
            self.text_color = WHITE

        # Data
        self.names = ["Ava", "Jasper", "Lila", "Nolan", "Maren", "Theo", "Elodie", "Kai", "Isla", "Silas", "Tessa", "Remy", "Soren", "Quinn", "Briar"]
        self.fruits = [
            "Apple", "Apricot", "Avocado", "Banana", "Blackberry", "Blackcurrant", "Blueberry", "Boysenberry",
            "Breadfruit", "Cantaloupe", "Carambola", "Cherimoya", "Cherry", "Clementine", "Cloudberry", "Coconut",
            "Cranberry", "Currant", "Date", "Dragonfruit", "Durian", "Elderberry", "Feijoa", "Fig", "Goji berry",
            "Gooseberry", "Grape", "Grapefruit", "Guava", "Hackberry", "Honeyberry", "Honeydew", "Huckleberry",
            "Imbe", "Indian Fig", "Jabuticaba", "Jackfruit", "Jambul", "Jujube", "Juniper berry", "Kaffir Lime",
            "Kiwano", "Kiwi", "Kumquat", "Langsat", "Lemon", "Lime", "Longan", "Loquat"
        ]

        # Store for reasoning trace
        self.column_events = []
        self.table_data = None
        self.headers = None
        self.rows = None
        self.chosen_fruits = None
        self.bought_fruits = None
        self.amounts = None
        self.amount_text = None
        self.answer = None
        self.question = None
        self.person_name = None

    def construct(self):
        # Set background color
        config.background_color = self.background_color

        # Generate table data
        random_headers = ["number", "weight (lb)", "calories", "protein", "sugar (g)", "score", "distractor"]
        self.chosen_fruits = np.random.choice(self.fruits, min(len(self.fruits), 3 + self.difficulty), replace=False)
        header_choices = np.random.choice(random_headers, min(len(random_headers), 1 + (self.difficulty+1)//2), replace=False)
        self.headers = list(header_choices)

        # Generate price data
        if self.difficulty >= 7:
            self.rows = [[self.chosen_fruits[k]] +
                    list(np.round(np.clip(np.random.normal(np.random.uniform(20, 80), 25/(1+self.difficulty), len(self.headers)+1), 0, 100), 2))
                    for k in range(len(self.chosen_fruits))]
        elif self.difficulty >= 4:
            self.rows = [[self.chosen_fruits[k]] +
                    list(np.round(np.clip(np.random.normal(np.random.uniform(20, 80), 25/(1+self.difficulty), len(self.headers)+1), 0, 100), 1))
                    for k in range(len(self.chosen_fruits))]
        else:
            self.rows = [[self.chosen_fruits[k]] +
                    list(np.random.randint(20, 80, len(self.headers)+1))
                    for k in range(len(self.chosen_fruits))]

        # Add dollar signs to price column
        for r in self.rows:
            r[1] = str(r[1]) + " dollars"

        # Select fruits to buy
        bought_indices = np.random.choice(range(len(self.rows)), self.difficulty+1, replace=False)
        self.bought_fruits = [(self.rows[i][0], float(str(self.rows[i][1]).split()[0])) for i in bought_indices]
        self.amounts = np.random.randint(2, 10, self.difficulty+1)

        # Generate amount text
        if self.difficulty == 0:
            self.amount_text = f"{self.amounts[0]} {self.bought_fruits[0][0]}s"
        else:
            amount_parts = []
            for i, (f, _) in enumerate(self.bought_fruits[:-1]):
                amount_parts.append(f"{self.amounts[i]} {f}s")
            if self.difficulty == 1:
                self.amount_text = ", ".join(amount_parts) + f" and {self.amounts[-1]} {self.bought_fruits[-1][0]}s"
            else:
                self.amount_text = ", ".join(amount_parts) + f", and {self.amounts[-1]} {self.bought_fruits[-1][0]}s"

        # Calculate answer
        self.answer = sum(self.amounts[i] * self.bought_fruits[i][1] for i in range(self.difficulty+1))
        if self.difficulty < 4:
            self.answer = int(self.answer)
        else:
            self.answer = round(self.answer, 2)

        # Generate question
        self.person_name = np.random.choice(self.names)
        self.question = f"Question: {self.person_name} wants to buy {self.amount_text}. How much does it cost?\nPlease answer with just a number and nothing else"

        # Display question
        title = Text(self.question, font_size=20, color=self.text_color)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(0.5)
        self.play(FadeOut(title))

        # Create table
        table_headers = ["\\text{fruit}", "\\text{price per kg}"] + ["\\text{" + h + "}" for h in self.headers]
        table_rows = []
        for row in self.rows:
            formatted_row = ["\\text{" + str(row[0]) + "}"] + [str(row[i]) if i == 1 else str(row[i]) for i in range(1, len(row))]
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
        with open(f"solutions/table_fruits_dissapearing_cols_d{self.difficulty}_seed{self.seed}.txt", "w") as f:
            f.write(str(self.answer))

        # Save question text
        question_text_content = self.question
        with open(f"question_text/table_fruits_dissapearing_cols_d{self.difficulty}_seed{self.seed}.txt", "w") as f:
            f.write(question_text_content)

        # Generate and save reasoning trace
        reasoning_trace = self.generate_reasoning_trace()
        with open(f"reasoning_traces/table_fruits_dissapearing_cols_d{self.difficulty}_seed{self.seed}.txt", "w") as f:
            f.write(reasoning_trace)

    def generate_reasoning_trace(self):
        """Generate a detailed reasoning trace describing the scene chronologically"""

        trace = []
        trace.append("=== TASK OVERVIEW ===\n")
        trace.append(f"Question: {self.question}")
        trace.append(f"Difficulty level: {self.difficulty}")
        trace.append(f"Number of fruit types in table: {len(self.chosen_fruits)}")
        trace.append(f"Number of columns (including fruit and price): {len(self.table_data[0])}")
        trace.append("")

        trace.append("\n=== CHRONOLOGICAL OBSERVATION ===\n")
        trace.append("The video shows a table where columns appear and disappear one at a time.")
        trace.append("Each column is displayed briefly (fades in, then fades out) before the next column appears.")
        trace.append("This requires remembering the information from the 'fruit' and 'price per kg' columns to answer the question.\n")

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
        trace.append("From the table, I need to extract the fruit names and their prices:\n")

        trace.append("Fruit-Price Mapping:")
        for row in self.rows:
            fruit_name = row[0]
            price_str = row[1]
            price = float(price_str.split()[0])
            trace.append(f"  {fruit_name}: ${price:.2f} per kg")
        trace.append("")

        trace.append("\n=== PURCHASE ANALYSIS ===\n")
        trace.append(f"{self.person_name} wants to buy: {self.amount_text}\n")

        trace.append("Breaking down the purchase:")
        total_cost = 0
        for i in range(len(self.bought_fruits)):
            fruit_name = self.bought_fruits[i][0]
            price = self.bought_fruits[i][1]
            quantity = self.amounts[i]
            cost = quantity * price
            total_cost += cost

            trace.append(f"  {quantity} {fruit_name}s × ${price:.2f} = ${cost:.2f}")
        trace.append("")

        trace.append("\n=== CALCULATION PROCESS ===\n")
        trace.append("Total cost calculation:")

        calculation_steps = []
        for i in range(len(self.bought_fruits)):
            fruit_name = self.bought_fruits[i][0]
            price = self.bought_fruits[i][1]
            quantity = self.amounts[i]
            cost = quantity * price
            calculation_steps.append(f"${cost:.2f}")

        if len(calculation_steps) > 1:
            trace.append(f"  {' + '.join(calculation_steps)} = ${self.answer}")
        else:
            trace.append(f"  ${calculation_steps[0]} = ${self.answer}")
        trace.append("")

        trace.append("\n=== FINAL ANSWER ===\n")
        if self.difficulty < 4:
            trace.append(f"The total cost is: {int(self.answer)}")
        else:
            trace.append(f"The total cost is: {self.answer:.2f}")
        trace.append("")

        trace.append("\n=== REASONING SUMMARY ===\n")
        trace.append(
            f"I observed a table with {len(self.chosen_fruits)} fruits and their prices displayed in a "
            f"format where columns appeared and disappeared sequentially. "
        )
        trace.append(
            f"The table showed {len(self.table_data[0])} columns in total, including the fruit names, "
            f"prices, and additional information columns. "
        )
        trace.append(
            f"To answer the question, I needed to remember the fruit prices from when they were briefly visible. "
        )
        trace.append(
            f"{self.person_name} wanted to buy {len(self.bought_fruits)} different types of fruit in various quantities. "
        )
        trace.append(
            f"By multiplying each quantity by its corresponding price and summing the results, "
        )
        if self.difficulty < 4:
            trace.append(
                f"I calculated the total cost to be ${int(self.answer)}."
            )
        else:
            trace.append(
                f"I calculated the total cost to be ${self.answer:.2f}."
            )

        return "\n".join(trace)


if __name__ == "__main__":
    # Generate the table video
    scene = TableFruitsDisappearingCols()
    scene.render()

    # Move the output file with descriptive name
    output = Path("manim_output/videos/1080p30/TableFruitsDisappearingCols.mp4")
    if output.exists():
        filename = f"table_fruits_dissapearing_cols_d{scene.difficulty}_seed{scene.seed}.mp4"
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
