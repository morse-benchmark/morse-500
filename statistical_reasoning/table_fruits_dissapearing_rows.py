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


class TableFruitsDisappearingRows(Scene):
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
            self.background_color = RED
            self.text_color = BLACK
        else:
            self.background_color = BLACK
            self.text_color = WHITE

        # Data lists
        self.names = [
            "Ava", "Jasper", "Lila", "Nolan", "Maren", "Theo", "Elodie",
            "Kai", "Isla", "Silas", "Tessa", "Remy", "Soren", "Quinn", "Briar"
        ]
        self.fruits = [
            "Apple", "Apricot", "Avocado", "Banana", "Blackberry", "Blackcurrant", "Blueberry", "Boysenberry",
            "Breadfruit", "Cantaloupe", "Carambola", "Cherimoya", "Cherry", "Clementine", "Cloudberry", "Coconut",
            "Cranberry", "Currant", "Date", "Dragonfruit", "Durian", "Elderberry", "Feijoa", "Fig", "Goji berry",
            "Gooseberry", "Grape", "Grapefruit", "Guava", "Hackberry", "Honeyberry", "Honeydew", "Huckleberry",
            "Imbe", "Indian Fig", "Jabuticaba", "Jackfruit", "Jambul", "Jujube", "Juniper berry", "Kaffir Lime",
            "Kiwano", "Kiwi", "Kumquat", "Langsat", "Lemon", "Lime", "Longan", "Loquat"
        ]

        # Store for reasoning trace
        self.table_events = []
        self.headers = []
        self.rows = []
        self.chosen_fruits = []
        self.bought_fruits = []
        self.amounts = []
        self.person_name = None
        self.question_text = ""
        self.answer = None

    def construct(self):
        # Set background color
        config.background_color = self.background_color

        # Generate table data
        random_headers = ["number", "weight (lb)", "calories", "protein", "sugar (g)", "score", "distractor"]

        # Choose fruits
        num_fruits = min(len(self.fruits), 3 + self.difficulty)
        self.chosen_fruits = np.random.choice(self.fruits, num_fruits, replace=False).tolist()

        # Choose headers
        num_headers = min(len(random_headers), 1 + (self.difficulty + 1) // 2)
        self.headers = np.random.choice(random_headers, num_headers, replace=False).tolist()

        # Generate row data based on difficulty
        self.rows = []
        for fruit in self.chosen_fruits:
            if self.difficulty >= 7:
                row_data = np.round(
                    np.clip(
                        np.random.normal(
                            np.random.uniform(20, 80),
                            25 / (1 + self.difficulty),
                            len(self.headers) + 1
                        ),
                        0,
                        100
                    ),
                    2
                ).tolist()
            elif self.difficulty >= 4:
                row_data = np.round(
                    np.clip(
                        np.random.normal(
                            np.random.uniform(20, 80),
                            25 / (1 + self.difficulty),
                            len(self.headers) + 1
                        ),
                        0,
                        100
                    ),
                    1
                ).tolist()
            else:
                row_data = np.random.randint(20, 80, len(self.headers) + 1).tolist()

            # Format price column (first column after fruit name)
            price = row_data[0]
            row_data[0] = price  # Keep numeric value for calculation
            self.rows.append([fruit, row_data])

        # Select fruits to buy
        num_to_buy = self.difficulty + 1
        bought_indices = np.random.choice(range(len(self.rows)), num_to_buy, replace=False)
        self.bought_fruits = [(self.rows[i][0], self.rows[i][1][0]) for i in bought_indices]

        # Generate amounts
        self.amounts = np.random.randint(2, 10, num_to_buy).tolist()

        # Generate question text
        amount_text = ""
        if self.difficulty == 0:
            amount_text = f"{self.amounts[0]} {self.bought_fruits[0][0]}s"
        else:
            for i in range(len(self.bought_fruits) - 1):
                amount_text += f"{self.amounts[i]} {self.bought_fruits[i][0]}s, "
            if self.difficulty == 1:
                amount_text = amount_text[:-2] + " "
            amount_text += f"and {self.amounts[-1]} {self.bought_fruits[-1][0]}s"

        self.person_name = np.random.choice(self.names)
        self.question_text = f"Question: {self.person_name} wants to buy {amount_text}. How much does it cost?\nPlease answer with just a number and nothing else"

        # Calculate answer
        self.answer = sum(self.amounts[i] * self.bought_fruits[i][1] for i in range(len(self.bought_fruits)))
        if self.difficulty < 4:
            self.answer = int(self.answer)
        else:
            self.answer = round(self.answer, 2)

        # Display question
        title = Text(self.question_text, font_size=24, color=self.text_color)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(0.5)
        self.play(FadeOut(title))

        # Record event
        self.table_events.append({
            "event": "question_displayed",
            "time": 0.0,
            "text": self.question_text
        })

        # Prepare table data for MathTable
        table_headers = ["\\text{fruit}", "\\text{price per kg}"] + [f"\\text{{{h}}}" for h in self.headers]
        table_rows = []

        cumulative_time = 1.5  # Account for question display time

        for fruit, row_data in self.rows:
            formatted_row = [
                f"\\text{{{fruit}}}",
                f"{row_data[0]}\\text{{ dollars}}"
            ] + [str(val) for val in row_data[1:]]
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
            else:
                row_content = table_rows[i - 1]

            self.table_events.append({
                "event": "row_displayed",
                "row_index": i,
                "content": row_content,
                "start_time": row_start_time,
                "end_time": row_end_time,
                "duration": 0.5
            })

        self.wait(1)

        # Save solution
        with open(
            f"solutions/table_fruits_disappearing_rows_d{self.difficulty}_seed{self.seed}.txt", "w"
        ) as f:
            f.write(str(self.answer))

        # Save question text
        with open(
            f"question_text/table_fruits_disappearing_rows_d{self.difficulty}_seed{self.seed}.txt", "w"
        ) as f:
            f.write(self.question_text)

        # Generate and save reasoning trace
        reasoning_trace = self.generate_reasoning_trace()
        with open(
            f"reasoning_traces/table_fruits_disappearing_rows_d{self.difficulty}_seed{self.seed}.txt", "w"
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
        trace.append(f"  Headers: fruit, price per kg, {', '.join(self.headers)}\n")

        trace.append("Rows that appeared (in order):")
        for i, event in enumerate(self.table_events):
            if event["event"] == "row_displayed":
                row_idx = event["row_index"]
                if row_idx == 0:
                    trace.append(f"  Row {row_idx + 1} (Header): {', '.join(event['content'])}")
                else:
                    trace.append(f"  Row {row_idx + 1}: {', '.join(map(str, event['content']))}")
                trace.append(f"    - Appeared at {event['start_time']:.2f}s, disappeared at {event['end_time']:.2f}s")
        trace.append("")

        trace.append("\n=== ANALYSIS ===\n")

        trace.append("From the table, I need to extract the relevant information:")
        trace.append("\nFruit prices (dollars per kg):")
        for fruit, row_data in self.rows:
            trace.append(f"  - {fruit}: ${row_data[0]}")

        trace.append(f"\nItems to purchase by {self.person_name}:")
        for i, (fruit, price) in enumerate(self.bought_fruits):
            trace.append(f"  - {self.amounts[i]} {fruit}s at ${price} per kg")
        trace.append("")

        trace.append("\n=== REASONING PROCESS ===\n")

        trace.append("To find the total cost, I need to:")
        trace.append("1. Identify which fruits need to be purchased")
        trace.append("2. Find the price per kg for each fruit from the table")
        trace.append("3. Multiply the quantity by the price for each fruit")
        trace.append("4. Sum all the individual costs\n")

        trace.append("Calculations:")
        total = 0
        for i, (fruit, price) in enumerate(self.bought_fruits):
            cost = self.amounts[i] * price
            trace.append(f"  {self.amounts[i]} {fruit}s × ${price} = ${cost}")
            total += cost

        trace.append(f"\nTotal: {' + '.join([f'${self.amounts[i] * self.bought_fruits[i][1]}' for i in range(len(self.bought_fruits))])}")
        trace.append(f"     = ${self.answer}")
        trace.append("")

        trace.append("\n=== FINAL ANSWER ===\n")
        trace.append(f"The total cost is: ${self.answer}")
        trace.append("")

        trace.append("\n=== REASONING SUMMARY ===\n")
        trace.append(f"The video presented a table showing various fruits and their prices per kilogram. ")
        trace.append(f"The question asked how much it would cost for {self.person_name} to buy specific quantities of certain fruits. ")
        trace.append(f"By extracting the prices from the table and multiplying each price by the corresponding quantity, ")
        trace.append(f"then summing all the individual costs, I calculated the total cost to be ${self.answer}.")

        return "\n".join(trace)


if __name__ == "__main__":
    # Generate the table video
    scene = TableFruitsDisappearingRows()
    scene.render()

    # Move the output file with descriptive name
    output = Path("manim_output/videos/1080p30/TableFruitsDisappearingRows.mp4")
    if output.exists():
        filename = f"table_fruits_disappearing_rows_d{scene.difficulty}_seed{scene.seed}.mp4"
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
