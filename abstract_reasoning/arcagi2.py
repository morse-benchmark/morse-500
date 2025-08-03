from manim import *
import json, itertools, random, copy
import os
from pathlib import Path
from google import genai
from pydantic import BaseModel
from collections import Counter
from itertools import chain

API_KEY = os.environ['GEMINI_KEY']


class ARCResponse(BaseModel):
    r_1: list[list[int]]


def grid_to_text(grid):
    """Convert a 2‑d int list to the required space‑separated row string."""
    return "\n".join(" ".join(map(str, row)) for row in grid)


def build_prompt(task):
    parts = [
        "Find the common rule that maps an input grid to an output grid, given the examples below.\n"
    ]
    for i, ex in enumerate(task["train"], 1):
        parts += [
            f"Example {i}:\n",
            "Input:",
            grid_to_text(ex["input"]),
            "Output:",
            grid_to_text(ex["output"]),
            "",
        ]

    test_input = task["test"][0]["input"]
    parts += [
        "Below is a test input grid. Predict the corresponding output grid by applying the rule you found. Your final answer should just be the output grid itself in the form of a 2D integer array.",
        "Input:",
        grid_to_text(test_input),
        "",
    ]

    prompt_examples = "\n".join(parts)
    return prompt_examples


def normalise_json(reply):
    if reply.startswith("```"):
        reply = reply.strip("`").lstrip("json").strip()
    data = json.loads(reply)
    print(data)
    return {0: data.get("r_1")}


def ask_gemini(prompt, model):
    client = genai.Client(api_key=API_KEY)

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": ARCResponse,
        },
    )

    return response.text


def get_variants(task, n=3, model=None):
    prompt = build_prompt(task)
    models = ["gemini-1.5-flash", "gemini-2.0-flash", "gemini-2.0-flash-lite"]

    answers = []

    for model in models:
        reply = ask_gemini(prompt, model)
        parsed = normalise_json(reply)[0]
        answers.append(parsed)

    return answers


class ARCScene(Scene):

    def __init__(self, data_path, p_type):
        super().__init__()
        self.p_type = p_type
        self.DATA_PATH = data_path  # path to ARC json file

        # ---------- Visual constants ----------
        self.CELL_SIZE = 0.35
        self.MAX_GRID = 3.0  # max side length after scaling for training grids
        self.TOP_SCALE = 0.8  # additional shrink factor for test (top) grids
        self.COLOR_TABLE = [
            BLACK,  # 0 – background / empty
            ManimColor.from_hex("#0074D9"),  # 1 – blue
            ManimColor.from_hex("#FF4136"),  # 2 – orange
            ManimColor.from_hex("#2ECC40"),  # 3 – green
            ManimColor.from_hex("#FFDC00"),  # 4 – yellow
            ManimColor.from_hex("#AAAAAA"),  # 5 – light gray
            ManimColor.from_hex("#F012BE"),  # 6 – pink
            ManimColor.from_hex("#7FDBFF"),  # 7 – light blue
            ManimColor.from_hex("#870C25"),  # 8 – dark red
            WHITE,  # 9 – white
        ]
        self.NUM_TO_STR = [
            "black",
            "blue",
            "orange",
            "green",
            "yellow",
            "light gray",
            "pink",
            "light blue",
            "dark red",
            "white",
        ]
        self.TRAIN_STAY = 1.5
        self.TRANSITION = 0.5
        self.TEST_BIG_STAY = 1.5  # how long to hold full-size test pair
        self.TEST_STAY = 4.0  # after options appear

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def grid_to_vgroup(self, grid):
        """Convert a 2‑D list of ints into a VGroup of colored squares."""
        print(grid)
        vg = VGroup()
        for r in range(len(grid)):
            for c in range(len(grid[r])):
                val = grid[r][c]
                val = val % 10
                color = self.COLOR_TABLE[val]
                sq = Square(side_length=self.CELL_SIZE, stroke_width=0)
                sq.set_fill(color, opacity=1)
                # top‑left origin mapping
                x = (c + 0.5) * self.CELL_SIZE
                y = -(r + 0.5) * self.CELL_SIZE
                sq.move_to([x, y, 0])
                vg.add(sq)
        # center on origin then return
        vg.move_to(ORIGIN)
        return vg

    def scale_grid(self, vg, max_side):
        """Scale *vg* so its larger dimension equals *max_side*."""
        sf = min(max_side / vg.width, max_side / vg.height)
        vg.scale(sf)
        return vg

    def load_task(self):
        with open(self.DATA_PATH, "r") as fp:
            task = json.load(fp)
        train_pairs = [(s["input"], s["output"]) for s in task["train"]]
        test_input = task["test"][0]["input"]
        test_output = task["test"][0]["output"]
        return task, train_pairs, test_input, test_output

    def color_perturb(self, grid, p_changes=0.1):
        g = copy.deepcopy(grid)
        rows, cols = len(g), len(g[0])
        n_changes = int(p_changes * rows * cols)
        for _ in range(n_changes):
            allowed = set()
            while len(allowed) == 0:
                allowed = set()
                r, c = random.randrange(rows), random.randrange(cols)
                orig = g[r][c]
                dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right

                for dr, dc in dirs:
                    if 0 <= r+dr < rows and 0 <= c+dc < cols and grid[r+dr][c+dc]!=orig:
                        allowed.add(grid[r+dr][c+dc])

            g[r][c] = random.choice(list(allowed))

        return g

    def show_colors(self, colors, names):

        cols = VGroup()
        all_squares = VGroup()
        all_arrows = VGroup()
        all_labels = VGroup()
        for start in range(0, len(colors), 4):

            squares = VGroup(
                *[
                    Square(1.0).set_fill(col, 1).set_stroke(color=WHITE, width=2)
                    for col in colors[start : start + 4]
                ]
            )
            squares.arrange(DOWN, buff=0.5, aligned_edge=LEFT).to_edge(LEFT, buff=2)
            labels = VGroup(
                *[Text(nm, font_size=32) for nm in names[start : start + 4]]
            )
            for sq, lbl in zip(squares, labels):
                lbl.next_to(sq, RIGHT, buff=1.2)
            arrows = VGroup(
                *[
                    Arrow(
                        start=sq.get_right(),
                        end=lbl.get_left(),
                        buff=0.05,
                        stroke_width=4,
                    )
                    for sq, lbl in zip(squares, labels)
                ]
            )
            col_group = VGroup(squares, arrows, labels)
            cols.add(col_group)
            all_squares.add(*squares)
            all_arrows.add(*arrows)
            all_labels.add(*labels)

        cols.arrange(buff=1.7, aligned_edge=UP).move_to(ORIGIN).scale_to_fit_width(
            0.9 * config.frame_width
        )

        # Title
        title = Text("Remember the following color names", font_size=40)
        title.to_edge(UP)

        # Animation sequence
        self.play(Write(title))
        self.play(
            Succession(
                FadeIn(all_squares),
                AnimationGroup(*[GrowArrow(ar) for ar in all_arrows], lag_ratio=0.1),
                AnimationGroup(*[Write(label) for label in all_labels]),
            ),
            run_time=1.5,
        )
        self.wait(2)
        self.play(FadeOut(title, cols))
        self.wait(0.5)

    # ---------- Main construct ----------
    def construct(self):
        self.add(
            Rectangle(height=config.frame_height, width=config.frame_width)
            .set_color(color_gradient(random.sample(self.COLOR_TABLE, 2), 5))
            .set_opacity(0.7)
        )
        self.show_colors(self.COLOR_TABLE, self.NUM_TO_STR)

        prompt = Text(
            "Observe the following inputs/outputs", color=WHITE, font_size=36
        ).move_to(ORIGIN)
        self.play(FadeIn(prompt), run_time=0.5)
        self.wait(1.5)
        self.play(FadeOut(prompt), run_time=0.5)
        self.wait(1)
        task_raw, train_pairs, test_input, test_output = self.load_task()

        arrow_proto = Arrow(LEFT, RIGHT, color=WHITE, buff=0.2)
        train_L = LEFT * 3
        train_R = RIGHT * 3

        # ---------- Training slideshow ----------
        for inp, out in train_pairs:
            lgrid = self.scale_grid(self.grid_to_vgroup(inp), self.MAX_GRID).move_to(
                train_L
            )
            rgrid = self.scale_grid(self.grid_to_vgroup(out), self.MAX_GRID).move_to(
                train_R
            )
            arr = arrow_proto.copy()
            self.play(
                FadeIn(lgrid, shift=DOWN * 0.2),
                FadeIn(arr),
                FadeIn(rgrid, shift=UP * 0.2),
                run_time=self.TRANSITION,
            )
            self.wait(self.TRAIN_STAY)
            self.play(
                FadeOut(lgrid), FadeOut(arr), FadeOut(rgrid), run_time=self.TRANSITION
            )

        # ---------- Test: show full‑size pair ----------
        L_big_anchor = LEFT * 3
        R_big_anchor = RIGHT * 3
        test_in_big = self.scale_grid(
            self.grid_to_vgroup(test_input), self.MAX_GRID
        ).move_to(L_big_anchor)

        blank_grid = [[0 for _ in row] for row in test_output]
        blank_big = self.scale_grid(
            self.grid_to_vgroup(blank_grid), self.MAX_GRID
        ).move_to(R_big_anchor)
        q_big = Text("?", font_size=160, color=WHITE).move_to(blank_big)
        arrow_big = arrow_proto.copy()

        self.play(
            FadeIn(test_in_big),
            FadeIn(arrow_big),
            FadeIn(q_big),
            run_time=self.TRANSITION,
        )
        self.wait(self.TEST_BIG_STAY)
        if self.p_type == "count":

            ca = Counter(chain.from_iterable(test_input))
            cb = Counter(chain.from_iterable(test_output))
            changed = {n for n in set(ca) | set(cb) if ca[n] != cb[n]} - {0, 9}
            color = random.choice(list(changed))
            color_str = self.NUM_TO_STR[color]
            title = f"How many {color_str} squares should appear in the output grid?\nAnswer with a single integer"
        else:
            title = "Which output grid should follow? Answer with one multiple choice option."
        lines = title.split("\n")

        para = Paragraph(*lines, alignment="center", font_size=36, line_spacing=0.8)
        para.to_edge(UP)
        if para.width > 0.9 * config.frame_width:
            para.scale_to_fit_width(config.frame_width * 0.9)
        self.play(Write(para), run_time=1)

        # self.wait(1)
        if self.p_type == "count":
            self.answer = sum(row.count(color) for row in test_output)
        else:
            # ---------- Shrink to top & introduce options ----------
            L_small_anchor = UP * 1.2 + LEFT * 3
            R_small_anchor = UP * 1.2 + RIGHT * 3
            test_in_small = self.scale_grid(
                self.grid_to_vgroup(test_input), self.MAX_GRID * self.TOP_SCALE
            ).move_to(L_small_anchor)
            blank_small = self.scale_grid(
                self.grid_to_vgroup(blank_grid), self.MAX_GRID * self.TOP_SCALE
            ).move_to(R_small_anchor)
            q_small = Text("?", font_size=120, color=WHITE).move_to(blank_small)
            arrow_small = arrow_proto.copy().move_to(UP * 1.7)

            # Transform big → small while fading in bottom row later
            self.play(
                ReplacementTransform(test_in_big, test_in_small),
                ReplacementTransform(q_big, q_small),
                arrow_big.animate.move_to(arrow_small.get_center()),
                run_time=self.TRANSITION,
            )

            # Build MCQ row
            distractors = get_variants(task_raw)
            none_answer = False
            if random.random() < 0.3:
                none_answer = True
                test_output = self.color_perturb(test_output)

            options = distractors + [test_output]
            random.shuffle(options)
            options.append("none")
            labels = ["a", "b", "c", "d", "e"]
            row_y = DOWN * 2.2
            spacing = 3.2
            options_vg = VGroup()
            for idx, (opt, lab) in enumerate(zip(options, labels)):
                option = VGroup()
                if lab == "e":
                    vg = Text(
                        "None of the above", font_size=36, color=WHITE
                    ).scale_to_fit_width(self.MAX_GRID * 0.75)
                else:
                    vg = self.scale_grid(self.grid_to_vgroup(opt), self.MAX_GRID * 0.75)
                label = Text(f"{lab})", font_size=40, color=WHITE).next_to(
                    vg, DOWN, buff=0.2
                )
                option.add(vg)
                option.add(label)
                options_vg.add(option)
            options_vg.arrange(buff=1.5, aligned_edge=DOWN)
            options_vg.to_edge(DOWN, buff=0.4).scale_to_fit_width(
                0.9 * config.frame_width
            )
            self.play(FadeIn(options_vg), run_time=self.TRANSITION)
            self.wait(self.TEST_STAY)
            if none_answer:
                self.answer = "e"
            else:
                self.answer = labels[options.index(test_output)]
        self.question_text = title.replace("\n", " ")


if __name__ == "__main__":
    N_EXAMPLES = 20
    p_type = "count" # {mc, count}
    path_name = f"arcagi2_{p_type}"

    os.makedirs(f"media/videos/1080p60/{path_name}/questions", exist_ok=True)
    os.makedirs(f"media/videos/1080p60/{path_name}/solutions", exist_ok=True)
    os.makedirs(f"media/videos/1080p60/{path_name}/question_text", exist_ok=True)

    folder = Path("arcagi2")
    random.seed(1)
    paths = random.sample(list(folder.iterdir()), N_EXAMPLES)
    for path in paths:
        config.output_file = f"{path_name}/questions/{path.stem}"
        scene = ARCScene(path, p_type)
        scene.render()
        with open(
            f"media/videos/1080p60/{path_name}/solutions/{path.stem}.txt", "w"
        ) as f:
            f.write(str(scene.answer))
        with open(
            f"media/videos/1080p60/{path_name}/question_text/{path.stem}.txt", "w"
        ) as f:
            f.write(scene.question_text)
