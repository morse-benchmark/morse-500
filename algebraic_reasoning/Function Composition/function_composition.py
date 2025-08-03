from manim import *
import numpy as np
import random
from pathlib import Path
import shutil
import os


Path("questions").mkdir(exist_ok=True)
Path("solutions").mkdir(exist_ok=True)
Path("question_text").mkdir(exist_ok=True)

config.media_dir = "manim_output"
config.verbosity = "WARNING"
config.pixel_height = 1080
config.pixel_width = 1920
config.frame_rate = 60


class FunctionComposition(Scene):
    def __init__(self, file_index, **kwargs):
        super().__init__(**kwargs)
        self.file_index = file_index
        self.f, self.g, self.f_str, self.g_str = self.generate_functions()
        self.correct_func = lambda x: self.f(self.g(x))
        self.y_min, self.y_max = -5, 5
        self.out_bounds = random.randint(self.y_min, self.y_max)

    def generate_functions(self):

        types = ["linear", "quadratic", "trig", "exp"]
        f_type, g_type = random.sample(types, 2)

        if f_type == "linear":
            a = random.choice([-2, -1, 1, 2])
            b = random.choice([-2, -1, 0, 1, 2])
            f = lambda x: a * x + b
            f_str = f"{a}x + {b}" if b >= 0 else f"{a}x - {abs(b)}"

        elif f_type == "quadratic":
            a = random.choice([-1, 1])
            b = random.choice([-2, -1, 0, 1, 2])
            c = random.choice([-2, -1, 0, 1, 2])
            f = lambda x: a * x**2 + b * x + c
            f_str = (
                f"{a}x² + {b}x + {c}"
                if b >= 0 and c >= 0
                else (
                    f"{a}x² + {b}x - {abs(c)}"
                    if b >= 0
                    else (
                        f"{a}x² - {abs(b)}x + {c}"
                        if c >= 0
                        else f"{a}x² - {abs(b)}x - {abs(c)}"
                    )
                )
            )

        elif f_type == "trig":
            a = random.choice([1, 2])
            b = random.choice([1, 2])
            c = random.choice([0, 1])
            f = lambda x: a * np.sin(b * x) + c
            f_str = f"{a}sin({b}x) + {c}" if c != 0 else f"{a}sin({b}x)"

        else:
            a = random.choice([1, 2])
            b = random.choice([-1, 1])
            c = random.choice([0, 1])
            f = lambda x: a * np.exp(b * x) + c
            f_str = f"{a}e^{b}x + {c}" if c != 0 else f"{a}e^{b}x"

        if g_type == "linear":
            a = random.choice([-2, -1, 1, 2])
            b = random.choice([-2, -1, 0, 1, 2])
            g = lambda x: a * x + b
            g_str = f"{a}x + {b}" if b >= 0 else f"{a}x - {abs(b)}"

        elif g_type == "quadratic":
            a = random.choice([-1, 1])
            b = random.choice([-2, -1, 0, 1, 2])
            c = random.choice([-1, 0, 1])
            g = lambda x: a * x**2 + b * x + c
            g_str = (
                f"{a}x² + {b}x + {c}"
                if b >= 0 and c >= 0
                else (
                    f"{a}x² + {b}x - {abs(c)}"
                    if b >= 0
                    else (
                        f"{a}x² - {abs(b)}x + {c}"
                        if c >= 0
                        else f"{a}x² - {abs(b)}x - {abs(c)}"
                    )
                )
            )

        elif g_type == "trig":
            a = random.choice([1, 2])
            b = random.choice([1, 2])
            g = lambda x: a * np.cos(b * x)
            g_str = f"{a}cos({b}x)"

        else:
            a = random.choice([1, 2])
            b = random.choice([-1, 1])
            g = lambda x: a * np.exp(-b * x**2)
            g_str = f"{a}e^(-{b}x^2)"

        return f, g, f_str, g_str

    def bounded_plot(self, func, color, label, position, x_range=(-3, 3)):

        axes = Axes(
            x_range=[x_range[0], x_range[1], 1],
            y_range=[self.y_min, self.y_max, 1],
            x_length=4,
            y_length=3,
            axis_config={"color": WHITE},
        ).move_to(position)

        def bounded_func(x):
            y = func(x)
            return y if self.y_min <= y <= self.y_max else self.out_bounds

        graph = axes.plot(bounded_func, color=color, use_smoothing=True)
        label = Text(label, color=color, font_size=24).next_to(axes, UP, buff=0.1)
        return VGroup(axes, graph, label)

    def construct(self):

        title = Text("Function Composition Challenge", font_size=36).to_edge(UP)
        self.play(Write(title))

        f_graph = self.bounded_plot(self.f, BLUE, "f(x)", LEFT * 3 + UP * 1)
        g_graph = self.bounded_plot(self.g, RED, "g(x)", RIGHT * 3 + UP * 1)

        self.play(Create(f_graph), Create(g_graph), run_time=2)
        self.wait(1.5)

        self.play(*[FadeOut(mob) for mob in self.mobjects])

        options = []
        letters = ["A", "B", "C", "D"]

        correct_graph = lambda x: self.f(self.g(x))

        options.append(lambda x: self.g(self.f(x)))
        options.append(lambda x: self.f(x) + self.g(x))
        options.append(lambda x: self.f(x) * self.g(x))

        correct_index = random.randint(0, 3)
        options.insert(correct_index, correct_graph)

        question = Text("Which graph shows f(g(x))?", font_size=36).to_edge(UP)
        second_part = Text(
            "Output just the letter of the correct answer.", font_size=24
        ).next_to(question, DOWN)

        self.play(Write(question))
        self.play(FadeIn(second_part))

        answer_graphs = VGroup()
        positions = [
            UP * 0.75 + LEFT * 3.5,
            UP * 0.75 + RIGHT * 3.5,
            DOWN * 2.75 + LEFT * 3.5,
            DOWN * 2.75 + RIGHT * 3.5,
        ]

        for i, (func, pos) in enumerate(zip(options, positions)):
            graph = self.bounded_plot(func, GREEN, letters[i], pos)
            answer_graphs.add(graph)

        self.play(
            LaggedStart(*[Create(g) for g in answer_graphs], lag_ratio=0.2), run_time=3
        )
        self.wait(3)

        with open(f"solutions/function_composition_{self.file_index}.txt", "w") as f:
            f.write(letters[correct_index])
        with open(
            f"question_text/function_composition_{self.file_index}.txt", "w"
        ) as f:
            f.write(
                "Which graph shows f(g(x))? Output just the letter of the correct answer."
            )


for i in range(3):
    if os.path.exists("manim_output"):
        shutil.rmtree("manim_output")

    scene = FunctionComposition(file_index=i)
    scene.render()

    output = Path("manim_output/videos/1080p60/FunctionComposition.mp4")
    if output.exists():
        shutil.move(str(output), f"questions/function_composition_{i}.mp4")

if os.path.exists("manim_output"):
    shutil.rmtree("manim_output")
