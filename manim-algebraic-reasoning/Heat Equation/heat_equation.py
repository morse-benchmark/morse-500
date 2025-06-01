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


class HeatEquationQuiz(Scene):
    def __init__(self, file_index, **kwargs):
        super().__init__(**kwargs)
        self.file_index = file_index
        self.k = random.randint(1, 10)
        self.correct_index = 0
        self.num_points = 100
        self.x_points = np.linspace(0, 10, self.num_points)
        self.time_points = np.linspace(0, 5, 50)

        self.fourier_coeffs = self.generate_random_coefficients()

    def generate_random_coefficients(self):

        n_terms = random.randint(2, 5)
        return [random.uniform(0.5, 2.0) for _ in range(n_terms)]

    def initial_condition(self, x):

        result = 0
        for n, coeff in enumerate(self.fourier_coeffs, start=1):
            result += coeff * np.sin(n * np.pi * x / 10)

        result = 50 * (result / max(1, np.max(np.abs(result)))) + 50
        return np.clip(result, 0, 100)

    def heat_solution(self, x, t):

        solution = 0
        for n, coeff in enumerate(self.fourier_coeffs, start=1):
            solution += (
                coeff
                * np.sin(n * np.pi * x / 10)
                * np.exp(-self.k * (n * np.pi / 10) ** 2 * t)
            )

        solution = 50 * (solution / max(1, np.max(np.abs(solution)))) + 50
        return np.clip(solution, 0, 100)

    def add_manual_ticks(self, axes):

        x_ticks = VGroup()
        for x in range(0, 11):
            tick = Line(
                start=axes.c2p(x, 0) - DOWN * 0.2,
                end=axes.c2p(x, 0) + DOWN * 0.2,
                color=WHITE,
            )
            number = Text(str(x), font_size=24).next_to(tick, DOWN, buff=0.1)
            x_ticks.add(tick, number)

        y_ticks = VGroup()
        for y in range(0, 101, 10):
            tick = Line(
                start=axes.c2p(0, y) - LEFT * 0.2,
                end=axes.c2p(0, y) + LEFT * 0.2,
                color=WHITE,
            )
            number = Text(str(y), font_size=24).next_to(tick, LEFT, buff=0.1)
            y_ticks.add(tick, number)

        return VGroup(x_ticks, y_ticks)

    def construct(self):

        axes = Axes(
            x_range=[0, 10],
            y_range=[0, 100],
            x_length=10,
            y_length=6,
            axis_config={"color": WHITE},
        ).shift(DOWN * 0.5)

        ticks = self.add_manual_ticks(axes)

        title = Text("1D Heat Equation Simulation", font_size=36).to_edge(UP)
        x_label = Text("Position (cm)", font_size=24).next_to(
            axes.x_axis, DOWN, buff=1.5
        )
        y_label = Text("(°C)", font_size=24).next_to(axes.y_axis, LEFT, buff=0.5)

        time_text = Text("t = 0.00 s", font_size=28).to_edge(UP).shift(DOWN * 0.5)
        time_text.set_color(YELLOW)

        initial_points = [axes.c2p(x, self.initial_condition(x)) for x in self.x_points]
        initial_temp = VMobject().set_points_as_corners(initial_points).set_color(BLUE)

        graph = VMobject().set_color(RED)
        graph.set_points_as_corners(initial_points)

        def update_graph(graph, dt):
            t = self.time_elapsed
            new_points = [axes.c2p(x, self.heat_solution(x, t)) for x in self.x_points]
            graph.set_points_as_corners(new_points)

            time_text.become(
                Text(f"t = {t:.2f} s", font_size=28).to_edge(UP).shift(DOWN * 0.5)
            )
            time_text.set_color(YELLOW)

        graph.add_updater(update_graph)
        self.time_elapsed = 0

        self.play(Write(title))
        self.play(Create(axes), Create(ticks), Write(x_label), Write(y_label))
        self.add(time_text)
        self.play(Create(initial_temp))
        self.wait(0.5)

        self.play(ReplacementTransform(initial_temp, graph))

        for t in np.linspace(0, 5, 50):
            self.time_elapsed = t
            self.wait(0.1)

        graph.remove_updater(update_graph)

        self.play(*[FadeOut(mob) for mob in self.mobjects])

        question = Text(
            "What is the thermal diffusivity constant k?\nChoose the letter of the correct answer:",
            font_size=30,
            line_spacing=1.5,
        ).to_edge(UP)

        options = sorted(
            {self.k, max(1, self.k - 1), min(10, self.k + 1), random.randint(1, 10)}
        )
        random.shuffle(options)
        self.correct_index = options.index(self.k)

        option_text = VGroup()
        for i, opt in enumerate(options):
            option = Text(
                f"{chr(65+i)}. k = {opt}", font_size=28, t2c={f"{opt}": YELLOW}
            )
            option_text.add(option)

        option_text.arrange(DOWN, aligned_edge=LEFT, buff=0.5).next_to(
            question, DOWN, buff=1
        )

        self.play(Write(question))
        self.play(LaggedStart(*[Write(opt) for opt in option_text], lag_ratio=0.2))
        self.wait(3)

        with open(f"solutions/heat_equation_{self.file_index}.txt", "w") as f:
            f.write(chr(65 + self.correct_index))
        with open(f"question_text/heat_equation_{self.file_index}.txt", "w") as f:
            f.write(
                "What is the thermal diffusivity constant k? Choose the letter of the correct answer:"
            )


for i in range(3):
    if os.path.exists("manim_output"):
        shutil.rmtree("manim_output")

    scene = HeatEquationQuiz(file_index=i)
    scene.render()

    output = Path("manim_output/videos/1080p60/HeatEquationQuiz.mp4")
    if output.exists():
        shutil.move(str(output), f"questions/heat_equation_{i}.mp4")

if os.path.exists("manim_output"):
    shutil.rmtree("manim_output")
