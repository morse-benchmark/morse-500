from manim import *
import numpy as np
import random
from pathlib import Path
import shutil
import os
from scipy.signal import fftconvolve


Path("questions").mkdir(exist_ok=True)
Path("solutions").mkdir(exist_ok=True)
Path("question_text").mkdir(exist_ok=True)

config.media_dir = "manim_output"
config.verbosity = "WARNING"
config.pixel_height = 1080
config.pixel_width = 1920
config.frame_rate = 60


class ConvolutionQuiz(Scene):
    def __init__(self, file_index, **kwargs):
        super().__init__(**kwargs)
        self.file_index = file_index
        self.functions = self.generate_functions()
        self.correct_convolution = self.calculate_convolution()

    def generate_functions(self):

        types = ["rect", "tri", "gauss", "exp"]
        f1_type, f2_type = random.sample(types, 2)

        def rect(x, center=0, width=1):
            return np.where((x >= center - width / 2) & (x <= center + width / 2), 1, 0)

        def tri(x, center=0, width=1):
            return np.maximum(0, 1 - 2 * np.abs(x - center) / width)

        def gauss(x, center=0, sigma=0.5):
            return np.exp(-(((x - center) / sigma) ** 2) / 2)

        def exp_decay(x, center=0):
            return np.where(x >= center, np.exp(-(x - center)), 0)

        x = np.linspace(-3, 3, 400)

        if f1_type == "rect":
            f1 = rect(x, center=random.uniform(-1, 0), width=random.uniform(0.8, 1.5))
        elif f1_type == "tri":
            f1 = tri(x, center=random.uniform(-1, 0), width=random.uniform(1, 2))
        elif f1_type == "gauss":
            f1 = gauss(x, center=random.uniform(-1, 0), sigma=random.uniform(0.3, 0.7))
        else:
            f1 = exp_decay(x, center=random.uniform(-1.5, -0.5))

        if f2_type == "rect":
            f2 = rect(x, center=random.uniform(0, 1), width=random.uniform(0.8, 1.5))
        elif f2_type == "tri":
            f2 = tri(x, center=random.uniform(0, 1), width=random.uniform(1, 2))
        elif f2_type == "gauss":
            f2 = gauss(x, center=random.uniform(0, 1), sigma=random.uniform(0.3, 0.7))
        else:
            f2 = exp_decay(x, center=random.uniform(0.5, 1.5))

        return x, f1, f2

    def calculate_convolution(self):

        x, f1, f2 = self.functions
        conv = fftconvolve(f1, f2, mode="same") * (x[1] - x[0])
        return x, conv

    def create_graph(self, axes, x, y, color, label):

        graph = axes.plot_line_graph(x, y, line_color=color, add_vertex_dots=False)
        label = Text(label, color=color, font_size=24).next_to(axes, UP, buff=0.1)
        return VGroup(axes, graph, label)

    def construct(self):
        x, f1, f2 = self.functions
        x_conv, conv = self.correct_convolution

        title = Text("Function Convolution", font_size=32).to_edge(UP)
        subtitle = Text("Original Functions", font_size=28).next_to(title, DOWN)

        f1_axes = (
            Axes(
                x_range=[-3, 3, 1],
                y_range=[0, 1.2, 0.2],
                x_length=6,
                y_length=3,
            )
            .next_to(subtitle, DOWN, buff=0.5)
            .shift(LEFT * 3)
        )

        f2_axes = (
            Axes(
                x_range=[-3, 3, 1],
                y_range=[0, 1.2, 0.2],
                x_length=6,
                y_length=3,
            )
            .next_to(subtitle, DOWN, buff=0.5)
            .shift(RIGHT * 3)
        )

        f1_graph = self.create_graph(f1_axes, x, f1, BLUE, r"f(t)")
        f2_graph = self.create_graph(f2_axes, x, f2, RED, r"g(t)")

        self.play(Write(title), Write(subtitle))
        self.play(Create(f1_graph), Create(f2_graph), run_time=2)
        self.wait(2)

        options = []
        for _ in range(3):
            if random.random() > 0.5:

                shift = random.uniform(-1, 1)
                wrong_conv = np.roll(conv, int(shift * len(conv) / 6))
            else:

                scale = random.uniform(0.7, 1.3)
                wrong_conv = conv * scale
            options.append(wrong_conv)

        correct_index = random.randint(0, len(options))
        options.insert(correct_index, conv)
        letters = ["A", "B", "C", "D"]

        self.play(
            FadeOut(title),
            FadeOut(subtitle),
            FadeOut(f1_graph),
            FadeOut(f2_graph),
        )

        question = (
            VGroup(
                Text("Which is the correct convolution f(t) * g(t)?", font_size=28),
                Text(
                    "Output just the letter of the correct answer:",
                    font_size=24,
                    color=YELLOW,
                ),
            )
            .arrange(DOWN, aligned_edge=LEFT)
            .to_edge(UP, buff=0.5)
        )

        option_graphs = VGroup()
        positions = [
            UP * 0.5 + LEFT * 3.5,
            UP * 0.5 + RIGHT * 3.5,
            DOWN * 2.75 + LEFT * 3.5,
            DOWN * 2.75 + RIGHT * 3.5,
        ]

        for i, (opt, pos) in enumerate(zip(options, positions)):
            ax = Axes(
                x_range=[-6, 6, 2],
                y_range=[0, round(np.max(options) * 1.2, 1), 0.5],
                x_length=5,
                y_length=2.5,
            ).move_to(pos)

            graph = ax.plot_line_graph(
                x_conv, opt, line_color=GREEN, add_vertex_dots=False
            )
            label = Text(letters[i], color=WHITE, font_size=36).next_to(
                ax, UP, buff=0.1
            )

            option_graphs.add(VGroup(ax, graph, label))

        self.play(Write(question))
        self.play(
            LaggedStart(*[Create(opt) for opt in option_graphs], lag_ratio=0.3),
            run_time=2,
        )
        self.wait(3)

        with open(f"solutions/convolution_{self.file_index}.txt", "w") as f:
            f.write(f"{letters[correct_index]}")
        with open(f"question_text/convolution_{self.file_index}.txt", "w") as f:
            f.write(
                f"Which is the correct convolution f(t) * g(t)? Output just the letter of the correct answer."
            )


for i in range(3):
    if os.path.exists("manim_output"):
        shutil.rmtree("manim_output")

    scene = ConvolutionQuiz(file_index=i)
    scene.render()

    output = Path("manim_output/videos/1080p60/ConvolutionQuiz.mp4")
    if output.exists():
        shutil.move(str(output), f"questions/convolution_{i}.mp4")

    if os.path.exists("manim_output"):
        shutil.rmtree("manim_output")
