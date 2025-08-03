from manim import *
import numpy as np
import random
from pathlib import Path
import shutil
import os
from scipy.integrate import quad


Path("questions").mkdir(exist_ok=True)
Path("solutions").mkdir(exist_ok=True)
Path("question_text").mkdir(exist_ok=True)

config.media_dir = "manim_output"
config.pixel_height = 1080
config.pixel_width = 1920
config.frame_rate = 60
config.verbosity = "WARNING"


class HardIntegral(Scene):
    def __init__(self, f, f_tex, a, b, integral_val, file_index, **kwargs):
        super().__init__(**kwargs)
        self.f = f
        self.f_tex = f_tex
        self.a = a
        self.b = b
        self.integral_val = integral_val
        self.file_index = file_index

    def construct(self):
        axes = Axes(
            x_range=[0, 10, 1],
            y_range=[-1, 6, 1],
            x_length=10,
            y_length=5,
        ).to_edge(DOWN)

        graph = axes.plot(self.f, color=BLUE)
        area = axes.get_area(graph, x_range=[self.a, self.b], color=BLUE, opacity=0.5)

        function_label = Text(self.f_tex, font_size=50).to_edge(UP)

        self.play(Create(axes), Write(function_label))
        self.play(Create(graph), run_time=2)
        self.wait(0.5)
        self.play(FadeIn(area))
        self.wait(1)
        self.play(FadeOut(area), FadeOut(graph), FadeOut(function_label), FadeOut(axes))

        question = Text(
            "What is the definite integral of the shown function\nbetween the two endpoints?",
            font_size=30,
        ).to_edge(UP)
        second_part = Text(
            "Output the answer rounded to 2 decimal places.", font_size=30
        ).next_to(question, DOWN)

        self.play(FadeIn(question))
        self.wait(2)
        self.play(FadeIn(second_part))
        self.wait(2)
        self.play(FadeOut(question), FadeOut(second_part))

        with open(f"solutions/definite_integral_{self.file_index}.txt", "w") as f_out:
            f_out.write(f"{round(self.integral_val, 2)}")
        with open(
            f"question_text/definite_integral_{self.file_index}.txt", "w"
        ) as f_out:
            f_out.write(
                f"What is the definite integral of the shown function between the two endpoints? Output the answer rounded to 2 decimal places."
            )


for i in range(3):
    funcs = [
        (lambda x: x**2 * np.sin(x), "f(x) = x^2 * sin(x)"),
        (lambda x: np.tanh(1.2 * x) * np.cos(x), "f(x) = tanh(1.2 * x) * cos(x)"),
        (lambda x: np.log(x + 1) * x, "f(x) = log(x+1) * x"),
    ]
    f, f_tex = funcs[i]

    a = random.randint(1, 5)
    b = random.randint(a + 1, 8)

    val, _ = quad(f, a, b)
    val = round(val, 2)

    if os.path.exists("manim_output"):
        shutil.rmtree("manim_output")

    scene = HardIntegral(
        f=f,
        f_tex=f_tex,
        a=a,
        b=b,
        integral_val=val,
        file_index=i,
    )
    scene.render()

    output = Path("manim_output/videos/1080p60/HardIntegral.mp4")
    if output.exists():
        shutil.move(str(output), f"questions/definite_integral_{i}.mp4")

    if os.path.exists("manim_output"):
        shutil.rmtree("manim_output")
