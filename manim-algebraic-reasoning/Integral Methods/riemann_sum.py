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


class LeftRiemannSum(Scene):
    def __init__(self, f, f_tex, a, b, n, integral_val, file_index, **kwargs):
        super().__init__(**kwargs)
        self.f = f
        self.f_tex = f_tex
        self.a = a
        self.b = b
        self.n = n
        self.integral_val = integral_val
        self.file_index = file_index

    def construct(self):

        axes = Axes(
            x_range=[self.a - 1, self.b + 1, 1],
            y_range=[0, max(self.f(self.a), self.f(self.b)) + 2, 1],
            x_length=10,
            y_length=6,
            axis_config={"color": WHITE},
        ).to_edge(DOWN)

        x_label = Text("x").next_to(axes.x_axis.get_end(), RIGHT)
        y_label = Text("y").next_to(axes.y_axis.get_end(), UP)

        graph = axes.plot(self.f, color=BLUE)

        dx = (self.b - self.a) / self.n
        x_values = [self.a + i * dx for i in range(self.n)]
        y_values = [self.f(x) for x in x_values]

        rectangles = VGroup()
        for i in range(self.n):

            rect = Rectangle(
                width=axes.x_axis.unit_size * dx,
                height=axes.y_axis.unit_size * y_values[i],
                fill_color=BLUE,
                fill_opacity=0.5,
                stroke_color=WHITE,
            )

            rect.move_to(axes.c2p(x_values[i] + dx / 2, 0), aligned_edge=DOWN)
            rectangles.add(rect)

        points = VGroup()
        point_labels = VGroup()
        for i in range(self.n):
            point = Dot(color=ORANGE, radius=0.06).move_to(
                axes.c2p(x_values[i], y_values[i])
            )
            points.add(point)

            label = Text(
                f"({x_values[i]:.2f}, {y_values[i]:.2f})", font_size=20
            ).next_to(point, UP, buff=0.1)
            point_labels.add(label)

        self.play(Create(axes), Write(x_label), Write(y_label))
        self.play(Create(graph), run_time=2)
        self.wait(1)

        for i in range(self.n):
            self.play(Create(points[i]), run_time=0.3)
            self.play(Write(point_labels[i]), run_time=0.3)
            self.play(Create(rectangles[i]), run_time=0.7)
            self.wait(0.2)

        self.play(
            FadeOut(rectangles),
            FadeOut(points),
            FadeOut(graph),
            FadeOut(axes),
            FadeOut(x_label),
            FadeOut(y_label),
            FadeOut(point_labels),
        )

        question = Text(
            "What is the definite integral using the Left Riemann Sum\nbetween two endpoints?",
            font_size=30,
        ).to_edge(UP)
        second_part = Text(
            "Output the answer rounded to 2 decimal places.", font_size=30
        ).next_to(question, DOWN)
        self.play(FadeIn(question))
        self.wait(0.5)
        self.play(FadeIn(second_part))
        self.wait(0.5)
        self.play(FadeOut(question), FadeOut(second_part))

        with open(f"solutions/riemann_sum_{self.file_index}.txt", "w") as f_out:
            f_out.write(f"{round(self.integral_val, 2)}")
        with open(f"question_text/riemann_sum_{self.file_index}.txt", "w") as f_out:
            f_out.write(
                f"What is the definite integral using the Left Riemann Sum between two endpoints? Output the answer rounded to 2 decimal places."
            )


for i in range(3):
    funcs = [
        (lambda x: np.log(x + 2) * x, "log(x+2) * x"),
        (lambda x: np.exp(x), "e^x"),
        (lambda x: np.exp(x), "tanh(x+4)"),
    ]
    f, f_tex = funcs[i]

    a = random.randint(1, 3)
    b = random.randint(a + 1, 5)
    n = random.randint(4, 6)

    val, _ = quad(f, a, b)
    val = round(val, 2)

    if os.path.exists("manim_output"):
        shutil.rmtree("manim_output")

    scene = LeftRiemannSum(
        f=f,
        f_tex=f_tex,
        a=a,
        b=b,
        n=n,
        integral_val=val,
        file_index=i,
    )
    scene.render()

    output = Path("manim_output/videos/1080p60/LeftRiemannSum.mp4")
    if output.exists():
        shutil.move(str(output), f"questions/riemann_sum_{i}.mp4")

    if os.path.exists("manim_output"):
        shutil.rmtree("manim_output")
