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


class MidpointRiemannSum(Scene):
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
        midpoints = [
            (
                (x_values[i] + x_values[i + 1]) / 2
                if i + 1 < self.n
                else x_values[i] + dx / 2
            )
            for i in range(self.n)
        ]
        y_values = [self.f(x) for x in midpoints]

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
                axes.c2p(midpoints[i], y_values[i])
            )
            points.add(point)

            label = Text(
                f"({midpoints[i]:.2f}, {y_values[i]:.2f})", font_size=20
            ).next_to(point, UP, buff=0.1)
            point_labels.add(label)

        title = Text("Midpoint Riemann Sum", font_size=30).to_edge(UP)
        function_label = Text(f"Function: {self.f_tex}", font_size=24).next_to(
            title, DOWN
        )
        bounds_label = Text(
            f"Interval: {self.a} to {self.b} with {self.n} rectangles", font_size=24
        ).next_to(function_label, DOWN)

        self.play(Create(axes), Write(x_label), Write(y_label))
        self.play(Write(title), Write(function_label), Write(bounds_label))
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
            FadeOut(title),
            FadeOut(function_label),
            FadeOut(bounds_label),
        )

        question = Text(
            "What is the definite integral using the Midpoint Riemann Sum\nbetween two endpoints?",
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

        with open(f"solutions/midpoint_sum_{self.file_index}.txt", "w") as f_out:
            f_out.write(f"{round(self.integral_val, 2)}")
        with open(f"question_text/midpoint_sum_{self.file_index}.txt", "w") as f_out:
            f_out.write(
                f"What is the definite integral using the Midpoint Riemann Sum between two endpoints? Output the answer rounded to 2 decimal places."
            )


funcs = [
    (lambda x: 0.5 * x**3 - 2 * x**2 + 3 * x + 1, "0.5x³ - 2x² + 3x + 1"),
    (lambda x: 2 * np.sin(x) + 3, "2sin(x) + 3"),
    (lambda x: np.log(x + 1) * np.exp(-x / 3), "ln(x+1)·e^(-x/3)"),
]

for i in range(3):
    f, f_tex = funcs[i]

    a = random.randint(1, 3)
    b = random.randint(a + 1, 5)
    n = random.randint(4, 6)

    val, _ = quad(f, a, b)
    val = round(val, 2)

    if os.path.exists("manim_output"):
        shutil.rmtree("manim_output")

    scene = MidpointRiemannSum(
        f=f,
        f_tex=f_tex,
        a=a,
        b=b,
        n=n,
        integral_val=val,
        file_index=i,
    )
    scene.render()

    output = Path("manim_output/videos/1080p60/MidpointRiemannSum.mp4")
    if output.exists():
        shutil.move(str(output), f"questions/midpoint_sum_{i}.mp4")

    if os.path.exists("manim_output"):
        shutil.rmtree("manim_output")
