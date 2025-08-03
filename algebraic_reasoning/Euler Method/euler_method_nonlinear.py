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


class EulerNonlinearVisual(Scene):
    def __init__(self, f, step_size, num_steps, x0, y0, coeffs, file_index, **kwargs):
        super().__init__(**kwargs)
        self.f = f
        self.step_size = step_size
        self.num_steps = num_steps
        self.x0 = x0
        self.y0 = y0
        self.coeffs = coeffs
        self.file_index = file_index

    def construct(self):
        f = self.f
        step_size = self.step_size
        num_steps = self.num_steps
        x0, y0 = self.x0, self.y0
        a, b, c, d = self.coeffs

        title = Text("Euler's Method Problem", font_size=36).to_edge(UP)
        step_info = Text(
            f"Step size (h) = {step_size}, Number of steps = {num_steps}", font_size=24
        ).next_to(title, DOWN)

        self.play(Write(title))
        self.play(Write(step_info))
        self.wait(2)

        self.play(
            FadeOut(title),
            FadeOut(step_info),
        )

        axes = Axes(
            x_range=[-0.5, 2.5, 0.5],
            y_range=[0, 4, 0.5],
            x_length=10,
            y_length=5,
        ).to_edge(DOWN)

        field = VGroup()
        for x in np.arange(-0.5, 2.5, 0.3):
            for y in np.arange(0, 4, 0.3):
                slope = f(x, y)
                dx = 1 / np.sqrt(1 + slope**2)
                dy = slope / np.sqrt(1 + slope**2)
                norm_slope = min(abs(slope) / 3, 1.0)
                if slope > 0:
                    color = interpolate_color(GREEN, RED, norm_slope)
                else:
                    color = interpolate_color(GREEN, BLUE, norm_slope)
                vec = Line(
                    start=axes.c2p(x - 0.1, y - 0.1 * slope),
                    end=axes.c2p(x + 0.1, y + 0.1 * slope),
                    stroke_width=2,
                    color=color,
                )
                field.add(vec)

        points = [(x0, y0)]
        for _ in range(num_steps):
            x_curr, y_curr = points[-1]
            y_next = y_curr + step_size * f(x_curr, y_curr)
            x_next = x_curr + step_size
            points.append((x_next, y_next))

        euler_dots = VGroup()
        euler_lines = VGroup()
        dot_labels = VGroup()

        for i in range(len(points) - 2):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]
            dot = Dot(axes.c2p(x1, y1), color=RED)
            line = Line(axes.c2p(x1, y1), axes.c2p(x2, y2), color=YELLOW)

            label = Text(f"({x1:.1f}, {y1:.1f})", font_size=20).next_to(
                dot, UP, buff=0.1
            )

            euler_dots.add(dot)
            euler_lines.add(line)
            dot_labels.add(label)

        x_last, y_last = points[len(points) - 2]
        final_dot = Dot(axes.c2p(x_last, y_last), color=RED)
        final_label = Text(f"({x_last:.1f}, {y_last:.1f})", font_size=20).next_to(
            final_dot, UP, buff=0.1
        )
        euler_dots.add(final_dot)
        dot_labels.add(final_label)

        self.play(Create(axes), Create(field))
        self.wait(0.5)

        self.play(Create(euler_dots[0]), Write(dot_labels[0]))

        for i in range(1, len(euler_lines) + 1):
            self.play(
                Create(euler_lines[i - 1]),
                Create(euler_dots[i]),
                Write(dot_labels[i]),
                run_time=0.5,
            )

        final_y = round(points[-1][1], 2)

        self.play(
            FadeOut(field),
            FadeOut(euler_lines),
            FadeOut(euler_dots),
            FadeOut(dot_labels),
            FadeOut(axes),
            run_time=1,
        )
        self.wait(0.5)

        correct_answer = final_y
        distractors = list(
            {
                round(correct_answer + delta, 2)
                for delta in [-0.7, -0.5, 0.5, 0.7]
                if round(correct_answer + delta, 2) != correct_answer
            }
        )
        while len(distractors) < 3:
            distractors.append(round(correct_answer + random.uniform(-0.4, 0.4), 2))
        distractors = random.sample(distractors, 3)

        options = distractors + [correct_answer]
        random.shuffle(options)
        labels = ["A", "B", "C", "D"]

        question_text = VGroup(
            Text(f"What is the y-value after {num_steps} Euler steps?", font_size=28),
            Text(
                "(Choose the closest answer, give just the letter)",
                font_size=22,
            ),
        ).arrange(DOWN)

        mc_choices = (
            VGroup(
                *[
                    Text(f"{label}. {value:.2f}", font_size=28)
                    for label, value in zip(labels, options)
                ]
            )
            .arrange(DOWN, aligned_edge=LEFT, buff=0.3)
            .next_to(question_text, DOWN, buff=0.5)
        )

        full_question = VGroup(question_text, mc_choices).to_edge(UP)

        self.play(Write(full_question))
        self.wait(2)

        correct_index = options.index(correct_answer)
        with open(
            f"solutions/euler_method_nonlinear_{self.file_index}.txt", "w"
        ) as f_out:
            f_out.write(f"{labels[correct_index]}")
        with open(
            f"question_text/euler_method_nonlinear_{self.file_index}.txt", "w"
        ) as f_out:
            f_out.write(
                f"What is the y-value after {num_steps} Euler steps? (Choose the closest answer, give just the letter)"
            )


for i in range(3):

    while True:
        a = random.choice([-2, -1, 0, 1, 2])
        b = random.choice([-2, -1, 0, 1, 2])
        c = random.choice([-2, -1, 0, 1, 2])
        d = random.choice([-2, -1, 0, 1, 2])
        if any([a, b, c, d]):
            break

    def f(x, y, a=a, b=b, c=c, d=d):
        return a * x**2 + b * y**2 + c * np.sin(x) + d * np.cos(y)

    if os.path.exists("manim_output"):
        shutil.rmtree("manim_output")

    scene = EulerNonlinearVisual(
        f=f,
        step_size=0.2,
        num_steps=3,
        x0=0.0,
        y0=1.0,
        coeffs=(a, b, c, d),
        file_index=i,
    )
    scene.render()

    output = Path("manim_output/videos/1080p60/EulerNonlinearVisual.mp4")
    if output.exists():
        shutil.move(str(output), f"questions/euler_method_nonlinear_{i}.mp4")

    if os.path.exists("manim_output"):
        shutil.rmtree("manim_output")
