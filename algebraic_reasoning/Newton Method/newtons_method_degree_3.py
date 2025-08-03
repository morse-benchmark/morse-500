from manim import *
import random
import numpy as np
from sympy import symbols, diff, lambdify
from pathlib import Path
import shutil
import os


Path("questions").mkdir(exist_ok=True)
Path("solutions").mkdir(exist_ok=True)
Path("question_text").mkdir(exist_ok=True)


class NewtonsMethodVisual(Scene):
    def __init__(self, degree, steps, x0, idx, **kwargs):
        super().__init__(**kwargs)
        self.degree = degree
        self.steps = steps
        self.x0 = x0
        self.idx = idx

    def construct(self):
        coeffs, poly_str, solution = self.generate_newton_problem(
            self.degree, self.steps, self.x0, self.idx
        )
        self.visualize_function(self.degree, coeffs, self.x0)
        self.display_question(self.degree, self.steps)

    def generate_random_polynomial(self, degree):
        coeffs = [random.randint(-3, 3) for _ in range(degree + 1)]
        while coeffs[-1] == 0:
            coeffs[-1] = random.randint(-3, 3)
        return coeffs

    def poly_to_str(self, coeffs):
        terms = []
        for power, coeff in enumerate(reversed(coeffs)):
            if coeff == 0:
                continue
            term = ""
            if coeff != 1 or power == 0:
                term += f"{coeff}"
            if power > 0:
                term += "x"
                if power > 1:
                    term += f"^{power}"
            terms.append(term)
        return " + ".join(terms).replace("+ -", " - ")

    def generate_newton_problem(self, degree, steps, x0, idx):
        coeffs = self.generate_random_polynomial(degree)
        poly_str = self.poly_to_str(coeffs)

        x = symbols("x")
        poly_expr = sum(c * x**i for i, c in enumerate(coeffs))
        deriv_expr = diff(poly_expr, x)

        f = lambdify(x, poly_expr)
        f_prime = lambdify(x, deriv_expr)

        current_x = x0
        for _ in range(steps):
            if abs(f_prime(current_x)) < 1e-6:
                current_x += 0.1
            current_x = current_x - f(current_x) / f_prime(current_x)

        solution = round(current_x, 2)
        with open(f"solutions/newtons_method_degree_3_{idx}.txt", "w") as f:
            f.write(f"{solution:.2f}")
        with open(f"question_text/newtons_method_degree_3_{idx}.txt", "w") as f:
            f.write(
                f"What is the approximation after {steps} steps of Newton's method of this degree {degree} polynomial? (rounded to 2 decimal places, start at the red dot)"
            )
        return coeffs, poly_str, solution

    def visualize_function(self, degree, coeffs, x0):
        x = symbols("x")
        poly_expr = sum(c * x**i for i, c in enumerate(coeffs))
        f = lambdify(x, poly_expr)

        axes = Axes(
            x_range=[-5, 5, 1],
            y_range=[
                np.min([f(x) for x in np.arange(-5, 5, 0.1)]) - 1,
                np.max([f(x) for x in np.arange(-5, 5, 0.1)]) + 1,
                2,
            ],
            axis_config={"color": BLUE},
            x_length=10,
            y_length=6,
        )

        try:
            graph = axes.plot(lambda x_val: f(x_val), color=GREEN)
        except:
            graph = axes.plot(lambda x_val: x_val**3 - 2 * x_val + 1, color=GREEN)

        x_vals = []
        while len(x_vals) < degree + 1:
            new_x = round(random.uniform(-5, 5), 2)
            if all(abs(new_x - x) > 0.2 for x in x_vals):
                x_vals.append(new_x)

        points = []
        labels = []
        offset_dirs = [UP, DOWN, LEFT, RIGHT, UR, UL, DR, DL]

        for i, x_val in enumerate(x_vals):
            y_val = f(x_val)
            point = Dot(axes.c2p(x_val, y_val), color=YELLOW)
            label = Text(f"({x_val:.2f}, {y_val:.2f})", font_size=14)
            direction = offset_dirs[i % len(offset_dirs)]
            label.next_to(point, direction, buff=0.2)

            jitter_x = 0.1 * random.uniform(-1, 1)
            jitter_y = 0.1 * random.uniform(-1, 1)
            label.shift(jitter_x * RIGHT + jitter_y * UP)

            points.append(point)
            labels.append(label)

        start_dot = Dot(axes.c2p(x0, 0), color=RED)
        start_label = Text(f"x₀ = {x0}", font_size=20, color=RED)
        start_label.next_to(start_dot, UP, buff=0.2)

        self.play(Create(axes), Create(graph))
        self.play(FadeIn(start_dot), Write(start_label))
        for p, l in zip(points, labels):
            self.play(FadeIn(p), FadeIn(l), run_time=0.3)
        self.wait(0.5)
        self.play(
            FadeOut(axes),
            FadeOut(graph),
            FadeOut(start_dot),
            FadeOut(start_label),
            *[FadeOut(p) for p in points],
            *[FadeOut(l) for l in labels],
        )

    def display_question(self, degree, steps):
        question = (
            VGroup(
                Text(
                    f"What is the approximation after {steps} steps of Newton's method of this degree {degree} polynomial?",
                    font_size=18,
                ),
                Text(
                    "(rounded to 2 decimal places, start at the red dot)", font_size=18
                ),
            )
            .arrange(DOWN, aligned_edge=LEFT, buff=0.4)
            .to_edge(UP)
        )
        self.play(Write(question))
        self.wait(0.5)
        self.play(FadeOut(question))


if os.path.exists("manim_output"):
    shutil.rmtree("manim_output")

config.media_dir = "manim_output"
config.verbosity = "WARNING"
config.pixel_height = 1080
config.pixel_width = 1920
config.frame_rate = 60


for idx in range(3):
    degree = 3
    steps = random.randint(2, 7)
    x0 = round(random.uniform(-5, 5), 0)

    scene = NewtonsMethodVisual(degree=degree, steps=steps, x0=x0, idx=idx)
    scene.render()

    output_file = Path("manim_output/videos/1080p60/NewtonsMethodVisual.mp4")
    if output_file.exists():
        shutil.move(str(output_file), f"questions/newtons_method_degree_3_{idx}.mp4")

    if os.path.exists("manim_output"):
        shutil.rmtree("manim_output")
