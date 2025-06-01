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


class FluxCalculation(Scene):
    def __init__(self, file_index, **kwargs):
        super().__init__(**kwargs)
        self.file_index = file_index
        self.field, self.equation_text, self.flux = self.generate_problem()

    def generate_problem(self):

        problem_type = random.choice(
            ["polynomial_2d", "trigonometric", "mixed_exponential"]
        )

        R = random.uniform(1.0, 2.0)

        if problem_type == "polynomial_2d":
            a, b = random.uniform(0.5, 2.0), random.uniform(0.5, 2.0)
            field_func = lambda p: np.array(
                [a * p[0] ** 2 + b * p[1], a * p[0] - b * p[1] ** 2, 0]
            )
            equation_text = Text(
                f"F(x,y) = ({a:.1f}x² + {b:.1f}y)i + ({a:.1f}x - {b:.1f}y²)j",
                font_size=30,
            )
            flux = np.pi * R**2 * (2 * a - 2 * b)

        elif problem_type == "trigonometric":
            k = random.uniform(0.5, 2.0)
            field_func = lambda p: np.array([np.sin(k * p[1]), np.cos(k * p[0]), 0])
            equation_text = Text(
                f"F(x,y) = sin({k:.1f}y)i + cos({k:.1f}x)j", font_size=30
            )
            flux = 0

        else:
            a = random.uniform(0.5, 1.5)
            field_func = lambda p: np.array(
                [np.exp(a * p[0]) - p[1] ** 3, np.exp(-a * p[1]) + p[0] ** 3, 0]
            )
            equation_text = Text(
                f"F(x,y) = (e^({a:.1f}x) - y³)i + (e^(-{a:.1f}y) + x³)j", font_size=30
            )
            flux = np.pi * R**2 * (a * np.exp(a * R) + a * np.exp(-a * R))

        return field_func, equation_text, round(flux, 2)

    def create_vector_field(self, field_func):

        grid = NumberPlane(x_range=[-4, 4], y_range=[-4, 4])

        def get_color(magnitude):
            normalized = np.tanh(magnitude / 2.0)
            return interpolate_color(BLUE_E, RED_E, normalized)

        arrows = VGroup()
        for x in np.arange(-3.5, 3.5, 0.4):
            for y in np.arange(-3.5, 3.5, 0.4):
                point = np.array([x, y, 0])
                vec = field_func(point)
                magnitude = np.linalg.norm(vec[:2])
                direction = vec / magnitude if magnitude > 0 else vec

                arrow = Arrow(
                    start=point,
                    end=point + direction * 0.5,
                    buff=0,
                    color=get_color(magnitude),
                    stroke_width=2,
                    max_tip_length_to_length_ratio=0.2,
                )
                arrows.add(arrow)

        magnitudes = [0, 1, 2, 4]
        legend = VGroup()
        for i, mag in enumerate(magnitudes):
            dot = Dot(color=get_color(mag)).shift(RIGHT * i + DOWN * 3)
            label = Text(f"{mag:.1f}", font_size=20).next_to(dot, DOWN)
            legend.add(dot, label)

        return VGroup(grid, arrows, legend)

    def construct(self):
        field_func, equation_text, flux = self.field, self.equation_text, self.flux

        title = Text("Flux Calculation", font_size=36).to_edge(UP)

        equation_text.next_to(title, DOWN)

        field_display = self.create_vector_field(field_func)
        surface = Circle(radius=1.5, color=BLUE, fill_opacity=0.2)

        self.play(Write(title))
        self.play(Write(equation_text))
        self.play(Create(field_display[0]))
        self.play(Create(field_display[1]), run_time=2)
        self.play(FadeIn(field_display[2]))
        self.play(Create(surface))
        self.wait(3)

        self.play(
            FadeOut(title),
            FadeOut(equation_text),
            FadeOut(field_display),
            FadeOut(surface),
            run_time=1.5,
        )

        self.wait(0.5)

        question = (
            VGroup(
                Text("What is the flux through the surface?", font_size=36),
                Text(
                    "Output only the number, round to 2 decimal places",
                    font_size=28,
                    color=YELLOW,
                ),
            )
            .arrange(DOWN, buff=0.5)
            .shift(UP * 0.5)
        )

        self.play(Write(question))
        self.wait(3)

        with open(f"solutions/flux_{self.file_index}.txt", "w") as f:
            f.write(f"{flux:.2f}")
        with open(f"question_text/flux_{self.file_index}.txt", "w") as f:
            f.write(
                "What is the flux through the surface? Output only the number, round to 2 decimal places"
            )


for i in range(3):
    if os.path.exists("manim_output"):
        shutil.rmtree("manim_output")

    scene = FluxCalculation(file_index=i)
    scene.render()

    output = Path("manim_output/videos/1080p60/FluxCalculation.mp4")
    if output.exists():
        shutil.move(str(output), f"questions/flux_{i}.mp4")

    if os.path.exists("manim_output"):
        shutil.rmtree("manim_output")
