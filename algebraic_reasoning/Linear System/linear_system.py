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
config.pixel_height = 1080
config.pixel_width = 1920
config.frame_rate = 60
config.verbosity = "WARNING"


class MatrixTransformationDemo(Scene):
    def __init__(self, file_index, **kwargs):
        super().__init__(**kwargs)
        self.file_index = file_index

        self.matrix = self.generate_random_matrix()
        self.matrix_str = self.matrix_to_str(self.matrix)

    def generate_random_matrix(self):

        while True:
            a, b, c, d = [random.choice(range(-5, 6)) for _ in range(4)]
            det = a * d - b * c
            if det != 0:
                return np.array([[a, b], [c, d]])

    def matrix_to_str(self, matrix):

        return f"[[{matrix[0][0]} {matrix[0][1]}], [{matrix[1][0]} {matrix[1][1]}]]"

    def construct(self):

        axes = Axes(
            x_range=[-4, 4, 1],
            y_range=[-4, 4, 1],
            x_length=7,
            y_length=7,
            axis_config={"color": BLUE, "stroke_width": 3},
        )

        title = Text("Matrix Transformation", font_size=36).to_edge(UP)
        self.play(Write(title))

        grid = VGroup()
        for x in np.arange(-4, 4.5, 0.5):
            line = axes.get_vertical_line(axes.c2p(x, 0), color=BLUE_E)
            if x.is_integer():
                line.set_stroke(width=2, opacity=0.8)
            else:
                line.set_stroke(width=1, opacity=0.4)
            grid.add(line)

        for y in np.arange(-4, 4.5, 0.5):
            line = axes.get_horizontal_line(axes.c2p(0, y), color=BLUE_E)
            if y.is_integer():
                line.set_stroke(width=2, opacity=0.8)
            else:
                line.set_stroke(width=1, opacity=0.4)
            grid.add(line)

        i_hat = Arrow(
            start=axes.c2p(0, 0),
            end=axes.c2p(1, 0),
            buff=0,
            color=GREEN,
            stroke_width=8,
            tip_length=0.25,
        )
        j_hat = Arrow(
            start=axes.c2p(0, 0),
            end=axes.c2p(0, 1),
            buff=0,
            color=YELLOW,
            stroke_width=8,
            tip_length=0.25,
        )
        i_label = Text("i", color=GREEN, font_size=28).next_to(
            i_hat.get_end(), RIGHT, buff=0.15
        )
        j_label = Text("j", color=YELLOW, font_size=28).next_to(
            j_hat.get_end(), UP, buff=0.15
        )

        self.play(Create(axes))
        self.play(Create(grid))
        self.play(GrowArrow(i_hat), GrowArrow(j_hat), Write(i_label), Write(j_label))
        self.wait(1)

        transformed_grid = VGroup()
        for line in grid:
            start, end = line.get_start()[:2], line.get_end()[:2]
            new_line = Line(
                axes.c2p(*(self.matrix @ start)),
                axes.c2p(*(self.matrix @ end)),
                color=RED_E,
                stroke_width=line.get_stroke_width(),
                stroke_opacity=line.get_stroke_opacity(),
            )
            transformed_grid.add(new_line)

        new_i = Arrow(
            start=axes.c2p(0, 0),
            end=axes.c2p(*self.matrix[:, 0]),
            buff=0,
            color=GREEN,
            stroke_width=8,
            tip_length=0.25,
        )
        new_j = Arrow(
            start=axes.c2p(0, 0),
            end=axes.c2p(*self.matrix[:, 1]),
            buff=0,
            color=YELLOW,
            stroke_width=8,
            tip_length=0.25,
        )

        new_i_label = Text("i", color=GREEN, font_size=28).next_to(
            new_i.get_end(), RIGHT, buff=0.15
        )
        new_j_label = Text("j", color=YELLOW, font_size=28).next_to(
            new_j.get_end(), UP, buff=0.15
        )

        self.play(
            Transform(grid, transformed_grid),
            Transform(i_hat, new_i),
            Transform(j_hat, new_j),
            Transform(i_label, new_i_label),
            Transform(j_label, new_j_label),
            run_time=2,
        )
        self.wait(2)

        self.play(*[FadeOut(mob) for mob in self.mobjects])

        question = Text(
            "What was the transformation matrix?\n"
            "Format as [[a b], [c d]] with integers.",
            font_size=30,
        ).to_edge(UP)

        self.play(Write(question))
        self.wait(3)

        with open(f"solutions/matrix_transform_{self.file_index}.txt", "w") as f:
            f.write(self.matrix_str)
        with open(f"question_text/matrix_transform_{self.file_index}.txt", "w") as f:
            f.write(
                "What was the transformation matrix? Format as [[a b], [c d]] with integers."
            )

        self.play(FadeOut(question))


for i in range(3):
    if os.path.exists("manim_output"):
        shutil.rmtree("manim_output")

    scene = MatrixTransformationDemo(file_index=i)
    scene.render()

    output = Path("manim_output/videos/1080p60/MatrixTransformationDemo.mp4")
    if output.exists():
        shutil.move(str(output), f"questions/matrix_transform_{i}.mp4")

    if os.path.exists("manim_output"):
        shutil.rmtree("manim_output")
