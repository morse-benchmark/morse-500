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


class CurlQuiz(Scene):
    def __init__(self, file_index, **kwargs):
        super().__init__(**kwargs)
        self.file_index = file_index
        self.field, self.equation_text, self.correct_curl = self.generate_problem()

    def generate_problem(self):

        problem_type = random.choice(
            ["rotational_field", "shear_field", "mixed_vortex", "nonlinear_rotation"]
        )

        x0, y0 = random.uniform(-1, 1), random.uniform(-1, 1)

        if problem_type == "rotational_field":
            a = random.uniform(0.5, 2.0)
            field_func = lambda p: np.array([-a * p[1], a * p[0], 0])
            equation_text = Text(f"F(x,y) = (-{a:.1f}y)i + ({a:.1f}x)j", font_size=30)
            correct_curl = 2 * a

        elif problem_type == "shear_field":
            b = random.uniform(0.5, 2.0)
            field_func = lambda p: np.array([b * p[1], 0, 0])
            equation_text = Text(f"F(x,y) = ({b:.1f}y)i", font_size=30)
            correct_curl = -b

        elif problem_type == "mixed_vortex":
            k = random.uniform(0.5, 2.0)
            field_func = lambda p: np.array(
                [
                    -k * p[1] / (p[0] ** 2 + p[1] ** 2 + 0.1),
                    k * p[0] / (p[0] ** 2 + p[1] ** 2 + 0.1),
                    0,
                ]
            )
            equation_text = Text(
                f"F(x,y) = (-{k:.1f}y/(x²+y²))i + ({k:.1f}x/(x²+y²))j", font_size=30
            )

            denom = (x0**2 + y0**2 + 0.1) ** 2
            correct_curl = 2 * k * (x0**2 + y0**2) / denom

        else:
            a, b = random.uniform(0.5, 1.5), random.uniform(0.5, 1.5)
            field_func = lambda p: np.array(
                [-a * p[1] + b * p[0] * p[1], a * p[0] - b * p[0] ** 2, 0]
            )
            equation_text = Text(
                f"F(x,y) = (-{a:.1f}y + {b:.1f}xy)i + ({a:.1f}x - {b:.1f}x²)j",
                font_size=30,
            )
            correct_curl = 2 * a - b * (x0 + y0)

        return field_func, equation_text, round(correct_curl, 2)

    def create_distractors(self, correct_curl):

        d1 = -correct_curl

        if abs(correct_curl) > 1:
            d2 = correct_curl * random.choice([0.5, 0.33, 0.25])
        else:
            d2 = correct_curl * random.choice([2, 3, 4])

        d3 = correct_curl + random.choice([-0.5, 0.5, -1, 1])

        if abs(correct_curl) < 1:
            d4 = random.choice([-2.5, -3.0, 2.5, 3.0])
        else:
            d4 = correct_curl / random.choice([5, 10])

        distractors = random.sample([d1, d2, d3, d4], 3)

        return [round(d, 2) for d in distractors]

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
        field_func, equation_text, correct_curl = (
            self.field,
            self.equation_text,
            self.correct_curl,
        )

        title = Text("Curl Problem", font_size=36).to_edge(UP)
        equation_text.next_to(title, DOWN)

        field_display = self.create_vector_field(field_func)
        point = Dot(color=GREEN, radius=0.1).move_to(
            np.array([random.uniform(-1, 1), random.uniform(-1, 1), 0])
        )

        self.play(Write(title))
        self.play(Write(equation_text))
        self.play(Create(field_display[0]))
        self.play(Create(field_display[1]), run_time=2)
        self.play(FadeIn(field_display[2]))
        self.play(GrowFromCenter(point))
        self.wait(3)

        distractors = self.create_distractors(correct_curl)
        options = distractors + [correct_curl]
        random.shuffle(options)
        correct_index = options.index(correct_curl)
        letters = ["A", "B", "C", "D"]

        self.play(
            FadeOut(title),
            FadeOut(equation_text),
            FadeOut(field_display),
            FadeOut(point),
            run_time=1.5,
        )

        question = (
            VGroup(
                Text(
                    "What is the z-component of curl at the green point?", font_size=32
                ),
                Text(
                    "Choose the letter of the correct answer:",
                    font_size=24,
                    color=YELLOW,
                ),
            )
            .arrange(DOWN)
            .to_edge(UP, buff=0.5)
        )

        mc_options = VGroup()
        positions = [
            UP * 0.5 + LEFT * 3.5,
            UP * 0.5 + RIGHT * 3.5,
            DOWN * 1.0 + LEFT * 3.5,
            DOWN * 1.0 + RIGHT * 3.5,
        ]

        for i, (opt, pos) in enumerate(zip(options, positions)):
            text = Text(f"{letters[i]}. {opt:.2f}", font_size=36).move_to(pos)
            mc_options.add(text)

        self.play(Write(question))
        self.play(LaggedStart(*[Write(opt) for opt in mc_options], lag_ratio=0.3))
        self.wait(3)

        with open(f"solutions/curl_{self.file_index}.txt", "w") as f:
            f.write(f"{letters[correct_index]}")
        with open(f"question_text/curl_{self.file_index}.txt", "w") as f:
            f.write(
                "What is the z-component of curl at the green point? Output only the letter of the correct answer"
            )


for i in range(3):
    if os.path.exists("manim_output"):
        shutil.rmtree("manim_output")

    scene = CurlQuiz(file_index=i)
    scene.render()

    output = Path("manim_output/videos/1080p60/CurlQuiz.mp4")
    if output.exists():
        shutil.move(str(output), f"questions/curl_{i}.mp4")

    if os.path.exists("manim_output"):
        shutil.rmtree("manim_output")
