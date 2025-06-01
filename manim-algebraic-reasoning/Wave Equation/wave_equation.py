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


class WaveEquationQuiz(Scene):
    def __init__(self, file_index, **kwargs):
        super().__init__(**kwargs)
        self.file_index = file_index
        self.c_squared = random.randint(1, 10)
        self.correct_index = 0
        self.num_points = 100
        self.x_points = np.linspace(0, 10, self.num_points)
        self.time_points = np.linspace(0, 5, 100)

        self.pluck_pos = random.uniform(2, 8)
        self.fly_pos = random.uniform(2, 8)
        while abs(self.fly_pos - self.pluck_pos) < 1.5:
            self.fly_pos = random.uniform(2, 8)

        self.amplitude = random.uniform(0.5, 2.0)
        self.width = random.uniform(0.5, 1.5)

    def initial_condition(self, x):

        return self.amplitude * np.exp(
            -((x - self.pluck_pos) ** 2) / (2 * self.width**2)
        )

    def wave_solution(self, x, t):

        term1 = self.initial_condition(x - np.sqrt(self.c_squared) * t)
        term2 = self.initial_condition(x + np.sqrt(self.c_squared) * t)
        return (term1 + term2) / 2

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
        for y in np.arange(-2, 2.1, 0.5):
            tick = Line(
                start=axes.c2p(0, y) - LEFT * 0.2,
                end=axes.c2p(0, y) + LEFT * 0.2,
                color=WHITE,
            )
            number = Text(f"{y:.1f}", font_size=24).next_to(tick, LEFT, buff=0.1)
            y_ticks.add(tick, number)

        return VGroup(x_ticks, y_ticks)

    def construct(self):

        axes = Axes(
            x_range=[0, 10],
            y_range=[0, 2],
            x_length=10,
            y_length=6,
            axis_config={"color": WHITE},
        ).shift(DOWN * 0.5)

        ticks = self.add_manual_ticks(axes)

        title = Text("Wave Equation Simulation", font_size=36).to_edge(UP)
        x_label = Text("Position (m)", font_size=24).next_to(
            axes.x_axis, DOWN, buff=1.5
        )
        y_label = Text("m", font_size=24).next_to(axes.y_axis, LEFT, buff=1)

        speed_text = Text(f"c² = {self.c_squared} m²/s²", font_size=28).to_corner(UR)
        speed_text.set_color(YELLOW)

        time_text = Text("t = 0.00 s", font_size=28).next_to(
            speed_text, DOWN, aligned_edge=RIGHT
        )
        time_text.set_color(YELLOW)

        pluck_indicator = Dot(axes.c2p(self.pluck_pos, 0), color=RED)
        pluck_label = Text("Pluck", font_size=24, color=RED).next_to(
            pluck_indicator, UP
        )

        fly_indicator = Dot(axes.c2p(self.fly_pos, 0), color=GREEN)
        fly_label = Text("Fly", font_size=24, color=GREEN).next_to(fly_indicator, UP)

        initial_points = [axes.c2p(x, self.initial_condition(x)) for x in self.x_points]
        initial_wave = VMobject().set_points_as_corners(initial_points).set_color(BLUE)

        graph = VMobject().set_color(BLUE)
        graph.set_points_as_corners(initial_points)

        def update_graph(graph, dt):
            t = self.time_elapsed
            new_points = [axes.c2p(x, self.wave_solution(x, t)) for x in self.x_points]
            graph.set_points_as_corners(new_points)

            time_text.become(
                Text(f"t = {t:.2f} s", font_size=28).next_to(
                    speed_text, DOWN, aligned_edge=RIGHT
                )
            )
            time_text.set_color(YELLOW)

        graph.add_updater(update_graph)
        self.time_elapsed = 0
        max_wave_height = 0

        for t in self.time_points:
            wave_height = abs(self.wave_solution(self.fly_pos, t))
            if wave_height > max_wave_height:
                max_wave_height = wave_height

        required_jump = max_wave_height + 0.1
        self.correct_answer = round(required_jump, 2)

        self.play(Write(title))
        self.play(Create(axes), Create(ticks), Write(x_label), Write(y_label))
        self.play(Write(speed_text), Write(time_text))
        self.play(Create(pluck_indicator), Write(pluck_label))
        self.play(Create(fly_indicator), Write(fly_label))
        self.play(Create(initial_wave))
        self.wait(0.5)

        self.play(ReplacementTransform(initial_wave, graph))

        for t in np.linspace(0, 5, 100):
            self.time_elapsed = t
            self.wait(0.05)

        graph.remove_updater(update_graph)

        self.play(*[FadeOut(mob) for mob in self.mobjects])

        question = Text(
            f"A fly is sitting on the string at x = {self.fly_pos:.1f} m.\n"
            f"Someone plucks the string at x = {self.pluck_pos:.1f} m.\n"
            "How high does the fly have to jump to avoid getting hit by the wave?\n"
            "Output just the correct letter:",
            font_size=24,
            line_spacing=1.5,
        ).to_edge(UP)

        options = sorted(
            {
                self.correct_answer,
                max(0.1, self.correct_answer - 0.2),
                min(2.0, self.correct_answer + 0.2),
                random.uniform(0.1, 2.0),
            }
        )
        random.shuffle(options)
        self.correct_index = options.index(self.correct_answer)

        option_text = VGroup()
        for i, opt in enumerate(options):
            option = Text(
                f"{chr(65+i)}. {opt:.2f} m", font_size=28, t2c={f"{opt:.2f}": YELLOW}
            )
            option_text.add(option)

        option_text.arrange(DOWN, aligned_edge=LEFT, buff=0.5).next_to(
            question, DOWN, buff=1
        )

        self.play(Write(question))
        self.play(LaggedStart(*[Write(opt) for opt in option_text], lag_ratio=0.2))
        self.wait(3)

        with open(f"solutions/wave_equation_{self.file_index}.txt", "w") as f:
            f.write(chr(65 + self.correct_index))
        with open(f"question_text/wave_equation_{self.file_index}.txt", "w") as f:
            f.write(
                f"A fly is sitting on the string at x = {self.fly_pos:.1f} m. "
                f"Someone plucks the string at x = {self.pluck_pos:.1f} m. "
                "How high does the fly have to jump to avoid getting hit by the wave? "
                "Output just the correct letter:"
            )


for i in range(3):
    if os.path.exists("manim_output"):
        shutil.rmtree("manim_output")

    scene = WaveEquationQuiz(file_index=i)
    scene.render()

    output = Path("manim_output/videos/1080p60/WaveEquationQuiz.mp4")
    if output.exists():
        shutil.move(str(output), f"questions/wave_equation_{i}.mp4")

if os.path.exists("manim_output"):
    shutil.rmtree("manim_output")
