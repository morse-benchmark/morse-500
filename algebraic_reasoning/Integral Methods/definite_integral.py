from manim import *
import numpy as np
import random
from pathlib import Path
import shutil
import os
from scipy.integrate import quad

# Setup directories
Path("questions").mkdir(exist_ok=True)
Path("solutions").mkdir(exist_ok=True)
Path("question_text").mkdir(exist_ok=True)
Path("reasoning_traces").mkdir(exist_ok=True)

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

        # Clean up tex string for label
        tex_label = self.f_tex.replace("*", "")  # basic cleanup
        function_label = Text(tex_label, font_size=40).to_edge(UP)

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

        # --- SAVE OUTPUTS ---
        ans_str = f"{round(self.integral_val, 2)}"

        with open(f"solutions/definite_integral_{self.file_index}.txt", "w") as f_out:
            f_out.write(ans_str)

        with open(
            f"question_text/definite_integral_{self.file_index}.txt", "w"
        ) as f_out:
            f_out.write(
                f"What is the definite integral of the function {self.f_tex} from x={self.a} to x={self.b}? Output the answer rounded to 2 decimal places."
            )

        trace = self.generate_reasoning_trace(ans_str)
        with open(
            f"reasoning_traces/definite_integral_{self.file_index}.txt", "w"
        ) as f_out:
            f_out.write(trace)

    def generate_reasoning_trace(self, ans_str):
        trace = []
        trace.append("=== Problem Statement ===")
        trace.append(f"Function: {self.f_tex}")
        trace.append(f"Lower Bound (a): {self.a}")
        trace.append(f"Upper Bound (b): {self.b}")
        trace.append("")

        trace.append("=== Calculation ===")
        trace.append(
            f"We are calculating the definite integral: ∫ from {self.a} to {self.b} of ({self.f_tex}) dx."
        )
        trace.append(
            "This represents the shaded area under the curve shown in the video."
        )
        trace.append(
            f"Using numerical integration (quadrature), the exact value is approximately {self.integral_val:.6f}."
        )
        trace.append("")

        trace.append("=== Final Answer ===")
        trace.append(f"Rounding the result to 2 decimal places:")
        trace.append(f"Answer: {ans_str}")

        return "\n".join(trace)


for i in range(3):
    funcs = [
        (lambda x: x**2 * np.sin(x), "f(x) = x^2 * sin(x)"),
        (lambda x: np.tanh(1.2 * x) * np.cos(x), "f(x) = tanh(1.2 * x) * cos(x)"),
        (lambda x: np.log(x + 1) * x, "f(x) = log(x+1) * x"),
    ]

    # Randomize function selection or cycle through
    f, f_tex = funcs[i % 3]

    a = random.randint(1, 5)
    b = random.randint(a + 1, 8)

    val, _ = quad(f, a, b)

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
    else:
        # Fallback search for manim output file
        found = list(Path("manim_output/videos").rglob("*.mp4"))
        if found:
            shutil.move(str(found[0]), f"questions/definite_integral_{i}.mp4")

    if os.path.exists("manim_output"):
        shutil.rmtree("manim_output")
