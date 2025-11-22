from manim import *
import numpy as np
import random
from pathlib import Path
import shutil
import os

# Setup directories
Path("questions").mkdir(exist_ok=True)
Path("solutions").mkdir(exist_ok=True)
Path("question_text").mkdir(exist_ok=True)
Path("reasoning_traces").mkdir(exist_ok=True)

config.media_dir = "manim_output"
config.verbosity = "WARNING"
config.pixel_height = 1080
config.pixel_width = 1920
config.frame_rate = 60


class FunctionComposition(Scene):
    def __init__(self, file_index, **kwargs):
        super().__init__(**kwargs)
        self.file_index = file_index
        # Store types to help with reasoning trace
        self.f_type = ""
        self.g_type = ""
        self.f, self.g, self.f_str, self.g_str = self.generate_functions()
        self.correct_func = lambda x: self.f(self.g(x))
        self.y_min, self.y_max = -5, 5
        self.out_bounds = random.randint(self.y_min, self.y_max)

    def generate_functions(self):
        types = ["linear", "quadratic", "trig", "exp"]
        self.f_type, self.g_type = random.sample(types, 2)

        # Generate f(x)
        if self.f_type == "linear":
            a = random.choice([-2, -1, 1, 2])
            b = random.choice([-2, -1, 0, 1, 2])
            f = lambda x: a * x + b
            f_str = f"{a}x + {b}" if b >= 0 else f"{a}x - {abs(b)}"
        elif self.f_type == "quadratic":
            a = random.choice([-1, 1])
            b = random.choice([-2, -1, 0, 1, 2])
            c = random.choice([-2, -1, 0, 1, 2])
            f = lambda x: a * x**2 + b * x + c
            f_str = f"{a}x^2 + {b}x + {c}".replace("+-", "-").replace("+ -", "-")
        elif self.f_type == "trig":
            a = random.choice([1, 2])
            b = random.choice([1, 2])
            c = random.choice([0, 1])
            f = lambda x: a * np.sin(b * x) + c
            f_str = f"{a}sin({b}x) + {c}" if c != 0 else f"{a}sin({b}x)"
        else:  # exp
            a = random.choice([1, 2])
            b = random.choice([-1, 1])
            c = random.choice([0, 1])
            f = lambda x: a * np.exp(b * x) + c
            f_str = f"{a}e^({b}x) + {c}" if c != 0 else f"{a}e^({b}x)"

        # Generate g(x)
        if self.g_type == "linear":
            a = random.choice([-2, -1, 1, 2])
            b = random.choice([-2, -1, 0, 1, 2])
            g = lambda x: a * x + b
            g_str = f"{a}x + {b}" if b >= 0 else f"{a}x - {abs(b)}"
        elif self.g_type == "quadratic":
            a = random.choice([-1, 1])
            b = random.choice([-2, -1, 0, 1, 2])
            c = random.choice([-1, 0, 1])
            g = lambda x: a * x**2 + b * x + c
            g_str = f"{a}x^2 + {b}x + {c}".replace("+-", "-").replace("+ -", "-")
        elif self.g_type == "trig":
            a = random.choice([1, 2])
            b = random.choice([1, 2])
            g = lambda x: a * np.cos(b * x)
            g_str = f"{a}cos({b}x)"
        else:  # exp (bell curve style for g to keep things bounded-ish)
            a = random.choice([1, 2])
            b = random.choice([-1, 1])
            g = lambda x: a * np.exp(-abs(b) * x**2)  # Force negative exp for g usually
            g_str = f"{a}e^(-{abs(b)}x^2)"

        return f, g, f_str, g_str

    def bounded_plot(self, func, color, label, position, x_range=(-3, 3)):
        axes = Axes(
            x_range=[x_range[0], x_range[1], 1],
            y_range=[self.y_min, self.y_max, 1],
            x_length=4,
            y_length=3,
            axis_config={"color": WHITE},
        ).move_to(position)

        # Wrap for safety
        def safe_func(x):
            try:
                y = func(x)
                return y
            except:
                return 0

        # Clamp for plotting
        def bounded_func(x):
            y = safe_func(x)
            if np.isnan(y) or np.isinf(y):
                return self.out_bounds
            return y if self.y_min <= y <= self.y_max else self.out_bounds

        graph = axes.plot(bounded_func, color=color, use_smoothing=True)
        label_obj = Text(label, color=color, font_size=24).next_to(axes, UP, buff=0.1)
        return VGroup(axes, graph, label_obj)

    def construct(self):
        title = Text("Function Composition Challenge", font_size=36).to_edge(UP)
        self.play(Write(title))

        f_graph = self.bounded_plot(
            self.f, BLUE, f"f(x) = {self.f_str}", LEFT * 3 + UP * 1
        )
        g_graph = self.bounded_plot(
            self.g, RED, f"g(x) = {self.g_str}", RIGHT * 3 + UP * 1
        )

        self.play(Create(f_graph), Create(g_graph), run_time=2)
        self.wait(1.5)

        self.play(*[FadeOut(mob) for mob in self.mobjects])

        # Generate Options
        options = []
        letters = ["A", "B", "C", "D"]

        # 1. Correct: f(g(x))
        correct_func = lambda x: self.f(self.g(x))

        # 2. Reverse: g(f(x))
        options.append({"func": lambda x: self.g(self.f(x)), "type": "reverse"})

        # 3. Sum: f(x) + g(x)
        options.append({"func": lambda x: self.f(x) + self.g(x), "type": "sum"})

        # 4. Product: f(x) * g(x)
        options.append({"func": lambda x: self.f(x) * self.g(x), "type": "product"})

        # Insert correct answer randomly
        correct_index = random.randint(0, 3)
        options.insert(correct_index, {"func": correct_func, "type": "correct"})

        question = Text("Which graph shows f(g(x))?", font_size=36).to_edge(UP)
        second_part = Text(
            "Output just the letter of the correct answer.", font_size=24
        ).next_to(question, DOWN)

        self.play(Write(question))
        self.play(FadeIn(second_part))

        answer_graphs = VGroup()
        positions = [
            UP * 0.75 + LEFT * 3.5,
            UP * 0.75 + RIGHT * 3.5,
            DOWN * 2.75 + LEFT * 3.5,
            DOWN * 2.75 + RIGHT * 3.5,
        ]

        # Store letter map for trace
        correct_letter = letters[correct_index]

        for i, (opt, pos) in enumerate(zip(options, positions)):
            graph = self.bounded_plot(opt["func"], GREEN, letters[i], pos)
            answer_graphs.add(graph)

        self.play(
            LaggedStart(*[Create(g) for g in answer_graphs], lag_ratio=0.2), run_time=3
        )
        self.wait(3)

        # --- SAVE OUTPUTS ---
        with open(f"solutions/function_composition_{self.file_index}.txt", "w") as f:
            f.write(correct_letter)

        with open(
            f"question_text/function_composition_{self.file_index}.txt", "w"
        ) as f:
            f.write(
                f"Given f(x) = {self.f_str} and g(x) = {self.g_str}. Which graph shows the composition f(g(x))? Output just the letter of the correct answer."
            )

        trace = self.generate_reasoning_trace(correct_letter)
        with open(
            f"reasoning_traces/function_composition_{self.file_index}.txt", "w"
        ) as f:
            f.write(trace)

    def generate_reasoning_trace(self, correct_letter):
        trace = []
        trace.append("=== Analyze Input Functions ===")
        trace.append(f"Function f(x): {self.f_str} (Type: {self.f_type})")
        trace.append(f"Function g(x): {self.g_str} (Type: {self.g_type})")

        trace.append("\n=== Analyze Composition f(g(x)) ===")
        trace.append("We need to find the graph of the composite function f(g(x)).")
        trace.append("This means we take the output of g(x) and plug it into f(x).")

        # Algebraic substitution logic
        sub_str = self.f_str.replace("x", f"({self.g_str})")
        trace.append(f"Algebraically, f(g(x)) ≈ {sub_str}")

        # Qualitative analysis
        trace.append("\n=== Qualitative Prediction ===")
        if self.f_type == "linear" and self.g_type == "linear":
            trace.append(
                "Since both functions are linear, the composition will be a line."
            )
            trace.append("The slope will be the product of the two slopes.")
        elif self.f_type == "quadratic" and self.g_type == "linear":
            trace.append(
                "Plugging a linear function into a quadratic preserves the quadratic nature."
            )
            trace.append("The result will be a parabola.")
        elif self.f_type == "linear" and self.g_type == "quadratic":
            trace.append(
                "Plugging a quadratic into a linear function scales and shifts the parabola."
            )
            trace.append(
                "The result will still be a parabola with the same concavity direction as g(x) (unless flipped by f's slope)."
            )
        elif "trig" in [self.f_type, self.g_type]:
            trace.append(
                "The composition involves a trigonometric function, so we expect oscillatory behavior."
            )
            if self.f_type == "trig":
                trace.append(
                    "Since f is trig, the output will be bounded (oscillating between min/max of f)."
                )
            else:
                trace.append(
                    "Since g is trig, the input to f oscillates, creating a complex wave pattern."
                )

        trace.append("\n=== Conclusion ===")
        trace.append(
            f"Visual inspection of the options confirms that Graph {correct_letter} matches the expected behavior of {sub_str}."
        )
        trace.append(f"Therefore, the correct answer is {correct_letter}.")

        return "\n".join(trace)


for i in range(3):
    if os.path.exists("manim_output"):
        shutil.rmtree("manim_output")

    scene = FunctionComposition(file_index=i)
    scene.render()

    output = Path("manim_output/videos/1080p60/FunctionComposition.mp4")
    if output.exists():
        shutil.move(str(output), f"questions/function_composition_{i}.mp4")

if os.path.exists("manim_output"):
    shutil.rmtree("manim_output")
