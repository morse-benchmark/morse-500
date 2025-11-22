from manim import *
import random
import numpy as np
from sympy import symbols, diff, lambdify
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


class NewtonsMethodVisual(Scene):
    def __init__(self, degree, steps, x0, idx, **kwargs):
        super().__init__(**kwargs)
        self.degree = degree
        self.steps = steps
        self.x0 = x0
        self.idx = idx
        # Storage for reasoning trace
        self.poly_coeffs = []
        self.poly_str = ""
        self.deriv_str = ""
        self.iterations = []  # Will store tuples of (x_n, f(x_n), f'(x_n), x_n+1)

    def construct(self):
        # 1. Generate the Problem
        self.generate_newton_problem()

        # 2. Visualize
        self.visualize_function()

        # 3. Pose Question
        self.display_question()

        # 4. Save Outputs
        self.save_outputs()

    def generate_random_polynomial(self):
        # Generate coefficients for: c_n*x^n + ... + c_1*x + c_0
        coeffs = [random.randint(-3, 3) for _ in range(self.degree + 1)]
        # Ensure leading coefficient isn't 0
        while coeffs[self.degree] == 0:
            coeffs[self.degree] = random.randint(-3, 3)
        return coeffs

    def format_poly_str(self, expr):
        # Clean up Sympy string output (e.g., "x**3 + -2*x" -> "x^3 - 2x")
        s = str(expr).replace("**", "^").replace("*", "")
        s = s.replace("+ -", "- ")
        return s

    def generate_newton_problem(self):
        self.poly_coeffs = self.generate_random_polynomial()

        # Sympy setup
        x = symbols("x")
        poly_expr = sum(c * x**i for i, c in enumerate(self.poly_coeffs))
        deriv_expr = diff(poly_expr, x)

        self.poly_str = self.format_poly_str(poly_expr)
        self.deriv_str = self.format_poly_str(deriv_expr)

        # Create numerical functions
        f = lambdify(x, poly_expr)
        f_prime = lambdify(x, deriv_expr)

        # Run Newton's Method
        current_x = self.x0
        self.iterations = []

        for _ in range(self.steps):
            val = f(current_x)
            slope = f_prime(current_x)

            # Avoid division by zero in random generation
            if abs(slope) < 1e-4:
                slope = 1e-4 if slope >= 0 else -1e-4

            next_x = current_x - val / slope
            self.iterations.append(
                {"x_curr": current_x, "f_val": val, "f_prime": slope, "x_next": next_x}
            )
            current_x = next_x

        self.solution = round(current_x, 2)

    def visualize_function(self):
        x = symbols("x")
        poly_expr = sum(c * x**i for i, c in enumerate(self.poly_coeffs))
        f = lambdify(x, poly_expr)

        # Determine plot bounds based on critical points + padding
        # Simple heuristic: check range around x0 and 0
        check_range = np.linspace(-5, 5, 100)
        y_vals = [f(v) for v in check_range]
        y_min = min(y_vals)
        y_max = max(y_vals)

        # Clamp reasonable view
        y_min = max(y_min, -15)
        y_max = min(y_max, 15)

        axes = Axes(
            x_range=[-5, 5, 1],
            y_range=[y_min - 1, y_max + 1, 2],
            axis_config={"color": BLUE},
            x_length=10,
            y_length=6,
        )

        # Draw curve
        graph = axes.plot(f, color=GREEN, x_range=[-5, 5], use_smoothing=True)

        # Highlight specific points to give clues about the shape
        points = []
        labels = []

        # Add a few integer points so the graph is readable
        clue_xs = sorted(list(set([int(self.x0), -2, 0, 2])))
        for cx in clue_xs:
            if -5 <= cx <= 5:
                cy = f(cx)
                if y_min <= cy <= y_max:
                    p = Dot(axes.c2p(cx, cy), color=YELLOW, radius=0.06)
                    l = Text(f"({cx}, {cy:.1f})", font_size=16).next_to(
                        p, UP + RIGHT, buff=0.05
                    )
                    points.append(p)
                    labels.append(l)

        # Start Point (Red Dot)
        start_y = f(self.x0)
        # Clamp start point to visible area for the visual (even if math continues)
        start_dot_y = min(max(start_y, y_min), y_max)

        start_dot = Dot(axes.c2p(self.x0, start_dot_y), color=RED, radius=0.1)
        start_label = Text(f"Start x₀ = {self.x0}", font_size=24, color=RED)
        start_label.next_to(start_dot, UP, buff=0.2)

        # Animations
        self.play(Create(axes), Create(graph))
        self.play(FadeIn(start_dot), Write(start_label))
        self.play(
            LaggedStart(*[FadeIn(p) for p in points], lag_ratio=0.1),
            LaggedStart(*[Write(l) for l in labels], lag_ratio=0.1),
        )
        self.wait(1)

        # Fade out helpers, keep main graph
        self.play(
            FadeOut(start_label),
            *[FadeOut(p) for p in points],
            *[FadeOut(l) for l in labels],
        )

        # Store mobjects to fade out later if needed, but keeping graph is good context
        self.axes_group = VGroup(axes, graph, start_dot)

    def display_question(self):
        question_box = (
            VGroup(
                Text(
                    f"Calculate the approximation after {self.steps} steps of Newton's Method.",
                    font_size=28,
                ),
                Text(f"Polynomial: f(x) = {self.poly_str}", font_size=24, color=BLUE),
                Text(
                    f"Start at x₀ = {self.x0}. Round final answer to 2 d.p.",
                    font_size=24,
                    color=RED,
                ),
            )
            .arrange(DOWN, aligned_edge=LEFT, buff=0.2)
            .to_edge(UP)
        )

        background = BackgroundRectangle(
            question_box, color=BLACK, fill_opacity=0.8, buff=0.2
        )

        self.play(FadeIn(background), Write(question_box))
        self.wait(4)  # Give time to read

    def save_outputs(self):
        # 1. Solution
        with open(
            f"solutions/newtons_method_degree_{self.degree}_{self.idx}.txt", "w"
        ) as f:
            f.write(f"{self.solution:.2f}")

        # 2. Question Text
        q_text = (
            f"Consider the polynomial f(x) = {self.poly_str}.\n"
            f"Using Newton's Method starting at x_0 = {self.x0}, "
            f"what is the approximation x_{self.steps} after {self.steps} steps?\n"
            "Output the answer rounded to 2 decimal places."
        )
        with open(
            f"question_text/newtons_method_degree_{self.degree}_{self.idx}.txt", "w"
        ) as f:
            f.write(q_text)

        # 3. Reasoning Trace
        trace = self.generate_reasoning_trace()
        with open(
            f"reasoning_traces/newtons_method_degree_{self.degree}_{self.idx}.txt", "w"
        ) as f:
            f.write(trace)

    def generate_reasoning_trace(self):
        trace = []
        trace.append("=== Problem Setup ===")
        trace.append(f"Function: f(x) = {self.poly_str}")
        trace.append(f"Start Value: x_0 = {self.x0}")
        trace.append(f"Iterations required: {self.steps}")
        trace.append("")

        trace.append("=== Derivative Calculation ===")
        trace.append(f"To apply Newton's method, we need the derivative f'(x).")
        trace.append(f"Applying the power rule to each term:")
        trace.append(f"f'(x) = {self.deriv_str}")
        trace.append("")

        trace.append("=== Iterations ===")
        trace.append("Formula: x_{n+1} = x_n - f(x_n) / f'(x_n)")
        trace.append("")

        for i, step in enumerate(self.iterations):
            x_n = step["x_curr"]
            fx = step["f_val"]
            fpx = step["f_prime"]
            x_next = step["x_next"]

            trace.append(f"--- Step {i+1} ---")
            trace.append(f"Current x_{i} = {x_n:.6f}")
            trace.append(f"1. Calculate f({x_n:.6f}) = {fx:.6f}")
            trace.append(f"2. Calculate f'({x_n:.6f}) = {fpx:.6f}")
            trace.append(f"3. Apply formula:")
            trace.append(f"   x_{i+1} = {x_n:.6f} - ({fx:.6f} / {fpx:.6f})")
            trace.append(f"   x_{i+1} = {x_next:.6f}")
            trace.append("")

        trace.append("=== Final Answer ===")
        trace.append(f"After {self.steps} steps, x ≈ {self.solution:.2f}")

        return "\n".join(trace)


if __name__ == "__main__":
    # Get degree from environment variable or default to 3
    try:
        degree_arg = int(os.environ.get("MANIM_DEGREE", 3))
    except ValueError:
        degree_arg = 3

    for idx in range(3):
        # Clean previous render for this specific index context
        if os.path.exists("manim_output"):
            shutil.rmtree("manim_output")

        # Randomize parameters
        steps = random.randint(
            2, 4
        )  # Keep steps lower so manual verification is feasible
        x0 = round(random.uniform(-3, 3), 0)  # Integers are nicer starting points

        scene = NewtonsMethodVisual(degree=degree_arg, steps=steps, x0=x0, idx=idx)
        scene.render()

        # Move file
        output_file = Path("manim_output/videos/1080p60/NewtonsMethodVisual.mp4")
        if output_file.exists():
            shutil.move(
                str(output_file),
                f"questions/newtons_method_degree_{degree_arg}_{idx}.mp4",
            )
        else:
            # Fallback search
            found = list(Path("manim_output/videos").rglob("*.mp4"))
            if found:
                shutil.move(
                    str(found[0]),
                    f"questions/newtons_method_degree_{degree_arg}_{idx}.mp4",
                )

        # Cleanup
        if os.path.exists("manim_output"):
            shutil.rmtree("manim_output")
