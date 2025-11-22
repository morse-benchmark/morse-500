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

        # --- INTRO ---
        title = Text("Euler's Method Problem", font_size=36).to_edge(UP)
        step_info = Text(
            f"Step size (h) = {step_size}, Number of steps = {num_steps}", font_size=24
        ).next_to(title, DOWN)

        self.play(Write(title), Write(step_info))
        self.wait(2)
        self.play(FadeOut(title), FadeOut(step_info))

        # --- VISUALIZATION ---
        axes = Axes(
            x_range=[-0.5, 2.5, 0.5],
            y_range=[0, 4, 0.5],
            x_length=10,
            y_length=5,
        ).to_edge(DOWN)

        # Vector Field
        field = VGroup()
        for x in np.arange(-0.5, 2.5, 0.3):
            for y in np.arange(0, 4, 0.3):
                slope = f(x, y)
                # Normalize for visual consistency
                angle = np.arctan(slope)
                vec_len = 0.2
                dx = np.cos(angle) * vec_len
                dy = np.sin(angle) * vec_len

                norm_slope = min(abs(slope) / 3, 1.0)
                if slope > 0:
                    col = interpolate_color(GREEN, RED, norm_slope)
                else:
                    col = interpolate_color(GREEN, BLUE, norm_slope)
                vec = Line(
                    start=axes.c2p(x - dx / 2, y - dy / 2),
                    end=axes.c2p(x + dx / 2, y + dy / 2),
                    stroke_width=2,
                    color=col,
                )
                field.add(vec)

        # Calculate Steps
        points = [(x0, y0)]
        for _ in range(num_steps):
            x_curr, y_curr = points[-1]
            slope = f(x_curr, y_curr)
            y_next = y_curr + step_size * slope
            x_next = x_curr + step_size
            points.append((x_next, y_next))

        # Animation Objects
        euler_dots = VGroup()
        euler_lines = VGroup()
        dot_labels = VGroup()

        for i in range(len(points) - 1):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]
            dot = Dot(axes.c2p(x1, y1), color=RED)
            line = Line(axes.c2p(x1, y1), axes.c2p(x2, y2), color=YELLOW)
            label = Text(f"({x1:.1f}, {y1:.2f})", font_size=20).next_to(
                dot, UP, buff=0.1
            )

            euler_dots.add(dot)
            euler_lines.add(line)
            dot_labels.add(label)

        # Final point
        x_last, y_last = points[-1]
        final_dot = Dot(axes.c2p(x_last, y_last), color=RED)
        final_label = Text(f"({x_last:.1f}, {y_last:.2f})", font_size=20).next_to(
            final_dot, UP, buff=0.1
        )
        euler_dots.add(final_dot)
        dot_labels.add(final_label)

        # Play Animation
        self.play(Create(axes), Create(field))
        self.wait(0.5)
        self.play(Create(euler_dots[0]), Write(dot_labels[0]))

        for i in range(len(euler_lines)):
            self.play(
                Create(euler_lines[i]),
                Create(euler_dots[i + 1]),
                Write(dot_labels[i + 1]),
                run_time=0.7,
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

        # --- GENERATE QUESTION & OPTIONS ---
        correct_answer = final_y

        # Generate distinct distractors
        distractors = set()
        offsets = [-0.7, -0.5, -0.2, 0.2, 0.5, 0.7]
        for off in offsets:
            val = round(correct_answer + off, 2)
            if val != correct_answer:
                distractors.add(val)

        distractors_list = list(distractors)
        while len(distractors_list) < 3:
            val = round(correct_answer + random.uniform(-1, 1), 2)
            if val != correct_answer and val not in distractors_list:
                distractors_list.append(val)

        options = random.sample(distractors_list, 3) + [correct_answer]
        random.shuffle(options)
        labels = ["A", "B", "C", "D"]
        correct_label = labels[options.index(correct_answer)]

        question_text_obj = VGroup(
            Text(f"What is the y-value after {num_steps} Euler steps?", font_size=28),
            Text("(Choose the closest answer, give just the letter)", font_size=22),
        ).arrange(DOWN)

        mc_choices = (
            VGroup(
                *[
                    Text(f"{label}. {value:.2f}", font_size=28)
                    for label, value in zip(labels, options)
                ]
            )
            .arrange(DOWN, aligned_edge=LEFT, buff=0.3)
            .next_to(question_text_obj, DOWN, buff=0.5)
        )

        full_question = VGroup(question_text_obj, mc_choices).to_edge(UP)

        self.play(Write(full_question))
        self.wait(2)

        # --- SAVE OUTPUTS ---
        # 1. Solution
        with open(
            f"solutions/euler_method_nonlinear_{self.file_index}.txt", "w"
        ) as f_out:
            f_out.write(correct_label)

        # 2. Question Text
        func_str = f"dy/dx = {a}x^2 + {b}y^2 + {c}sin(x) + {d}cos(y)"
        q_text = (
            f"Consider the differential equation: {func_str}\n"
            f"Starting at ({x0}, {y0}) with step size h={step_size}.\n"
            f"What is the y-value after {num_steps} Euler steps? (Choose the closest answer, give just the letter)"
        )
        with open(
            f"question_text/euler_method_nonlinear_{self.file_index}.txt", "w"
        ) as f_out:
            f_out.write(q_text)

        # 3. Reasoning Trace
        trace = self.generate_reasoning_trace(points, correct_label, final_y)
        with open(
            f"reasoning_traces/euler_method_nonlinear_{self.file_index}.txt", "w"
        ) as f_out:
            f_out.write(trace)

    def generate_reasoning_trace(self, points, correct_label, final_val):
        a, b, c, d = self.coeffs
        trace = []
        trace.append("=== Problem Setup ===")
        trace.append(
            f"Equation: dy/dx = f(x,y) = {a}x^2 + {b}y^2 + {c}sin(x) + {d}cos(y)"
        )
        trace.append(f"Initial: (x0, y0) = ({self.x0}, {self.y0})")
        trace.append(f"Step size: h = {self.step_size}")
        trace.append(f"Steps: {self.num_steps}")
        trace.append("")
        trace.append("=== Step-by-Step Calculation ===")

        for i in range(self.num_steps):
            x_curr, y_curr = points[i]
            x_next, y_next = points[i + 1]

            # Recompute terms for the trace explanation
            term1 = a * (x_curr**2)
            term2 = b * (y_curr**2)
            term3 = c * np.sin(x_curr)
            term4 = d * np.cos(y_curr)
            slope = term1 + term2 + term3 + term4

            trace.append(f"Step {i+1}:")
            trace.append(f"  Current point: ({x_curr:.1f}, {y_curr:.3f})")
            trace.append(f"  Calculate Slope f({x_curr:.1f}, {y_curr:.3f}):")
            trace.append(
                f"    = {a}({x_curr:.1f})^2 + {b}({y_curr:.3f})^2 + {c}sin({x_curr:.1f}) + {d}cos({y_curr:.3f})"
            )
            trace.append(f"    = {term1:.3f} + {term2:.3f} + {term3:.3f} + {term4:.3f}")
            trace.append(f"    = {slope:.4f}")
            trace.append(f"  Update y: y_new = y_old + h * slope")
            trace.append(f"    = {y_curr:.3f} + {self.step_size} * {slope:.4f}")
            trace.append(f"    = {y_next:.3f}")
            trace.append("")

        trace.append("=== Conclusion ===")
        trace.append(f"The final calculated value is approximately {final_val:.2f}.")
        trace.append(f"This corresponds to option {correct_label}.")
        return "\n".join(trace)


if __name__ == "__main__":
    for i in range(3):
        # Ensure at least one non-zero coefficient so it's not trivial
        while True:
            a = random.choice([-2, -1, 0, 1, 2])
            b = random.choice([-2, -1, 0, 1, 2])
            c = random.choice([-2, -1, 0, 1, 2])
            d = random.choice([-2, -1, 0, 1, 2])
            if any([a, b, c, d]):
                break

        def f_func(x, y, a=a, b=b, c=c, d=d):
            return a * x**2 + b * y**2 + c * np.sin(x) + d * np.cos(y)

        if os.path.exists("manim_output"):
            shutil.rmtree("manim_output")

        scene = EulerNonlinearVisual(
            f=f_func,
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
        else:
            # Fallback search
            found = list(Path("manim_output/videos").rglob("*.mp4"))
            if found:
                shutil.move(str(found[0]), f"questions/euler_method_nonlinear_{i}.mp4")

        if os.path.exists("manim_output"):
            shutil.rmtree("manim_output")
