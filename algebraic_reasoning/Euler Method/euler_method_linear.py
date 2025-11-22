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


class EulerMethodVisual(Scene):
    def __init__(self, f, step_size, num_steps, x0, y0, a, b, file_index, **kwargs):
        super().__init__(**kwargs)
        self.f = f
        self.step_size = step_size
        self.num_steps = num_steps
        self.x0 = x0
        self.y0 = y0
        self.a = a
        self.b = b
        self.file_index = file_index

    def construct(self):
        f = self.f
        step_size = self.step_size
        num_steps = self.num_steps
        x0, y0 = self.x0, self.y0

        # --- INTRO TEXT ---
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

        # --- AXES AND VECTOR FIELD ---
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
                # Normalize for display
                angle = np.arctan(slope)
                dx = np.cos(angle) * 0.2  # fixed length for visual uniformity
                dy = np.sin(angle) * 0.2

                # Color based on slope magnitude for visual flair
                norm_slope = min(abs(slope) / 3, 1.0)
                if slope > 0:
                    color = interpolate_color(GREEN, RED, norm_slope)
                else:
                    color = interpolate_color(GREEN, BLUE, norm_slope)

                vec = Line(
                    start=axes.c2p(x - dx / 2, y - dy / 2),
                    end=axes.c2p(x + dx / 2, y + dy / 2),
                    stroke_width=2,
                    color=color,
                )
                field.add(vec)

        # --- CALCULATE POINTS ---
        points = [(x0, y0)]
        for _ in range(num_steps):
            x_curr, y_curr = points[-1]
            slope = f(x_curr, y_curr)
            y_next = y_curr + step_size * slope
            x_next = x_curr + step_size
            points.append((x_next, y_next))

        # --- PREPARE ANIMATION OBJECTS ---
        euler_dots = VGroup()
        euler_lines = VGroup()
        dot_labels = VGroup()

        # Generate dots and lines for all segments
        for i in range(len(points) - 1):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]

            # Dot at start of segment
            dot = Dot(axes.c2p(x1, y1), color=RED)
            euler_dots.add(dot)

            # Label for dot
            label = Text(f"({x1:.1f}, {y1:.2f})", font_size=20).next_to(
                dot, UP, buff=0.1
            )
            dot_labels.add(label)

            # Line to next point
            line = Line(axes.c2p(x1, y1), axes.c2p(x2, y2), color=YELLOW)
            euler_lines.add(line)

        # Final dot
        x_last, y_last = points[-1]
        final_dot = Dot(axes.c2p(x_last, y_last), color=RED)
        euler_dots.add(final_dot)
        final_label = Text(f"({x_last:.1f}, {y_last:.2f})", font_size=20).next_to(
            final_dot, UP, buff=0.1
        )
        dot_labels.add(final_label)

        # --- ANIMATE SCENE ---
        self.play(Create(axes), Create(field))
        self.wait(0.5)

        # Animate first dot
        self.play(Create(euler_dots[0]), Write(dot_labels[0]))

        # Animate steps
        for i in range(len(euler_lines)):
            self.play(
                Create(euler_lines[i]),
                Create(euler_dots[i + 1]),
                Write(dot_labels[i + 1]),
                run_time=1.0,
            )

        final_y = round(points[-1][1], 2)

        # --- TRANSITION TO QUESTION ---
        self.play(
            FadeOut(field),
            FadeOut(euler_lines),
            FadeOut(euler_dots),
            FadeOut(dot_labels),
            FadeOut(axes),
            run_time=1,
        )
        self.wait(0.5)

        # --- GENERATE OPTIONS ---
        correct_answer = final_y
        distractors = set()

        # Generate plausible distractors (arithmetic errors)
        distractors.add(
            round(correct_answer + step_size, 2)
        )  # Off by one full step addition
        distractors.add(round(correct_answer - step_size, 2))
        distractors.add(round(correct_answer + 0.5, 2))
        distractors.add(round(correct_answer - 0.5, 2))

        distractors.discard(correct_answer)
        distractor_list = list(distractors)

        while len(distractor_list) < 3:
            new_val = round(correct_answer + random.uniform(-1.0, 1.0), 2)
            if new_val != correct_answer and new_val not in distractor_list:
                distractor_list.append(new_val)

        distractor_list = distractor_list[:3]
        options = distractor_list + [correct_answer]
        random.shuffle(options)

        labels = ["A", "B", "C", "D"]
        correct_label = labels[options.index(correct_answer)]

        # --- DISPLAY QUESTION ---
        question_text = VGroup(
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
            .next_to(question_text, DOWN, buff=0.5)
        )

        full_question = VGroup(question_text, mc_choices).to_edge(UP)

        self.play(Write(full_question))
        self.wait(2)

        # --- SAVE OUTPUTS ---

        # 1. Save Solution
        with open(f"solutions/euler_method_linear_{self.file_index}.txt", "w") as f_out:
            f_out.write(f"{correct_label}")

        # 2. Save Question Text
        with open(
            f"question_text/euler_method_linear_{self.file_index}.txt", "w"
        ) as f_out:
            f_out.write(
                f"The vector field shows the differential equation dy/dx = {self.a}x + {self.b}y.\n"
                f"Starting at (0, 1) with a step size of h={self.step_size}, "
                f"what is the y-value after {num_steps} Euler steps?\n"
                "(Choose the closest answer, give just the letter)"
            )

        # 3. Save Reasoning Trace
        reasoning_trace = self.generate_reasoning_trace(
            points, correct_label, correct_answer
        )
        with open(
            f"reasoning_traces/euler_method_linear_{self.file_index}.txt", "w"
        ) as f_out:
            f_out.write(reasoning_trace)

    def generate_reasoning_trace(self, points, correct_label, final_answer):
        """Generates a step-by-step explanation of the math."""
        trace = []

        trace.append("=== Problem Breakdown ===")
        trace.append(f"We are solving an initial value problem using Euler's Method.")
        trace.append(f"Differential Equation: dy/dx = f(x, y) = {self.a}x + {self.b}y")
        trace.append(f"Initial Condition: (x₀, y₀) = ({self.x0}, {self.y0})")
        trace.append(f"Step size: h = {self.step_size}")
        trace.append(f"Number of steps: {self.num_steps}")
        trace.append("")
        trace.append("=== Step-by-Step Calculation ===")
        trace.append("Euler's Method formula: y_{n+1} = y_n + h * f(x_n, y_n)")
        trace.append("")

        for i in range(self.num_steps):
            x_curr, y_curr = points[i]
            x_next, y_next = points[i + 1]

            # Calculate slope manually for the trace to show the arithmetic
            slope = self.a * x_curr + self.b * y_curr

            trace.append(f"Step {i + 1}:")
            trace.append(
                f"  Current Point: (x_{i}, y_{i}) = ({x_curr:.1f}, {y_curr:.3f})"
            )
            trace.append(
                f"  Calculate Slope: m = {self.a}({x_curr:.1f}) + {self.b}({y_curr:.3f})"
            )
            trace.append(f"                   m = {slope:.3f}")
            trace.append(
                f"  Calculate Next y: y_{i+1} = {y_curr:.3f} + {self.step_size} * {slope:.3f}"
            )
            trace.append(f"                    y_{i+1} = {y_next:.3f}")
            trace.append(
                f"  Next x: x_{i+1} = {x_curr:.1f} + {self.step_size} = {x_next:.1f}"
            )
            trace.append("")

        trace.append("=== Conclusion ===")
        trace.append(
            f"After {self.num_steps} steps, the approximate value of y is {final_answer:.2f}."
        )
        trace.append(
            f"Comparing this result to the given options, the correct choice is {correct_label}."
        )

        return "\n".join(trace)


if __name__ == "__main__":
    for i in range(3):
        # Randomize coefficients for the differential equation dy/dx = ax + by
        a = random.randint(0, 3)
        b = random.randint(
            1, 3
        )  # Avoid b=0 to make it slightly more interesting than just integrating x

        # Define the function for the simulation
        def f_func(x, y, a_val=a, b_val=b):
            return a_val * x + b_val * y

        # Clean previous render
        if os.path.exists("manim_output"):
            shutil.rmtree("manim_output")

        # Instantiate and render scene
        scene = EulerMethodVisual(
            f=f_func, step_size=0.2, num_steps=3, x0=0.0, y0=1.0, a=a, b=b, file_index=i
        )
        scene.render()

        # Move video output
        output_path = Path("manim_output/videos/1080p60/EulerMethodVisual.mp4")
        if output_path.exists():
            shutil.move(str(output_path), f"questions/euler_method_linear_{i}.mp4")
        else:
            # Fallback for different Manim versions/configs
            videos_dir = Path("manim_output/videos")
            found = list(videos_dir.rglob("*.mp4"))
            if found:
                shutil.move(str(found[0]), f"questions/euler_method_linear_{i}.mp4")

        # Cleanup
        if os.path.exists("manim_output"):
            shutil.rmtree("manim_output")
