from manim import *
import numpy as np
import random
from pathlib import Path
import shutil
import os
from scipy.signal import fftconvolve

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


class ConvolutionQuiz(Scene):
    def __init__(self, file_index, **kwargs):
        super().__init__(**kwargs)
        self.file_index = file_index

        # Store metadata for reasoning trace
        self.f1_meta = {}
        self.f2_meta = {}
        self.options_meta = (
            []
        )  # Will store {'label': 'A', 'type': 'correct'|'shifted'|'scaled', ...}

        self.functions = self.generate_functions()
        self.correct_convolution = self.calculate_convolution()

    def generate_functions(self):
        types = ["rect", "tri", "gauss", "exp"]
        f1_type, f2_type = random.sample(types, 2)

        # Define mathematical functions
        def rect(x, center=0, width=1):
            return np.where((x >= center - width / 2) & (x <= center + width / 2), 1, 0)

        def tri(x, center=0, width=1):
            return np.maximum(0, 1 - 2 * np.abs(x - center) / width)

        def gauss(x, center=0, sigma=0.5):
            return np.exp(-(((x - center) / sigma) ** 2) / 2)

        def exp_decay(x, center=0):
            return np.where(x >= center, np.exp(-(x - center)), 0)

        x = np.linspace(-3, 3, 400)

        # Generate F1 and store metadata
        if f1_type == "rect":
            c, w = random.uniform(-1, 0), random.uniform(0.8, 1.5)
            f1 = rect(x, center=c, width=w)
            self.f1_meta = {
                "type": "Rectangle",
                "center": c,
                "param": w,
                "param_name": "width",
            }
        elif f1_type == "tri":
            c, w = random.uniform(-1, 0), random.uniform(1, 2)
            f1 = tri(x, center=c, width=w)
            self.f1_meta = {
                "type": "Triangle",
                "center": c,
                "param": w,
                "param_name": "width",
            }
        elif f1_type == "gauss":
            c, s = random.uniform(-1, 0), random.uniform(0.3, 0.7)
            f1 = gauss(x, center=c, sigma=s)
            self.f1_meta = {
                "type": "Gaussian",
                "center": c,
                "param": s,
                "param_name": "sigma",
            }
        else:  # exp
            c = random.uniform(-1.5, -0.5)
            f1 = exp_decay(x, center=c)
            self.f1_meta = {
                "type": "Exponential Decay",
                "center": c,
                "param": 1.0,
                "param_name": "decay_rate",
            }

        # Generate F2 and store metadata
        if f2_type == "rect":
            c, w = random.uniform(0, 1), random.uniform(0.8, 1.5)
            f2 = rect(x, center=c, width=w)
            self.f2_meta = {
                "type": "Rectangle",
                "center": c,
                "param": w,
                "param_name": "width",
            }
        elif f2_type == "tri":
            c, w = random.uniform(0, 1), random.uniform(1, 2)
            f2 = tri(x, center=c, width=w)
            self.f2_meta = {
                "type": "Triangle",
                "center": c,
                "param": w,
                "param_name": "width",
            }
        elif f2_type == "gauss":
            c, s = random.uniform(0, 1), random.uniform(0.3, 0.7)
            f2 = gauss(x, center=c, sigma=s)
            self.f2_meta = {
                "type": "Gaussian",
                "center": c,
                "param": s,
                "param_name": "sigma",
            }
        else:  # exp
            c = random.uniform(0.5, 1.5)
            f2 = exp_decay(x, center=c)
            self.f2_meta = {
                "type": "Exponential Decay",
                "center": c,
                "param": 1.0,
                "param_name": "decay_rate",
            }

        return x, f1, f2

    def calculate_convolution(self):
        x, f1, f2 = self.functions
        # Calculate convolution. Note: fftconvolve size is len(f1)+len(f2)-1
        # mode='same' returns output of length max(M, N).
        # Since our dx is constant, we multiply by dx to approximate integral.
        dx = x[1] - x[0]
        conv = fftconvolve(f1, f2, mode="same") * dx
        return x, conv

    def create_graph(self, axes, x, y, color, label):
        graph = axes.plot_line_graph(x, y, line_color=color, add_vertex_dots=False)
        label_obj = Text(label, color=color, font_size=24).next_to(axes, UP, buff=0.1)
        return VGroup(axes, graph, label_obj)

    def construct(self):
        x, f1, f2 = self.functions
        x_conv, conv = self.correct_convolution

        # --- VISUALIZATION PART 1: INPUTS ---
        title = Text("Function Convolution", font_size=32).to_edge(UP)
        subtitle = Text("Original Functions", font_size=28).next_to(title, DOWN)

        f1_axes = (
            Axes(x_range=[-3, 3, 1], y_range=[0, 1.2, 0.2], x_length=6, y_length=3)
            .next_to(subtitle, DOWN, buff=0.5)
            .shift(LEFT * 3)
        )
        f2_axes = (
            Axes(x_range=[-3, 3, 1], y_range=[0, 1.2, 0.2], x_length=6, y_length=3)
            .next_to(subtitle, DOWN, buff=0.5)
            .shift(RIGHT * 3)
        )

        f1_graph = self.create_graph(f1_axes, x, f1, BLUE, r"f(t)")
        f2_graph = self.create_graph(f2_axes, x, f2, RED, r"g(t)")

        self.play(Write(title), Write(subtitle))
        self.play(Create(f1_graph), Create(f2_graph), run_time=2)
        self.wait(2)

        # --- GENERATE OPTIONS & DISTRACTORS ---
        # We generate 3 wrong options and 1 right option
        raw_options = []

        # Create 3 wrong answers
        for _ in range(3):
            if random.random() > 0.5:
                # Shift Error
                shift_factor = random.choice([-1, 1]) * random.uniform(0.5, 1.2)
                shift_pixels = int(shift_factor * len(conv) / 6)
                wrong_conv = np.roll(conv, shift_pixels)

                # Record metadata (convert pixels to approximate x-units for trace)
                # The x-range is 6 units (-3 to 3) over 400 pixels.
                x_shift_val = (shift_pixels / 400) * 6
                raw_options.append(
                    {"data": wrong_conv, "type": "shifted", "val": x_shift_val}
                )
            else:
                # Scale Error
                scale = (
                    random.uniform(0.5, 0.8)
                    if random.random() < 0.5
                    else random.uniform(1.3, 1.8)
                )
                wrong_conv = conv * scale
                raw_options.append({"data": wrong_conv, "type": "scaled", "val": scale})

        # Add correct answer
        raw_options.append({"data": conv, "type": "correct", "val": 0})

        # Shuffle
        random.shuffle(raw_options)
        letters = ["A", "B", "C", "D"]

        # Store the final mapping of Label -> Data/Meta
        self.options_meta = []
        for i, opt in enumerate(raw_options):
            opt["label"] = letters[i]
            self.options_meta.append(opt)
            if opt["type"] == "correct":
                correct_letter = letters[i]

        # --- VISUALIZATION PART 2: QUIZ ---
        self.play(
            FadeOut(title),
            FadeOut(subtitle),
            FadeOut(f1_graph),
            FadeOut(f2_graph),
        )

        question = (
            VGroup(
                Text("Which is the correct convolution f(t) * g(t)?", font_size=28),
                Text(
                    "Output just the letter of the correct answer:",
                    font_size=24,
                    color=YELLOW,
                ),
            )
            .arrange(DOWN, aligned_edge=LEFT)
            .to_edge(UP, buff=0.5)
        )

        option_graphs = VGroup()
        positions = [
            UP * 0.5 + LEFT * 3.5,
            UP * 0.5 + RIGHT * 3.5,
            DOWN * 2.75 + LEFT * 3.5,
            DOWN * 2.75 + RIGHT * 3.5,
        ]

        for i, pos in enumerate(positions):
            opt_data = self.options_meta[i]["data"]
            # Calculate appropriate y-range for display so graphs don't clip
            max_y = max(np.max(opt_data) * 1.2, 1.0)

            ax = Axes(
                x_range=[-6, 6, 2],
                y_range=[0, max_y, 0.5],  # Dynamic height
                x_length=5,
                y_length=2.5,
            ).move_to(pos)

            graph = ax.plot_line_graph(
                x_conv, opt_data, line_color=GREEN, add_vertex_dots=False
            )
            label = Text(letters[i], color=WHITE, font_size=36).next_to(
                ax, UP, buff=0.1
            )
            option_graphs.add(VGroup(ax, graph, label))

        self.play(Write(question))
        self.play(
            LaggedStart(*[Create(opt) for opt in option_graphs], lag_ratio=0.3),
            run_time=2,
        )
        self.wait(3)

        # --- SAVE OUTPUTS ---

        # 1. Solution
        with open(f"solutions/convolution_{self.file_index}.txt", "w") as f:
            f.write(f"{correct_letter}")

        # 2. Question Text
        with open(f"question_text/convolution_{self.file_index}.txt", "w") as f:
            f.write(
                "Which is the correct convolution f(t) * g(t)? Output just the letter of the correct answer."
            )

        # 3. Reasoning Trace
        trace_text = self.generate_reasoning_trace(correct_letter)
        with open(f"reasoning_traces/convolution_{self.file_index}.txt", "w") as f:
            f.write(trace_text)

    def generate_reasoning_trace(self, correct_letter):
        # Extract parameters
        f1 = self.f1_meta
        f2 = self.f2_meta

        # Calculate expected theoretical center
        # Note: For exponential decay generated in this script, 'center' is the start point.
        # The mass is centered roughly at start + 1/lambda (lambda=1 here).
        # For symmetry, the code uses 'center' as the peak location for rect/tri/gauss.

        def get_effective_center(meta):
            if meta["type"] == "Exponential Decay":
                return meta["center"] + 1.0  # rough centroid shift for exp(-x)
            return meta["center"]

        c1 = get_effective_center(f1)
        c2 = get_effective_center(f2)
        expected_center = c1 + c2

        trace = []
        trace.append("step_1: Analyze the input functions.")
        trace.append(f"The scene presents two functions to be convolved.")
        trace.append(
            f"Function f(t) is a {f1['type']} positioned at approximately t = {f1['center']:.2f}."
        )
        trace.append(
            f"Function g(t) is a {f2['type']} positioned at approximately t = {f2['center']:.2f}."
        )

        trace.append("\nstep_2: Predict properties of the convolution.")
        trace.append(
            "Convolution involves sliding one function past the other and integrating the product."
        )
        trace.append(
            "A key property of convolution is the additivity of means (centers)."
        )
        trace.append(
            f"The center of the resulting function should be at the sum of the input centers: {c1:.2f} + {c2:.2f} = {expected_center:.2f}."
        )

        # Shape prediction
        shape_desc = ""
        if f1["type"] == "Rectangle" and f2["type"] == "Rectangle":
            shape_desc = "The convolution of two rectangles is a trapezoid (or triangle if widths are equal)."
        elif "Gaussian" in [f1["type"], f2["type"]]:
            shape_desc = "Since one input is a Gaussian, the output will be smoothed, resembling a broader Gaussian."
        elif "Exponential" in [f1["type"], f2["type"]]:
            shape_desc = "The presence of an exponential decay will result in a function with an asymmetric tail."
        trace.append(shape_desc)

        trace.append("\nstep_3: Evaluate the options.")

        # Sort options by letter to make the trace logical
        sorted_options = sorted(self.options_meta, key=lambda x: x["label"])

        for opt in sorted_options:
            label = opt["label"]
            if opt["type"] == "correct":
                trace.append(
                    f"Option {label}: This graph is centered near t = {expected_center:.2f} and has the expected amplitude/shape. This matches our prediction."
                )
            elif opt["type"] == "shifted":
                direction = "right" if opt["val"] > 0 else "left"
                trace.append(
                    f"Option {label}: This graph is shifted significantly to the {direction} compared to the expected center of {expected_center:.2f}. It is incorrect."
                )
            elif opt["type"] == "scaled":
                problem = (
                    "too tall (amplified)"
                    if opt["val"] > 1
                    else "too short (attenuated)"
                )
                trace.append(
                    f"Option {label}: While the position might be correct, the amplitude is {problem} compared to the expected area conservation. It is incorrect."
                )

        trace.append("\nstep_4: Conclusion.")
        trace.append(
            f"Based on the center position and shape analysis, Option {correct_letter} is the correct convolution."
        )

        return "\n".join(trace)


if __name__ == "__main__":
    # Generate 3 variations
    for i in range(3):
        # Clean up previous renders to ensure no conflict
        if os.path.exists("manim_output"):
            shutil.rmtree("manim_output")

        scene = ConvolutionQuiz(file_index=i)
        scene.render()

        # Locate output
        # Note: Manim default output structure can vary based on config.
        # We check the specific path defined by the 1080p60 folder.
        output = Path("manim_output/videos/1080p60/ConvolutionQuiz.mp4")

        if output.exists():
            shutil.move(str(output), f"questions/convolution_{i}.mp4")
        else:
            print(f"Warning: Could not find video output for index {i}")

        # Cleanup
        if os.path.exists("manim_output"):
            shutil.rmtree("manim_output")
