from manim import *
import numpy as np
import random
from pathlib import Path
import shutil
import os
from scipy.fft import fft, fftfreq


Path("questions").mkdir(exist_ok=True)
Path("solutions").mkdir(exist_ok=True)
Path("question_text").mkdir(exist_ok=True)

config.media_dir = "manim_output"
config.verbosity = "WARNING"
config.pixel_height = 1080
config.pixel_width = 1920
config.frame_rate = 60


class FourierTransformQuiz(Scene):
    def __init__(self, file_index, **kwargs):
        super().__init__(**kwargs)
        self.file_index = file_index
        self.signal_params = self.generate_signal()
        self.correct_spectrum = self.calculate_spectrum()

    def generate_signal(self):

        components = []
        n_components = random.randint(2, 4)
        fundamental_freq = random.uniform(0.5, 2.0)

        for i in range(1, n_components + 1):
            freq = fundamental_freq * i
            amp = random.uniform(0.2, 1.0) / i
            phase = random.uniform(0, 2 * np.pi)
            components.append((freq, amp, phase))

        return {"components": components, "duration": 4, "sample_rate": 100}

    def calculate_spectrum(self):

        t = np.linspace(
            0,
            self.signal_params["duration"],
            self.signal_params["duration"] * self.signal_params["sample_rate"],
            endpoint=False,
        )
        signal = sum(
            amp * np.sin(2 * np.pi * freq * t + phase)
            for freq, amp, phase in self.signal_params["components"]
        )

        n = len(t)
        yf = fft(signal)
        xf = fftfreq(n, 1 / self.signal_params["sample_rate"])

        idx = np.where(xf >= 0)
        xf_pos = xf[idx]
        yf_pos = np.abs(yf[idx]) / n

        sorted_indices = np.argsort(yf_pos)[::-1]
        top_indices = sorted_indices[:10]

        return xf_pos[top_indices], yf_pos[top_indices]

    def create_signal_graph(self, axes, t, signal, color, label):

        graph = axes.plot_line_graph(t, signal, line_color=color, add_vertex_dots=False)
        label = Text(label, color=color, font_size=24).next_to(axes, UP, buff=0.1)
        return VGroup(axes, graph, label)

    def create_spectrum_graph(self, axes, freqs, amps, color, label):

        max_freq = max(freqs) if len(freqs) > 0 else 1
        axes.x_range = [0, max(10, max_freq * 1.2), max(2, int(max_freq / 2))]

        stems = VGroup()
        for f, a in zip(freqs, amps):
            stem = Line(
                start=axes.c2p(f, 0), end=axes.c2p(f, a), stroke_width=3, color=color
            )
            stems.add(stem)

        dots = VGroup(
            *[
                Dot(axes.c2p(f, a), color=color, radius=0.05)
                for f, a in zip(freqs, amps)
            ]
        )

        label = Text(label, color=color, font_size=24).next_to(axes, UP, buff=0.1)
        return VGroup(axes, stems, dots, label)

    def construct(self):

        t = np.linspace(
            0,
            self.signal_params["duration"],
            self.signal_params["duration"] * self.signal_params["sample_rate"],
            endpoint=False,
        )
        signal = sum(
            amp * np.sin(2 * np.pi * freq * t + phase)
            for freq, amp, phase in self.signal_params["components"]
        )

        time_axes = Axes(
            x_range=[0, self.signal_params["duration"], 1],
            y_range=[-1.5, 1.5, 0.5],
            x_length=10,
            y_length=3,
        ).shift(DOWN * 1.5)

        time_graph = self.create_signal_graph(
            time_axes, t, signal, BLUE, "Time Domain Signal"
        )

        title = Text("Fourier Transform Quiz", font_size=32).to_edge(UP, buff=0.5)
        subtitle = Text(
            "Identify the correct frequency spectrum", font_size=24
        ).next_to(title, DOWN)

        self.play(Write(title), Write(subtitle))
        self.play(Create(time_graph), run_time=2)
        self.wait(2)

        options = []
        xf, yf = self.correct_spectrum
        for _ in range(3):
            if random.random() > 0.5:

                shift = random.choice([-1, 1]) * random.uniform(0.2, 0.5)
                wrong_freqs = xf + shift
                wrong_freqs[wrong_freqs < 0] = 0
                wrong_amps = yf
            else:

                wrong_freqs = xf
                scale = random.uniform(0.5, 1.5)
                wrong_amps = yf * scale

            options.append((wrong_freqs, wrong_amps))

        correct_index = random.randint(0, len(options))
        options.insert(correct_index, (xf, yf))
        letters = ["A", "B", "C", "D"]

        self.play(
            FadeOut(title),
            FadeOut(subtitle),
            FadeOut(time_graph),
        )

        question = (
            VGroup(
                Text("Which spectrum matches the time-domain signal?", font_size=28),
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

        for i, ((freqs, amps), pos) in enumerate(zip(options, positions)):

            max_freq = max(freqs) if len(freqs) > 0 else 1
            x_max = max(10, max_freq * 1.2)

            ax = Axes(
                x_range=[0, x_max, max(2, int(x_max / 5))],
                y_range=[0, 1.2, 0.2],
                x_length=5,
                y_length=2.5,
            ).move_to(pos)

            graph = self.create_spectrum_graph(ax, freqs, amps, GREEN, letters[i])
            option_graphs.add(graph)

        self.play(Write(question))
        self.play(
            LaggedStart(*[Create(opt) for opt in option_graphs], lag_ratio=0.3),
            run_time=2,
        )
        self.wait(3)

        with open(f"solutions/fourier_{self.file_index}.txt", "w") as f:
            f.write(f"{letters[correct_index]}")
        with open(f"question_text/fourier_{self.file_index}.txt", "w") as f:
            f.write(
                f"Which spectrum matches the time-domain signal? Output just the letter of the correct answer."
            )


for i in range(3):
    if os.path.exists("manim_output"):
        shutil.rmtree("manim_output")

    scene = FourierTransformQuiz(file_index=i)
    scene.render()

    output = Path("manim_output/videos/1080p60/FourierTransformQuiz.mp4")
    if output.exists():
        shutil.move(str(output), f"questions/fourier_{i}.mp4")

    if os.path.exists("manim_output"):
        shutil.rmtree("manim_output")
