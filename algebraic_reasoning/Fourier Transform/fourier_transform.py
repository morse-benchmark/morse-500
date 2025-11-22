from manim import *
import numpy as np
import random
from pathlib import Path
import shutil
import os
from scipy.fft import fft, fftfreq

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


class FourierTransformQuiz(Scene):
    def __init__(self, file_index, **kwargs):
        super().__init__(**kwargs)
        self.file_index = file_index
        self.signal_params = self.generate_signal()
        self.correct_spectrum = self.calculate_spectrum()

        # To store reasoning info about options
        self.options_meta = []

    def generate_signal(self):
        components = []
        n_components = random.randint(2, 4)
        fundamental_freq = random.uniform(0.5, 2.0)

        for i in range(1, n_components + 1):
            freq = round(fundamental_freq * i, 2)
            amp = round(random.uniform(0.2, 1.0) / i, 2)
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
        yf_pos = (
            np.abs(yf[idx]) / n * 2
        )  # Multiply by 2 for single-sided spectrum amplitude normalization

        # Just return the arrays, we'll plot them
        return xf_pos, yf_pos

    def create_signal_graph(self, axes, t, signal, color, label):
        graph = axes.plot_line_graph(t, signal, line_color=color, add_vertex_dots=False)
        label_obj = Text(label, color=color, font_size=24).next_to(axes, UP, buff=0.1)
        return VGroup(axes, graph, label_obj)

    def create_spectrum_graph(self, axes, freqs, amps, color, label):
        # Filter for significant peaks for plotting clean lines
        max_freq = 10
        if self.signal_params["components"]:
            max_freq = max(c[0] for c in self.signal_params["components"]) * 1.5

        axes.x_range = [0, max(10, max_freq), 1]

        stems = VGroup()
        dots = VGroup()

        # We only plot indices where amp is significant to save rendering time/clutter
        indices = np.where((freqs <= axes.x_range[1]) & (amps > 0.05))[0]

        for idx in indices:
            f = freqs[idx]
            a = amps[idx]
            stem = Line(
                start=axes.c2p(f, 0), end=axes.c2p(f, a), stroke_width=3, color=color
            )
            dot = Dot(axes.c2p(f, a), color=color, radius=0.05)
            stems.add(stem)
            dots.add(dot)

        label_obj = Text(label, color=color, font_size=24).next_to(axes, UP, buff=0.1)
        return VGroup(axes, stems, dots, label_obj)

    def construct(self):
        # Generate Time Domain Data
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

        # --- PART 1: SHOW TIME DOMAIN ---
        time_axes = Axes(
            x_range=[0, self.signal_params["duration"], 1],
            y_range=[-2, 2, 0.5],
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

        # --- PART 2: GENERATE OPTIONS ---
        raw_options = []
        xf, yf = self.correct_spectrum

        # 1. Correct Option
        raw_options.append(
            {"freqs": xf, "amps": yf, "type": "correct", "desc": "Matches components"}
        )

        # 2. Distractors
        for _ in range(3):
            if random.random() > 0.5:
                # Shifted Frequencies
                shift = random.choice([-1, 1]) * random.uniform(0.5, 1.0)
                wrong_freqs = xf + shift
                wrong_freqs[wrong_freqs < 0] = 0  # Clamp

                raw_options.append(
                    {
                        "freqs": wrong_freqs,
                        "amps": yf,
                        "type": "shifted",
                        "desc": f"Frequencies shifted by {shift:.2f}Hz",
                    }
                )
            else:
                # Scaled Amplitudes
                scale = random.choice([0.5, 2.0])
                wrong_amps = yf * scale

                raw_options.append(
                    {
                        "freqs": xf,
                        "amps": wrong_amps,
                        "type": "scaled",
                        "desc": f"Amplitudes scaled by {scale}x",
                    }
                )

        random.shuffle(raw_options)
        letters = ["A", "B", "C", "D"]

        # Map letters to options
        self.options_meta = []
        for i, opt in enumerate(raw_options):
            opt["label"] = letters[i]
            self.options_meta.append(opt)
            if opt["type"] == "correct":
                correct_letter = letters[i]

        # --- PART 3: SHOW QUIZ ---
        self.play(FadeOut(title), FadeOut(subtitle), FadeOut(time_graph))

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

        max_freq_disp = max(c[0] for c in self.signal_params["components"]) * 1.5

        for i, pos in enumerate(positions):
            opt = self.options_meta[i]
            ax = Axes(
                x_range=[0, max_freq_disp, 1],
                y_range=[0, 1.5, 0.5],
                x_length=5,
                y_length=2.5,
            ).move_to(pos)

            graph = self.create_spectrum_graph(
                ax, opt["freqs"], opt["amps"], GREEN, opt["label"]
            )
            option_graphs.add(graph)

        self.play(Write(question))
        self.play(
            LaggedStart(*[Create(opt) for opt in option_graphs], lag_ratio=0.3),
            run_time=2,
        )
        self.wait(3)

        # --- SAVE OUTPUTS ---
        with open(f"solutions/fourier_{self.file_index}.txt", "w") as f:
            f.write(correct_letter)

        with open(f"question_text/fourier_{self.file_index}.txt", "w") as f:
            f.write(
                "Which spectrum matches the time-domain signal? Output just the letter of the correct answer."
            )

        trace = self.generate_reasoning_trace(correct_letter)
        with open(f"reasoning_traces/fourier_{self.file_index}.txt", "w") as f:
            f.write(trace)

    def generate_reasoning_trace(self, correct_letter):
        trace = []
        trace.append("=== Signal Analysis ===")
        trace.append(
            "The time-domain signal was constructed by summing the following sinusoidal components:"
        )

        # Sort by frequency for clarity
        comps = sorted(self.signal_params["components"], key=lambda x: x[0])
        for f, a, p in comps:
            trace.append(f"  - Frequency: {f:.2f} Hz, Amplitude: {a:.2f}")

        trace.append("\n=== Frequency Spectrum Prediction ===")
        trace.append(
            "The Fourier Transform decomposes a signal into its constituent frequencies."
        )
        trace.append(
            "Therefore, the correct magnitude spectrum should show distinct vertical spikes (peaks) at exactly the frequencies listed above."
        )
        trace.append(
            "The height of these spikes should be proportional to the amplitudes."
        )

        trace.append("\n=== Evaluating Options ===")
        # Sort options by label
        sorted_opts = sorted(self.options_meta, key=lambda x: x["label"])

        for opt in sorted_opts:
            label = opt["label"]
            desc = opt["desc"]
            if opt["type"] == "correct":
                trace.append(
                    f"Option {label}: This spectrum shows peaks at {', '.join([str(c[0]) for c in comps])} Hz with correct relative heights. This matches our prediction."
                )
            elif opt["type"] == "shifted":
                trace.append(
                    f"Option {label}: The peaks in this spectrum are shifted along the x-axis (frequency). They do not align with the components of the signal. ({desc})"
                )
            elif opt["type"] == "scaled":
                trace.append(
                    f"Option {label}: The frequencies are correct, but the heights (amplitudes) are wrong. ({desc})"
                )

        trace.append("\n=== Conclusion ===")
        trace.append(
            f"Option {correct_letter} is the only graph that accurately represents the frequency content of the generated signal."
        )

        return "\n".join(trace)


if __name__ == "__main__":
    for i in range(3):
        if os.path.exists("manim_output"):
            shutil.rmtree("manim_output")

        scene = FourierTransformQuiz(file_index=i)
        scene.render()

        output = Path("manim_output/videos/1080p60/FourierTransformQuiz.mp4")
        if output.exists():
            shutil.move(str(output), f"questions/fourier_{i}.mp4")
        else:
            # Fallback search
            found = list(Path("manim_output/videos").rglob("*.mp4"))
            if found:
                shutil.move(str(found[0]), f"questions/fourier_{i}.mp4")

        if os.path.exists("manim_output"):
            shutil.rmtree("manim_output")
