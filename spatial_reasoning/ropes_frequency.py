from manim import *
import random
import numpy as np
import os
import shutil
from pathlib import Path

# ============================================================================
# FREQUENCY-PARAMETERIZED VERSION - Global Size Parameter (0.0 to 1.0)
# ============================================================================
# This version includes a FREQUENCY parameter that controls the number of ropes
# and their complexity (bends), making the problem harder as size increases.
#
# FREQUENCY parameter mapping:
# - 0.0: 2 ropes with 1 bend each (simple, easy to count/track)
# - 0.5: 5 ropes with 3 bends each (medium complexity)
# - 1.0: 8 ropes with 5 bends each (very complex, harder to analyze)
# ============================================================================

# ============================================================================
# Setup directories for output files
# ============================================================================
Path("questions").mkdir(exist_ok=True)
Path("solutions").mkdir(exist_ok=True)
Path("question_text").mkdir(exist_ok=True)
Path("reasoning_traces").mkdir(exist_ok=True)

# ============================================================================
# Manim configuration
# ============================================================================
config.media_dir = "manim_output"
config.verbosity = "WARNING"
config.pixel_height = 1080
config.pixel_width = 1920
config.frame_rate = 30
config.preview = False

# ============================================================================
# Try to import shapely for accurate geometric calculations
# ============================================================================
try:
    from shapely.geometry import LineString
    from shapely.ops import unary_union, polygonize
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False
    print("Warning: Shapely not available. Some geometric calculations may be approximate.")

def ordinal(n):
    """Convert integer to ordinal string (e.g., 1 -> '1st', 2 -> '2nd')"""
    if 11 <= (n % 100) <= 13:
        return f"{n}th"
    return f"{n}{ {1:'st', 2:'nd', 3:'rd'}.get(n%10, 'th') }"


class RopeCutSceneFrequency(Scene):
    """
    FREQUENCY-PARAMETERIZED version of RopeCutScene.

    A scene that generates rope/line intersection puzzles with adjustable complexity.

    The FREQUENCY parameter (0.0 to 1.0) controls both the number of ropes and their
    curve complexity, making analysis progressively harder.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Set random seed for reproducibility
        self.seed = random.randint(1000, 9999)

        # ====================================================================
        # FREQUENCY PARAMETER (0.0 to 1.0) - VISUAL COMPLEXITY CONTROL
        # ====================================================================
        self.frequency_param = float(os.getenv("FREQUENCY", 0.5))  # Default to medium

        # Clamp to valid range
        self.frequency_param = max(0.0, min(1.0, self.frequency_param))

        # FREQUENCY controls wave/oscillation frequency (NOT number of ropes)
        # Number of ropes is FIXED, wave complexity varies
        # FREQUENCY 0.0 (coarse) → gentle curves, low oscillation rate
        # FREQUENCY 0.5 (medium) → standard wave complexity
        # FREQUENCY 1.0 (fine)   → rapid oscillations, high frequency waves

        # Fixed number of ropes for FREQUENCY variant (same as DENSITY=0.5)
        self.num_ropes = 6  # Fixed at medium count

        # FREQUENCY controls bend/wave frequency (1 to 8 oscillations)
        # Higher frequency = more rapid bends/waves
        self.bends_per_rope = int(1 + (self.frequency_param * 7))  # 1 to 8

        # Fixed drawing area scale - stays constant
        self.drawing_area_scale = 0.75  # 75% of screen area

        # Load other parameters from environment
        self.p_type = os.getenv("P_TYPE", "count")

        # Define valid colors
        self.VALID_COLORS = {
            "blue": BLUE,
            "red": RED,
            "orange": ORANGE,
            "green": GREEN,
            "yellow": YELLOW,
            "purple": PURPLE,
            "white": WHITE,
        }

        # Question templates
        self.cfg = {
            "text": {
                "count": [
                    "How many lines are drawn in the video?\nAnswer with a single integer."
                ],
                "cut": [
                    "How many sections does the dotted line cut the colored lines into?\nAnswer with a single integer."
                ],
                "closed": [
                    "How many closed shapes do the intersecting lines create?\nAnswer with a single integer."
                ],
                "order": [
                    "What was the color of the <N> line to be drawn?\nAnswer with only the color name."
                ]
            }
        }

        # Initialize timing and reasoning trace
        self.scene_events = []

        self.reasoning_metadata = {
            "problem_type": self.p_type,
            "num_ropes": self.num_ropes,
            "bends_per_rope": self.bends_per_rope,
            "seed": self.seed,
            "shapely_available": SHAPELY_AVAILABLE,
            "frequency_param": self.frequency_param
        }

    def log_event(self, description):
        """Log a scene event with video timestamp."""
        current_time = self.renderer.time

        self.scene_events.append({
            'time': current_time,
            'description': description
        })

    def format_time(self, seconds):
        """Format seconds as M:SS for display in reasoning trace."""
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}:{secs:02d}"

    def show_colors(self, colors, names):
        """Display color legend at the beginning of the video."""
        squares1 = VGroup(
            *[Square(0.5).set_fill(col, 1).set_stroke(width=0) for col in colors[:4]]
        )
        squares1.arrange(DOWN, buff=0.3, aligned_edge=LEFT).to_edge(LEFT, buff=2)

        labels1 = VGroup(*[Text(nm, font_size=20) for nm in names[:4]])
        for sq, lbl in zip(squares1, labels1):
            lbl.next_to(sq, RIGHT, buff=0.8)

        arrows1 = VGroup(
            *[
                Arrow(
                    start=sq.get_right(), end=lbl.get_left(), buff=0.05, stroke_width=2
                )
                for sq, lbl in zip(squares1, labels1)
            ]
        )

        squares2 = VGroup(
            *[Square(0.5).set_fill(col, 1).set_stroke(width=0) for col in colors[4:]]
        )
        squares2.arrange(DOWN, buff=0.3, aligned_edge=LEFT).to_edge(LEFT, buff=2)

        labels2 = VGroup(*[Text(nm, font_size=20) for nm in names[4:]])
        for sq, lbl in zip(squares2, labels2):
            lbl.next_to(sq, RIGHT, buff=0.8)

        arrows2 = VGroup(
            *[
                Arrow(
                    start=sq.get_right(), end=lbl.get_left(), buff=0.05, stroke_width=2
                )
                for sq, lbl in zip(squares2, labels2)
            ]
        )

        left = VGroup(squares1, arrows1, labels1)
        right = VGroup(squares2, arrows2, labels2)
        both = VGroup(left, right)
        both.arrange(buff=1.7, aligned_edge=UP)
        both.move_to(ORIGIN)

        title = Text("Remember the following color names", font_size=32)
        title.to_edge(UP)

        self.log_event("Title appears: 'Remember the following color names'")
        self.play(Write(title))

        self.log_event("Color legend begins to display")
        self.play(
            Succession(
                FadeIn(squares1, squares2),
                AnimationGroup(*([GrowArrow(ar) for ar in arrows1]+[GrowArrow(ar) for ar in arrows2]), lag_ratio=0.1),
                FadeIn(labels1, labels2)
            ),
            run_time=1.5
        )
        self.log_event(f"Color legend fully displayed with {len(colors)} colors: {', '.join(names)}")

        self.wait(2)

        self.log_event("Color legend fades out")
        self.play(FadeOut(title, both))
        self.wait(0.5)

    def count_closed_shapes_fallback(self, ropes):
        """Fallback method when shapely is not available."""
        self.log_event("Using fallback method for counting closed shapes (Shapely not available)")

        intersection_count = 0
        for i, rope1 in enumerate(ropes):
            for j, rope2 in enumerate(ropes[i+1:], i+1):
                samples1 = [rope1.point_from_proportion(t)[:2] for t in np.linspace(0, 1, 50)]
                samples2 = [rope2.point_from_proportion(t)[:2] for t in np.linspace(0, 1, 50)]

                for p1 in samples1:
                    for p2 in samples2:
                        if np.linalg.norm(np.array(p1) - np.array(p2)) < 0.1:
                            intersection_count += 1
                            break

        return max(0, intersection_count - len(ropes) + 1)

    def construct(self):
        """Main scene construction method."""
        random.seed(self.seed)

        # Create animated gradient background
        bg = (
            Rectangle(height=config.frame_height, width=config.frame_width)
            .set_color(
                color_gradient([random_bright_color(), random_bright_color()], 5)
            )
            .set_opacity(0.6)
            .set_z_index(-2)
        )
        self.add(bg)

        # Show color legend (only for "order" problem type)
        if self.p_type == "order":
            self.show_colors(list(self.VALID_COLORS.values()), list(self.VALID_COLORS.keys()))

        # Display initial prompt
        prompt = Text(
            "Observe the following scene", color=WHITE, font_size=36
        ).move_to(ORIGIN)

        self.log_event("Initial prompt appears: 'Observe the following scene'")
        self.play(FadeIn(prompt), run_time=0.5)
        self.wait(1.5)

        self.log_event("Initial prompt fades out")
        self.play(FadeOut(prompt), run_time=0.5)
        self.wait(1)

        # Randomize color order
        colors = list(self.VALID_COLORS.keys())
        random.shuffle(colors)
        self.color_sequence = colors

        # Calculate FIXED drawing area bounds (same across all density levels)
        frame_width = config.frame_width
        frame_height = config.frame_height

        # Fixed physical area - density controls how many ropes pack into this space
        area_width = frame_width * self.drawing_area_scale
        area_height = frame_height * self.drawing_area_scale
        x_min, x_max = -area_width / 2, area_width / 2
        y_min, y_max = -area_height / 2, area_height / 2

        # Buffer zones for random positioning
        h_buf = area_width / 2
        v_buf = area_height / 2

        # ====================================================================
        # Generate ropes with FREQUENCY-DEPENDENT parameters
        # Higher density = more ropes in the SAME fixed area
        # ====================================================================
        ropes = []
        lengths = []
        self.rope_details = []

        self.log_event(f"Beginning to generate {self.num_ropes} ropes with FREQUENCY={self.frequency_param:.2f} (bends={self.bends_per_rope}) in fixed area")

        for i in range(self.num_ropes):
            start = np.array(
                [random.uniform(x_min, x_max), random.uniform(y_min, y_max), 0]
            )
            end = np.array(
                [random.uniform(x_min, x_max), random.uniform(y_min, y_max), 0]
            )

            length = np.sqrt(np.sum((start - end) ** 2))
            lengths.append(length)

            # SIZE-DEPENDENT: More bends = more complex curves
            bend_points = [
                np.array(
                    [
                        np.interp(j, [0, self.bends_per_rope + 1], [x_min, x_max]),
                        random.uniform(y_min, y_max),
                        0,
                    ]
                )
                for j in range(1, self.bends_per_rope + 1)
            ]

            pts = [start, *bend_points, end]

            color_name = colors[i % len(colors)]
            color = self.VALID_COLORS[color_name]

            rope = VMobject()
            rope.set_points_smoothly(pts)
            rope.set_stroke(color, 3)
            ropes.append(rope)

            self.rope_details.append({
                'index': i + 1,
                'color': color_name,
                'length': length,
                'start': start[:2],
                'end': end[:2],
                'bend_points': [p[:2] for p in bend_points],
                'num_bends': len(bend_points)
            })

        # Position rope group - keep in FIXED area (no scaling beyond generation)
        # The drawing area was already fixed during generation
        # This maintains consistent physical space across all density levels
        rope_group = VGroup(*ropes)
        rope_group.move_to(ORIGIN)

        self.log_event(f"Ropes positioned in fixed drawing area ({area_width:.1f}×{area_height:.1f})")

        # Animate ropes being drawn
        self.log_event("Ropes begin drawing on screen")
        self.play(Create(rope_group, run_time=2))
        self.log_event(f"All {self.num_ropes} ropes have been drawn")

        # Calculate answer based on problem type
        N = random.randint(1, self.num_ropes)

        if self.p_type == "count":
            self.answer = self.num_ropes
            self.log_event(f"Answer calculated: {self.answer} lines drawn")

        elif self.p_type == "closed":
            self.log_event("Beginning calculation of closed shapes")

            if SHAPELY_AVAILABLE:
                line_strings = []
                for i, rope in enumerate(ropes):
                    samples = [
                        tuple(rope.point_from_proportion(t)[:2])
                        for t in np.linspace(0, 1, 200)
                    ]
                    line_strings.append(LineString(samples))

                merged = unary_union(line_strings)
                polygons = list(polygonize(merged))

                num_closed_shapes = len(polygons)
                self.answer = num_closed_shapes
                self.log_event(f"Using Shapely: found {self.answer} closed shapes from {len(line_strings)} line segments")
            else:
                self.answer = self.count_closed_shapes_fallback(ropes)
                self.log_event(f"Using fallback method: estimated {self.answer} closed shapes")

        elif self.p_type == "cut":
            self.log_event("Generating cutting line")

            angle = random.uniform(0, TAU)
            x_off = random.uniform(-h_buf, h_buf)
            y_off = random.uniform(-v_buf, v_buf)
            center_pt = np.array([x_off, y_off, 0])

            diag = np.hypot(frame_width, frame_height)
            half_len = diag / 2
            direction = np.array([np.cos(angle), np.sin(angle), 0])

            start_line = center_pt - direction * half_len
            end_line = center_pt + direction * half_len

            dashed_line = DashedLine(start_line, end_line).set_color(WHITE)

            self.log_event(f"Cutting line created with angle {np.degrees(angle):.1f}°")
            self.play(Create(dashed_line))
            self.log_event("Cutting line fully drawn")

            self.cut_line_details = {
                'angle_radians': angle,
                'angle_degrees': np.degrees(angle),
                'center': center_pt[:2],
                'direction': direction[:2]
            }

            total_sections = 0
            self.section_details = []

            for i, rope in enumerate(ropes):
                ts = np.linspace(0, 1, 300)
                vals = []

                for t in ts:
                    p = rope.point_from_proportion(t)
                    val = direction[0] * (p[1] - center_pt[1]) - direction[1] * (
                        p[0] - center_pt[0]
                    )
                    vals.append(val)

                signs = np.sign(vals)
                crossings = sum(abs(np.diff(signs)) > 0)

                sections = crossings + 1
                total_sections += sections

                self.section_details.append({
                    'rope_index': i + 1,
                    'color': colors[i % len(colors)],
                    'crossings': crossings,
                    'sections': sections
                })

            self.answer = total_sections
            self.log_event(f"Calculated total sections: {self.answer}")

        elif self.p_type == "order":
            self.answer = colors[N - 1]
            self.ordinal_position = N
            self.log_event(f"The {ordinal(N)} rope color is: {self.answer}")

        # Display final question
        self.wait(3)

        self.log_event("All scene elements begin to fade out")
        self.play(*[FadeOut(mob) for mob in self.mobjects if mob != bg])
        self.log_event("Scene cleared, only background remains")

        title_text = random.choice(self.cfg["text"][self.p_type])
        title_text = title_text.replace("<N>", ordinal(N))

        lines = title_text.split('\n')
        para = Paragraph(
            *lines, alignment="center", font_size=36, line_spacing=0.8
        )
        para.move_to(ORIGIN)

        if para.width > 0.9*config.frame_width:
            para.scale_to_fit_width(config.frame_width * 0.9)

        self.log_event("Question text appears on screen")
        self.play(Write(para), run_time=1.5)
        self.wait(3)
        self.log_event("Question remains visible for user to read")

        self.question_text = f"Observe the following scene. {title_text.replace(chr(10), ' ')}"

        # Generate reasoning trace
        self.build_reasoning_trace()

        # Save output files
        with open(f"solutions/ropes_frequency_{self.p_type}_density{self.frequency_param:.2f}_seed{self.seed}.txt", "w") as f:
            f.write(str(self.answer))

        with open(f"question_text/ropes_frequency_{self.p_type}_density{self.frequency_param:.2f}_seed{self.seed}.txt", "w") as f:
            f.write(self.question_text)

        with open(f"reasoning_traces/ropes_frequency_{self.p_type}_density{self.frequency_param:.2f}_seed{self.seed}.txt", "w") as f:
            f.write("\n".join(self.reasoning_trace))

    def build_reasoning_trace(self):
        """Build a comprehensive, step-by-step reasoning trace."""
        self.reasoning_trace = []

        self.reasoning_trace.append("**Question:** " + self.question_text)
        self.reasoning_trace.append("")
        self.reasoning_trace.append(f"**FREQUENCY Parameter:** {self.frequency_param:.2f} (wave/oscillation frequency: {self.bends_per_rope} bends per rope)")
        self.reasoning_trace.append("")
        self.reasoning_trace.append("Let's solve this step by step.")
        self.reasoning_trace.append("")

        # Scene description with timestamps
        self.reasoning_trace.append("### Scene Description")
        self.reasoning_trace.append("")
        self.reasoning_trace.append("The video shows the following sequence of events:")
        self.reasoning_trace.append("")

        for event in self.scene_events:
            time_str = self.format_time(event['time'])
            self.reasoning_trace.append(f"At {time_str}, {event['description']}")

        self.reasoning_trace.append("")

        # Problem setup
        self.reasoning_trace.append("### Problem Setup")
        self.reasoning_trace.append("")
        self.reasoning_trace.append(f"- **Problem Type**: {self.p_type}")
        self.reasoning_trace.append(f"- **Number of Ropes**: {self.num_ropes}")
        self.reasoning_trace.append(f"- **Bends per Rope**: {self.bends_per_rope}")
        self.reasoning_trace.append(f"- **Random Seed**: {self.seed}")
        self.reasoning_trace.append("")

        # Detailed rope information
        self.reasoning_trace.append("### Step 1: Understand the Ropes")
        self.reasoning_trace.append("")
        self.reasoning_trace.append(f"The scene contains {self.num_ropes} colored lines (ropes) drawn in the following order:")
        self.reasoning_trace.append("")

        for detail in self.rope_details:
            self.reasoning_trace.append(f"**Rope {detail['index']}** ({detail['color']}):")
            self.reasoning_trace.append(f"  - Number of curve control points: {detail['num_bends']}")
            self.reasoning_trace.append("")

        # Problem-specific reasoning
        if self.p_type == "count":
            self.reasoning_trace.append("### Step 2: Count the Lines")
            self.reasoning_trace.append("")
            self.reasoning_trace.append(f"The total count is **{self.answer}** lines.")

        # Final answer
        self.reasoning_trace.append("")
        self.reasoning_trace.append("### Final Answer")
        self.reasoning_trace.append("")
        self.reasoning_trace.append(f"\\boxed{{{self.answer}}}")


# ============================================================================
# Main execution
# ============================================================================
if __name__ == "__main__":
    scene = RopeCutSceneFrequency()
    scene.render()

    output = Path("manim_output/videos/1080p30/RopeCutSceneFrequency.mp4")
    if output.exists():
        filename = f"ropes_frequency_{scene.p_type}_frequency{scene.frequency_param:.2f}_seed{scene.seed}.mp4"
        shutil.move(str(output), f"questions/{filename}")
        print(f"✓ Video saved: questions/{filename}")
        print(f"✓ FREQUENCY parameter: {scene.frequency_param:.2f} (wave frequency: {scene.bends_per_rope} bends per rope)")
    else:
        print("Error: Expected output file not found")

    if os.path.exists("manim_output"):
        shutil.rmtree("manim_output")
        print("✓ Temporary files cleaned up")
