from manim import *
import random
import numpy as np
import os
import shutil
from pathlib import Path

# ============================================================================
# Setup directories for output files
# ============================================================================
Path("questions").mkdir(exist_ok=True)  # Video files
Path("solutions").mkdir(exist_ok=True)  # Answer text files
Path("question_text").mkdir(exist_ok=True)  # Question text files
Path("reasoning_traces").mkdir(exist_ok=True)  # Step-by-step reasoning

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
# Shapely provides robust computational geometry algorithms
# If unavailable, we fall back to approximation methods
try:
    from shapely.geometry import LineString
    from shapely.ops import unary_union, polygonize

    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False
    print(
        "Warning: Shapely not available. Some geometric calculations may be approximate."
    )


# ============================================================================
# Helper function for ordinal numbers (1st, 2nd, 3rd, etc.)
# ============================================================================
def ordinal(n):
    """Convert integer to ordinal string (e.g., 1 -> '1st', 2 -> '2nd')"""
    if 11 <= (n % 100) <= 13:
        return f"{n}th"
    return f"{n}{ {1:'st', 2:'nd', 3:'rd'}.get(n%10, 'th') }"


class RopeCutScene(Scene):
    """
    A scene that generates rope/line intersection puzzles:
    - Shows colored lines being drawn on screen
    - Optionally shows a cutting line through the ropes
    - User must count lines, sections, closed shapes, or identify order
    - Generates question video, solution, and detailed reasoning trace
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # ====================================================================
        # Set random seed for reproducibility
        # ====================================================================
        # This ensures the same seed produces identical puzzles
        self.seed = os.getenv("SEED", random.randint(1000, 9999))

        # ====================================================================
        # Load parameters from environment variables
        # ====================================================================
        self.p_type = os.getenv(
            "P_TYPE", "count"
        )  # Problem type: count/cut/closed/order
        self.num_ropes = int(os.getenv("NUM_ROPES", 3))  # Number of lines to draw
        self.bends_per_rope = int(
            os.getenv("BENDS_PER_ROPE", 2)
        )  # Curvature control points

        # ====================================================================
        # Define valid colors for the ropes
        # ====================================================================
        # Using distinct, easily distinguishable colors
        self.VALID_COLORS = {
            "blue": BLUE,
            "red": RED,
            "orange": ORANGE,
            "green": GREEN,
            "yellow": YELLOW,
            "purple": PURPLE,
            "white": WHITE,
        }

        # ====================================================================
        # Question templates for different problem types
        # ====================================================================
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
                ],
            }
        }

        # ====================================================================
        # Initialize timing and reasoning trace
        # ====================================================================
        # Track scene events with video timestamps for detailed reasoning
        self.scene_events = []

        # Store metadata for reasoning trace
        self.reasoning_metadata = {
            "problem_type": self.p_type,
            "num_ropes": self.num_ropes,
            "bends_per_rope": self.bends_per_rope,
            "seed": self.seed,
            "shapely_available": SHAPELY_AVAILABLE,
        }

    def log_event(self, description):
        """
        Log a scene event with video timestamp.
        Uses Manim's internal renderer.time which tracks actual video playback time,
        not wall-clock execution time.

        Args:
            description: Human-readable description of the event
        """
        # Get current video time from Manim's renderer
        # self.renderer.time tracks the cumulative duration of all animations/waits
        current_time = self.renderer.time

        self.scene_events.append({"time": current_time, "description": description})

    def format_time(self, seconds):
        """
        Format seconds as M:SS for display in reasoning trace.

        Args:
            seconds: Time in seconds (float)

        Returns:
            String formatted as "M:SS" (e.g., "2:37")
        """
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}:{secs:02d}"

    def show_colors(self, colors, names):
        """
        Display color legend at the beginning of the video.
        Shows colored squares with arrows pointing to their names.
        This is crucial for the "order" problem type where users must remember colors.

        Args:
            colors: List of Manim color objects
            names: List of color name strings
        """
        # ====================================================================
        # Create first column of color swatches (first 4 colors)
        # ====================================================================
        squares1 = VGroup(
            *[Square(0.5).set_fill(col, 1).set_stroke(width=0) for col in colors[:4]]
        )
        squares1.arrange(DOWN, buff=0.3, aligned_edge=LEFT).to_edge(LEFT, buff=2)

        # Position labels to the right of each square
        labels1 = VGroup(*[Text(nm, font_size=20) for nm in names[:4]])
        for sq, lbl in zip(squares1, labels1):
            lbl.next_to(sq, RIGHT, buff=0.8)

        # Create arrows connecting squares to labels for clear association
        arrows1 = VGroup(
            *[
                Arrow(
                    start=sq.get_right(), end=lbl.get_left(), buff=0.05, stroke_width=2
                )
                for sq, lbl in zip(squares1, labels1)
            ]
        )

        # ====================================================================
        # Create second column of color swatches (remaining colors)
        # ====================================================================
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

        # ====================================================================
        # Arrange both columns side by side
        # ====================================================================
        left = VGroup(squares1, arrows1, labels1)
        right = VGroup(squares2, arrows2, labels2)
        both = VGroup(left, right)
        both.arrange(buff=1.7, aligned_edge=UP)
        both.move_to(ORIGIN)

        # Add title
        title = Text("Remember the following color names", font_size=32)
        title.to_edge(UP)

        # ====================================================================
        # Animate the color legend
        # ====================================================================
        self.log_event("Title appears: 'Remember the following color names'")
        self.play(Write(title))

        self.log_event("Color legend begins to display")
        self.play(
            Succession(
                FadeIn(squares1, squares2),
                AnimationGroup(
                    *(
                        [GrowArrow(ar) for ar in arrows1]
                        + [GrowArrow(ar) for ar in arrows2]
                    ),
                    lag_ratio=0.1,
                ),
                FadeIn(labels1, labels2),
            ),
            run_time=1.5,
        )
        self.log_event(
            f"Color legend fully displayed with {len(colors)} colors: {', '.join(names)}"
        )

        self.wait(2)

        # Clean up the legend
        self.log_event("Color legend fades out")
        self.play(FadeOut(title, both))
        self.wait(0.5)

    def count_closed_shapes_fallback(self, ropes):
        """
        Fallback method when shapely is not available.
        Uses proximity-based intersection detection to estimate closed shapes.

        This is an approximation and may not be as accurate as shapely's
        polygonization algorithm.

        Args:
            ropes: List of VMobject rope curves

        Returns:
            Estimated number of closed shapes
        """
        self.log_event(
            "Using fallback method for counting closed shapes (Shapely not available)"
        )

        # Simple approximation: count potential intersections
        intersection_count = 0
        for i, rope1 in enumerate(ropes):
            for j, rope2 in enumerate(ropes[i + 1 :], i + 1):
                # Sample points along each rope
                samples1 = [
                    rope1.point_from_proportion(t)[:2] for t in np.linspace(0, 1, 50)
                ]
                samples2 = [
                    rope2.point_from_proportion(t)[:2] for t in np.linspace(0, 1, 50)
                ]

                # Simple proximity-based intersection detection
                # If two sample points are very close, consider it an intersection
                for p1 in samples1:
                    for p2 in samples2:
                        if np.linalg.norm(np.array(p1) - np.array(p2)) < 0.1:
                            intersection_count += 1
                            break

        # Very rough approximation using Euler's formula: V - E + F = 2
        # closed shapes ≈ intersections - ropes + 1
        return max(0, intersection_count - len(ropes) + 1)

    def construct(self):
        """
        Main scene construction method.
        This is called by Manim to build and render the entire scene.
        """
        # ====================================================================
        # Use the seed set in __init__ for reproducible randomness
        # ====================================================================
        random.seed(self.seed)

        # ====================================================================
        # Create animated gradient background
        # ====================================================================
        # Random bright colors make each video visually unique
        bg = (
            Rectangle(height=config.frame_height, width=config.frame_width)
            .set_color(
                color_gradient([random_bright_color(), random_bright_color()], 5)
            )
            .set_opacity(0.6)
            .set_z_index(-2)  # Ensure it stays behind all other objects
        )
        self.add(bg)

        # ====================================================================
        # Show color legend (only for "order" problem type)
        # ====================================================================
        # For "order" questions, users need to remember which color was drawn when
        if self.p_type == "order":
            self.show_colors(
                list(self.VALID_COLORS.values()), list(self.VALID_COLORS.keys())
            )

        # ====================================================================
        # Display initial prompt
        # ====================================================================
        prompt = Text("Observe the following scene", color=WHITE, font_size=36).move_to(
            ORIGIN
        )

        self.log_event("Initial prompt appears: 'Observe the following scene'")
        self.play(FadeIn(prompt), run_time=0.5)
        self.wait(1.5)

        self.log_event("Initial prompt fades out")
        self.play(FadeOut(prompt), run_time=0.5)
        self.wait(1)

        # ====================================================================
        # Randomize color order for drawing ropes
        # ====================================================================
        colors = list(self.VALID_COLORS.keys())
        random.shuffle(colors)
        self.color_sequence = colors  # Store for reasoning trace

        # ====================================================================
        # Calculate screen bounds for rope generation
        # ====================================================================
        frame_width = config.frame_width
        frame_height = config.frame_height
        x_min, x_max = -frame_width / 2, frame_width / 2
        y_min, y_max = -frame_height / 2, frame_height / 2

        # ====================================================================
        # Generate ropes with random paths
        # ====================================================================
        ropes = []
        lengths = []
        self.rope_details = []  # Store details for reasoning trace

        self.log_event(f"Beginning to generate {self.num_ropes} ropes")

        for i in range(self.num_ropes):
            # Random start and end points within screen bounds
            start = np.array(
                [random.uniform(x_min, x_max), random.uniform(y_min, y_max), 0]
            )
            end = np.array(
                [random.uniform(x_min, x_max), random.uniform(y_min, y_max), 0]
            )

            # Calculate length for potential use in reasoning
            length = np.sqrt(np.sum((start - end) ** 2))
            lengths.append(length)

            # Generate bend points to create curved paths
            # These are distributed horizontally to create smooth curves
            bend_points = [
                np.array(
                    [
                        # x progression from left to right across screen
                        np.interp(j, [0, self.bends_per_rope + 1], [x_min, x_max]),
                        random.uniform(y_min, y_max),  # Random y position
                        0,
                    ]
                )
                for j in range(1, self.bends_per_rope + 1)
            ]

            # Assemble full point list: start -> bend points -> end
            pts = [start, *bend_points, end]

            # Get color for this rope
            color_name = colors[i % len(colors)]
            color = self.VALID_COLORS[color_name]

            # Build a VMobject and smooth it through all control points
            # This creates a smooth Bezier curve through the points
            rope = VMobject()
            rope.set_points_smoothly(pts)
            rope.set_stroke(color, 3)
            ropes.append(rope)

            # Store details for reasoning trace
            self.rope_details.append(
                {
                    "index": i + 1,
                    "color": color_name,
                    "length": length,
                    "start": start[:2],
                    "end": end[:2],
                    "bend_points": [p[:2] for p in bend_points],
                    "num_bends": len(bend_points),
                }
            )

        # ====================================================================
        # Scale and position rope group to fit screen
        # ====================================================================
        rope_group = VGroup(*ropes)

        # Define proportional buffers (margins around the edges)
        h_prop, v_prop = 0.1, 0.2  # 10% horizontal, 20% vertical buffer
        h_buf = frame_width * h_prop
        v_buf = frame_height * v_prop
        max_w = frame_width - 2 * h_buf
        max_h = frame_height - 2 * v_buf

        # Calculate scale factor to fit within bounds
        scale = min(max_w / rope_group.width, max_h / rope_group.height)
        rope_group.scale(scale).move_to(ORIGIN)

        # Additional stretching for better screen utilization
        rope_group.stretch_to_fit_height(0.7 * frame_height)
        rope_group.stretch_to_fit_width(0.9 * frame_width)

        self.log_event(f"Ropes scaled by factor {scale:.3f} to fit screen")

        # ====================================================================
        # Animate ropes being drawn
        # ====================================================================
        self.log_event("Ropes begin drawing on screen")
        self.play(Create(rope_group, run_time=2))
        self.log_event(f"All {self.num_ropes} ropes have been drawn")

        # ====================================================================
        # Calculate answer based on problem type
        # ====================================================================
        # Choose a random ordinal for "order" problems
        N = random.randint(1, self.num_ropes)

        if self.p_type == "count":
            # ================================================================
            # COUNT: Simply count the number of lines
            # ================================================================
            self.answer = self.num_ropes
            self.log_event(f"Answer calculated: {self.answer} lines drawn")

        elif self.p_type == "closed":
            # ================================================================
            # CLOSED: Count closed shapes formed by intersecting lines
            # ================================================================
            self.log_event("Beginning calculation of closed shapes")

            if SHAPELY_AVAILABLE:
                # Use shapely's robust geometric algorithms
                line_strings = []
                for i, rope in enumerate(ropes):
                    # Sample many points along the Manim curve
                    # More samples = more accurate representation
                    samples = [
                        tuple(rope.point_from_proportion(t)[:2])
                        for t in np.linspace(0, 1, 200)
                    ]
                    line_strings.append(LineString(samples))

                # Merge all lines into one geometry
                merged = unary_union(line_strings)

                # Polygonize: extract finite closed regions
                # This finds all the "pockets" formed by intersecting lines
                polygons = list(polygonize(merged))

                # Count the closed shapes
                num_closed_shapes = len(polygons)
                self.answer = num_closed_shapes
                self.log_event(
                    f"Using Shapely: found {self.answer} closed shapes from {len(line_strings)} line segments"
                )
            else:
                # Use approximation method when shapely unavailable
                self.answer = self.count_closed_shapes_fallback(ropes)
                self.log_event(
                    f"Using fallback method: estimated {self.answer} closed shapes"
                )

        elif self.p_type == "cut":
            # ================================================================
            # CUT: Count sections created by a cutting line
            # ================================================================
            self.log_event("Generating cutting line")

            # Create a random cutting line across the screen
            angle = random.uniform(0, TAU)
            x_off = random.uniform(-h_buf, h_buf)
            y_off = random.uniform(-v_buf, v_buf)
            center_pt = np.array([x_off, y_off, 0])

            # Make line long enough to cross entire screen
            diag = np.hypot(frame_width, frame_height)
            half_len = diag / 2
            direction = np.array([np.cos(angle), np.sin(angle), 0])

            start_line = center_pt - direction * half_len
            end_line = center_pt + direction * half_len

            # Create dashed line for visual distinction
            dashed_line = DashedLine(start_line, end_line).set_color(WHITE)

            self.log_event(f"Cutting line created with angle {np.degrees(angle):.1f}°")
            self.play(Create(dashed_line))
            self.log_event("Cutting line fully drawn")

            # Store cutting line details for reasoning
            self.cut_line_details = {
                "angle_radians": angle,
                "angle_degrees": np.degrees(angle),
                "center": center_pt[:2],
                "direction": direction[:2],
            }

            # ================================================================
            # Calculate how many sections the cutting line creates
            # ================================================================
            # For each rope, count how many times it crosses the cutting line
            total_sections = 0
            self.section_details = []

            for i, rope in enumerate(ropes):
                # Sample many points along the rope
                ts = np.linspace(0, 1, 300)
                vals = []

                for t in ts:
                    p = rope.point_from_proportion(t)
                    # Calculate signed distance from point to cutting line
                    # Using 2D cross product: positive on one side, negative on other
                    val = direction[0] * (p[1] - center_pt[1]) - direction[1] * (
                        p[0] - center_pt[0]
                    )
                    vals.append(val)

                # Detect sign changes (crossings)
                signs = np.sign(vals)
                crossings = sum(abs(np.diff(signs)) > 0)

                # Number of sections = number of crossings + 1
                sections = crossings + 1
                total_sections += sections

                # Store for reasoning trace
                self.section_details.append(
                    {
                        "rope_index": i + 1,
                        "color": colors[i % len(colors)],
                        "crossings": crossings,
                        "sections": sections,
                    }
                )

            self.answer = total_sections
            self.log_event(f"Calculated total sections: {self.answer}")

        elif self.p_type == "order":
            # ================================================================
            # ORDER: Identify which color was drawn at position N
            # ================================================================
            self.answer = colors[N - 1]
            self.ordinal_position = N
            self.log_event(f"The {ordinal(N)} rope color is: {self.answer}")

        # ====================================================================
        # Display final question to the user
        # ====================================================================
        self.wait(3)

        self.log_event("All scene elements begin to fade out")
        self.play(*[FadeOut(mob) for mob in self.mobjects if mob != bg])
        self.log_event("Scene cleared, only background remains")

        # Get question text and insert ordinal if needed
        title_text = random.choice(self.cfg["text"][self.p_type])
        title_text = title_text.replace("<N>", ordinal(N))

        # Split into lines and create paragraph
        lines = title_text.split("\n")
        para = Paragraph(*lines, alignment="center", font_size=36, line_spacing=0.8)
        para.move_to(ORIGIN)

        # Scale if too wide for screen
        if para.width > 0.9 * config.frame_width:
            para.scale_to_fit_width(config.frame_width * 0.9)

        self.log_event("Question text appears on screen")
        self.play(Write(para), run_time=1.5)
        self.wait(3)
        self.log_event("Question remains visible for user to read")

        # Store question text for output file
        self.question_text = (
            f"Observe the following scene. {title_text.replace(chr(10), ' ')}"
        )

        # ====================================================================
        # Generate comprehensive reasoning trace
        # ====================================================================
        self.build_reasoning_trace()

        # ====================================================================
        # Save all output files
        # ====================================================================
        # Solution file (just the answer)
        with open(
            f"solutions/ropes_{self.p_type}_ropes{self.num_ropes}_seed{self.seed}.txt",
            "w",
        ) as f:
            f.write(str(self.answer))

        # Question text file
        with open(
            f"question_text/ropes_{self.p_type}_ropes{self.num_ropes}_seed{self.seed}.txt",
            "w",
        ) as f:
            f.write(self.question_text)

        # Detailed reasoning trace file
        with open(
            f"reasoning_traces/ropes_{self.p_type}_ropes{self.num_ropes}_seed{self.seed}.txt",
            "w",
        ) as f:
            f.write("\n".join(self.reasoning_trace))

    def build_reasoning_trace(self):
        """
        Build a comprehensive, step-by-step reasoning trace.
        This explains how to solve the puzzle systematically.
        """
        self.reasoning_trace = []

        # ====================================================================
        # Introduction with question
        # ====================================================================
        self.reasoning_trace.append("**Question:** " + self.question_text)
        self.reasoning_trace.append("")
        self.reasoning_trace.append("Let's solve this step by step.")
        self.reasoning_trace.append("")

        # ====================================================================
        # Scene description with timestamps
        # ====================================================================
        self.reasoning_trace.append("### Scene Description")
        self.reasoning_trace.append("")
        self.reasoning_trace.append("The video shows the following sequence of events:")
        self.reasoning_trace.append("")

        for event in self.scene_events:
            time_str = self.format_time(event["time"])
            self.reasoning_trace.append(f"At {time_str}, {event['description']}")

        self.reasoning_trace.append("")

        # ====================================================================
        # Metadata about the problem
        # ====================================================================
        self.reasoning_trace.append("### Problem Setup")
        self.reasoning_trace.append("")
        self.reasoning_trace.append(f"- **Problem Type**: {self.p_type}")
        self.reasoning_trace.append(f"- **Number of Ropes**: {self.num_ropes}")
        self.reasoning_trace.append(f"- **Bends per Rope**: {self.bends_per_rope}")
        self.reasoning_trace.append(
            f"- **Random Seed**: {self.seed} (for reproducibility)"
        )
        self.reasoning_trace.append(f"- **Shapely Available**: {SHAPELY_AVAILABLE}")
        self.reasoning_trace.append("")

        # ====================================================================
        # Detailed rope information
        # ====================================================================
        self.reasoning_trace.append("### Step 1: Understand the Ropes")
        self.reasoning_trace.append("")
        self.reasoning_trace.append(
            f"The scene contains {self.num_ropes} colored lines (ropes) drawn in the following order:"
        )
        self.reasoning_trace.append("")

        for detail in self.rope_details:
            self.reasoning_trace.append(
                f"**Rope {detail['index']}** ({detail['color']}):"
            )
            self.reasoning_trace.append(
                f"  - Start point: ({detail['start'][0]:.2f}, {detail['start'][1]:.2f})"
            )
            self.reasoning_trace.append(
                f"  - End point: ({detail['end'][0]:.2f}, {detail['end'][1]:.2f})"
            )
            self.reasoning_trace.append(
                f"  - Approximate length: {detail['length']:.2f}"
            )
            self.reasoning_trace.append(
                f"  - Number of curve control points: {detail['num_bends']}"
            )
            self.reasoning_trace.append("")

        # ====================================================================
        # Problem-specific reasoning
        # ====================================================================
        if self.p_type == "count":
            self.reasoning_trace.append("### Step 2: Count the Lines")
            self.reasoning_trace.append("")
            self.reasoning_trace.append(
                "To solve this problem, we simply need to count how many lines were drawn."
            )
            self.reasoning_trace.append("")
            self.reasoning_trace.append(
                f"Looking at the video, we see {self.num_ropes} distinct colored lines:"
            )
            for i, detail in enumerate(self.rope_details, 1):
                self.reasoning_trace.append(f"{i}. {detail['color'].capitalize()} line")
            self.reasoning_trace.append("")
            self.reasoning_trace.append(
                f"Therefore, the total count is **{self.answer}** lines."
            )

        elif self.p_type == "cut":
            self.reasoning_trace.append("### Step 2: Understand the Cutting Line")
            self.reasoning_trace.append("")
            self.reasoning_trace.append(
                "A dotted white line is drawn across the screen to cut through the colored ropes."
            )
            self.reasoning_trace.append("")
            self.reasoning_trace.append(
                f"- **Angle**: {self.cut_line_details['angle_degrees']:.1f}° from horizontal"
            )
            self.reasoning_trace.append(
                f"- **Center point**: ({self.cut_line_details['center'][0]:.2f}, {self.cut_line_details['center'][1]:.2f})"
            )
            self.reasoning_trace.append("")

            self.reasoning_trace.append("### Step 3: Count Sections Created by the Cut")
            self.reasoning_trace.append("")
            self.reasoning_trace.append(
                "For each rope, we need to determine how many times the cutting line crosses it."
            )
            self.reasoning_trace.append(
                "Each rope is divided into (crossings + 1) sections."
            )
            self.reasoning_trace.append("")

            total_crossings = 0
            for detail in self.section_details:
                self.reasoning_trace.append(
                    f"**{detail['color'].capitalize()} rope (Rope {detail['rope_index']})**:"
                )
                self.reasoning_trace.append(f"  - Crossings: {detail['crossings']}")
                self.reasoning_trace.append(
                    f"  - Sections created: {detail['sections']}"
                )
                self.reasoning_trace.append("")
                total_crossings += detail["crossings"]

            self.reasoning_trace.append(f"Total crossings: {total_crossings}")
            self.reasoning_trace.append(f"Total sections: {self.answer}")
            self.reasoning_trace.append("")
            self.reasoning_trace.append("### Step 4: Calculate Final Answer")
            self.reasoning_trace.append("")
            self.reasoning_trace.append("Sum all the sections from each rope:")
            sum_str = " + ".join([str(d["sections"]) for d in self.section_details])
            self.reasoning_trace.append(f"{sum_str} = **{self.answer}**")

        elif self.p_type == "closed":
            self.reasoning_trace.append("### Step 2: Identify Intersections")
            self.reasoning_trace.append("")
            self.reasoning_trace.append(
                "When multiple curved lines cross each other, they can form closed shapes (regions completely bounded by the lines)."
            )
            self.reasoning_trace.append("")

            if SHAPELY_AVAILABLE:
                self.reasoning_trace.append(
                    "We use the Shapely geometric library to accurately detect these closed regions:"
                )
                self.reasoning_trace.append("")
                self.reasoning_trace.append(
                    "1. Sample 200 points along each rope to create line segments"
                )
                self.reasoning_trace.append(
                    "2. Merge all line segments into a single geometric object"
                )
                self.reasoning_trace.append(
                    "3. Use polygonization algorithm to extract all closed regions"
                )
                self.reasoning_trace.append("4. Count the number of polygons found")
                self.reasoning_trace.append("")
                self.reasoning_trace.append(
                    f"This algorithm found **{self.answer}** closed shapes."
                )
            else:
                self.reasoning_trace.append(
                    "Note: Shapely library not available, using approximation method."
                )
                self.reasoning_trace.append("")
                self.reasoning_trace.append("The approximation method:")
                self.reasoning_trace.append("1. Sample 50 points along each rope")
                self.reasoning_trace.append(
                    "2. Detect intersections by finding nearby points from different ropes"
                )
                self.reasoning_trace.append(
                    "3. Estimate closed shapes using Euler's formula approximation"
                )
                self.reasoning_trace.append("")
                self.reasoning_trace.append(
                    f"This estimated **{self.answer}** closed shapes (this may not be exact)."
                )

        elif self.p_type == "order":
            self.reasoning_trace.append("### Step 2: Recall the Drawing Order")
            self.reasoning_trace.append("")
            self.reasoning_trace.append(
                "The color legend at the beginning showed all possible colors."
            )
            self.reasoning_trace.append("The ropes were then drawn in this order:")
            self.reasoning_trace.append("")

            for i, detail in enumerate(self.rope_details, 1):
                self.reasoning_trace.append(f"{i}. {detail['color'].capitalize()}")

            self.reasoning_trace.append("")
            self.reasoning_trace.append(
                f"### Step 3: Identify the {ordinal(self.ordinal_position)} Rope"
            )
            self.reasoning_trace.append("")
            self.reasoning_trace.append(
                f"Looking at the sequence above, the {ordinal(self.ordinal_position)} rope to be drawn was **{self.answer}**."
            )

        # ====================================================================
        # Final answer
        # ====================================================================
        self.reasoning_trace.append("")
        self.reasoning_trace.append("### Final Answer")
        self.reasoning_trace.append("")
        self.reasoning_trace.append(f"\\boxed{{{self.answer}}}")


# ============================================================================
# Main execution
# ============================================================================
# Generate the ropes video
scene = RopeCutScene()
scene.render()

# ============================================================================
# Move output file to questions directory with descriptive name
# ============================================================================
output = Path("manim_output/videos/1080p30/RopeCutScene.mp4")
if output.exists():
    filename = f"ropes_{scene.p_type}_ropes{scene.num_ropes}_seed{scene.seed}.mp4"
    shutil.move(str(output), f"questions/{filename}")
    print(f"✓ Video saved: questions/{filename}")
    print(
        f"✓ Solution saved: solutions/ropes_{scene.p_type}_ropes{scene.num_ropes}_seed{scene.seed}.txt"
    )
    print(
        f"✓ Question saved: question_text/ropes_{scene.p_type}_ropes{scene.num_ropes}_seed{scene.seed}.txt"
    )
    print(
        f"✓ Reasoning saved: reasoning_traces/ropes_{scene.p_type}_ropes{scene.num_ropes}_seed{scene.seed}.txt"
    )
else:
    # Debug: Print what files actually exist
    videos_dir = Path("manim_output/videos")
    if videos_dir.exists():
        print(f"Available folders in videos/: {list(videos_dir.iterdir())}")
        for folder in videos_dir.iterdir():
            if folder.is_dir():
                subfolder = folder / "1080p30"
                if subfolder.exists():
                    print(f"Files in {subfolder}: {list(subfolder.iterdir())}")
    else:
        print("manim_output/videos directory doesn't exist")

# Final cleanup
if os.path.exists("manim_output"):
    shutil.rmtree("manim_output")
    print("✓ Temporary files cleaned up")
