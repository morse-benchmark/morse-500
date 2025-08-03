from manim import *
import random
import numpy as np
import json
import os
import shutil
from pathlib import Path

# Setup directories
Path("questions").mkdir(exist_ok=True)
Path("solutions").mkdir(exist_ok=True)
Path("question_text").mkdir(exist_ok=True)
Path("reasoning_traces").mkdir(exist_ok=True)

config.media_dir = "manim_output"
config.verbosity = "WARNING"
config.pixel_height = 1080
config.pixel_width = 1920
config.frame_rate = 30
config.preview = False

# Try to import shapely, provide fallback if not available
try:
    from shapely.geometry import LineString
    from shapely.ops import unary_union, polygonize
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False
    print("Warning: Shapely not available. Some geometric calculations may be approximate.")

def ordinal(n):
    if 11 <= (n % 100) <= 13:
        return f"{n}th"
    return f"{n}{ {1:'st', 2:'nd', 3:'rd'}.get(n%10, 'th') }"

class RopeCutScene(Scene):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Set random seed for reproducibility
        self.seed = random.randint(1000, 9999)
        
        # Parameters from environment variables
        self.p_type = os.getenv("P_TYPE", "count")
        self.num_ropes = int(os.getenv("NUM_ROPES", 3))
        self.bends_per_rope = int(os.getenv("BENDS_PER_ROPE", 2))
        
        self.VALID_COLORS = {
            "blue": BLUE,
            "red": RED,
            "orange": ORANGE,
            "green": GREEN,
            "yellow": YELLOW,
            "purple": PURPLE,
            "white": WHITE,
        }

        # Configuration for different problem types
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
        
        # Initialize reasoning trace
        self.reasoning_trace = []
        self.reasoning_trace.append(f"Problem Type: {self.p_type}")
        self.reasoning_trace.append(f"Number of Ropes: {self.num_ropes}")
        self.reasoning_trace.append(f"Bends per Rope: {self.bends_per_rope}")
        self.reasoning_trace.append(f"Random Seed: {self.seed}")
        self.reasoning_trace.append(f"Shapely Available: {SHAPELY_AVAILABLE}")
        self.reasoning_trace.append("")

    def show_colors(self, colors, names):
        squares1 = VGroup(
            *[Square(0.5).set_fill(col, 1).set_stroke(width=0) for col in colors[:4]]
        )
        squares1.arrange(DOWN, buff=0.3, aligned_edge=LEFT).to_edge(LEFT, buff=2)

        # Labels positioned relative to squares
        labels1 = VGroup(*[Text(nm, font_size=20) for nm in names[:4]])
        for sq, lbl in zip(squares1, labels1):
            lbl.next_to(sq, RIGHT, buff=0.8)

        # Arrows connecting squares to labels
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

        # Labels positioned relative to squares
        labels2 = VGroup(*[Text(nm, font_size=20) for nm in names[4:]])
        for sq, lbl in zip(squares2, labels2):
            lbl.next_to(sq, RIGHT, buff=0.8)

        # Arrows connecting squares to labels
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

        # Title
        title = Text("Remember the following color names", font_size=32)
        title.to_edge(UP)

        # Animation sequence
        self.play(Write(title))
        self.play(
            Succession(
                FadeIn(squares1, squares2),
                AnimationGroup(*([GrowArrow(ar) for ar in arrows1]+[GrowArrow(ar) for ar in arrows2]), lag_ratio=0.1),
                FadeIn(labels1, labels2)
            ),
            run_time=1.5
        )
        self.wait(2)
        self.play(FadeOut(title, both))
        self.wait(0.5)

    def count_closed_shapes_fallback(self, ropes):
        """Fallback method when shapely is not available"""
        # Simple approximation: count potential intersections
        # This is a rough estimate and may not be accurate
        intersection_count = 0
        for i, rope1 in enumerate(ropes):
            for j, rope2 in enumerate(ropes[i+1:], i+1):
                # Sample points and check for approximate intersections
                samples1 = [rope1.point_from_proportion(t)[:2] for t in np.linspace(0, 1, 50)]
                samples2 = [rope2.point_from_proportion(t)[:2] for t in np.linspace(0, 1, 50)]
                
                # Simple proximity-based intersection detection
                for p1 in samples1:
                    for p2 in samples2:
                        if np.linalg.norm(np.array(p1) - np.array(p2)) < 0.1:
                            intersection_count += 1
                            break
        
        # Very rough approximation: closed shapes ≈ intersections - ropes + 1 (Euler's formula approximation)
        return max(0, intersection_count - len(ropes) + 1)

    def construct(self):
        # Use the seed set in __init__
        random.seed(self.seed)
        
        bg = (
            Rectangle(height=config.frame_height, width=config.frame_width)
            .set_color(
                color_gradient([random_bright_color(), random_bright_color()], 5)
            )
            .set_opacity(0.6)
            .set_z_index(-2)
        )
        self.add(bg)
        
        if self.p_type == "order":
            self.show_colors(list(self.VALID_COLORS.values()), list(self.VALID_COLORS.keys()))
            
        prompt = Text(
            "Observe the following scene", color=WHITE, font_size=36
        ).move_to(ORIGIN)
        self.play(FadeIn(prompt), run_time=0.5)
        self.wait(1.5)
        self.play(FadeOut(prompt), run_time=0.5)
        self.wait(1)
        
        colors = list(self.VALID_COLORS.keys())
        random.shuffle(colors)
        self.reasoning_trace.append(f"Shuffled colors: {colors}")

        # Full screen bounds
        frame_width = config.frame_width
        frame_height = config.frame_height
        x_min, x_max = -frame_width / 2, frame_width / 2
        y_min, y_max = -frame_height / 2, frame_height / 2

        ropes = []
        lengths = []
        
        self.reasoning_trace.append("Creating ropes:")
        for i in range(self.num_ropes):
            start = np.array(
                [random.uniform(x_min, x_max), random.uniform(y_min, y_max), 0]
            )
            end = np.array(
                [random.uniform(x_min, x_max), random.uniform(y_min, y_max), 0]
            )
            length = np.sqrt(np.sum((start - end) ** 2))
            lengths.append(length)
            
            bend_points = [
                np.array(
                    [
                        # x progression from start→end
                        np.interp(j, [0, self.bends_per_rope + 1], [x_min, x_max]),
                        random.uniform(y_min, y_max),
                        0,
                    ]
                )
                for j in range(1, self.bends_per_rope + 1)
            ]
            # assemble full point list
            pts = [start, *bend_points, end]
            color_name = colors[i % len(colors)]
            color = self.VALID_COLORS[color_name]

            # Build a VMobject and smooth it through all these points
            rope = VMobject()
            rope.set_points_smoothly(pts)
            rope.set_stroke(color, 3)
            ropes.append(rope)
            
            self.reasoning_trace.append(f"  Rope {i + 1}: {color_name}, length {length:.2f}")
            self.reasoning_trace.append(f"    Start: {start[:2]}, End: {end[:2]}")
            self.reasoning_trace.append(f"    Bend points: {[p[:2] for p in bend_points]}")

        # Group ropes and scale to fit with proportional buffer
        rope_group = VGroup(*ropes)
        h_prop, v_prop = 0.1, 0.2  # buffers as proportion of frame
        h_buf = frame_width * h_prop
        v_buf = frame_height * v_prop
        max_w = frame_width - 2 * h_buf
        max_h = frame_height - 2 * v_buf
        scale = min(max_w / rope_group.width, max_h / rope_group.height)
        rope_group.scale(scale).move_to(ORIGIN)
        rope_group.stretch_to_fit_height(0.7 * frame_height)
        rope_group.stretch_to_fit_width(0.9 * frame_width)
        
        self.reasoning_trace.append(f"Scaled rope group by factor: {scale}")
        self.reasoning_trace.append("")

        # Animate ropes drawing
        self.play(Create(rope_group, run_time=2))
        
        # Calculate answer based on problem type
        N = random.randint(1, self.num_ropes)
        self.reasoning_trace.append("Calculating answer:")
        
        if self.p_type == "count":
            self.answer = self.num_ropes
            self.reasoning_trace.append(f"Number of lines drawn: {self.answer}")
            
        elif self.p_type == "closed":
            if SHAPELY_AVAILABLE:
                line_strings = []
                for i, rope in enumerate(ropes):
                    # sample 200 points along the Manim curve
                    samples = [
                        tuple(rope.point_from_proportion(t)[:2])
                        for t in np.linspace(0, 1, 200)
                    ]
                    line_strings.append(LineString(samples))
                    self.reasoning_trace.append(f"  Rope {i + 1}: {len(samples)} sample points")

                # 2) Merge them into one geometry
                merged = unary_union(line_strings)

                # 3) Polygonize: this returns only the finite faces (i.e. your closed loops)
                polygons = list(polygonize(merged))

                # 4) Count them
                num_closed_shapes = len(polygons)
                self.answer = num_closed_shapes
                self.reasoning_trace.append(f"Using Shapely: found {self.answer} closed shapes")
            else:
                self.answer = self.count_closed_shapes_fallback(ropes)
                self.reasoning_trace.append(f"Using fallback method: estimated {self.answer} closed shapes")
                
        elif self.p_type == "cut":
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
            self.play(Create(dashed_line))
            
            self.reasoning_trace.append(f"Cutting line: angle {angle:.2f}, center {center_pt[:2]}")

            # Compute intersections and section counts
            total_sections = 0
            for i, rope in enumerate(ropes):
                ts = np.linspace(0, 1, 300)
                vals = []
                for t in ts:
                    p = rope.point_from_proportion(t)
                    # signed distance (2D cross) to line
                    val = direction[0] * (p[1] - center_pt[1]) - direction[1] * (
                        p[0] - center_pt[0]
                    )
                    vals.append(val)
                signs = np.sign(vals)
                crossings = sum(abs(np.diff(signs)) > 0)
                sections = crossings + 1
                total_sections += sections
                self.reasoning_trace.append(f"  Rope {i + 1}: {crossings} crossings, {sections} sections")
            
            self.answer = total_sections
            self.reasoning_trace.append(f"Total sections after cutting: {self.answer}")
            
        elif self.p_type == "order":
            self.answer = colors[N - 1]
            self.reasoning_trace.append(f"The {ordinal(N)} rope color: {self.answer}")

        # Display result prompt
        self.wait(3)
        self.play(*[FadeOut(mob) for mob in self.mobjects if mob != bg])
        
        # Display question
        title_text = random.choice(self.cfg["text"][self.p_type])
        title_text = title_text.replace("<N>", ordinal(N))
        lines = title_text.split('\n')
        para = Paragraph(
            *lines, alignment="center", font_size=36, line_spacing=0.8
        )

        para.move_to(ORIGIN)
        if para.width > 0.9*config.frame_width:
            para.scale_to_fit_width(config.frame_width * 0.9)
        self.play(Write(para), run_time=1.5)
        self.wait(3)

        self.question_text = f"Observe the following scene. {title_text.replace(chr(10), ' ')}"
        
        # Save files
        with open(f"solutions/ropes_{self.p_type}_ropes{self.num_ropes}_seed{self.seed}.txt", "w") as f:
            f.write(str(self.answer))
        
        with open(f"question_text/ropes_{self.p_type}_ropes{self.num_ropes}_seed{self.seed}.txt", "w") as f:
            f.write(self.question_text)
            
        # Save detailed reasoning trace
        self.reasoning_trace.append("")
        self.reasoning_trace.append(f"Final Answer: {self.answer}")
        with open(f"reasoning_traces/ropes_{self.p_type}_ropes{self.num_ropes}_seed{self.seed}.txt", "w") as f:
            f.write("\n".join(self.reasoning_trace))

# Generate the ropes video
scene = RopeCutScene()
scene.render()

# Move the output file with descriptive name
output = Path("manim_output/videos/1080p30/RopeCutScene.mp4")
if output.exists():
    filename = f"ropes_{scene.p_type}_ropes{scene.num_ropes}_seed{scene.seed}.mp4"
    shutil.move(str(output), f"questions/{filename}")
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