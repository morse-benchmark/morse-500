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

def ordinal(n):
    if 11 <= (n % 100) <= 13:
        return f"{n}th"
    return f"{n}{ {1:'st', 2:'nd', 3:'rd'}.get(n%10, 'th') }"

class Paths(Scene):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Set random seed for reproducibility
        self.seed = random.randint(1000, 9999)
        
        # Parameters from environment variables
        self.p_type = os.getenv("P_TYPE", "order")
        self.num_shapes = int(os.getenv("NUM_SHAPES", 4))
        self.max_placement_tries = int(os.getenv("MAX_PLACEMENT_TRIES", 250))
        
        # Scene parameters
        self.SHAPE_SCALE = 0.1
        self.MARGIN = config.frame_height * 0.2
        self.min_time = 0.5
        self.time_step = 2

        self.VALID_COLORS = {
            BLUE: "blue", 
            RED: "red", 
            ORANGE: "orange",
            GREEN: "green", 
            YELLOW: "yellow",
            PURPLE: "purple",
            WHITE: "white",
        }

        # Configuration for different problem types
        self.cfg = {
            "text": {
                "order": [
                    "What was the color of the <N> shape to be visited?\nAnswer with only the color name."
                ],
                "max_dist": [
                    "Which color shape was the furthest from\nthe previous shape in the path?\nAnswer with only the color name."
                ],
                "min_dist": [
                    "Which color shape was the closest to\nthe previous shape in the path?\nAnswer with only the color name."
                ],
                "min_time": [
                    "Which color shape took the shortest amount\nof time to reach from the previous shape?\nAnswer with only the color name."
                ],
                "max_time": [
                    "Which color shape took the longest amount\nof time to reach from the previous shape?\nAnswer with only the color name."
                ]
            }
        }
        
        # Initialize reasoning trace
        self.reasoning_trace = []
        self.reasoning_trace.append(f"Problem Type: {self.p_type}")
        self.reasoning_trace.append(f"Number of Shapes: {self.num_shapes}")
        self.reasoning_trace.append(f"Random Seed: {self.seed}")
        self.reasoning_trace.append("")

    def _non_overlapping_position(self, new_shape, existing, x_min, x_max, y_min, y_max):
        for attempt in range(self.max_placement_tries):
            new_shape.move_to([
                random.uniform(x_min, x_max),
                random.uniform(y_min, y_max),
                0,
            ])
            if all(
                np.linalg.norm(new_shape.get_center() - s.get_center())
                > (new_shape.width + s.width) * 0.55
                for s in existing
            ):
                return True
        return False

    def _random_point_outside_shapes(self, shapes, x_min, x_max, y_min, y_max):
        attempts = 0
        while attempts < 100:  # Prevent infinite loop
            attempts += 1
            p = np.array([
                random.uniform(x_min, x_max),
                random.uniform(y_min, y_max),
                0,
            ])
            if all(np.linalg.norm(p - s.get_center()) > s.width * 0.6 for s in shapes):
                return p
        # Fallback if no good position found
        return np.array([0, 0, 0])

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
        
        self.show_colors(list(self.VALID_COLORS.keys()), list(self.VALID_COLORS.values()))
        
        prompt = Text(
            "Observe the trajectory of the arrow", color=WHITE, font_size=36
        ).move_to(ORIGIN)
        self.play(FadeIn(prompt), run_time=0.5)
        self.wait(1.5)
        self.play(FadeOut(prompt), run_time=0.5)
        self.wait(1)
        
        # Frame bounds (inner safe zone)
        x_min = -config.frame_width / 2 + self.MARGIN
        x_max =  config.frame_width / 2 - self.MARGIN
        y_min = -config.frame_height / 2 + self.MARGIN
        y_max =  config.frame_height / 2 - self.MARGIN

        # 1) Random non‑overlapping shapes --------------------------------------
        shapes = VGroup()
        shape_colors = list(self.VALID_COLORS.keys())
        random.shuffle(shape_colors)
        
        self.reasoning_trace.append("Creating shapes:")
        for i in range(self.num_shapes):
            shape_cls = random.choice([
                Circle,
                Square,
                Triangle,
                lambda: RegularPolygon(n=random.randint(5, 8)),
            ])
            shp = shape_cls()
            color = shape_colors[i % len(shape_colors)]
            shp.set_fill(opacity=0).set_stroke(width=4, color=color)
            shp.scale(self.SHAPE_SCALE)

            if self._non_overlapping_position(shp, shapes, x_min, x_max, y_min, y_max):
                shapes.add(shp)
                self.reasoning_trace.append(f"  Shape {i + 1}: {type(shp).__name__} in {self.VALID_COLORS[color]} at {shp.get_center()}")
            else:
                self.reasoning_trace.append(f"  Shape {i + 1}: Failed to place after {self.max_placement_tries} attempts")

        # Scale shapes to fit screen
        frame_width = config.frame_width
        frame_height = config.frame_height
        h_prop, v_prop = 0.1, 0.2  # buffers as proportion of frame
        h_buf = frame_width * h_prop
        v_buf = frame_height * v_prop
        max_w = frame_width - 2*h_buf
        max_h = frame_height - 2*v_buf
        scale = min(max_w / shapes.width, max_h / shapes.height)
        shapes.scale(scale).move_to(ORIGIN)
        for shape in shapes:
            shape.scale(1/scale)

        self.reasoning_trace.append(f"Scaled shapes by factor: {scale}")
        self.reasoning_trace.append("")

        self.play(LaggedStart(*[Create(s) for s in shapes], lag_ratio=0.15))
        self.wait(0.3)

        # 2) Start point + visit order ------------------------------------------
        start_pt = self._random_point_outside_shapes(shapes, x_min, x_max, y_min, y_max)
        visit_order = random.sample(list(shapes), len(shapes))
        
        self.reasoning_trace.append("Path planning:")
        self.reasoning_trace.append(f"Start point: {start_pt}")
        self.reasoning_trace.append("Visit order:")
        for i, shape in enumerate(visit_order):
            color_name = self.VALID_COLORS[shape.stroke_color]
            self.reasoning_trace.append(f"  {i + 1}. {color_name} {type(shape).__name__} at {shape.get_center()}")

        # 3) Smooth path through centres ----------------------------------------
        control_pts = [start_pt] + [s.get_center() for s in visit_order]
        path = VMobject().set_points_smoothly(control_pts)
        path.set_stroke(opacity=0)
        self.add(path)

        # 4) Arrowhead only ------------------------------------------------------
        arrow_len = 0.9
        arrow = Arrow(
            start_pt - RIGHT * arrow_len,
            start_pt,
            buff=0,
            color=WHITE,
            max_tip_length_to_length_ratio=0.2,
            stroke_width=0,
        )
        if hasattr(arrow, "body"):
            arrow.body.set_stroke(width=0)
        if hasattr(arrow, "tip"):
            arrow.tip.set_stroke(width=0)
            arrow.tip.set_fill(WHITE, 1)
        arrow.set_z_index(2)
        self.add(arrow)

        # 5) Trail flush with arrow tip -----------------------------------------
        trail = TracedPath(arrow.get_end, dissipating_time=0.7, stroke_color=WHITE, stroke_width=2)
        self.add(trail)

        # 6) Orientation updater: use tangent ------------------------------------
        tracker = ValueTracker(0)

        def orient_arrow(mob):
            alpha = tracker.get_value()
            pos = path.point_from_proportion(alpha)

            delta = 1e-3
            # Use central difference when possible, else forward/backward diff
            if alpha < delta:
                q = path.point_from_proportion(alpha + delta)
                tangent = q - pos
            elif alpha > 1 - delta:
                q = path.point_from_proportion(alpha - delta)
                tangent = pos - q
            else:
                q1 = path.point_from_proportion(alpha + delta)
                q0 = path.point_from_proportion(alpha - delta)
                tangent = q1 - q0

            if np.linalg.norm(tangent) == 0:
                tangent = RIGHT * 0.001

            unit = tangent / np.linalg.norm(tangent)
            tail = pos - unit * arrow_len
            mob.put_start_and_end_on(tail, pos)

        arrow.add_updater(orient_arrow)

        # 7) Variable‑duration hops ---------------------------------------------
        hops = len(visit_order)
        durations = [1.9] + [self.min_time*(self.time_step**i) for i in range(hops-1)]
        random.shuffle(durations)
        alpha_step = 1 / hops

        self.reasoning_trace.append("")
        self.reasoning_trace.append("Animation durations:")
        self.reasoning_trace.append(f"  Start to first shape: {durations[0]:.2f}s")
        for i in range(1, len(durations)):
            shape_color = self.VALID_COLORS[visit_order[i-1].stroke_color]
            self.reasoning_trace.append(f"  To {shape_color} shape: {durations[i]:.2f}s")

        for i, dur in enumerate(durations):
            self.play(
                tracker.animate.set_value((i + 1) * alpha_step),
                run_time=dur,
                rate_func=linear,
            )

        # --- Snap exactly to final centre with correct orientation -------------
        tracker.set_value(1)
        orient_arrow(arrow)  # one last manual update ensures tangent‑based orient
        arrow.remove_updater(orient_arrow)
        
        # Calculate answer based on problem type
        N = random.randint(1, self.num_shapes)
        centers = np.array([poly.get_center() for poly in list(visit_order)])  # shape (n, 3) or (n, 2)
        deltas = centers[1:] - centers[:-1]  # shape (n-1, 3)
        dists = list(np.linalg.norm(deltas, axis=1))  # shape (n-1,)
        shape_colors = [poly.stroke_color for poly in list(visit_order)]
        
        self.reasoning_trace.append("")
        self.reasoning_trace.append("Calculating answer:")
        self.reasoning_trace.append(f"Distances between consecutive shapes: {[f'{d:.2f}' for d in dists]}")
        self.reasoning_trace.append(f"Animation durations (excluding start): {[f'{d:.2f}' for d in durations[1:]]}")
        
        if self.p_type == "order": 
            self.answer = self.VALID_COLORS[shape_colors[N-1]]
            self.reasoning_trace.append(f"The {ordinal(N)} shape to be visited: {self.answer}")
        elif self.p_type == "min_dist":
            idx = dists.index(min(dists))
            self.answer = self.VALID_COLORS[shape_colors[idx+1]]
            self.reasoning_trace.append(f"Closest shape (min distance {min(dists):.2f}): {self.answer}")
        elif self.p_type == "max_dist":
            idx = dists.index(max(dists))
            self.answer = self.VALID_COLORS[shape_colors[idx+1]]
            self.reasoning_trace.append(f"Furthest shape (max distance {max(dists):.2f}): {self.answer}")
        elif self.p_type == "min_time":
            idx = durations[1:].index(min(durations[1:]))
            self.answer = self.VALID_COLORS[shape_colors[idx]]
            self.reasoning_trace.append(f"Shortest time (min {min(durations[1:]):.2f}s): {self.answer}")
        elif self.p_type == "max_time":
            idx = durations[1:].index(max(durations[1:]))
            self.answer = self.VALID_COLORS[shape_colors[idx]]
            self.reasoning_trace.append(f"Longest time (max {max(durations[1:]):.2f}s): {self.answer}")

        self.wait(2)
        self.play(*[FadeOut(mob) for mob in self.mobjects if mob!=bg])
        
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

        self.question_text = f"Observe the trajectory of the arrow. {title_text.replace(chr(10), ' ')}"
        
        # Save files
        with open(f"solutions/path_{self.p_type}_shapes{self.num_shapes}_seed{self.seed}.txt", "w") as f:
            f.write(str(self.answer))
        
        with open(f"question_text/path_{self.p_type}_shapes{self.num_shapes}_seed{self.seed}.txt", "w") as f:
            f.write(self.question_text)
            
        # Save detailed reasoning trace
        self.reasoning_trace.append("")
        self.reasoning_trace.append(f"Final Answer: {self.answer}")
        with open(f"reasoning_traces/path_{self.p_type}_shapes{self.num_shapes}_seed{self.seed}.txt", "w") as f:
            f.write("\n".join(self.reasoning_trace))

# Generate the path video
scene = Paths()
scene.render()

# Move the output file with descriptive name
output = Path("manim_output/videos/1080p30/Paths.mp4")
if output.exists():
    filename = f"path_{scene.p_type}_shapes{scene.num_shapes}_seed{scene.seed}.mp4"
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