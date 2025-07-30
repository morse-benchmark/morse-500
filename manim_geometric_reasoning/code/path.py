from manim import *
import random
import numpy as np
import json
import os 

def ordinal(n):
    if 11 <= (n % 100) <= 13:
        return f"{n}th"
    return f"{n}{ {1:'st', 2:'nd', 3:'rd'}.get(n%10, 'th') }"

class Paths(Scene):
    def __init__(self, p_type, num_shapes, cfg_path='../templates/path.json'):
        super().__init__()
        self.p_type=p_type
        self.NUM_SHAPES = num_shapes # how many shapes to draw
        self.SHAPE_SCALE = 0.1
        self.MARGIN = config.frame_height*0.2

        self.min_time = 0.5
        self.time_step = 2

        self.MAX_PLACEMENT_TRIES = 250
        self.VALID_COLORS = {
            BLUE: "blue", 
            RED: "red", 
            ORANGE: "orange",
            GREEN: "green", 
            YELLOW: "yellow",
            PURPLE: "purple",
            WHITE: "white",
        }

        with open(cfg_path, 'r') as f:
            self.cfg = json.load(f)

    def _non_overlapping_position(self, new_shape, existing, x_min, x_max, y_min, y_max):
        for _ in range(self.MAX_PLACEMENT_TRIES):
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
        while True:
            p = np.array([
                random.uniform(x_min, x_max),
                random.uniform(y_min, y_max),
                0,
            ])
            if all(np.linalg.norm(p - s.get_center()) > s.width * 0.6 for s in shapes):
                return p

    def show_colors(self, colors, names):

        squares1 = VGroup(
            *[Square(1.0).set_fill(col, 1).set_stroke(width=0) for col in colors[:4]]
        )
        squares1.arrange(DOWN, buff=0.5, aligned_edge=LEFT).to_edge(LEFT, buff=2)

        # Labels positioned relative to squares
        labels1 = VGroup(*[Text(nm, font_size=32) for nm in names[:4]])
        for sq, lbl in zip(squares1, labels1):
            lbl.next_to(sq, RIGHT, buff=1.2)

        # Arrows connecting squares to labels
        arrows1 = VGroup(
            *[
                Arrow(
                    start=sq.get_right(), end=lbl.get_left(), buff=0.05, stroke_width=4
                )
                for sq, lbl in zip(squares1, labels1)
            ]
        )

        squares2 = VGroup(
            *[Square(1.0).set_fill(col, 1).set_stroke(width=0) for col in colors[4:]]
        )
        squares2.arrange(DOWN, buff=0.4, aligned_edge=LEFT).to_edge(LEFT, buff=2)

        # Labels positioned relative to squares
        labels2 = VGroup(*[Text(nm, font_size=32) for nm in names[4:]])
        for sq, lbl in zip(squares2, labels2):
            lbl.next_to(sq, RIGHT, buff=1.2)

        # Arrows connecting squares to labels
        arrows2 = VGroup(
            *[
                Arrow(
                    start=sq.get_right(), end=lbl.get_left(), buff=0.05, stroke_width=4
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
        title = Text("Remember the following color names", font_size=40)
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

    # ---------------------------------------------------------------------
    # Main animation
    # ---------------------------------------------------------------------
    def construct(self):
        bg = (
            Rectangle(height=config.frame_height, width=config.frame_width)
            .set_color(
                color_gradient([random_bright_color(), random_bright_color()], 5)
            )
            .set_opacity(0.6)
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
        for i in range(self.NUM_SHAPES):
            shape_cls = random.choice([
                Circle,
                Square,
                Triangle,
                lambda: RegularPolygon(n=random.randint(5, 8)),
            ])
            shp = shape_cls()
            shp.set_fill(opacity=0).set_stroke(width=4, color=shape_colors[i])
            shp.scale(self.SHAPE_SCALE)

            if self._non_overlapping_position(shp, shapes, x_min, x_max, y_min, y_max):
                shapes.add(shp)

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

        self.play(LaggedStart(*[Create(s) for s in shapes], lag_ratio=0.15))
        self.wait(0.3)

        # 2) Start point + visit order ------------------------------------------
        start_pt = self._random_point_outside_shapes(shapes, x_min, x_max, y_min, y_max)
        visit_order = random.sample(list(shapes), len(shapes))

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
        
        N = random.randint(1, self.NUM_SHAPES)

        centers = np.array([poly.get_center() for poly in list(visit_order)])  # shape (n, 3) or (n, 2)

        deltas = centers[1:] - centers[:-1]  # shape (n-1, 3)
        dists = list(np.linalg.norm(deltas, axis=1))  # shape (n-1,)
        shape_colors = [poly.stroke_color for poly in list(visit_order)]
        if self.p_type == "order": 
            self.answer = self.VALID_COLORS[shape_colors[N-1]]
        elif self.p_type == "min_dist":
            idx = dists.index(min(dists))
            self.answer = self.VALID_COLORS[shape_colors[idx+1]]
        elif self.p_type == "max_dist":
            idx = dists.index(max(dists))
            self.answer = self.VALID_COLORS[shape_colors[idx+1]]
        elif self.p_type == "min_time":
            idx = durations.index(min(durations[1:]))
            self.answer = self.VALID_COLORS[shape_colors[idx]]
        elif self.p_type == "max_time":
            idx = durations.index(max(durations[1:]))
            self.answer = self.VALID_COLORS[shape_colors[idx]]

        self.wait(2)
        self.play(*[FadeOut(mob) for mob in self.mobjects if mob!=bg])
        title = random.choice(self.cfg["text"][self.p_type])
        title = title.replace("<N>", ordinal(N))
        lines = title.split('\n')
        
        para = Paragraph(
            *lines, alignment="center", font_size=36, line_spacing=0.8
        )

        para.move_to(ORIGIN)
        if para.width > 0.9*config.frame_width:
            para.scale_to_fit_width(config.frame_width * 0.9)
        self.play(Write(para), run_time=1.5)

        self.wait()
        self.question_text = "Observe the trajectory of the arrow. {}".format(title.replace('\n', ' '))


def create_problem(p_type, num_shapes, path, file_name):
    config.output_file = f'{path}/questions/{file_name}'
    scene = Paths(p_type, num_shapes)
    scene.render()
    with open(f'media/videos/1080p60/{path}/solutions/{file_name}.txt', 'w') as f:
        f.write(str(scene.answer))
    with open(f"media/videos/1080p60/{path}/question_text/{file_name}.txt", "w") as f:
        f.write(scene.question_text)

    
if __name__ == "__main__":
    os.makedirs('media/videos/1080p60/path/questions', exist_ok=True)
    os.makedirs('media/videos/1080p60/path/solutions', exist_ok=True)
    os.makedirs("media/videos/1080p60/path/question_text", exist_ok=True)
    types = ["order", "min_dist", "max_dist", "min_time", "max_time"]
    answers = []
    for p_type in types:
        create_problem(p_type, 4, 'path', f'{p_type}_1')
        create_problem(p_type, 5, 'path', f'{p_type}_2')
        create_problem(p_type, 6, 'path', f'{p_type}_3')