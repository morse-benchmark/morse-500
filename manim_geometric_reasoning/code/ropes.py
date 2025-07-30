from manim import *
import random
import numpy as np
import json
from shapely.geometry import LineString
from shapely.ops import unary_union, polygonize
import os


def ordinal(n):
    if 11 <= (n % 100) <= 13:
        return f"{n}th"
    return f"{n}{ {1:'st', 2:'nd', 3:'rd'}.get(n%10, 'th') }"


class RopeCutScene(Scene):
    def __init__(
        self, p_type, cfg_path="../templates/ropes.json", num_ropes=2, bends_per_rope=2
    ):
        super().__init__()
        self.p_type = p_type
        self.num_ropes = num_ropes
        self.bends_per_rope = bends_per_rope

        with open(cfg_path, "r") as f:
            self.cfg = json.load(f)

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

    def construct(self):
        bg = (
            Rectangle(height=config.frame_height, width=config.frame_width)
            .set_color(
                color_gradient([random_bright_color(), random_bright_color()], 5)
            )
            .set_opacity(0.6)
        )
        self.add(bg)
        VALID_COLORS = {
            "blue": BLUE,
            "red": RED,
            "orange": ORANGE,
            "green": GREEN,
            "yellow": YELLOW,
            "purple": PURPLE,
            "white": WHITE,
        }
        if self.p_type == "order":
            self.show_colors(list(VALID_COLORS.values()), list(VALID_COLORS.keys()))
        prompt = Text(
            "Observe the following scene", color=WHITE, font_size=36
        ).move_to(ORIGIN)
        self.play(FadeIn(prompt), run_time=0.5)
        self.wait(1.5)
        self.play(FadeOut(prompt), run_time=0.5)
        self.wait(1)
        colors = list(VALID_COLORS.keys())
        random.shuffle(colors)

        # Full screen bounds
        frame_width = config.frame_width
        frame_height = config.frame_height
        x_min, x_max = -frame_width / 2, frame_width / 2
        y_min, y_max = -frame_height / 2, frame_height / 2

        ropes = []
        lengths = []
        for i in range(self.num_ropes):
            start = np.array(
                [random.uniform(x_min, x_max), random.uniform(y_min, y_max), 0]
            )
            end = np.array(
                [random.uniform(x_min, x_max), random.uniform(y_min, y_max), 0]
            )
            lengths.append(np.sqrt(np.sum((start - end) ** 2)))
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
            color = VALID_COLORS[colors[i]]

            # Build a VMobject and smooth it through all these points
            rope = VMobject()
            rope.set_points_smoothly(pts)
            rope.set_stroke(color, 3)
            ropes.append(rope)

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

        # Animate ropes drawing
        self.play(Create(rope_group, run_time=2))
        N = random.randint(1, self.num_ropes)
        if self.p_type == "count":
            self.answer = self.num_ropes
        elif self.p_type == "closed":
            line_strings = []
            for rope in ropes:
                # sample 200 points along the Manim curve
                samples = [
                    tuple(rope.point_from_proportion(t)[:2])
                    for t in np.linspace(0, 1, 200)
                ]
                line_strings.append(LineString(samples))

            # 2) Merge them into one geometry
            merged = unary_union(line_strings)

            # 3) Polygonize: this returns only the finite faces (i.e. your closed loops)
            polygons = list(polygonize(merged))

            # 4) Count them
            num_closed_shapes = len(polygons)
            self.answer = num_closed_shapes
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

            # Compute intersections and section counts

            total_sections = 0
            for rope in ropes:
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
                total_sections += crossings + 1
            self.answer = total_sections
        elif self.p_type == "order":
            self.answer = colors[N - 1]

        # Display result prompt
        self.wait(3)
        self.play(*[FadeOut(mob) for mob in self.mobjects if mob != bg])
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
        self.question_text = "Observe the following scene. {}".format(title.replace('\n', ' '))



def create_problem(num_ropes, p_type, bends_per_rope, path, file_name):
    config.output_file = f"{path}/questions/{file_name}"
    scene = RopeCutScene(p_type, num_ropes=num_ropes, bends_per_rope=bends_per_rope)
    scene.render()
    with open(f"media/videos/1080p60/{path}/solutions/{file_name}.txt", "w") as f:
        f.write(str(scene.answer))
    with open(f"media/videos/1080p60/{path}/question_text/{file_name}.txt", "w") as f:
        f.write(scene.question_text)


if __name__ == "__main__":
    os.makedirs("media/videos/1080p60/ropes/questions", exist_ok=True)
    os.makedirs("media/videos/1080p60/ropes/solutions", exist_ok=True)
    os.makedirs("media/videos/1080p60/ropes/question_text", exist_ok=True)
    types = ["count", "cut", "closed", "order"]
    for p_type in types:
        if p_type == "closed":
            create_problem(3, p_type, 1, "ropes", f"{p_type}_1")
            create_problem(4, p_type, 1, "ropes", f"{p_type}_2")
            create_problem(5, p_type, 1, "ropes", f"{p_type}_3")
        else:
            create_problem(3, p_type, 2, "ropes", f"{p_type}_1")
            create_problem(4, p_type, 2, "ropes", f"{p_type}_2")
            create_problem(5, p_type, 3, "ropes", f"{p_type}_3")
