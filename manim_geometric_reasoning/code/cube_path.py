from manim import *
import random
import os
from util import fit_to_camera_3d


class CubeRollScene(ThreeDScene):
    BASE_SIZE = 1.5  # base square + cube size (will be rescaled)
    FACE_COLORS = {
        BLUE: "blue",
        RED: "red",
        GREEN: "green",
        YELLOW: "yellow",
        ORANGE: "orange",
        PURPLE: "purple",
    }

    DIRS = {
        1: RIGHT,
        -1: LEFT,
        2: DOWN,
        -2: UP,
    }

    def __init__(self, path_len):
        super().__init__()
        self.PATH_LENGTH = path_len

    def generate_valid_path(self, length):
        """Return a list of direction codes creating a simple chain."""
        attempts = 0
        while True:
            attempts += 1
            path_dirs = []
            occupied = {(0, 0)}
            pos = (0, 0)  # use 2‑D grid (x,z). y is constant 0.

            def neighbours(pt):
                x, z = pt
                return [
                    n
                    for (dx, dz) in [(1, 0), (-1, 0), (0, 1), (0, -1)]
                    if (n := (x + dx, z + dz)) in occupied
                ]

            success = True
            for _ in range(length - 1):
                possible = list(self.DIRS.keys())
                random.shuffle(possible)
                move_chosen = None
                for d in possible:
                    dx, dz = self.DIRS[d][:2]
                    nxt = (pos[0] + int(dx), pos[1] + int(dz))
                    if nxt in occupied:
                        continue
                    if len(neighbours(nxt)) > 1:
                        continue  # new square would have >1 existing neighbour
                    move_chosen = d
                    break
                if move_chosen is None:
                    success = False
                    break
                path_dirs.append(move_chosen)
                dx, dz = self.DIRS[move_chosen][:2]
                pos = (pos[0] + int(dx), pos[1] + int(dz))
                occupied.add(pos)
            if success:
                print(f"Generated path after {attempts} attempts: {path_dirs}")
                return path_dirs

    def show_colors(self, colors, names, col_size=4):

        cols = VGroup()
        all_squares = VGroup()
        all_arrows = VGroup()
        all_labels = VGroup()
        for start in range(0, len(colors), col_size):

            squares = VGroup(
                *[
                    Square(1.0).set_fill(col, 1).set_stroke(color=WHITE, width=2)
                    for col in colors[start : start + col_size]
                ]
            )
            squares.arrange(DOWN, buff=0.5, aligned_edge=LEFT).to_edge(LEFT, buff=2)
            labels = VGroup(
                *[Text(nm, font_size=32) for nm in names[start : start + col_size]]
            )
            for sq, lbl in zip(squares, labels):
                lbl.next_to(sq, RIGHT, buff=1.2)
            arrows = VGroup(
                *[
                    Arrow(
                        start=sq.get_right(),
                        end=lbl.get_left(),
                        buff=0.05,
                        stroke_width=4,
                    )
                    for sq, lbl in zip(squares, labels)
                ]
            )
            col_group = VGroup(squares, arrows, labels)
            cols.add(col_group)
            all_squares.add(*squares)
            all_arrows.add(*arrows)
            all_labels.add(*labels)

        cols.arrange(buff=1.7, aligned_edge=UP).move_to(ORIGIN).scale_to_fit_width(
            0.9 * config.frame_width
        )

        # Title
        title = Text("Remember the following color names", font_size=40)
        title.to_edge(UP)

        self.play(Write(title))
        self.play(
            Succession(
                FadeIn(all_squares),
                AnimationGroup(*[GrowArrow(ar) for ar in all_arrows], lag_ratio=0.1),
                AnimationGroup(*[Write(label) for label in all_labels]),
            ),
            run_time=1.5,
        )
        self.wait(2)
        self.play(FadeOut(title, cols))
        self.wait(0.5)


    def construct(self):
        bg = (
            Rectangle(height=config.frame_height, width=config.frame_width)
            .set_color(
                color_gradient([random_bright_color(), random_bright_color()], 5)
            )
            .set_opacity(0.6)
        )
        self.add_fixed_in_frame_mobjects(bg)

        self.show_colors(list(self.FACE_COLORS.keys()), list(self.FACE_COLORS.values()))

        self.set_camera_orientation(phi=70 * DEGREES, theta=-45 * DEGREES)

        demo_cube = Cube(side_length=2)
        for face, col in zip(demo_cube, list(self.FACE_COLORS.keys())):
            face.set_fill(col, 0.9).set_stroke(BLACK, 0.9)
        demo_cube.move_to(ORIGIN)

        self.play(FadeIn(demo_cube, run_time=0.7))
        self.play(
            Rotate(
                demo_cube,
                angle=2 * PI,
                axis=[1, 1, 1],
                about_point=demo_cube.get_center(),
            ),
            run_time=5,
            rate_func=smooth,
        )

        self.PATH_DIRS = self.generate_valid_path(self.PATH_LENGTH)
        positions = [ORIGIN]
        for d in self.PATH_DIRS:
            positions.append(positions[-1] + self.DIRS[d] * self.BASE_SIZE)

        squares = VGroup()
        for pos in positions:
            sq = Square(self.BASE_SIZE)
            sq.set_fill(GREY_E, 0.9).set_stroke(WHITE, 0.9)
            sq.move_to(pos)
            squares.add(sq)
        squares[-1].set_fill(RED, 0.5)

        cube = Cube(side_length=self.BASE_SIZE)
        for face, col in zip(cube, list(self.FACE_COLORS.keys())):
            face.set_fill(col, 1.0).set_stroke(BLACK, 1.0)
        cube.move_to(positions[0] + OUT * self.BASE_SIZE / 2)

        track = VGroup(squares, cube)
        fit_to_camera_3d(track, self, xbuffer=0.1, ybuffer=0.15)
        self.play(ReplacementTransform(demo_cube, cube), run_time=1.5)
        self.play(FadeIn(squares), run_time=0.75)

        self.wait(0.3)

        self.move_camera(
            phi=70 * DEGREES, theta=-45 * DEGREES + TAU, run_time=3, rate_func=smooth
        )
        self.wait(0.3)

        def bottom_edge(direction: int):
            if direction == 1:  # right
                p1 = cube.get_corner(DOWN + RIGHT + IN)
                p2 = cube.get_corner(UP + RIGHT + IN)
            elif direction == -1:  # left
                p1 = cube.get_corner(DOWN + LEFT + IN)
                p2 = cube.get_corner(UP + LEFT + IN)
            elif direction == 2:  # forward (OUT)
                p1 = cube.get_corner(DOWN + LEFT + IN)
                p2 = cube.get_corner(DOWN + RIGHT + IN)
            else:  # back (IN)
                p1 = cube.get_corner(UP + LEFT + IN)
                p2 = cube.get_corner(UP + RIGHT + IN)
            return p1, p2

        def roll(mob, direction: int = 1, animate=True):
            p1, p2 = bottom_edge(direction)
            axis_vec = (p2 - p1) / np.linalg.norm(p2 - p1)
            angle = PI / 2 if direction > 0 else -PI / 2
            if animate:
                self.play(
                    Rotate(mob, angle=angle, axis=axis_vec, about_point=p1),
                    run_time=0.6,
                )
            else:
                mob.rotate(angle=angle, axis=axis_vec, about_point=p1)

        n_rolls = int(0.2 * self.PATH_LENGTH) + 1
        for d in self.PATH_DIRS[:n_rolls]:
            roll(cube, d)

        cube_sim = cube.copy()
        for d in self.PATH_DIRS[n_rolls:]:
            roll(cube_sim, d, animate=False)

        def unit_normal(face: Mobject):
            v = face.get_vertices()[:3]  # any three non-collinear vertices
            n = np.cross(v[1] - v[0], v[2] - v[0])
            return n / np.linalg.norm(n)

        score = lambda f: np.dot(unit_normal(f), IN)
        top_face = max(cube_sim, key=score)
        self.answer = self.FACE_COLORS[top_face.get_fill_color()]

        self.wait(1.5)
        self.play(*[FadeOut(mob) for mob in self.mobjects if mob != bg])
        title = "If the cube continues along the path, what color face\nwill be on top once it reaches the red square?\nAnswer with a single color name"
        lines = title.split("\n")
        para = Paragraph(*lines, alignment="center", font_size=36, line_spacing=0.8)
        self.add_fixed_in_frame_mobjects(para)
        para.move_to(ORIGIN)
        if para.width > 0.9 * config.frame_width:
            para.scale_to_fit_width(config.frame_width * 0.9)
        self.play(Write(para))
        self.question_text = title


def create_problem(path_len, path, file_name):
    config.output_file = f"{path}/questions/{file_name}"
    scene = CubeRollScene(path_len)
    scene.render()
    with open(f"media/videos/1080p60/{path}/solutions/{file_name}.txt", "w") as f:
        f.write(str(scene.answer))
    with open(f"media/videos/1080p60/{path}/question_text/{file_name}.txt", "w") as f:
        f.write(scene.question_text)


if __name__ == "__main__":
    os.makedirs("media/videos/1080p60/cube_rotation/questions", exist_ok=True)
    os.makedirs("media/videos/1080p60/cube_rotation/solutions", exist_ok=True)
    os.makedirs("media/videos/1080p60/cube_rotation/question_text", exist_ok=True)

    create_problem(5, "cube_rotation", f"cube_path_1")
    create_problem(10, "cube_rotation", f"cube_path_2")
    create_problem(15, "cube_rotation", f"cube_path_3")
