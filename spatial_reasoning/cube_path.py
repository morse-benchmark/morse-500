from manim import *
import random
import math
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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Set random seed for reproducibility
        self.seed = random.randint(1000, 9999)
        
        # Parameters from environment variables
        self.path_length = int(os.getenv("PATH_LENGTH", 8))
        self.max_attempts = int(os.getenv("MAX_ATTEMPTS", 100))
        
        # Problem type is always "path" for this scene
        self.p_type = "path"
        
        # Initialize reasoning trace
        self.reasoning_trace = []
        self.reasoning_trace.append(f"Problem Type: {self.p_type}")
        self.reasoning_trace.append(f"Path Length: {self.path_length}")
        self.reasoning_trace.append(f"Random Seed: {self.seed}")
        self.reasoning_trace.append("")

    def generate_valid_path(self, length):
        """Return a list of direction codes creating a simple chain."""
        self.reasoning_trace.append("Generating valid path:")
        attempts = 0
        while attempts < self.max_attempts:
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
            for step in range(length - 1):
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
                self.reasoning_trace.append(f"  Step {step + 1}: Move {move_chosen} to position {pos}")
            
            if success:
                self.reasoning_trace.append(f"Successfully generated path after {attempts} attempts")
                self.reasoning_trace.append(f"Final path: {path_dirs}")
                self.reasoning_trace.append("")
                return path_dirs
                
        raise ValueError(f"Failed to generate valid path in {self.max_attempts} attempts")

    def show_colors(self, colors, names, col_size=4):
        cols = VGroup()
        all_squares = VGroup()
        all_arrows = VGroup()
        all_labels = VGroup()
        for start in range(0, len(colors), col_size):
            squares = VGroup(
                *[
                    Square(0.5).set_fill(col, 1).set_stroke(color=WHITE, width=2)
                    for col in colors[start : start + col_size]
                ]
            )
            squares.arrange(DOWN, buff=0.3, aligned_edge=LEFT).to_edge(LEFT, buff=2)
            labels = VGroup(
                *[Text(nm, font_size=20) for nm in names[start : start + col_size]]
            )
            for sq, lbl in zip(squares, labels):
                lbl.next_to(sq, RIGHT, buff=0.8)
            arrows = VGroup(
                *[
                    Arrow(
                        start=sq.get_right(),
                        end=lbl.get_left(),
                        buff=0.05,
                        stroke_width=2,
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
        title = Text("Remember the following color names", font_size=32)
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

    def fit_to_camera_3d(self, mobject, xbuffer=0.0, ybuffer=0.0):
        """Simple 3D fitting function"""
        frame_height = config.frame_height
        frame_width = config.frame_width
        
        if mobject.height > frame_height * (1 - ybuffer):
            mobject.scale((frame_height * (1 - ybuffer)) / mobject.height)
        if mobject.width > frame_width * (1 - xbuffer):
            mobject.scale((frame_width * (1 - xbuffer)) / mobject.width)

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

        # Generate path and calculate positions
        self.path_dirs = self.generate_valid_path(self.path_length)
        positions = [ORIGIN]
        for d in self.path_dirs:
            positions.append(positions[-1] + self.DIRS[d] * self.BASE_SIZE)

        self.reasoning_trace.append("Creating path visualization:")
        self.reasoning_trace.append(f"Number of squares: {len(positions)}")
        
        squares = VGroup()
        for i, pos in enumerate(positions):
            sq = Square(self.BASE_SIZE)
            if i == len(positions) - 1:  # Last square
                sq.set_fill(RED, 0.5).set_stroke(WHITE, 0.9)
                self.reasoning_trace.append(f"  Square {i}: Target square (red) at {pos}")
            else:
                sq.set_fill(GREY_E, 0.9).set_stroke(WHITE, 0.9)
                self.reasoning_trace.append(f"  Square {i}: Path square at {pos}")
            sq.move_to(pos)
            squares.add(sq)

        cube = Cube(side_length=self.BASE_SIZE)
        for face, col in zip(cube, list(self.FACE_COLORS.keys())):
            face.set_fill(col, 1.0).set_stroke(BLACK, 1.0)
        cube.move_to(positions[0] + OUT * self.BASE_SIZE / 2)

        track = VGroup(squares, cube)
        self.fit_to_camera_3d(track, xbuffer=0.1, ybuffer=0.15)
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

        # Animate initial rolls
        n_rolls = int(0.2 * self.path_length) + 1
        self.reasoning_trace.append(f"Animating first {n_rolls} rolls:")
        for i, d in enumerate(self.path_dirs[:n_rolls]):
            self.reasoning_trace.append(f"  Roll {i + 1}: Direction {d}")
            roll(cube, d)

        # Simulate remaining rolls
        cube_sim = cube.copy()
        self.reasoning_trace.append(f"Simulating remaining {len(self.path_dirs[n_rolls:])} rolls:")
        for i, d in enumerate(self.path_dirs[n_rolls:], n_rolls + 1):
            self.reasoning_trace.append(f"  Roll {i}: Direction {d}")
            roll(cube_sim, d, animate=False)

        # Calculate final answer
        def unit_normal(face: Mobject):
            v = face.get_vertices()[:3]  # any three non-collinear vertices
            n = np.cross(v[1] - v[0], v[2] - v[0])
            return n / np.linalg.norm(n)

        score = lambda f: np.dot(unit_normal(f), IN)
        top_face = max(cube_sim, key=score)
        self.answer = self.FACE_COLORS[top_face.get_fill_color()]
        
        self.reasoning_trace.append("")
        self.reasoning_trace.append("Calculating final answer:")
        self.reasoning_trace.append(f"Top face color after all rolls: {self.answer}")

        self.wait(1.5)
        self.play(*[FadeOut(mob) for mob in self.mobjects if mob != bg])
        
        # Display question
        title_text = "If the cube continues along the path, what color face\nwill be on top once it reaches the red square?\nAnswer with a single color name"
        lines = title_text.split("\n")
        para = Paragraph(*lines, alignment="center", font_size=36, line_spacing=0.8)
        para.move_to(ORIGIN)
        if para.width > 0.9 * config.frame_width:
            para.scale_to_fit_width(config.frame_width * 0.9)
        self.add_fixed_in_frame_mobjects(para)
        self.play(Write(para), run_time=1.5)
        self.wait(3)
        
        self.question_text = f"Observe the following structure. {title_text.replace(chr(10), ' ')}"
        
        # Save files
        with open(f"solutions/cube_path_{self.p_type}_len{self.path_length}_seed{self.seed}.txt", "w") as f:
            f.write(str(self.answer))
        
        with open(f"question_text/cube_path_{self.p_type}_len{self.path_length}_seed{self.seed}.txt", "w") as f:
            f.write(self.question_text)
            
        # Save detailed reasoning trace
        self.reasoning_trace.append("")
        self.reasoning_trace.append(f"Final Answer: {self.answer}")
        with open(f"reasoning_traces/cube_path_{self.p_type}_len{self.path_length}_seed{self.seed}.txt", "w") as f:
            f.write("\n".join(self.reasoning_trace))

# Generate the cube path video
scene = CubeRollScene()
scene.render()

# Move the output file with descriptive name
output = Path("manim_output/videos/1080p30/CubeRollScene.mp4")
if output.exists():
    filename = f"cube_path_{scene.p_type}_len{scene.path_length}_seed{scene.seed}.mp4"
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