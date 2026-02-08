from manim import *
import random
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


class CubeRollScene(ThreeDScene):
    """
    A 3D scene that generates a cube rolling puzzle:
    - Shows a colored cube rolling along a path of squares
    - User must determine which color face ends up on top
    - Generates question video, solution, and reasoning trace
    """

    BASE_SIZE = 1.5  # Size of base squares and cube (will be rescaled to fit camera)

    # ========================================================================
    # Face color configuration
    # ========================================================================
    # Manim Cube faces are ordered: [front(+Y), back(-Y), right(+X), left(-X), top(+Z), bottom(-Z)]
    FACE_COLORS_LIST = [BLUE, RED, GREEN, YELLOW, ORANGE, PURPLE]
    FACE_COLOR_NAMES = ["blue", "red", "green", "yellow", "orange", "purple"]

    # Color to name mapping for easy lookup
    FACE_COLORS = {
        BLUE: "blue",
        RED: "red",
        GREEN: "green",
        YELLOW: "yellow",
        ORANGE: "orange",
        PURPLE: "purple",
    }

    # ========================================================================
    # Direction mapping for cube movement
    # ========================================================================
    # Maps direction codes to 3D vectors
    # 1/-1: right/left, 2/-2: forward/backward
    DIRS = {
        1: RIGHT,  # +X direction
        -1: LEFT,  # -X direction
        2: DOWN,  # +Y direction (in Manim's default 3D view, this is "forward")
        -2: UP,  # -Y direction ("backward")
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Set random seed for reproducibility of path generation
        self.seed = os.getenv("SEED", random.randint(1000, 9999))

        # Parameters from environment variables (with defaults)
        self.path_length = int(os.getenv("PATH_LENGTH", 8))  # Number of squares in path
        self.max_attempts = int(
            os.getenv("MAX_ATTEMPTS", 100)
        )  # Max tries to generate valid path

        # Problem type is always "path" for this scene
        self.p_type = "path"

        # Initialize reasoning trace storage
        self.reasoning_trace = []

        # Track timing for reasoning trace using Manim's internal video time
        # This will be set when construct() is called and renderer is available
        self.scene_events = []

    def log_event(self, description):
        """
        Log a scene event with video timestamp.
        Uses Manim's internal renderer.time which tracks actual video playback time,
        not wall-clock execution time.
        """
        # Get current video time from Manim's renderer
        # self.renderer.time tracks the cumulative duration of all animations/waits
        current_time = self.renderer.time

        self.scene_events.append({"time": current_time, "description": description})

    def generate_valid_path(self, length):
        """
        Generate a valid path of connected squares without branching.

        Returns a list of direction codes that create a simple chain where:
        - Each new square connects to exactly one existing square
        - No square has more than 2 neighbors (prevents branching)

        Args:
            length: Number of squares in the path

        Returns:
            List of direction codes (from DIRS) representing the path

        Raises:
            ValueError: If unable to generate valid path after max_attempts
        """
        attempts = 0
        while attempts < self.max_attempts:
            attempts += 1
            path_dirs = []  # Store the sequence of moves
            occupied = {(0, 0)}  # Track which grid positions are filled
            pos = (0, 0)  # Current position on 2D grid (x, z)

            def neighbours(pt):
                """Count how many existing neighbors a point would have"""
                x, z = pt
                return [
                    n
                    for (dx, dz) in [(1, 0), (-1, 0), (0, 1), (0, -1)]
                    if (n := (x + dx, z + dz)) in occupied
                ]

            success = True
            # Generate length-1 moves (n squares need n-1 moves between them)
            for step in range(length - 1):
                # Try directions in random order
                possible = list(self.DIRS.keys())
                random.shuffle(possible)
                move_chosen = None

                for d in possible:
                    # Calculate next position
                    dx, dz = self.DIRS[d][:2]
                    nxt = (pos[0] + int(dx), pos[1] + int(dz))

                    # Skip if position already occupied
                    if nxt in occupied:
                        continue

                    # Skip if this would create branching (>1 neighbor)
                    if len(neighbours(nxt)) > 1:
                        continue

                    # Valid move found
                    move_chosen = d
                    break

                # If no valid move exists, restart
                if move_chosen is None:
                    success = False
                    break

                # Apply the move
                path_dirs.append(move_chosen)
                dx, dz = self.DIRS[move_chosen][:2]
                pos = (pos[0] + int(dx), pos[1] + int(dz))
                occupied.add(pos)

            if success:
                return path_dirs

        raise ValueError(
            f"Failed to generate valid path in {self.max_attempts} attempts"
        )

    def show_colors(self, colors, names, col_size=4):
        """
        Display color legend at the beginning of the video.
        Shows colored squares with arrows pointing to their names.

        Args:
            colors: List of Manim color objects
            names: List of color name strings
            col_size: Number of colors per column
        """
        cols = VGroup()
        all_squares = VGroup()
        all_arrows = VGroup()
        all_labels = VGroup()

        # Create columns of color swatches
        for start in range(0, len(colors), col_size):
            # Create colored squares for this column
            squares = VGroup(
                *[
                    Square(0.5).set_fill(col, 1).set_stroke(color=WHITE, width=2)
                    for col in colors[start : start + col_size]
                ]
            )
            squares.arrange(DOWN, buff=0.3, aligned_edge=LEFT).to_edge(LEFT, buff=2)

            # Create labels for each color
            labels = VGroup(
                *[Text(nm, font_size=20) for nm in names[start : start + col_size]]
            )
            for sq, lbl in zip(squares, labels):
                lbl.next_to(sq, RIGHT, buff=0.8)

            # Create arrows connecting squares to labels
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

        # Arrange all columns and fit to screen
        cols.arrange(buff=1.7, aligned_edge=UP).move_to(ORIGIN).scale_to_fit_width(
            0.9 * config.frame_width
        )

        # Add title
        title = Text("Remember the following color names", font_size=32)
        title.to_edge(UP)

        # Animate the color legend
        self.log_event("Title appears: 'Remember the following color names'")
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
        self.log_event(
            f"Color legend displayed with {len(colors)} colors: {', '.join(names)}"
        )

        # Clean up
        self.play(FadeOut(title, cols))
        self.wait(0.5)

    def fit_to_camera_3d(self, mobject, xbuffer=0.0, ybuffer=0.0):
        """
        Scale a 3D object to fit within the camera frame.

        Args:
            mobject: The Manim object to scale
            xbuffer: Fraction of width to leave as buffer (0.1 = 10% margin)
            ybuffer: Fraction of height to leave as buffer
        """
        frame_height = config.frame_height
        frame_width = config.frame_width

        # Scale down if too tall
        if mobject.height > frame_height * (1 - ybuffer):
            mobject.scale((frame_height * (1 - ybuffer)) / mobject.height)

        # Scale down if too wide
        if mobject.width > frame_width * (1 - xbuffer):
            mobject.scale((frame_width * (1 - xbuffer)) / mobject.width)

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

    def get_cube_face_info(self, cube):
        """
        Determine which color is on each face of the cube in its current orientation.

        This is crucial for tracking the cube state as it rolls.
        Uses face centers to identify which face points in which direction.
        This is more robust than using normal vectors.

        Args:
            cube: The Manim Cube object

        Returns:
            Dict mapping direction names to color names:
            {'top': 'blue', 'bottom': 'red', 'front': 'green', ...}
        """
        # Get the center of the entire cube
        cube_center = cube.get_center()

        face_info = {}

        # Define the 6 cardinal directions in Manim's 3D coordinate system
        # Important: Manim's Y axis points UP (not forward!)
        directions = {
            "top": OUT,  # +Z direction [0, 0, 1] (upward)
            "bottom": IN,  # -Z direction [0, 0, -1] (downward)
            "right": RIGHT,  # +X direction [1, 0, 0]
            "left": LEFT,  # -X direction [-1, 0, 0]
            "front": UP,  # +Y direction [0, 1, 0] (this is the "front" in Manim's initial cube orientation)
            "back": DOWN,  # -Y direction [0, -1, 0]
        }

        # For each direction, find which face is pointing that way
        for direction_name, direction_vec in directions.items():
            # Find the face whose center is furthest in this direction
            # We measure this by the dot product of (face_center - cube_center) with direction_vec
            best_face = max(
                cube, key=lambda f: np.dot(f.get_center() - cube_center, direction_vec)
            )

            # Get the color of this face
            color = self.FACE_COLORS[best_face.get_fill_color()]
            face_info[direction_name] = color

        return face_info

    def construct(self):
        """
        Main scene construction method.
        This is called by Manim to build and render the entire scene.
        """
        # Use the seed set in __init__ for reproducible randomness
        random.seed(self.seed)

        # ====================================================================
        # Create animated gradient background
        # ====================================================================
        bg = (
            Rectangle(height=config.frame_height, width=config.frame_width)
            .set_color(
                color_gradient([random_bright_color(), random_bright_color()], 5)
            )
            .set_opacity(0.6)
            .set_z_index(-2)  # Ensure it stays in background
        )
        self.add_fixed_in_frame_mobjects(
            bg
        )  # Keep background fixed during camera movement

        # ====================================================================
        # Show color legend
        # ====================================================================
        self.show_colors(self.FACE_COLORS_LIST, self.FACE_COLOR_NAMES)

        # ====================================================================
        # Setup 3D camera view
        # ====================================================================
        # phi: angle from the z-axis (70° = looking down at ~20° from horizontal)
        # theta: rotation around z-axis (-45° = viewing from front-right)
        self.set_camera_orientation(phi=70 * DEGREES, theta=-45 * DEGREES)

        # ====================================================================
        # Create and show demonstration cube
        # ====================================================================
        demo_cube = Cube(side_length=2)
        # Apply colors to each face (order matches FACE_COLORS_LIST)
        for face, col in zip(demo_cube, self.FACE_COLORS_LIST):
            face.set_fill(col, 0.9).set_stroke(BLACK, 0.9)
        demo_cube.move_to(ORIGIN)

        self.play(FadeIn(demo_cube, run_time=0.7))
        self.log_event("Cube appears at center showing all six colored faces")

        # Rotate cube to show all faces
        self.play(
            Rotate(
                demo_cube,
                angle=2 * PI,  # Full rotation
                axis=[1, 1, 1],  # Diagonal axis for interesting rotation
                about_point=demo_cube.get_center(),
            ),
            run_time=5,
            rate_func=smooth,
        )
        self.log_event("Cube finishes rotating to display all faces")

        # ====================================================================
        # Generate path and create squares
        # ====================================================================
        self.path_dirs = self.generate_valid_path(self.path_length)

        # Calculate 3D position for each square along the path
        positions = [ORIGIN]  # Start at origin
        for d in self.path_dirs:
            positions.append(positions[-1] + self.DIRS[d] * self.BASE_SIZE)

        # Track square counts for reasoning trace
        self.num_grey_squares = len(positions) - 1  # All but last
        self.num_red_squares = 1  # Just the target
        self.total_squares = len(positions)

        # Create the path squares
        squares = VGroup()
        for i, pos in enumerate(positions):
            sq = Square(self.BASE_SIZE)
            if i == len(positions) - 1:  # Last square is the target
                sq.set_fill(RED, 0.5).set_stroke(WHITE, 0.9)
            else:  # Other squares are grey path
                sq.set_fill(GREY_E, 0.9).set_stroke(WHITE, 0.9)
            sq.move_to(pos)
            squares.add(sq)

        # ====================================================================
        # Create the rolling cube
        # ====================================================================
        cube = Cube(side_length=self.BASE_SIZE)
        for face, col in zip(cube, self.FACE_COLORS_LIST):
            face.set_fill(col, 1.0).set_stroke(BLACK, 1.0)
        # Position cube on top of first square (half cube height above)
        cube.move_to(positions[0] + OUT * self.BASE_SIZE / 2)

        # ====================================================================
        # Fit everything to camera
        # ====================================================================
        track = VGroup(squares, cube)
        self.fit_to_camera_3d(track, xbuffer=0.1, ybuffer=0.15)

        # Transform demo cube into the rolling cube
        self.play(ReplacementTransform(demo_cube, cube), run_time=1.5)
        self.log_event("Cube transforms and moves to starting position on path")

        # Record initial cube orientation for reasoning trace
        self.initial_face_info = self.get_cube_face_info(cube)

        # Show the path
        self.play(FadeIn(squares), run_time=0.75)
        self.log_event(
            f"Path appears with {self.num_grey_squares} grey squares and {self.num_red_squares} red target square"
        )

        self.wait(0.3)

        # ====================================================================
        # Camera rotation for better view
        # ====================================================================
        self.move_camera(
            phi=70 * DEGREES,
            theta=-45 * DEGREES + TAU,  # Rotate 360° around the scene
            run_time=3,
            rate_func=smooth,
        )
        self.log_event("Camera completes rotation around the scene")
        self.wait(0.3)

        # ====================================================================
        # Define cube rolling mechanics
        # ====================================================================
        def bottom_edge(direction: int):
            """
            Find the edge of the cube that will act as the pivot for rolling.

            Args:
                direction: Direction code (1=right, -1=left, 2=forward, -2=back)

            Returns:
                Tuple of (point1, point2) defining the pivot edge
            """
            if direction == 1:  # Rolling right
                p1 = cube.get_corner(DOWN + RIGHT + IN)
                p2 = cube.get_corner(UP + RIGHT + IN)
            elif direction == -1:  # Rolling left
                p1 = cube.get_corner(DOWN + LEFT + IN)
                p2 = cube.get_corner(UP + LEFT + IN)
            elif direction == 2:  # Rolling forward
                p1 = cube.get_corner(DOWN + LEFT + IN)
                p2 = cube.get_corner(DOWN + RIGHT + IN)
            else:  # Rolling backward (-2)
                p1 = cube.get_corner(UP + LEFT + IN)
                p2 = cube.get_corner(UP + RIGHT + IN)
            return p1, p2

        def roll(mob, direction: int = 1, animate=True):
            """
            Roll the cube in the specified direction.

            Args:
                mob: The cube object to roll
                direction: Direction code
                animate: If True, show animation; if False, apply instantly
            """
            # Get the pivot edge for this roll direction
            p1, p2 = bottom_edge(direction)

            # Calculate rotation axis (along the pivot edge)
            axis_vec = (p2 - p1) / np.linalg.norm(p2 - p1)

            # 90° rotation (positive or negative depending on direction)
            angle = PI / 2 if direction > 0 else -PI / 2

            if animate:
                # Animated roll
                self.play(
                    Rotate(mob, angle=angle, axis=axis_vec, about_point=p1),
                    run_time=0.6,
                )
            else:
                # Instant roll (used for simulation)
                mob.rotate(angle=angle, axis=axis_vec, about_point=p1)

        # ====================================================================
        # Animate first portion of rolls and simulate the rest
        # ====================================================================
        self.roll_details = []  # Store info about each roll

        # Calculate how many rolls to animate (20% of path)
        n_rolls = int(0.2 * self.path_length) + 1
        direction_names = {1: "right", -1: "left", 2: "forward", -2: "backward"}

        # Animate the first n_rolls
        for i, d in enumerate(self.path_dirs[:n_rolls]):
            face_info_before = self.get_cube_face_info(cube)
            roll(cube, d)
            face_info_after = self.get_cube_face_info(cube)
            self.log_event(f"Cube completes roll {i + 1} ({direction_names[d]})")

            # Record this roll for reasoning trace
            self.roll_details.append(
                {
                    "roll_num": i + 1,
                    "direction": direction_names[d],
                    "face_info_before": face_info_before,
                    "face_info_after": face_info_after,
                }
            )

        # Simulate remaining rolls (no animation, just calculate final state)
        cube_sim = cube.copy()
        for i, d in enumerate(self.path_dirs[n_rolls:], n_rolls + 1):
            face_info_before = self.get_cube_face_info(cube_sim)
            roll(cube_sim, d, animate=False)  # Instant roll
            face_info_after = self.get_cube_face_info(cube_sim)

            self.roll_details.append(
                {
                    "roll_num": i,
                    "direction": direction_names[d],
                    "face_info_before": face_info_before,
                    "face_info_after": face_info_after,
                }
            )

        # ====================================================================
        # Determine final answer
        # ====================================================================
        final_face_info = self.get_cube_face_info(cube_sim)
        self.answer = final_face_info["top"]

        # ====================================================================
        # Transition to question display
        # ====================================================================
        self.wait(1.5)
        self.play(*[FadeOut(mob) for mob in self.mobjects if mob != bg])
        self.log_event("All objects fade out")

        # ====================================================================
        # Display the question
        # ====================================================================
        title_text = "If the cube continues along the path, what color face\nwill be on top once it reaches the red square?\nAnswer with a single color name"
        lines = title_text.split("\n")
        para = Paragraph(*lines, alignment="center", font_size=36, line_spacing=0.8)
        para.move_to(ORIGIN)

        # Scale if too wide
        if para.width > 0.9 * config.frame_width:
            para.scale_to_fit_width(config.frame_width * 0.9)

        self.add_fixed_in_frame_mobjects(para)  # Keep text fixed during camera movement

        self.play(Write(para), run_time=1.5)
        self.wait(3)
        self.log_event("Question displayed and remains on screen")

        # Store question text for output file
        self.question_text = (
            f"Observe the following structure. {title_text.replace(chr(10), ' ')}"
        )

        # ====================================================================
        # Generate reasoning trace
        # ====================================================================
        self.build_reasoning_trace()

        # ====================================================================
        # Save output files
        # ====================================================================
        # Solution file (just the answer)
        with open(
            f"solutions/cube_path_{self.p_type}_len{self.path_length}_seed{self.seed}.txt",
            "w",
        ) as f:
            f.write(str(self.answer))

        # Question text file
        with open(
            f"question_text/cube_path_{self.p_type}_len{self.path_length}_seed{self.seed}.txt",
            "w",
        ) as f:
            f.write(self.question_text)

        # Reasoning trace file
        with open(
            f"reasoning_traces/cube_path_{self.p_type}_len{self.path_length}_seed{self.seed}.txt",
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
        # Introduction
        # ====================================================================
        self.reasoning_trace.append(
            "**Question:** If the cube continues along the path, what color face will be on top once it reaches the red square?"
        )
        self.reasoning_trace.append("")
        self.reasoning_trace.append("Let's solve this step by step.")
        self.reasoning_trace.append("")

        # ====================================================================
        # Scene description with timestamps
        # ====================================================================
        self.reasoning_trace.append("### Scene Description")
        self.reasoning_trace.append("")

        for event in self.scene_events:
            time_str = self.format_time(event["time"])
            self.reasoning_trace.append(f"At {time_str}, {event['description']}")

        self.reasoning_trace.append("")

        # ====================================================================
        # Step 1: Initial cube configuration
        # ====================================================================
        self.reasoning_trace.append(
            "### Step 1: Understand the initial cube configuration"
        )
        self.reasoning_trace.append(
            f"At the beginning, the cube is positioned on the first square. The cube has six colored faces with the following arrangement:"
        )
        self.reasoning_trace.append("")
        self.reasoning_trace.append(f"- **Top face**: {self.initial_face_info['top']}")
        self.reasoning_trace.append(
            f"- **Bottom face**: {self.initial_face_info['bottom']} (opposite to top)"
        )
        self.reasoning_trace.append(
            f"- **Front face**: {self.initial_face_info['front']}"
        )
        self.reasoning_trace.append(
            f"- **Back face**: {self.initial_face_info['back']} (opposite to front)"
        )
        self.reasoning_trace.append(
            f"- **Right face**: {self.initial_face_info['right']}"
        )
        self.reasoning_trace.append(
            f"- **Left face**: {self.initial_face_info['left']} (opposite to right)"
        )
        self.reasoning_trace.append("")
        self.reasoning_trace.append(
            "This is the starting orientation before any movements."
        )
        self.reasoning_trace.append("")

        # ====================================================================
        # Step 2: Path structure
        # ====================================================================
        self.reasoning_trace.append("### Step 2: Understand the path structure")
        self.reasoning_trace.append(
            f"The path consists of **{self.total_squares} squares** in total:"
        )
        self.reasoning_trace.append(
            f"- {self.num_grey_squares} grey squares that form the path"
        )
        self.reasoning_trace.append(
            f"- {self.num_red_squares} red square at the end (the target)"
        )
        self.reasoning_trace.append("")
        self.reasoning_trace.append(
            f"To reach the red square, the cube must make **{len(self.path_dirs)} rolls** (one roll to move from each square to the next)."
        )
        self.reasoning_trace.append("")

        # ====================================================================
        # Step 3: Track movement
        # ====================================================================
        self.reasoning_trace.append("### Step 3: Track the cube's movement")

        # Show animated rolls in detail
        n_animated = int(0.2 * self.path_length) + 1
        if n_animated > 0:
            self.reasoning_trace.append(f"We observe the first {n_animated} rolls:")
            self.reasoning_trace.append("")
            for detail in self.roll_details[:n_animated]:
                before = detail["face_info_before"]
                after = detail["face_info_after"]
                self.reasoning_trace.append(
                    f"**Roll {detail['roll_num']}:** The cube rolls **{detail['direction']}**."
                )
                self.reasoning_trace.append(
                    f"  - Before: top = {before['top']}, bottom = {before['bottom']}"
                )
                self.reasoning_trace.append(
                    f"  - After: top = {after['top']}, bottom = {after['bottom']}"
                )
            self.reasoning_trace.append("")

        # Show summary of remaining rolls
        if len(self.roll_details) > n_animated:
            self.reasoning_trace.append(
                "The cube continues rolling along the path. Tracking the top face after each subsequent roll:"
            )
            self.reasoning_trace.append("")

            remaining = self.roll_details[n_animated:]
            if len(remaining) <= 5:
                # Show all if there are few remaining
                for detail in remaining:
                    after = detail["face_info_after"]
                    self.reasoning_trace.append(
                        f"- After roll {detail['roll_num']} ({detail['direction']}): top = {after['top']}, bottom = {after['bottom']}"
                    )
            else:
                # Show checkpoints for longer sequences
                checkpoints = [0, len(remaining) // 2, -1]
                for idx in checkpoints:
                    detail = remaining[idx]
                    after = detail["face_info_after"]
                    self.reasoning_trace.append(
                        f"- After roll {detail['roll_num']} ({detail['direction']}): top = {after['top']}, bottom = {after['bottom']}"
                    )
            self.reasoning_trace.append("")

        # ====================================================================
        # Step 4: Final determination
        # ====================================================================
        self.reasoning_trace.append("### Step 4: Determine the final orientation")
        final_detail = self.roll_details[-1]
        final_after = final_detail["face_info_after"]
        self.reasoning_trace.append(
            f"After completing all {len(self.path_dirs)} rolls, the cube reaches the red target square."
        )
        self.reasoning_trace.append(f"At this point:")
        self.reasoning_trace.append(f"  - Top face: **{final_after['top']}**")
        self.reasoning_trace.append(f"  - Bottom face: **{final_after['bottom']}**")
        self.reasoning_trace.append("")

        # ====================================================================
        # Final answer
        # ====================================================================
        self.reasoning_trace.append("### Final Answer")
        self.reasoning_trace.append(
            f"The color face on top when the cube reaches the red square is **{self.answer}**."
        )
        self.reasoning_trace.append("")
        self.reasoning_trace.append(f"\\boxed{{{self.answer}}}")


# ============================================================================
# Main execution
# ============================================================================
# Generate the cube path video
scene = CubeRollScene()
scene.render()

# ============================================================================
# Move output file to questions directory with descriptive name
# ============================================================================
output = Path("manim_output/videos/1080p30/CubeRollScene.mp4")
if output.exists():
    filename = f"cube_path_{scene.p_type}_len{scene.path_length}_seed{scene.seed}.mp4"
    shutil.move(str(output), f"questions/{filename}")
    print(f"✓ Video saved: questions/{filename}")
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
