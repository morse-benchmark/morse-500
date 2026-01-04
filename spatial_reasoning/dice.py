from manim import *
import random
import numpy as np
import os
import shutil
from pathlib import Path

# ============================================================================
# Setup directories for output files
# ============================================================================
Path("questions").mkdir(exist_ok=True)          # Video files
Path("solutions").mkdir(exist_ok=True)          # Answer text files
Path("question_text").mkdir(exist_ok=True)      # Question text files
Path("reasoning_traces").mkdir(exist_ok=True)   # Step-by-step reasoning

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
# Constants for cube orientation and rotation
# ============================================================================
ORDER = ["IN", "OUT", "LEFT", "RIGHT", "UP", "DOWN"]
NAME_TO_VEC = {
    "IN":   np.array([ 0,  0, -1]),
    "OUT":  np.array([ 0,  0,  1]),
    "LEFT": np.array([-1,  0,  0]),
    "RIGHT":np.array([ 1,  0,  0]),
    "UP":   np.array([ 0,  1,  0]),
    "DOWN": np.array([ 0, -1,  0]),
}
NAME_TO_IDX = {name: i for i, name in enumerate(ORDER)}
VEC_TO_NAME = {tuple(v): n for n, v in NAME_TO_VEC.items()}

# ============================================================================
# Cube rotation enumeration
# ============================================================================
def enumerate_rotations():
    """
    Enumerate all 24 possible cube orientations as index permutations.
    
    Each permutation maps ORDER[i] (original face) to world axis i after rotation.
    Example: p[3] = 0 means the original RIGHT face ends up in the IN position.
    
    Returns:
        List of 24 permutations representing all valid cube rotations
    """
    rots = []

    for up_name, up_vec in NAME_TO_VEC.items():
        for front_name, front_vec in NAME_TO_VEC.items():
            # Skip if same face or not perpendicular
            if front_name == up_name:
                continue
            if not np.isclose(np.dot(up_vec, front_vec), 0):
                continue

            # Calculate right face using cross product
            right_vec = np.cross(up_vec, front_vec)
            if np.linalg.norm(right_vec) < 0.5:
                continue
            right_name = VEC_TO_NAME[tuple(right_vec.astype(int))]

            # Build complete face mapping for this orientation
            mapping_names = {
                "UP": up_name,
                "DOWN": VEC_TO_NAME[tuple((-up_vec).astype(int))],
                "OUT": front_name,
                "IN": VEC_TO_NAME[tuple((-front_vec).astype(int))],
                "RIGHT": right_name,
                "LEFT": VEC_TO_NAME[tuple((-right_vec).astype(int))],
            }
            # Convert to index-based permutation
            mapping = [NAME_TO_IDX[mapping_names[axis]] for axis in ORDER]
            if mapping not in rots:
                rots.append(mapping)
    return rots

ROTATIONS = enumerate_rotations() 

def build_cube_map(color_list, side=0.55):
    """
    Build a T-shaped cube net visualization.
    
    The net layout is:
           UP
    LEFT  OUT  RIGHT  IN
           DOWN
    
    Args:
        color_list: List of 6 colors in ORDER (IN, OUT, LEFT, RIGHT, UP, DOWN)
        side: Size of each square in the net
        
    Returns:
        VGroup containing the colored squares arranged as a cube net
    """
    # Define grid positions (x, y) for each face
    offsets = {
        "OUT": (0, 0),
        "UP":  (0, 1),
        "DOWN":(0,-1),
        "LEFT":(-1,0),
        "RIGHT":(1,0),
        "IN":  (2,0),
    }
    g = VGroup()
    base = Square(side_length=side)
    gap = 0.05 * side  # Small gap between squares for clarity
    
    for name, (dx, dy) in offsets.items():
        idx = NAME_TO_IDX[name]
        sq = base.copy().set_fill(color_list[idx], 1).set_stroke(BLACK, 0.5)
        sq.shift(RIGHT * dx * (side + gap) + UP * dy * (side + gap))
        g.add(sq)
    return g

class Dice(ThreeDScene):
    """
    A 3D scene that generates various dice-related puzzles:
    - hidden: Count faces of a color that are hidden from view
    - max_hidden: Find most frequent color among hidden faces
    - match: Identify which cube matches the shown orientation
    - roll: Find most frequently rolled color
    - n_roll: Count rolls of a specific color
    - fold: Identify which net folds into the shown cube
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Set random seed for reproducibility
        self.seed = random.randint(1000, 9999)
        
        # Parameters from environment variables (with defaults)
        self.p_type = os.getenv("P_TYPE", "hidden")
        self.num_dice = int(os.getenv("NUM_DICE", 3))
        self.n_roll = int(os.getenv("N_ROLL", 3))
        
        # Valid color palette with names
        self.VALID_COLORS = {
            BLUE: "blue", 
            RED: "red", 
            WHITE: "white",
            GREEN: "green", 
            YELLOW: "yellow",
            PURPLE: "purple",
        }
        
        # Question templates for different problem types
        self.cfg = {
            "text": {
                "hidden": [
                    "How many <C> faces are currently hidden from the camera?\nAnswer with a single integer."
                ],
                "max_hidden": [
                    "Out of the faces currently hidden from the\ncamera, which color appears most often?\nAnswer with only the color name."
                ],
                "match": [
                    "Which cube matches the one shown in the video?\nAnswer with one multiple choice option."
                ],
                "roll": [
                    "Which color was rolled most often?\nAnswer with only the color name."
                ],
                "n_roll": [
                    "How many times was <C> rolled in total?\nAnswer with a single integer."
                ],
                "fold": [
                    "Which cube net folds into the cube shown in the video?\nAnswer with one multiple choice option."
                ]
            }
        }
        
        # Initialize tracking for reasoning trace
        self.reasoning_trace = []
        self.scene_events = []  # Video timeline events
        self.cube_details = []  # Detailed cube configurations

    def log_event(self, description):
        """
        Log a scene event with video timestamp.
        
        Uses Manim's renderer.time which tracks actual video playback time,
        not wall-clock execution time. This provides accurate timestamps for
        the reasoning trace.
        
        Args:
            description: Human-readable description of the event
        """
        current_time = self.renderer.time
        self.scene_events.append({
            'time': current_time,
            'description': description
        })

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

    def axis_to_idx_exact(self, v):
        """
        Convert a direction vector to its index in the cube faces array.
        
        This is used to determine which face is pointing in a given direction
        after the cube has been rotated or rolled.
        
        Args:
            v: Direction vector (one of IN, OUT, LEFT, RIGHT, UP, DOWN)
            
        Returns:
            Index (0-5) of the corresponding face
            
        Raises:
            ValueError: If vector doesn't match any standard axis
        """
        AXES = [IN, OUT, LEFT, RIGHT, UP, DOWN] 
        for i, axis in enumerate(AXES):
            if np.array_equal(v, axis):
                return i
        raise ValueError(f"Axis {v} not found in standard axes")

    # === BEGIN FIX: track cube orientation during rolls ===
    def update_dir_map(self, dir_map, curr_face, new_face):
        """
        Update a face->world-axis mapping after a roll from curr_face to new_face.

        Args:
            dir_map: Dict mapping face names to world-space direction vectors
            curr_face: Current top-facing direction vector
            new_face: Desired top-facing direction vector

        Returns:
            Updated mapping with all face directions rotated
        """
        axis = np.cross(curr_face, new_face)
        if np.all(axis == 0):
            axis = RIGHT if not np.array_equal(curr_face, RIGHT) else OUT
            angle = -PI if not np.array_equal(curr_face, new_face) else 0
        else:
            axis = axis / np.linalg.norm(axis)
            angle = -PI / 2
        rot = utils.space_ops.rotation_matrix(angle=angle, axis=axis)
        return {k: np.round(rot @ v) for k, v in dir_map.items()}

    def get_face_color_from_dir_map(self, cube, dir_map, world_axis):
        """
        Resolve the color of the face currently pointing in world_axis.
        """
        for name, vec in dir_map.items():
            if np.allclose(vec, world_axis, atol=1e-6):
                idx = self.axis_to_idx_exact(NAME_TO_VEC[name])
                return cube[idx].get_fill_color()
        raise ValueError(f"World axis {world_axis} not found in dir map")
    # === END FIX: track cube orientation during rolls ===
        
    def roll_to_face(self, cube, curr_face, new_face):
        """
        Generate animations to roll a cube from one face to another.
        
        First does a random tumble for visual interest, then rotates to
        place the desired face on top.
        
        Args:
            cube: The cube object to animate
            curr_face: Current top-facing direction vector
            new_face: Desired top-facing direction vector
            
        Returns:
            List of Animation objects to execute
        """
        animations = []
        
        # Random tumble for visual variety (doesn't affect final orientation)
        axis = np.random.normal(size=3)
        axis /= np.linalg.norm(axis)
        animations.append(
            Rotate(
                cube, 
                angle=2*PI,
                axis=axis,
                about_point=cube.get_center()
            )
        )
        
        # If already on correct face, we're done
        if np.array_equal(curr_face, new_face):
            return animations
            
        # Calculate rotation needed to reach new face
        axis = np.cross(curr_face, new_face)
        if np.all(axis == 0):
            # Opposite face case (e.g., DOWN → UP requires 180° rotation)
            # Choose any perpendicular axis
            axis = RIGHT if not np.array_equal(curr_face, RIGHT) else OUT
            angle = -PI
        else:
            # Normal case: 90° rotation around perpendicular axis
            axis = axis / np.linalg.norm(axis)
            angle = -PI/2

        animations.append(
            Rotate(
                cube, 
                angle=angle,
                axis=axis,
                about_point=cube.get_center()
            )
        )
        return animations
        
    def apply_perm(self, colors, mapping):
        """
        Apply a permutation to reorder colors according to a rotation.
        
        Args:
            colors: List of 6 colors in standard order
            mapping: Permutation indices (from ROTATIONS)
            
        Returns:
            Reordered list of colors after rotation
        """
        return [colors[i] for i in mapping]
        
    def build_cube(self, color_list, side=2.0):
        """
        Build a Manim cube with specified face colors.
        
        Args:
            color_list: List of 6 colors in order [IN, OUT, LEFT, RIGHT, UP, DOWN]
            side: Side length of the cube
            
        Returns:
            Manim Cube object with colored faces
        """
        cube = Cube(side_length=side)
        for face, col in zip(cube, color_list):
            face.set_fill(col, 1).set_stroke(BLACK, 0.5)
        return cube
        
    def show_colors(self, colors, names):
        """
        Display color legend showing color squares with their names.
        
        Creates two columns of color swatches with arrows pointing to labels.
        This helps users associate colors with their names before the puzzle.
        
        Args:
            colors: List of Manim color objects
            names: List of corresponding color name strings
        """
        # Create first column (first 3 colors)
        squares1 = VGroup(
            *[Square(0.5).set_fill(col, 1).set_stroke(width=0) for col in colors[:3]]
        )
        squares1.arrange(DOWN, buff=0.3, aligned_edge=LEFT).to_edge(LEFT, buff=2)

        labels1 = VGroup(*[Text(nm, font_size=20) for nm in names[:3]])
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

        # Create second column (last 3 colors)
        squares2 = VGroup(
            *[Square(0.5).set_fill(col, 1).set_stroke(width=0) for col in colors[3:]]
        )
        squares2.arrange(DOWN, buff=0.3, aligned_edge=LEFT).to_edge(LEFT, buff=2)

        labels2 = VGroup(*[Text(nm, font_size=20) for nm in names[3:]])
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

        # Arrange both columns side by side
        left = VGroup(squares1, arrows1, labels1)
        right = VGroup(squares2, arrows2, labels2)
        both = VGroup(left, right)
        both.arrange(buff=1.7, aligned_edge=UP).move_to(ORIGIN)

        # Add title
        title = Text("Remember the following color names", font_size=32)
        title.to_edge(UP)

        # Animation sequence: title → squares → arrows → labels
        self.log_event("Title appears: 'Remember the following color names'")
        self.play(Write(title))
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
        self.log_event(f"Color legend displayed with {len(colors)} colors: {', '.join(names)}")
        self.wait(2)
        self.play(FadeOut(title, both))
        self.log_event("Color legend fades out")
        self.wait(0.5)
        
    def construct(self):
        """
        Main scene construction method.
        Called by Manim to build and render the entire scene.
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
        self.add_fixed_in_frame_mobjects(bg)  # Keep background fixed during camera movement
        
        # ====================================================================
        # Show color legend (except for match/fold which use standard cubes)
        # ====================================================================
        if self.p_type not in ["match", "fold"]:
            self.show_colors(list(self.VALID_COLORS.keys()), list(self.VALID_COLORS.values()))
        
        # ====================================================================
        # Display initial prompt
        # ====================================================================
        if "roll" in self.p_type:
            prompt_text = "After each roll, record the color on the top of each cube"
            self.question_text = "After each roll, record the color on the top of each cube. "
        else:
            prompt_text = "Observe the following cubes"
            self.question_text = "Observe the following cubes. "

        prompt = Text(prompt_text, color=WHITE, font_size=36).move_to(ORIGIN)
        
        self.log_event(f"Prompt appears: '{prompt_text}'")
        self.play(FadeIn(prompt), run_time=0.5)
        self.wait(1.5)
        self.play(FadeOut(prompt), run_time=0.5)
        self.log_event("Prompt fades out")
        self.wait(1)
        
        # ====================================================================
        # Setup 3D camera view
        # ====================================================================
        # phi: angle from z-axis (70° = looking down at ~20° from horizontal)
        # theta: rotation around z-axis (-45° = viewing from front-right)
        self.set_camera_orientation(phi=70 * DEGREES, theta=-45 * DEGREES)
        
        # ====================================================================
        # Create cubes with random colors
        # ====================================================================
        cubes = VGroup()
        cube_colors = []  # Store color configuration for each cube
        
        self.log_event(f"Creating {self.num_dice} cubes with random colors")
        for i in range(self.num_dice):
            c = Cube(side_length=1.5)

            # Assign random colors to each face (IN, OUT, LEFT, RIGHT, UP, DOWN)
            c_color = []
            color_names = []
            for j, face in enumerate(c):
                color = random.choice(list(self.VALID_COLORS.keys()))
                face.set_fill(color, 1).set_stroke(BLACK, 1)
                c_color.append(color)
                color_names.append(self.VALID_COLORS[color])
            cube_colors.append(c_color)
            
            # Store cube configuration for reasoning trace
            self.cube_details.append({
                'cube_num': i + 1,
                'colors': color_names,
                'color_objects': c_color
            })
            
            # Position cubes in a row (3 units apart)
            c.shift(RIGHT * i * 3)
            cubes.add(c)
            
        # Center the entire row on screen
        cubes.move_to(ORIGIN)
        self.add(cubes)
        self.log_event(f"All {self.num_dice} cubes appear on screen")
        
        # ====================================================================
        # Select random color for color-specific questions
        # ====================================================================
        color = random.choice(list(self.VALID_COLORS.keys()))
        self.selected_color_name = self.VALID_COLORS[color]

        # ====================================================================
        # Calculate hidden faces (for hidden/max_hidden problems)
        # ====================================================================
        # From camera view at phi=70°, theta=-45°:
        # - IN (index 0) is hidden (facing away)
        # - LEFT (index 2) is hidden (facing left)
        # - UP (index 4) is hidden (facing top, hard to see at this angle)
        hidden = {}
        for i, c in enumerate(cube_colors):
            hidden_faces = [c[0], c[2], c[4]]  # IN, LEFT, UP
            for face_color in hidden_faces:
                hidden[face_color] = hidden.get(face_color, 0) + 1

        # ====================================================================
        # Execute problem-specific animations and calculations
        # ====================================================================
        choices = None  # For multiple choice options
        
        if self.p_type == "fold" or self.p_type == "match":
            # ================================================================
            # Match/Fold: Rotate cube to show all faces
            # ================================================================
            self.log_event("Cubes begin rotating to display all faces")
            self.play(*[
                Rotate(
                    cube,
                    angle=2 * PI,
                    axis=[1,1,1],  # Diagonal axis for interesting rotation
                    about_point=cube.get_center(),
                )
                for cube in cubes
            ],
                run_time=5,
                rate_func=smooth
            )
            self.log_event("Cubes finish rotation")
            
        elif self.p_type == "hidden":
            # ================================================================
            # Hidden: Count specific color in hidden faces
            # ================================================================
            self.log_event("Cubes begin rotating to display orientation")
            self.play(*[
                Rotate(
                    cube,
                    angle=2 * PI,
                    axis=[1,1,1],
                    about_point=cube.get_center(),
                )
                for cube in cubes
            ],
                run_time=3,
                rate_func=smooth
            )
            self.log_event("Cubes finish rotation, hidden faces determined")
            self.wait()
            
            self.answer = hidden.get(color, 0)
            
        elif self.p_type == "max_hidden":
            # ================================================================
            # Max Hidden: Find most frequent color in hidden faces
            # ================================================================
            self.log_event("Cubes begin rotating to display orientation")
            self.play(*[
                Rotate(
                    cube,
                    angle=2 * PI,
                    axis=[1,1,1],
                    about_point=cube.get_center(),
                )
                for cube in cubes
            ],
                run_time=3,
                rate_func=smooth
            )
            self.log_event("Cubes finish rotation, hidden faces determined")
            self.wait()
            
            # Verify unique maximum
            max_val = max(hidden.values())
            max_keys = [k for k, v in hidden.items() if v == max_val]
            if len(max_keys) > 1:
                raise ValueError(f"Multiple colors have same max value: {[self.VALID_COLORS[k] for k in max_keys]}")
            
            self.answer = self.VALID_COLORS[max(hidden, key=hidden.get)]
            
        elif "roll" in self.p_type:
            # ================================================================
            # Roll: Track colors after each roll
            # ================================================================
            # === BEGIN FIX: track cube orientation during rolls ===
            curr_faces = [OUT for _ in range(self.num_dice)]  # Initially all cubes show OUT face on top
            curr_dirs = [NAME_TO_VEC.copy() for _ in range(self.num_dice)]
            # === END FIX: track cube orientation during rolls ===
            rolls = {c: 0 for c in list(self.VALID_COLORS.keys())}  # Count each color
            
            self.roll_details = []  # Store details of each roll
            
            for i in range(self.n_roll):
                # Choose random target face for each cube
                final_faces = [random.choice([IN, OUT, LEFT, RIGHT, UP, DOWN]) for _ in range(self.num_dice)]
                print("Final faces: ", final_faces, "for roll", i)
                # Generate roll animations for all cubes
                animations = []
                for j in range(self.num_dice):
                    animations += self.roll_to_face(cubes[j], curr_faces[j], final_faces[j])
                
                self.log_event(f"Roll {i + 1} begins")
                self.play(*animations, run_time=1)
                self.log_event(f"Roll {i + 1} completes")
                self.wait(1)
                
                # Update current faces
                # === BEGIN FIX: track cube orientation during rolls ===
                for j in range(self.num_dice):
                    curr_dirs[j] = self.update_dir_map(curr_dirs[j], curr_faces[j], final_faces[j])
                # === END FIX: track cube orientation during rolls ===
                curr_faces = final_faces.copy()

                # Record results of this roll
                roll_results = []
                # === BEGIN FIX: track cube orientation during rolls ===
                for j in range(self.num_dice):
                    face_color = self.get_face_color_from_dir_map(cubes[j], curr_dirs[j], OUT)
                    rolls[face_color] += 1
                    roll_results.append(self.VALID_COLORS[face_color])
                # === END FIX: track cube orientation during rolls ===
                
                self.roll_details.append({
                    'roll_num': i + 1,
                    'results': roll_results
                })
            
            # Determine answer based on roll type
            if self.p_type == "roll":
                # Most frequently rolled color
                self.answer = self.VALID_COLORS[max(rolls, key=rolls.get)]
            else:  # n_roll
                # Count of specific color
                self.answer = rolls[color]

        # ====================================================================
        # Generate multiple choice options (for match/fold problems)
        # ====================================================================
        if self.p_type == "fold" or self.p_type == "match":
            # Reorient camera to 2D view for displaying options
            self.set_camera_orientation(phi=0 * DEGREES, theta=-90 * DEGREES)
            cubes.rotate(-90*DEGREES, axis=RIGHT, about_point=ORIGIN)
            cubes.rotate(-45*DEGREES, axis=UP, about_point=ORIGIN)
            cubes.rotate(20*DEGREES, axis=RIGHT, about_point=ORIGIN)
            
            # Use first cube as reference
            face_colors = random.choice(cube_colors)
            
            # Generate all legal rotations of this cube
            legal_layouts = {tuple(self.apply_perm(face_colors, p)) for p in ROTATIONS}
            correct_layout = random.choice(tuple(legal_layouts))

            # Generate 3 distractor options (impossible rotations)
            distractors = []
            attempts = 0
            while len(distractors) < 3 and attempts < 3000:
                attempts += 1
                # Random permutation that's not a valid rotation
                perm = random.sample(range(6), 6)
                cand = tuple(face_colors[i] for i in perm)
                if cand in legal_layouts or cand in distractors or cand == correct_layout:
                    continue
                distractors.append(cand)

            if len(distractors) < 3:
                raise ValueError(
                    "Could not generate 3 distinct impossible layouts; try with more varied colors.")

            # Shuffle options
            options = [correct_layout] + distractors
            random.shuffle(options)
            correct_index = options.index(correct_layout)
            
            # Build visual representations of options
            choices = VGroup()
            for i, layout in enumerate(options):
                choice = VGroup()
                if self.p_type == "fold":
                    # Show cube net
                    m = build_cube_map(list(layout))
                else:
                    # Show 3D cube
                    m = self.build_cube(list(layout), side=1.5)
                    m.move_to(ORIGIN)
                    m.rotate(-90*DEGREES, axis=RIGHT, about_point=ORIGIN)
                    m.rotate(-45*DEGREES, axis=UP, about_point=ORIGIN)
                    m.rotate(20*DEGREES, axis=RIGHT, about_point=ORIGIN)
                choice.add(m)
                
                # Add label (A, B, C, D)
                lbl = Text(chr(ord('A')+i), font_size=28)
                lbl.next_to(m, DOWN, buff=0.15)
                choice.add(lbl)
                choices.add(choice)
                
            # Arrange options in a row
            choices.arrange(buff=1.2, aligned_edge=DOWN)
            choices.to_edge(DOWN, buff=0.4)

            self.answer = chr(ord('A')+correct_index)

        # ====================================================================
        # Transition to question display
        # ====================================================================
        self.wait(1.5)
        self.log_event("All objects fade out before question display")
        self.play(*[FadeOut(mob) for mob in self.mobjects if mob != bg])
        
        # ====================================================================
        # Display the question
        # ====================================================================
        title_text = random.choice(self.cfg["text"][self.p_type])
        title_text = title_text.replace("<C>", self.selected_color_name)
        lines = title_text.split('\n')
        para = Paragraph(
            *lines, alignment="center", font_size=36, line_spacing=0.8
        )

        para.move_to(ORIGIN)
        if para.width > 0.9*config.frame_width:
            para.scale_to_fit_width(config.frame_width * 0.9)
        self.add_fixed_in_frame_mobjects(para)
        
        self.log_event("Question text appears on screen")
        self.play(Write(para), run_time=1.5)

        self.wait(0.5)
        if choices is not None:
            # Show multiple choice options
            self.play(para.animate.to_edge(UP, buff=0.2*config.frame_height))
            if self.p_type == "fold":
                self.add_fixed_in_frame_mobjects(choices)
            self.log_event(f"Multiple choice options appear ({len(options)} options)")
            self.play(FadeIn(choices, run_time=1.5))
            self.wait(0.5)
            
            if self.p_type == "match":
                # Rotate match options to show all sides
                self.log_event("Multiple choice cubes begin rotating")
                self.play(*[
                    Rotate(
                        g[0],
                        angle=2 * PI,
                        axis=UP,
                        about_point=g[0].get_center(),
                    )
                    for g in choices
                ],
                    run_time=3,
                    rate_func=smooth
                )
                self.log_event("Multiple choice cubes finish rotating")
                
        self.wait(3)
        self.log_event("Question remains on screen")
        
        self.question_text += title_text.replace('\n', ' ')
        
        # ====================================================================
        # Generate comprehensive reasoning trace
        # ====================================================================
        self.build_reasoning_trace(hidden, cube_colors, color)
        
        # ====================================================================
        # Save output files
        # ====================================================================
        # Solution file (just the answer)
        with open(f"solutions/dice_{self.p_type}_dice{self.num_dice}_seed{self.seed}.txt", "w") as f:
            f.write(str(self.answer))
        
        # Question text file
        with open(f"question_text/dice_{self.p_type}_dice{self.num_dice}_seed{self.seed}.txt", "w") as f:
            f.write(self.question_text)
            
        # Detailed reasoning trace file
        with open(f"reasoning_traces/dice_{self.p_type}_dice{self.num_dice}_seed{self.seed}.txt", "w") as f:
            f.write("\n".join(self.reasoning_trace))

    def build_reasoning_trace(self, hidden, cube_colors, query_color):
        """
        Build a comprehensive, step-by-step reasoning trace.
        Explains how to solve the puzzle systematically with proper timestamps.
        
        Args:
            hidden: Dictionary mapping colors to hidden face counts
            cube_colors: List of color configurations for each cube
            query_color: The color object selected for color-specific questions
        """
        self.reasoning_trace = []
        
        # ====================================================================
        # Introduction
        # ====================================================================
        question_text = self.cfg["text"][self.p_type][0].replace("<C>", self.selected_color_name)
        self.reasoning_trace.append(f"**Question:** {question_text.replace(chr(10), ' ')}")
        self.reasoning_trace.append("")
        self.reasoning_trace.append("Let's solve this step by step.")
        self.reasoning_trace.append("")
        
        # ====================================================================
        # Scene description with timestamps
        # ====================================================================
        self.reasoning_trace.append("### Scene Description")
        self.reasoning_trace.append("")
        
        for event in self.scene_events:
            time_str = self.format_time(event['time'])
            self.reasoning_trace.append(f"At {time_str}, {event['description']}")
        
        self.reasoning_trace.append("")
        
        # ====================================================================
        # Step 1: Understand the setup
        # ====================================================================
        self.reasoning_trace.append("### Step 1: Understand the cube configuration")
        self.reasoning_trace.append(f"There are **{self.num_dice} cube(s)** shown in the video.")
        self.reasoning_trace.append("")
        
        for detail in self.cube_details:
            self.reasoning_trace.append(f"**Cube {detail['cube_num']}** has the following colors on its faces:")
            self.reasoning_trace.append(f"- IN (back): {detail['colors'][0]}")
            self.reasoning_trace.append(f"- OUT (front): {detail['colors'][1]}")
            self.reasoning_trace.append(f"- LEFT: {detail['colors'][2]}")
            self.reasoning_trace.append(f"- RIGHT: {detail['colors'][3]}")
            self.reasoning_trace.append(f"- UP (top): {detail['colors'][4]}")
            self.reasoning_trace.append(f"- DOWN (bottom): {detail['colors'][5]}")
            self.reasoning_trace.append("")
        
        # ====================================================================
        # Problem-specific reasoning
        # ====================================================================
        
        if self.p_type == "hidden":
            # ================================================================
            # Hidden face counting
            # ================================================================
            self.reasoning_trace.append("### Step 2: Identify hidden faces")
            self.reasoning_trace.append("From the camera angle (phi=70°, theta=-45°), the following faces are hidden from view:")
            self.reasoning_trace.append("- **IN** face (facing away from camera)")
            self.reasoning_trace.append("- **LEFT** face (on the left side)")
            self.reasoning_trace.append("- **UP** face (on top, hard to see from this angle)")
            self.reasoning_trace.append("")
            
            for i, detail in enumerate(self.cube_details):
                hidden_colors = [detail['colors'][0], detail['colors'][2], detail['colors'][4]]
                self.reasoning_trace.append(f"Cube {i + 1} hidden faces: {hidden_colors[0]}, {hidden_colors[1]}, {hidden_colors[2]}")
            
            self.reasoning_trace.append("")
            self.reasoning_trace.append("### Step 3: Count faces of each color")
            for color_obj, count in sorted(hidden.items(), key=lambda x: x[1], reverse=True):
                color_name = self.VALID_COLORS[color_obj]
                self.reasoning_trace.append(f"- **{color_name}**: {count} hidden face(s)")
            
            self.reasoning_trace.append("")
            self.reasoning_trace.append(f"### Step 4: Determine answer for {self.selected_color_name}")
            self.reasoning_trace.append(f"The color **{self.selected_color_name}** appears **{self.answer}** time(s) among the hidden faces.")
            
        elif self.p_type == "max_hidden":
            # ================================================================
            # Most frequent hidden color
            # ================================================================
            self.reasoning_trace.append("### Step 2: Identify hidden faces")
            self.reasoning_trace.append("From the camera angle, these faces are hidden: IN, LEFT, UP")
            self.reasoning_trace.append("")
            
            for i, detail in enumerate(self.cube_details):
                hidden_colors = [detail['colors'][0], detail['colors'][2], detail['colors'][4]]
                self.reasoning_trace.append(f"Cube {i + 1} hidden faces: {', '.join(hidden_colors)}")
            
            self.reasoning_trace.append("")
            self.reasoning_trace.append("### Step 3: Count hidden face colors")
            for color_obj, count in sorted(hidden.items(), key=lambda x: x[1], reverse=True):
                color_name = self.VALID_COLORS[color_obj]
                self.reasoning_trace.append(f"- **{color_name}**: {count} hidden face(s)")
            
            self.reasoning_trace.append("")
            self.reasoning_trace.append("### Step 4: Find maximum")
            max_count = max(hidden.values())
            self.reasoning_trace.append(f"The maximum count is **{max_count}**, which corresponds to **{self.answer}**.")
            
        elif "roll" in self.p_type:
            # ================================================================
            # Roll tracking
            # ================================================================
            self.reasoning_trace.append(f"### Step 2: Track rolls ({self.n_roll} total)")
            self.reasoning_trace.append("After each roll, we record which color appears on top of each cube:")
            self.reasoning_trace.append("")
            
            for detail in self.roll_details:
                colors_str = ", ".join(f"Cube {i+1}: {c}" for i, c in enumerate(detail['results']))
                self.reasoning_trace.append(f"**Roll {detail['roll_num']}:** {colors_str}")
            
            self.reasoning_trace.append("")
            self.reasoning_trace.append("### Step 3: Count total occurrences")
            
            # Calculate totals from roll details
            roll_counts = {}
            for detail in self.roll_details:
                for color_name in detail['results']:
                    roll_counts[color_name] = roll_counts.get(color_name, 0) + 1
            
            for color_name in sorted(roll_counts.keys()):
                count = roll_counts[color_name]
                self.reasoning_trace.append(f"- **{color_name}**: {count} time(s)")
            
            self.reasoning_trace.append("")
            if self.p_type == "roll":
                self.reasoning_trace.append("### Step 4: Find most frequent")
                max_count = max(roll_counts.values())
                self.reasoning_trace.append(f"The color **{self.answer}** was rolled most often ({max_count} times).")
            else:  # n_roll
                self.reasoning_trace.append(f"### Step 4: Count {self.selected_color_name} occurrences")
                self.reasoning_trace.append(f"The color **{self.selected_color_name}** was rolled **{self.answer}** time(s) in total.")
        
        elif self.p_type == "fold":
            # ================================================================
            # Cube net folding
            # ================================================================
            self.reasoning_trace.append("### Step 2: Understand cube nets")
            self.reasoning_trace.append("A cube net is a 2D pattern that folds into a 3D cube.")
            self.reasoning_trace.append("The shown net uses a T-shape layout:")
            self.reasoning_trace.append("```")
            self.reasoning_trace.append("       UP")
            self.reasoning_trace.append("LEFT  OUT  RIGHT  IN")
            self.reasoning_trace.append("       DOWN")
            self.reasoning_trace.append("```")
            self.reasoning_trace.append("")
            
            self.reasoning_trace.append("### Step 3: Match net to cube orientation")
            self.reasoning_trace.append("By observing the rotating cube, we can determine which colors are on opposite faces:")
            
            # Use first cube's configuration
            colors = self.cube_details[0]['colors']
            self.reasoning_trace.append(f"- IN ({colors[0]}) ↔ OUT ({colors[1]})")
            self.reasoning_trace.append(f"- LEFT ({colors[2]}) ↔ RIGHT ({colors[3]})")
            self.reasoning_trace.append(f"- UP ({colors[4]}) ↔ DOWN ({colors[5]})")
            self.reasoning_trace.append("")
            
            self.reasoning_trace.append("### Step 4: Identify correct net")
            self.reasoning_trace.append(f"Only option **{self.answer}** matches all the observed face relationships.")
            
        elif self.p_type == "match":
            # ================================================================
            # Cube matching
            # ================================================================
            self.reasoning_trace.append("### Step 2: Observe reference cube")
            self.reasoning_trace.append("The video shows a rotating cube. We need to remember its color configuration.")
            self.reasoning_trace.append("")
            
            colors = self.cube_details[0]['colors']
            self.reasoning_trace.append("From the rotation, we can identify:")
            self.reasoning_trace.append(f"- Front: {colors[1]}")
            self.reasoning_trace.append(f"- Top: {colors[4]}")
            self.reasoning_trace.append(f"- Right: {colors[3]}")
            self.reasoning_trace.append("")
            
            self.reasoning_trace.append("### Step 3: Check each option")
            self.reasoning_trace.append("We examine each multiple choice cube to see if it matches the reference.")
            self.reasoning_trace.append("The correct match must have the same colors in the same relative positions.")
            self.reasoning_trace.append("")
            
            self.reasoning_trace.append("### Step 4: Identify match")
            self.reasoning_trace.append(f"Option **{self.answer}** exactly matches the reference cube's configuration.")
        
        # ====================================================================
        # Final answer
        # ====================================================================
        self.reasoning_trace.append("")
        self.reasoning_trace.append("### Final Answer")
        self.reasoning_trace.append(f"**{self.answer}**")
        self.reasoning_trace.append("")
        self.reasoning_trace.append(f"\\boxed{{{self.answer}}}")

# === BEGIN FIX: guard main execution for tests ===
# ============================================================================
# Main execution
# ============================================================================
if __name__ == "__main__":
    # Generate the dice video
    scene = Dice()
    scene.render()

    # ============================================================================
    # Move output file to questions directory with descriptive name
    # ============================================================================
    output = Path("manim_output/videos/1080p30/Dice.mp4")
    if output.exists():
        filename = f"dice_{scene.p_type}_dice{scene.num_dice}_seed{scene.seed}.mp4"
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
# === END FIX: guard main execution for tests ===
