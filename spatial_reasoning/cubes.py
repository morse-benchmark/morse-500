from manim import *
import random
import math
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

class Cubes(ThreeDScene):
    """
    A 3D scene that generates cube counting and analysis puzzles:
    - Shows a 3D grid of colored cubes with some removed
    - User must count, analyze surface area, or identify properties
    - Generates question video, solution, and detailed reasoning trace
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # ====================================================================
        # Random seed for reproducibility
        # ====================================================================
        # Set seed for this problem instance - all randomness will be deterministic
        self.seed = random.randint(1000, 9999)
        
        # ====================================================================
        # Parameters from environment variables (with defaults)
        # ====================================================================
        self.p_type = os.getenv("P_TYPE", "count")  # Problem type
        self.max_size = int(os.getenv("MAX_SIZE", 5))  # Maximum grid dimension
        self.max_iters = int(os.getenv("MAX_ITERS", 25))  # Max attempts to generate valid config
        
        # ====================================================================
        # Generate random grid dimensions
        # ====================================================================
        # Strategy: max_size is the longest side, other two sides are within 2 units
        # This creates varied but not too extreme aspect ratios
        sizes = [self.max_size]
        for _ in range(2):
            min_val = max(2, self.max_size - 2)  # At least 2, at most max_size-2 smaller
            max_val = self.max_size
            sizes.append(random.randint(min_val, max_val))
        
        # Randomly assign which dimension gets which size
        # This creates variety in orientation (tall vs wide vs deep grids)
        random.shuffle(sizes)
        self.grid_x, self.grid_y, self.grid_z = sizes
        self.grid_size = (self.grid_x, self.grid_y, self.grid_z)
        
        # ====================================================================
        # Calculate removal parameters
        # ====================================================================
        # Remove between 40-60% of cubes to create interesting visual structure
        # Too few removed = boring, too many = hard to analyze
        self.p_removed = random.uniform(0.4, 0.6)
        
        self.total = math.prod(self.grid_size)  # Total cubes in full grid
        self.n_removed = int(self.total * self.p_removed)  # Number to remove

        # ====================================================================
        # Problem-specific question templates
        # ====================================================================
        self.cfg = {
            "text": {
                "count": [
                    "How many cubes are left?\nAnswer with a single integer."
                ],
                "missing": [
                    "How many cubes are missing from this figure?\nAnswer with a single integer."
                ],
                "surface_area": [
                    "What is the surface area of the figure assuming\nall sides are 1 unit?\nAnswer with a single integer."
                ],
                "exposed": [
                    "How many cubes have exactly <N> faces exposed?\nAnswer with a single integer."
                ],
                "colors": [
                    "How many <C> cubes are visible in this figure?\nAnswer with a single integer."
                ],
                "max_color": [
                    "Which color cube appears most often?\nAnswer with only the color name."
                ],
                "project": [
                    "What is the maximum number of visible square\nfaces that can be seen in a parallel\n2D projection, considering only the side faces?\nAnswer with a single integer."
                ],
                "missing_shape": [
                    "Which shape matches the missing cubes?\nAnswer with only one multiple choice option."
                ],
                "matching": [
                    "Which shape matches the one shown in the figure?\nAnswer with only one multiple choice option."
                ]
            }
        }
        
        # ====================================================================
        # Initialize reasoning trace storage
        # ====================================================================
        self.reasoning_trace = []
        self.scene_events = []  # Track scene events with timestamps

    def log_event(self, description):
        """
        Log a scene event with video timestamp.
        Uses Manim's internal renderer.time which tracks actual video playback time,
        not wall-clock execution time.
        
        Args:
            description: Human-readable description of what's happening
        """
        # Get current video time from Manim's renderer
        # This is cumulative duration of all animations/waits so far
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

    def surface_area(self, removed):
        """
        Calculate the surface area of the 3D structure.
        
        Surface area is the count of cube faces that are exposed to the outside.
        A face is exposed if:
        1. It's on the boundary of the grid, OR
        2. The adjacent cube in that direction was removed
        
        Args:
            removed: Set of (x, y, z) tuples representing removed cubes
            
        Returns:
            Total number of exposed unit square faces
        """
        rows, cols, layers = self.grid_size
        # Six directions to check neighbors (±X, ±Y, ±Z)
        dirs = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
        area = 0

        for r in range(rows):
            for c in range(cols):
                for z in range(layers):
                    # Skip removed cubes - they don't contribute to surface area
                    if (r, c, z) in removed:
                        continue
                    
                    # Check each of the 6 faces of this cube
                    for dr, dc, dz in dirs:
                        nr, nc, nz = r + dr, c + dc, z + dz
                        
                        # Face is exposed if neighbor is out of bounds or removed
                        if (
                            not (0 <= nr < rows and 0 <= nc < cols and 0 <= nz < layers)
                            or (nr, nc, nz) in removed
                        ):
                            area += 1
        return area

    def count_cube_colors(self, color, colors, removed):
        """
        Count how many cubes of a specific color are visible.
        
        A cube is "visible" if at least one of its faces is exposed to the outside.
        This means it has at least one neighbor that is either:
        - Out of bounds (on the edge of the grid), OR
        - A removed cube
        
        Args:
            color: Color name to count (string like "red", "blue")
            colors: 3D array of color assignments [x][y][z] -> color name
            removed: Set of removed cube positions
            
        Returns:
            Count of visible cubes of this color
        """
        nx, ny, nz = self.grid_size
        # Six neighbor directions
        neighbors = [
            (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
        ]
        visible_count = 0

        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    # Skip if this cube was removed
                    if (i, j, k) in removed:
                        continue
                    # Skip if this cube is not the target color
                    if colors[i][j][k] != color:
                        continue

                    # Check if this cube has at least one exposed face
                    for dx, dy, dz in neighbors:
                        ni, nj, nk = i + dx, j + dy, k + dz
                        out_of_bounds = not (
                            0 <= ni < nx and 0 <= nj < ny and 0 <= nk < nz
                        )
                        # If neighbor is out of bounds or removed, this cube is visible
                        if out_of_bounds or (ni, nj, nk) in removed:
                            visible_count += 1
                            break  # Only need one exposed face to count

        return visible_count

    def count_project(self, removed):
        """
        Calculate maximum number of visible faces in a 2D parallel projection.
        
        A parallel projection means viewing the 3D structure from a specific direction
        as if all rays were parallel (orthographic projection, not perspective).
        We check projections from 4 side directions: +X, -X, +Y, -Y.
        
        For each direction, we count visible square faces by:
        - For each (j, k) position in the projection plane
        - Find the first non-removed cube along that ray
        - Check if its face in the viewing direction is exposed
        
        Args:
            removed: Set of removed cube positions
            
        Returns:
            Maximum face count across all 4 projections
        """
        nx, ny, nz = self.grid_size
        counts = {}

        def count_faces(axis, sign):
            """
            Count visible faces for projection along a specific axis.
            
            Args:
                axis: 'x' or 'y' (we only count side faces, not top/bottom)
                sign: +1 for positive direction, -1 for negative direction
                
            Returns:
                Count of visible faces in this projection
            """
            cnt = 0
            if axis == "x":
                # Projecting along X axis - iterate over Y-Z plane
                for j in range(ny):
                    for k in range(nz):
                        # Ray direction: if sign > 0, view from +X (start from far end)
                        i_range = range(nx - 1, -1, -1) if sign > 0 else range(0, nx)
                        for i in i_range:
                            # Skip removed cubes
                            if (i, j, k) in removed:
                                continue
                            # Found first cube along this ray
                            # Check if its face in viewing direction is exposed
                            ii = i + sign
                            if ii < 0 or ii >= nx or (ii, j, k) in removed:
                                cnt += 1
                            break  # Only count first cube along ray
            else:  # axis == 'y'
                # Projecting along Y axis - iterate over X-Z plane
                for i in range(nx):
                    for k in range(nz):
                        j_range = range(ny - 1, -1, -1) if sign > 0 else range(0, ny)
                        for j in j_range:
                            if (i, j, k) in removed:
                                continue
                            jj = j + sign
                            if jj < 0 or jj >= ny or (i, jj, k) in removed:
                                cnt += 1
                            break
            return cnt

        # Calculate for all 4 side projections
        counts["+X"] = count_faces("x", +1)
        counts["-X"] = count_faces("x", -1)
        counts["+Y"] = count_faces("y", +1)
        counts["-Y"] = count_faces("y", -1)

        return max(counts.values())

    def count_cubes_with_exposed_faces(self, removed, n):
        """
        Count cubes that have exactly n exposed faces.
        
        An exposed face is one where the neighbor in that direction is either:
        - Out of bounds (edge of grid), OR
        - A removed cube
        
        Args:
            removed: Set of removed cube positions
            n: Exact number of exposed faces to match
            
        Returns:
            Count of cubes with exactly n exposed faces
        """
        X, Y, Z = self.grid_size
        directions = [
            (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
        ]

        count = 0
        for i in range(X):
            for j in range(Y):
                for k in range(Z):
                    # Skip removed cubes
                    if (i, j, k) in removed:
                        continue

                    # Count exposed faces for this cube
                    exposed = 0
                    for dx, dy, dz in directions:
                        ni, nj, nk = i + dx, j + dy, k + dz
                        # Face is exposed if neighbor is out of bounds or removed
                        if not (0 <= ni < X and 0 <= nj < Y and 0 <= nk < Z):
                            exposed += 1
                        elif (ni, nj, nk) in removed:
                            exposed += 1

                    # Check if this cube has exactly n exposed faces
                    if exposed == n:
                        count += 1

        return count

    def show_colors(self, colors, names):
        """
        Display color legend at the beginning of the video.
        Shows colored squares with arrows pointing to their names.
        
        This helps users learn color names for later questions.
        
        Args:
            colors: List of Manim color objects
            names: List of color name strings
        """
        # Create colored squares
        squares = VGroup(
            *[Square(1.0).set_fill(col, 1).set_stroke(width=0) for col in colors]
        )
        squares.arrange(DOWN, buff=0.5, aligned_edge=LEFT).to_edge(LEFT, buff=2)

        # Create color name labels
        labels = VGroup(*[Text(nm, font_size=32) for nm in names])
        for sq, lbl in zip(squares, labels):
            lbl.next_to(sq, RIGHT, buff=1.2)

        # Create arrows connecting squares to labels
        arrows = VGroup(
            *[
                Arrow(
                    start=sq.get_right(), end=lbl.get_left(), buff=0.05, stroke_width=4
                )
                for sq, lbl in zip(squares, labels)
            ]
        )

        # Title for color legend
        title = Text("Remember the following color names", font_size=40)
        title.to_edge(UP)
        all_mobjects = VGroup(squares, arrows, labels)
        all_mobjects.move_to(ORIGIN)
        
        # Animate color legend
        self.log_event("Color legend title appears")
        self.play(FadeIn(title))
        self.log_event("Color squares fade in")
        self.play(FadeIn(squares))
        self.log_event("Arrows and labels animate in")
        self.play(
            AnimationGroup(*[GrowArrow(ar) for ar in arrows], lag_ratio=0.1),
            Write(labels),
        )
        self.wait(2)
        self.log_event(f"Color legend displayed: {', '.join(names)}")

        # Clean up
        all_mobjects.add(title)
        self.play(FadeOut(all_mobjects))
        self.wait(0.5)
        self.log_event("Color legend fades out")

    def construct(self):
        """
        Main scene construction method.
        This is called by Manim to build and render the entire scene.
        """
        # ====================================================================
        # Initialize random seed
        # ====================================================================
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
            .set_z_index(-2)  # Keep in background
        )
        self.add_fixed_in_frame_mobjects(bg)  # Fixed during camera movement
        
        # ====================================================================
        # Define valid colors for cubes
        # ====================================================================
        VALID_COLORS = {"blue": BLUE, "red": RED, "green": GREEN, "yellow": YELLOW}
        
        # ====================================================================
        # Show color legend if needed for this problem type
        # ====================================================================
        if "color" in self.p_type:
            self.show_colors(list(VALID_COLORS.values()), list(VALID_COLORS.keys()))

        # ====================================================================
        # Display initial prompt
        # ====================================================================
        prompt = Text(
            "Observe the following structure", color=WHITE, font_size=36
        ).move_to(ORIGIN)
        self.log_event("Initial prompt appears: 'Observe the following structure'")
        self.play(FadeIn(prompt), run_time=0.5)
        self.wait(1.5)
        self.play(FadeOut(prompt), run_time=0.5)
        self.log_event("Initial prompt fades out")
        self.wait(1)
        
        rows, cols, depth = self.grid_size

        # ====================================================================
        # Create 3D cube structure
        # ====================================================================
        cubes_vgroup = VGroup()
        cube_list = [
            [[None for z in range(depth)] for y in range(cols)] for x in range(rows)
        ]
        colors = [
            [[None for z in range(depth)] for y in range(cols)] for x in range(rows)
        ]
        unique_colors = set()

        # Create all cubes with random colors
        for x in range(rows):
            for y in range(cols):
                for z in range(depth):
                    # Randomly assign color
                    color = random.choice(list(VALID_COLORS.keys()))
                    
                    # Create cube with 0.75 unit side length
                    cube = Cube(side_length=0.75)
                    cube.set_fill(color=VALID_COLORS[color], opacity=1)
                    cube.set_stroke(color=VALID_COLORS[color], width=2)
                    
                    # Store grid indices for later reference
                    cube.x_idx = x
                    cube.y_idx = y
                    cube.z_idx = z
                    
                    # Position in 3D grid
                    cube.shift(0.75 * (x * RIGHT + y * UP + z * OUT))

                    # Store references
                    cube_list[x][y][z] = cube
                    colors[x][y][z] = color
                    cubes_vgroup.add(cube)
                    unique_colors.add(color)

        # ====================================================================
        # Scale to fit camera
        # ====================================================================
        # Ensure the cube structure fits nicely in frame
        max_height = config.frame_height * 0.6
        if cubes_vgroup.height > max_height:
            cubes_vgroup.scale(max_height / cubes_vgroup.height)
        cubes_vgroup.move_to(ORIGIN)
        
        # ====================================================================
        # Setup camera orientation based on problem type
        # ====================================================================
        if self.p_type == "project":
            # For projection problems, use rotating camera
            self.set_camera_orientation(phi=70 * DEGREES, theta=-45 * DEGREES)
            self.begin_ambient_camera_rotation(rate=1)
            self.log_event("Camera begins rotating for projection view")
        else:
            # For other problems, apply static rotation with updater
            cubes_vgroup.rotate(45 * DEGREES, axis=UP)
            cubes_vgroup.rotate(20 * DEGREES, axis=RIGHT)
            # Add continuous slow rotation
            cubes_vgroup.add_updater(
                lambda m, dt: m.rotate(1 * dt, axis=UP, about_point=ORIGIN)
            )
        
        # Animate cube structure appearing
        self.log_event(f"Full cube structure appears ({self.grid_x}×{self.grid_y}×{self.grid_z} = {self.total} cubes)")
        self.play(Write(cubes_vgroup), run_time=1)
        self.wait(1)
        self.log_event("Cube structure finishes appearing")

        # ====================================================================
        # Remove cubes to create interesting structure
        # ====================================================================
        # Strategy: Randomly lower the "height" of each column
        # This creates a natural-looking erosion pattern
        
        heights = [[depth - 1 for j in range(cols)] for i in range(rows)]
        iters = 0
        cubes_to_remove = set()
        idxs_to_remove = set()
        done = False
        
        # Iteratively remove cubes until we reach target count
        while iters < self.max_iters and not done:
            for row in range(rows):
                for col in range(cols):
                    old_height = heights[row][col]
                    
                    # 50% chance to lower this column
                    if random.uniform(0, 1) < 0.5:
                        new_height = random.randint(0, heights[row][col])
                    else:
                        new_height = old_height

                    heights[row][col] = new_height
                    
                    # Remove cubes from old_height down to new_height
                    for layer in range(old_height, new_height - 1, -1):
                        cubes_to_remove.add(cube_list[row][col][layer])
                        idxs_to_remove.add((row, col, layer))
                        
                        # Stop if we've removed enough
                        if len(cubes_to_remove) == self.n_removed:
                            done = True
                            break
                    if done:
                        break
                if done:
                    break
            iters += 1

        # Verify we successfully created a valid configuration
        if not done:
            raise ValueError(f"Failed to find valid configuration in {iters} attempts")

        # ====================================================================
        # Animate cube removal
        # ====================================================================
        self.log_event(f"Cube removal begins ({len(idxs_to_remove)} cubes to remove)")
        
        if self.p_type == "project":
            # Simple fade out for projection problems
            self.play(FadeOut(*cubes_to_remove), run_time=1.5)
        else:
            # Smooth opacity fade for other problems
            vt = ValueTracker(10)  # Start at full opacity
            for cube in cubes_to_remove:
                cube.add_updater(lambda m: m.set_opacity(vt.get_value() / 10))
            self.play(vt.animate.set_value(0), run_time=1.5, rate_func=smooth)
            # Clean up updaters
            for cube in cubes_to_remove:
                cube.clear_updaters()
            cubes_vgroup.remove(*cubes_to_remove)
        
        self.log_event(f"Cube removal complete - {self.total - len(idxs_to_remove)} cubes remaining")

        # ====================================================================
        # Calculate answer based on problem type
        # ====================================================================
        unique_colors = list(unique_colors)
        
        # For some problem types, we need random parameters
        color = random.choice(unique_colors)  # Random color for color-based questions
        N = random.randint(1, 4)  # Random number of exposed faces

        if self.p_type == "count":
            # Simple counting: total - removed
            self.answer = self.total - self.n_removed
            
        elif self.p_type == "missing":
            # Count missing cubes
            self.answer = self.n_removed
            
        elif self.p_type == "surface_area":
            # Calculate total exposed surface area
            self.answer = self.surface_area(idxs_to_remove)
            
        elif self.p_type == "exposed":
            # Count cubes with exactly N exposed faces
            self.answer = self.count_cubes_with_exposed_faces(idxs_to_remove, N)
            
        elif self.p_type == "colors":
            # Count visible cubes of a specific color
            self.answer = self.count_cube_colors(color, colors, idxs_to_remove)
            
        elif self.p_type == "max_color":
            # Find which color appears most frequently
            counts = [
                self.count_cube_colors(c, colors, idxs_to_remove) for c in unique_colors
            ]
            # Ensure unique maximum (regenerate if tie)
            if counts.count(max(counts)) > 1:
                raise ValueError(
                    "Multiple max colors found, please regenerate problem"
                )
            self.answer = unique_colors[counts.index(max(counts))]
            
        elif self.p_type == "project":
            # Maximum visible faces in parallel projection
            self.answer = self.count_project(idxs_to_remove)
            
        elif self.p_type == "missing_shape":
            # Multiple choice: which shape matches the REMOVED cubes
            correct = idxs_to_remove
            all_idxs = [
                (i, j, k)
                for i in range(rows)
                for j in range(cols)
                for k in range(depth)
            ]
            avail_to_add = set(all_idxs) - correct

            # Generate 3 incorrect variants
            variants = []
            while len(variants) < 3:
                # Randomly add/remove cubes to create similar but wrong answer
                na = random.randint(
                    1, min(len(avail_to_add), max(1, len(correct))) // 4
                )
                nr = (
                    random.randint(1, min(len(correct), max(1, len(correct) - 1)) // 4)
                    if len(correct) > 1
                    else 1
                )
                to_add = set(random.sample(list(avail_to_add), na))
                to_remove = set(random.sample(list(correct), nr))
                variant = (correct - to_remove) | to_add
                if variant != correct and variant not in variants:
                    variants.append(variant)

            # Randomize option order
            options = [correct] + variants
            random.shuffle(options)
            labels = ["A", "B", "C", "D"]

            # Draw each option as a mini-diagram
            option_groups = VGroup()
            positions = [LEFT * 3, LEFT, RIGHT, RIGHT * 3]
            for pos, inds, lbl in zip(positions, options, labels):
                mini = VGroup()
                for i, j, k in inds:
                    c = Cube(side_length=0.2)
                    c.set_fill(VALID_COLORS[colors[i][j][k]], opacity=1)
                    c.set_stroke(VALID_COLORS[colors[i][j][k]], width=1)
                    c.shift(i * 0.2 * RIGHT + j * 0.2 * UP + k * 0.2 * OUT)
                    mini.add(c)
                mini.scale(0.8)
                mini.move_to(pos + DOWN * 2)
                mini.rotate(-45 * DEGREES, axis=UP)
                mini.rotate(20 * DEGREES, axis=RIGHT)
                mini.add_updater(lambda m, dt: m.rotate(1.57 * dt, axis=UP))

                label = Text(lbl).scale(0.2).next_to(mini, UP)
                option_groups.add(VGroup(mini, label))

            # Add option E
            option_e_text = Text("E. None of the above").scale(0.2)
            option_e_text.move_to(DOWN * 2.8)
            option_groups.add(option_e_text)

            option_groups.scale_to_fit_height(config.frame_height * 0.4)
            option_groups.scale_to_fit_width(config.frame_width * 0.9)
            option_groups.move_to(DOWN * 1.2)
            
            self.answer = labels[options.index(correct)]
            
        elif self.p_type == "matching":
            # Multiple choice: which shape matches the REMAINING cubes
            all_idxs = [
                (i, j, k)
                for i in range(rows)
                for j in range(cols)
                for k in range(depth)
            ]
            correct = set(all_idxs) - idxs_to_remove
            avail_to_add = idxs_to_remove

            # Generate 3 incorrect variants
            variants = []
            while len(variants) < 3:
                na = random.randint(
                    1, min(len(avail_to_add), max(1, len(correct))) // 4
                )
                nr = (
                    random.randint(1, min(len(correct), max(1, len(correct) - 1)) // 4)
                    if len(correct) > 1
                    else 1
                )
                to_add = set(random.sample(list(avail_to_add), na))
                to_remove = set(random.sample(list(correct), nr))
                variant = (correct - to_remove) | to_add
                if variant != correct and variant not in variants:
                    variants.append(variant)

            options = [correct] + variants
            random.shuffle(options)
            labels = ["A", "B", "C", "D"]

            # Draw each option as a mini-diagram
            option_groups = VGroup()
            positions = [LEFT * 3, LEFT, RIGHT, RIGHT * 3]
            for pos, inds, lbl in zip(positions, options, labels):
                mini = VGroup()
                for i, j, k in inds:
                    c = Cube(side_length=0.2)
                    c.set_fill(VALID_COLORS[colors[i][j][k]], opacity=0.8)
                    c.set_stroke(VALID_COLORS[colors[i][j][k]], width=1)
                    c.shift(i * 0.2 * RIGHT + j * 0.2 * UP + k * 0.2 * OUT)
                    mini.add(c)
                mini.scale(0.8)
                mini.move_to(pos + DOWN * 2)
                mini.rotate(-45 * DEGREES, axis=UP)
                mini.rotate(20 * DEGREES, axis=RIGHT)
                mini.add_updater(lambda m, dt: m.rotate(1.57 * dt, axis=UP))

                label = Text(lbl).scale(0.2).next_to(mini, UP)
                option_groups.add(VGroup(mini, label))

            # Add option E
            option_e_text = Text("E. None of the above").scale(0.2)
            option_e_text.move_to(DOWN * 2.8)
            option_groups.add(option_e_text)

            option_groups.scale_to_fit_height(config.frame_height * 0.4)
            option_groups.scale_to_fit_width(config.frame_width * 0.9)
            option_groups.move_to(DOWN * 1.2)
            
            self.answer = labels[options.index(correct)]
        else:
            raise ValueError(f"Invalid problem type: {self.p_type}")

        # ====================================================================
        # Transition to question display
        # ====================================================================
        self.wait(3)
        self.log_event("Structure viewing time ends")
        self.play(*[FadeOut(mob) for mob in self.mobjects if mob != bg])
        self.log_event("All objects fade out")
        
        # ====================================================================
        # Display question text
        # ====================================================================
        title_text = random.choice(self.cfg["text"][self.p_type])
        # Substitute problem-specific parameters
        title_text = title_text.replace("<C>", color)
        title_text = title_text.replace("<N>", str(N))
        lines = title_text.split("\n")

        para = Paragraph(*lines, alignment="center", font_size=36, line_spacing=0.8)
        para.move_to(ORIGIN)
        if para.width > 0.9 * config.frame_width:
            para.scale_to_fit_width(config.frame_width * 0.9)
        self.add_fixed_in_frame_mobjects(para)
        
        self.log_event("Question text appears")
        self.play(Write(para), run_time=1.5)

        # Show multiple choice options if applicable
        if self.p_type == "matching" or self.p_type == "missing_shape":
            self.play(para.animate.to_edge(UP, buff=0.2 * config.frame_height))
            self.wait(0.5)
            self.log_event("Multiple choice options appear")
            self.play(Write(option_groups, run_time=1))

        self.wait(3)
        self.log_event("Question remains on screen")
        
        # ====================================================================
        # Prepare output text
        # ====================================================================
        self.question_text = f"Observe the following structure. {title_text.replace(chr(10), ' ')}"
        
        # Store metadata for reasoning trace
        self.unique_colors_used = unique_colors
        self.removed_indices = idxs_to_remove
        self.color_assignments = colors
        self.problem_color = color if self.p_type in ["colors", "max_color"] else None
        self.problem_n = N if self.p_type == "exposed" else None
        
        # ====================================================================
        # Build comprehensive reasoning trace
        # ====================================================================
        self.build_reasoning_trace()
        
        # ====================================================================
        # Save output files
        # ====================================================================
        # Solution file (just the answer)
        with open(f"solutions/cubes_{self.p_type}_max{self.max_size}_seed{self.seed}.txt", "w") as f:
            f.write(str(self.answer))
        
        # Question text file
        with open(f"question_text/cubes_{self.p_type}_max{self.max_size}_seed{self.seed}.txt", "w") as f:
            f.write(self.question_text)
            
        # Detailed reasoning trace
        with open(f"reasoning_traces/cubes_{self.p_type}_max{self.max_size}_seed{self.seed}.txt", "w") as f:
            f.write("\n".join(self.reasoning_trace))

    def build_reasoning_trace(self):
        """
        Build a comprehensive, step-by-step reasoning trace.
        This explains how to solve the puzzle systematically.
        
        The trace includes:
        1. Question statement
        2. Scene description with timestamps
        3. Step-by-step solution process
        4. Final answer
        """
        self.reasoning_trace = []
        
        # ====================================================================
        # Introduction
        # ====================================================================
        self.reasoning_trace.append(f"**Question:** {self.question_text}")
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
        # Step 1: Grid structure
        # ====================================================================
        self.reasoning_trace.append("### Step 1: Understand the grid structure")
        self.reasoning_trace.append(f"The complete grid has dimensions **{self.grid_x} × {self.grid_y} × {self.grid_z}**.")
        self.reasoning_trace.append(f"This gives a total of **{self.grid_x} × {self.grid_y} × {self.grid_z} = {self.total} cubes** if all positions were filled.")
        self.reasoning_trace.append("")
        
        # ====================================================================
        # Step 2: Cube removal
        # ====================================================================
        self.reasoning_trace.append("### Step 2: Identify removed cubes")
        self.reasoning_trace.append(f"From the video, we observe that **{len(self.removed_indices)} cubes have been removed** from the structure.")
        self.reasoning_trace.append(f"This means **{self.total - len(self.removed_indices)} cubes remain** in the structure.")
        self.reasoning_trace.append("")
        
        # ====================================================================
        # Step 3: Color information (if relevant)
        # ====================================================================
        if "color" in self.p_type:
            self.reasoning_trace.append("### Step 3: Color distribution")
            self.reasoning_trace.append(f"The cubes use {len(self.unique_colors_used)} colors: {', '.join(self.unique_colors_used)}.")
            
            # Count each color
            color_counts = {}
            for color in self.unique_colors_used:
                count = self.count_cube_colors(color, self.color_assignments, self.removed_indices)
                color_counts[color] = count
            
            self.reasoning_trace.append("")
            self.reasoning_trace.append("Visible cube counts by color:")
            for color, count in sorted(color_counts.items(), key=lambda x: -x[1]):
                self.reasoning_trace.append(f"- **{color}**: {count} visible cubes")
            self.reasoning_trace.append("")
        
        # ====================================================================
        # Step 4: Problem-specific solution
        # ====================================================================
        if self.p_type == "count":
            self.reasoning_trace.append("### Step 3: Calculate remaining cubes")
            self.reasoning_trace.append("To find how many cubes are left:")
            self.reasoning_trace.append(f"- Total cubes in full grid: {self.total}")
            self.reasoning_trace.append(f"- Cubes removed: {len(self.removed_indices)}")
            self.reasoning_trace.append(f"- **Remaining cubes = {self.total} - {len(self.removed_indices)} = {self.answer}**")
            
        elif self.p_type == "missing":
            self.reasoning_trace.append("### Step 3: Count missing cubes")
            self.reasoning_trace.append(f"The question asks for the number of **missing cubes**.")
            self.reasoning_trace.append(f"From our observation in Step 2, we identified **{len(self.removed_indices)} removed positions**.")
            self.reasoning_trace.append(f"Therefore, **{self.answer} cubes are missing**.")
            
        elif self.p_type == "surface_area":
            self.reasoning_trace.append("### Step 3: Calculate surface area")
            self.reasoning_trace.append("Surface area is the total number of exposed cube faces.")
            self.reasoning_trace.append("A face is exposed if:")
            self.reasoning_trace.append("- It's on the boundary of the grid, OR")
            self.reasoning_trace.append("- The adjacent position in that direction was removed")
            self.reasoning_trace.append("")
            self.reasoning_trace.append("For each of the remaining cubes, we check all 6 faces:")
            self.reasoning_trace.append(f"- Total remaining cubes: {self.total - len(self.removed_indices)}")
            self.reasoning_trace.append(f"- After checking all cubes and their neighbors:")
            self.reasoning_trace.append(f"- **Total surface area = {self.answer} unit squares**")
            
        elif self.p_type == "exposed":
            self.reasoning_trace.append(f"### Step 3: Count cubes with exactly {self.problem_n} exposed faces")
            self.reasoning_trace.append(f"We need to find cubes with **exactly {self.problem_n} faces exposed**.")
            self.reasoning_trace.append("")
            self.reasoning_trace.append("For each remaining cube, count how many of its 6 faces are exposed:")
            self.reasoning_trace.append("- A face is exposed if the neighbor in that direction is removed or out of bounds")
            self.reasoning_trace.append("")
            
            # Sample a few cubes to show the process
            sample_size = min(3, self.total - len(self.removed_indices))
            self.reasoning_trace.append(f"Example analysis (checking first {sample_size} cubes):")
            
            checked = 0
            for i in range(self.grid_x):
                for j in range(self.grid_y):
                    for k in range(self.grid_z):
                        if (i, j, k) in self.removed_indices:
                            continue
                        
                        # Count exposed faces for this cube
                        exposed = 0
                        directions = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]
                        for dx, dy, dz in directions:
                            ni, nj, nk = i + dx, j + dy, k + dz
                            if not (0 <= ni < self.grid_x and 0 <= nj < self.grid_y and 0 <= nk < self.grid_z):
                                exposed += 1
                            elif (ni, nj, nk) in self.removed_indices:
                                exposed += 1
                        
                        self.reasoning_trace.append(f"- Cube at ({i}, {j}, {k}): {exposed} exposed faces")
                        
                        checked += 1
                        if checked >= sample_size:
                            break
                    if checked >= sample_size:
                        break
                if checked >= sample_size:
                    break
            
            self.reasoning_trace.append("...")
            self.reasoning_trace.append(f"After checking all cubes: **{self.answer} cubes have exactly {self.problem_n} exposed faces**")
            
        elif self.p_type == "colors":
            self.reasoning_trace.append(f"### Step 4: Count visible {self.problem_color} cubes")
            self.reasoning_trace.append(f"We need to count how many **{self.problem_color} cubes are visible**.")
            self.reasoning_trace.append("")
            self.reasoning_trace.append("A cube is visible if at least one of its faces is exposed.")
            self.reasoning_trace.append(f"From Step 3, we found that **{color_counts[self.problem_color]} {self.problem_color} cubes** have at least one exposed face.")
            
        elif self.p_type == "max_color":
            self.reasoning_trace.append("### Step 4: Find most common color")
            self.reasoning_trace.append("Comparing the counts from Step 3:")
            max_count = max(color_counts.values())
            for color, count in sorted(color_counts.items(), key=lambda x: -x[1]):
                marker = " ← Maximum" if count == max_count else ""
                self.reasoning_trace.append(f"- {color}: {count}{marker}")
            self.reasoning_trace.append(f"")
            self.reasoning_trace.append(f"**{self.answer}** appears most often with {max_count} visible cubes.")
            
        elif self.p_type == "project":
            self.reasoning_trace.append("### Step 3: Calculate maximum projection")
            self.reasoning_trace.append("A parallel projection views the structure from a specific direction.")
            self.reasoning_trace.append("We check 4 side projections: +X, -X, +Y, -Y")
            self.reasoning_trace.append("")
            self.reasoning_trace.append("For each direction, count visible faces:")
            self.reasoning_trace.append("- Look along parallel rays through the structure")
            self.reasoning_trace.append("- Count exposed faces of first non-removed cube on each ray")
            self.reasoning_trace.append("")
            
            # Calculate all projections for reasoning
            counts = {}
            counts["+X"] = self.count_project_direction("x", +1)
            counts["-X"] = self.count_project_direction("x", -1)
            counts["+Y"] = self.count_project_direction("y", +1)
            counts["-Y"] = self.count_project_direction("y", -1)
            
            for direction, count in counts.items():
                marker = " ← Maximum" if count == self.answer else ""
                self.reasoning_trace.append(f"- {direction} projection: {count} faces{marker}")
            
            self.reasoning_trace.append("")
            self.reasoning_trace.append(f"**Maximum projection = {self.answer} faces**")
            
        elif self.p_type in ["missing_shape", "matching"]:
            target = "missing cubes" if self.p_type == "missing_shape" else "remaining structure"
            self.reasoning_trace.append(f"### Step 3: Identify the correct option")
            self.reasoning_trace.append(f"We need to find which multiple choice option matches the {target}.")
            self.reasoning_trace.append("")
            self.reasoning_trace.append("By carefully comparing each option to what we observed:")
            self.reasoning_trace.append(f"- Options A, B, C, D show different cube arrangements")
            self.reasoning_trace.append(f"- Option E is 'None of the above'")
            self.reasoning_trace.append("")
            self.reasoning_trace.append(f"The correct match is **option {self.answer}**.")
        
        # ====================================================================
        # Final answer
        # ====================================================================
        self.reasoning_trace.append("")
        self.reasoning_trace.append("### Final Answer")
        self.reasoning_trace.append(f"**{self.answer}**")
        self.reasoning_trace.append("")
        self.reasoning_trace.append(f"\\boxed{{{self.answer}}}")

    def count_project_direction(self, axis, sign):
        """
        Helper method to count faces in a specific projection direction.
        Used only in reasoning trace generation.
        
        Args:
            axis: 'x' or 'y'
            sign: +1 or -1
            
        Returns:
            Count of visible faces
        """
        nx, ny, nz = self.grid_size
        removed = self.removed_indices
        cnt = 0
        
        if axis == "x":
            for j in range(ny):
                for k in range(nz):
                    i_range = range(nx - 1, -1, -1) if sign > 0 else range(0, nx)
                    for i in i_range:
                        if (i, j, k) in removed:
                            continue
                        ii = i + sign
                        if ii < 0 or ii >= nx or (ii, j, k) in removed:
                            cnt += 1
                        break
        else:  # axis == 'y'
            for i in range(nx):
                for k in range(nz):
                    j_range = range(ny - 1, -1, -1) if sign > 0 else range(0, ny)
                    for j in j_range:
                        if (i, j, k) in removed:
                            continue
                        jj = j + sign
                        if jj < 0 or jj >= ny or (i, jj, k) in removed:
                            cnt += 1
                        break
        return cnt

# ============================================================================
# Main execution
# ============================================================================
if __name__ == "__main__":
    # Generate the cube video
    scene = Cubes()
    scene.render()

    # ========================================================================
    # Move output file to questions directory with descriptive name
    # ========================================================================
    output = Path("manim_output/videos/1080p30/Cubes.mp4")
    if output.exists():
        filename = f"cubes_{scene.p_type}_max{scene.max_size}_seed{scene.seed}.mp4"
        shutil.move(str(output), f"questions/{filename}")
        print(f"✓ Video saved: questions/{filename}")
        print(f"✓ Solution saved: solutions/cubes_{scene.p_type}_max{scene.max_size}_seed{scene.seed}.txt")
        print(f"✓ Question saved: question_text/cubes_{scene.p_type}_max{scene.max_size}_seed{scene.seed}.txt")
        print(f"✓ Reasoning saved: reasoning_traces/cubes_{scene.p_type}_max{scene.max_size}_seed{scene.seed}.txt")
    else:
        # Debug: Print what files actually exist
        print("Error: Expected output file not found")
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

    # ========================================================================
    # Final cleanup
    # ========================================================================
    if os.path.exists("manim_output"):
        shutil.rmtree("manim_output")
        print("✓ Cleanup complete")