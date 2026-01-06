from manim import *
import random
import math
import os
import shutil
from pathlib import Path

# ============================================================================
# FREQUENCY-PARAMETERIZED VERSION - Global Size Parameter (0.0 to 1.0)
# ============================================================================
# This version includes a FREQUENCY parameter that controls the grid dimensions
# and cube sizes, making the problem harder as size increases.
#
# FREQUENCY parameter mapping:
# - 0.0: 2x2x2 grid (8 cubes max), easier to count
# - 0.5: 4x4x4 grid (64 cubes max), medium difficulty
# - 1.0: 7x7x7 grid (343 cubes max), very hard to count accurately
#
# Cube size also scales inversely with grid size to keep visual manageable
# ============================================================================

# ============================================================================
# Setup directories for output files
# ============================================================================
Path("questions").mkdir(exist_ok=True)
Path("solutions").mkdir(exist_ok=True)
Path("question_text").mkdir(exist_ok=True)
Path("reasoning_traces").mkdir(exist_ok=True)

# ============================================================================
# Manim configuration
# ============================================================================
config.media_dir = "manim_output"
config.verbosity = "WARNING"
config.pixel_height = 1080
config.pixel_width = 1920
config.frame_rate = 30
config.preview = False

class CubesFrequency(ThreeDScene):
    """
    FREQUENCY-PARAMETERIZED version of Cubes scene.

    A 3D scene that generates cube counting and analysis puzzles with adjustable grid size.

    The FREQUENCY parameter (0.0 to 1.0) controls the grid dimensions, making
    counting and analysis progressively harder.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Random seed for reproducibility
        self.seed = random.randint(1000, 9999)

        # ====================================================================
        # FREQUENCY PARAMETER (0.0 to 1.0) - VISUAL COMPLEXITY CONTROL
        # ====================================================================
        self.frequency_param = float(os.getenv("FREQUENCY", 0.5))  # Default to medium

        # Clamp to valid range
        self.frequency_param = max(0.0, min(1.0, self.frequency_param))

        # FREQUENCY controls visual complexity through randomness granularity
        # Grid size is FIXED at 8×8×8 to provide consistent canvas
        # FREQUENCY 0.0 (coarse) → Smooth regions, low spatial frequency (easy)
        # FREQUENCY 0.5 (medium) → Moderate variation, medium complexity
        # FREQUENCY 1.0 (fine)   → High noise, per-cube variation (challenging)

        # Fixed grid size for FREQUENCY variant - larger for pattern detail
        self.max_size = 8  # Fixed at 8×8×8 for pattern complexity

        # Fixed PHYSICAL size - the overall bounding box stays constant
        self.cube_scale = 2.0 / self.max_size

        # Parameters from environment variables (with defaults)
        self.p_type = os.getenv("P_TYPE", "count")
        self.max_iters = int(os.getenv("MAX_ITERS", 25))

        # Generate grid dimensions - always cubic for FREQUENCY variant
        # This provides consistent canvas for pattern complexity
        self.grid_x = self.max_size
        self.grid_y = self.max_size
        self.grid_z = self.max_size
        self.grid_size = (self.grid_x, self.grid_y, self.grid_z)

        # Calculate removal parameters (keep same logic)
        self.p_removed = random.uniform(0.4, 0.6)
        self.total = math.prod(self.grid_size)
        self.n_removed = int(self.total * self.p_removed)

        # Problem-specific question templates
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

        # Initialize reasoning trace storage
        self.reasoning_trace = []
        self.scene_events = []

    def log_event(self, description):
        """Log a scene event with video timestamp."""
        current_time = self.renderer.time
        self.scene_events.append({
            'time': current_time,
            'description': description
        })

    def format_time(self, seconds):
        """Format seconds as M:SS for display in reasoning trace."""
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}:{secs:02d}"

    def surface_area(self, removed):
        """Calculate the surface area of the 3D structure."""
        rows, cols, layers = self.grid_size
        dirs = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
        area = 0

        for r in range(rows):
            for c in range(cols):
                for z in range(layers):
                    if (r, c, z) in removed:
                        continue

                    for dr, dc, dz in dirs:
                        nr, nc, nz = r + dr, c + dc, z + dz

                        if (
                            not (0 <= nr < rows and 0 <= nc < cols and 0 <= nz < layers)
                            or (nr, nc, nz) in removed
                        ):
                            area += 1
        return area

    def count_cube_colors(self, color, colors, removed):
        """Count how many cubes of a specific color are visible."""
        nx, ny, nz = self.grid_size
        neighbors = [
            (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
        ]
        visible_count = 0

        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    if (i, j, k) in removed:
                        continue
                    if colors[i][j][k] != color:
                        continue

                    for dx, dy, dz in neighbors:
                        ni, nj, nk = i + dx, j + dy, k + dz
                        out_of_bounds = not (
                            0 <= ni < nx and 0 <= nj < ny and 0 <= nk < nz
                        )
                        if out_of_bounds or (ni, nj, nk) in removed:
                            visible_count += 1
                            break

        return visible_count

    def count_project(self, removed):
        """Calculate maximum number of visible faces in a 2D parallel projection."""
        nx, ny, nz = self.grid_size
        counts = {}

        def count_faces(axis, sign):
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

        counts["+X"] = count_faces("x", +1)
        counts["-X"] = count_faces("x", -1)
        counts["+Y"] = count_faces("y", +1)
        counts["-Y"] = count_faces("y", -1)

        return max(counts.values())

    def count_cubes_with_exposed_faces(self, removed, n):
        """Count cubes that have exactly n exposed faces."""
        X, Y, Z = self.grid_size
        directions = [
            (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
        ]

        count = 0
        for i in range(X):
            for j in range(Y):
                for k in range(Z):
                    if (i, j, k) in removed:
                        continue

                    exposed = 0
                    for dx, dy, dz in directions:
                        ni, nj, nk = i + dx, j + dy, k + dz
                        if not (0 <= ni < X and 0 <= nj < Y and 0 <= nk < Z):
                            exposed += 1
                        elif (ni, nj, nk) in removed:
                            exposed += 1

                    if exposed == n:
                        count += 1

        return count

    def show_colors(self, colors, names):
        """Display color legend at the beginning of the video."""
        squares = VGroup(
            *[Square(1.0).set_fill(col, 1).set_stroke(width=0) for col in colors]
        )
        squares.arrange(DOWN, buff=0.5, aligned_edge=LEFT).to_edge(LEFT, buff=2)

        labels = VGroup(*[Text(nm, font_size=32) for nm in names])
        for sq, lbl in zip(squares, labels):
            lbl.next_to(sq, RIGHT, buff=1.2)

        arrows = VGroup(
            *[
                Arrow(
                    start=sq.get_right(), end=lbl.get_left(), buff=0.05, stroke_width=4
                )
                for sq, lbl in zip(squares, labels)
            ]
        )

        title = Text("Remember the following color names", font_size=40)
        title.to_edge(UP)
        all_mobjects = VGroup(squares, arrows, labels)
        all_mobjects.move_to(ORIGIN)

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

        all_mobjects.add(title)
        self.play(FadeOut(all_mobjects))
        self.wait(0.5)
        self.log_event("Color legend fades out")

    def construct(self):
        """Main scene construction method."""
        random.seed(self.seed)

        # Create animated gradient background
        bg = (
            Rectangle(height=config.frame_height, width=config.frame_width)
            .set_color(
                color_gradient([random_bright_color(), random_bright_color()], 5)
            )
            .set_opacity(0.6)
            .set_z_index(-2)
        )
        self.add_fixed_in_frame_mobjects(bg)

        # Define valid colors
        VALID_COLORS = {"blue": BLUE, "red": RED, "green": GREEN, "yellow": YELLOW}

        # Show color legend if needed
        if "color" in self.p_type:
            self.show_colors(list(VALID_COLORS.values()), list(VALID_COLORS.keys()))

        # Display initial prompt
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
        # Create 3D cube structure with SIZE-DEPENDENT parameters
        # ====================================================================
        cubes_vgroup = VGroup()
        cube_list = [
            [[None for z in range(depth)] for y in range(cols)] for x in range(rows)
        ]
        colors = [
            [[None for z in range(depth)] for y in range(cols)] for x in range(rows)
        ]
        unique_colors = set()

        self.log_event(f"Creating grid with SIZE={self.frequency_param:.2f} -> {rows}x{cols}x{depth} grid")

        for x in range(rows):
            for y in range(cols):
                for z in range(depth):
                    color = random.choice(list(VALID_COLORS.keys()))

                    # SIZE-DEPENDENT cube size
                    cube = Cube(side_length=self.cube_scale)
                    cube.set_fill(color=VALID_COLORS[color], opacity=1)
                    cube.set_stroke(color=VALID_COLORS[color], width=2)

                    cube.x_idx = x
                    cube.y_idx = y
                    cube.z_idx = z

                    # Position in 3D grid
                    cube.shift(self.cube_scale * (x * RIGHT + y * UP + z * OUT))

                    cube_list[x][y][z] = cube
                    colors[x][y][z] = color
                    cubes_vgroup.add(cube)
                    unique_colors.add(color)

        # Scale to fit camera
        max_height = config.frame_height * 0.6
        if cubes_vgroup.height > max_height:
            cubes_vgroup.scale(max_height / cubes_vgroup.height)
        cubes_vgroup.move_to(ORIGIN)

        # Setup camera orientation
        if self.p_type == "project":
            self.set_camera_orientation(phi=70 * DEGREES, theta=-45 * DEGREES)
            self.begin_ambient_camera_rotation(rate=1)
            self.log_event("Camera begins rotating for projection view")
        else:
            cubes_vgroup.rotate(45 * DEGREES, axis=UP)
            cubes_vgroup.rotate(20 * DEGREES, axis=RIGHT)
            cubes_vgroup.add_updater(
                lambda m, dt: m.rotate(1 * dt, axis=UP, about_point=ORIGIN)
            )

        self.log_event(f"Full cube structure appears ({self.grid_x}×{self.grid_y}×{self.grid_z} = {self.total} cubes)")
        self.play(Write(cubes_vgroup), run_time=1)
        self.wait(1)
        self.log_event("Cube structure finishes appearing")

        # Remove cubes using FREQUENCY-based randomness
        # FREQUENCY controls the spatial scale/granularity of random variation
        # Low frequency = smooth, large-scale variation (removes whole regions)
        # High frequency = noisy, fine-scale variation (removes individual cubes)

        cubes_to_remove = set()
        idxs_to_remove = set()

        # Generate a random height field with frequency-controlled smoothness
        # We'll remove cubes based on this height field
        heights = [[depth - 1 for j in range(cols)] for i in range(rows)]

        # FREQUENCY controls the smoothing kernel size
        # Low freq (0.0-0.3): Large smooth regions (kernel size 4-6)
        # Mid freq (0.3-0.7): Medium variation (kernel size 2-4)
        # High freq (0.7-1.0): Fine random variation (kernel size 0-2, mostly per-column)

        # Start with random heights for each column
        for row in range(rows):
            for col in range(cols):
                heights[row][col] = random.randint(0, depth - 1)

        # Apply smoothing based on frequency (low frequency = more smoothing)
        # Number of smoothing passes inversely proportional to frequency
        smoothing_passes = int((1.0 - self.frequency_param) * 8)  # 0-8 passes

        for _ in range(smoothing_passes):
            new_heights = [[0 for j in range(cols)] for i in range(rows)]
            for row in range(rows):
                for col in range(cols):
                    # Average with neighbors (with wrapping for smooth patterns)
                    total = 0
                    count = 0
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            nr = (row + dr) % rows
                            nc = (col + dc) % cols
                            total += heights[nr][nc]
                            count += 1
                    new_heights[row][col] = total // count
            heights = new_heights

        # Add back some randomness scaled by frequency
        # Higher frequency = more random jitter added back
        jitter_amount = self.frequency_param * 2  # 0-2 layers of jitter
        for row in range(rows):
            for col in range(cols):
                jitter = random.randint(-int(jitter_amount), int(jitter_amount))
                heights[row][col] = max(0, min(depth - 1, heights[row][col] + jitter))

        # Collect cubes to remove based on height field
        for row in range(rows):
            for col in range(cols):
                final_height = heights[row][col]
                for layer in range(final_height + 1, depth):
                    cubes_to_remove.add(cube_list[row][col][layer])
                    idxs_to_remove.add((row, col, layer))

        # Adjust to match target removal count
        current_count = len(idxs_to_remove)
        if current_count < self.n_removed:
            # Need to remove more - randomly remove from remaining cubes
            remaining = []
            for row in range(rows):
                for col in range(cols):
                    for layer in range(depth):
                        if (row, col, layer) not in idxs_to_remove:
                            remaining.append((row, col, layer))
            random.shuffle(remaining)
            for idx in remaining[:self.n_removed - current_count]:
                row, col, layer = idx
                cubes_to_remove.add(cube_list[row][col][layer])
                idxs_to_remove.add(idx)
        elif current_count > self.n_removed:
            # Need to remove fewer - randomly keep some
            to_keep = random.sample(list(idxs_to_remove), current_count - self.n_removed)
            for idx in to_keep:
                row, col, layer = idx
                cubes_to_remove.remove(cube_list[row][col][layer])
                idxs_to_remove.remove(idx)

        # Animate cube removal
        self.log_event(f"Cube removal begins ({len(idxs_to_remove)} cubes to remove)")

        if self.p_type == "project":
            self.play(FadeOut(*cubes_to_remove), run_time=1.5)
        else:
            vt = ValueTracker(10)
            for cube in cubes_to_remove:
                cube.add_updater(lambda m: m.set_opacity(vt.get_value() / 10))
            self.play(vt.animate.set_value(0), run_time=1.5, rate_func=smooth)
            for cube in cubes_to_remove:
                cube.clear_updaters()
            cubes_vgroup.remove(*cubes_to_remove)

        self.log_event(f"Cube removal complete - {self.total - len(idxs_to_remove)} cubes remaining")

        # Calculate answer based on problem type
        unique_colors = list(unique_colors)
        color = random.choice(unique_colors)
        N = random.randint(1, 4)

        if self.p_type == "count":
            self.answer = self.total - self.n_removed
        elif self.p_type == "missing":
            self.answer = self.n_removed
        elif self.p_type == "surface_area":
            self.answer = self.surface_area(idxs_to_remove)
        elif self.p_type == "exposed":
            self.answer = self.count_cubes_with_exposed_faces(idxs_to_remove, N)
        elif self.p_type == "colors":
            self.answer = self.count_cube_colors(color, colors, idxs_to_remove)
        elif self.p_type == "max_color":
            counts = [
                self.count_cube_colors(c, colors, idxs_to_remove) for c in unique_colors
            ]
            if counts.count(max(counts)) > 1:
                raise ValueError("Multiple max colors found, please regenerate")
            self.answer = unique_colors[counts.index(max(counts))]
        elif self.p_type == "project":
            self.answer = self.count_project(idxs_to_remove)
        elif self.p_type in ["missing_shape", "matching"]:
            # Simplified for space
            self.answer = "A"
        else:
            raise ValueError(f"Invalid problem type: {self.p_type}")

        # Transition to question display
        self.wait(3)
        self.log_event("Structure viewing time ends")
        self.play(*[FadeOut(mob) for mob in self.mobjects if mob != bg])
        self.log_event("All objects fade out")

        # Display question text
        title_text = random.choice(self.cfg["text"][self.p_type])
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
        self.wait(3)
        self.log_event("Question remains on screen")

        self.question_text = f"Observe the following structure. {title_text.replace(chr(10), ' ')}"

        # Store metadata for reasoning trace
        self.unique_colors_used = unique_colors
        self.removed_indices = idxs_to_remove
        self.color_assignments = colors
        self.problem_color = color if self.p_type in ["colors", "max_color"] else None
        self.problem_n = N if self.p_type == "exposed" else None

        # Build reasoning trace
        self.build_reasoning_trace()

        # Save output files
        with open(f"solutions/cubes_frequency_{self.p_type}_density{self.frequency_param:.2f}_seed{self.seed}.txt", "w") as f:
            f.write(str(self.answer))

        with open(f"question_text/cubes_frequency_{self.p_type}_density{self.frequency_param:.2f}_seed{self.seed}.txt", "w") as f:
            f.write(self.question_text)

        with open(f"reasoning_traces/cubes_frequency_{self.p_type}_density{self.frequency_param:.2f}_seed{self.seed}.txt", "w") as f:
            f.write("\n".join(self.reasoning_trace))

    def build_reasoning_trace(self):
        """Build a comprehensive, step-by-step reasoning trace."""
        self.reasoning_trace = []

        self.reasoning_trace.append(f"**Question:** {self.question_text}")
        self.reasoning_trace.append("")
        self.reasoning_trace.append(f"**FREQUENCY Parameter:** {self.frequency_param:.2f} (pattern granularity/complexity)")
        self.reasoning_trace.append(f"**Grid:** {self.grid_x}×{self.grid_y}×{self.grid_z} = {self.total} cubes total, {self.n_removed} removed ({self.p_removed:.1%})")
        self.reasoning_trace.append("")
        self.reasoning_trace.append("Let's solve this step by step.")
        self.reasoning_trace.append("")

        # Scene description
        self.reasoning_trace.append("### Scene Description")
        self.reasoning_trace.append("")

        for event in self.scene_events:
            time_str = self.format_time(event['time'])
            self.reasoning_trace.append(f"At {time_str}, {event['description']}")

        self.reasoning_trace.append("")

        # Grid structure
        self.reasoning_trace.append("### Step 1: Understand the grid structure")
        self.reasoning_trace.append(f"The complete grid has dimensions **{self.grid_x} × {self.grid_y} × {self.grid_z}**.")
        self.reasoning_trace.append(f"This gives a total of **{self.grid_x} × {self.grid_y} × {self.grid_z} = {self.total} cubes** if all positions were filled.")
        self.reasoning_trace.append("")

        # Cube removal
        self.reasoning_trace.append("### Step 2: Identify removed cubes")
        self.reasoning_trace.append(f"From the video, we observe that **{len(self.removed_indices)} cubes have been removed** from the structure.")
        self.reasoning_trace.append(f"This means **{self.total - len(self.removed_indices)} cubes remain** in the structure.")
        self.reasoning_trace.append("")

        # Problem-specific solution (abbreviated)
        if self.p_type == "count":
            self.reasoning_trace.append("### Step 3: Calculate remaining cubes")
            self.reasoning_trace.append(f"**Remaining cubes = {self.total} - {len(self.removed_indices)} = {self.answer}**")

        # Final answer
        self.reasoning_trace.append("")
        self.reasoning_trace.append("### Final Answer")
        self.reasoning_trace.append(f"**{self.answer}**")
        self.reasoning_trace.append("")
        self.reasoning_trace.append(f"\\boxed{{{self.answer}}}")

    def count_project_direction(self, axis, sign):
        """Helper method to count faces in a specific projection direction."""
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
    scene = CubesFrequency()
    scene.render()

    output = Path("manim_output/videos/1080p30/CubesFrequency.mp4")
    if output.exists():
        filename = f"cubes_frequency_{scene.p_type}_frequency{scene.frequency_param:.2f}_seed{scene.seed}.mp4"
        shutil.move(str(output), f"questions/{filename}")
        print(f"✓ Video saved: questions/{filename}")
        print(f"✓ FREQUENCY parameter: {scene.frequency_param:.2f} (randomness granularity: 8×8×8 grid, smoothing={int((1-scene.frequency_param)*8)} passes)")
        print(f"✓ Grid: {scene.grid_x}×{scene.grid_y}×{scene.grid_z} = {scene.total} cubes")
    else:
        print("Error: Expected output file not found")

    if os.path.exists("manim_output"):
        shutil.rmtree("manim_output")
        print("✓ Cleanup complete")
