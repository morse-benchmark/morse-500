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

class Cubes(ThreeDScene):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Set random seed for reproducibility
        self.seed = random.randint(1000, 9999)
        
        # Parameters from environment variables
        self.p_type = os.getenv("P_TYPE", "count")
        self.max_size = int(os.getenv("MAX_SIZE", 5))
        self.max_iters = int(os.getenv("MAX_ITERS", 25))
        
        # Generate random grid dimensions (max_size is the longest side)
        # Other two sides are within 2 units of max_size
        sizes = [self.max_size]
        for _ in range(2):
            min_val = max(2, self.max_size - 2)
            max_val = self.max_size
            sizes.append(random.randint(min_val, max_val))
        
        # Randomly assign which dimension gets which size
        random.shuffle(sizes)
        self.grid_x, self.grid_y, self.grid_z = sizes
        self.grid_size = (self.grid_x, self.grid_y, self.grid_z)
        
        # Random removal percentage between 0.4 and 0.6
        self.p_removed = random.uniform(0.4, 0.6)
        
        self.total = math.prod(self.grid_size)
        self.n_removed = int(self.total * self.p_removed)

        # Configuration for different problem types
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

    def surface_area(self, removed):
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
        
        self.play(FadeIn(title))
        self.play(FadeIn(squares))
        self.play(
            AnimationGroup(*[GrowArrow(ar) for ar in arrows], lag_ratio=0.1),
            Write(labels),
        )
        self.wait(2)

        all_mobjects.add(title)
        self.play(FadeOut(all_mobjects))
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

        self.add_fixed_in_frame_mobjects(bg)
        VALID_COLORS = {"blue": BLUE, "red": RED, "green": GREEN, "yellow": YELLOW}
        
        if "color" in self.p_type:
            self.show_colors(list(VALID_COLORS.values()), list(VALID_COLORS.keys()))

        prompt = Text(
            "Observe the following structure", color=WHITE, font_size=36
        ).move_to(ORIGIN)
        self.play(FadeIn(prompt), run_time=0.5)
        self.wait(1.5)
        self.play(FadeOut(prompt), run_time=0.5)
        self.wait(1)
        
        rows, cols, depth = self.grid_size

        # Initialize reasoning trace
        reasoning_trace = []
        reasoning_trace.append(f"Problem Type: {self.p_type}")
        reasoning_trace.append(f"Max Size Parameter: {self.max_size}")
        reasoning_trace.append(f"Generated Grid Size: {self.grid_x}x{self.grid_y}x{self.grid_z} = {self.total} cubes")
        # reasoning_trace.append(f"Removal Rate: {self.p_removed:.3f} ({self.n_removed} cubes to remove)")
        reasoning_trace.append("")

        cubes_vgroup = VGroup()
        cube_list = [
            [[None for z in range(depth)] for y in range(cols)] for z in range(rows)
        ]
        colors = [
            [[None for z in range(depth)] for y in range(cols)] for z in range(rows)
        ]
        unique_colors = set()

        # Create cubes and assign colors
        reasoning_trace.append("Creating cube structure:")
        for x in range(rows):
            for y in range(cols):
                for z in range(depth):
                    color = random.choice(list(VALID_COLORS.keys()))
                    cube = Cube(side_length=0.75)
                    cube.set_fill(color=VALID_COLORS[color], opacity=1)
                    cube.set_stroke(color=VALID_COLORS[color], width=2)
                    cube.x_idx = x
                    cube.y_idx = y
                    cube.z_idx = z
                    cube.shift(0.75 * (x * RIGHT + y * UP + z * OUT))

                    cube_list[x][y][z] = cube
                    colors[x][y][z] = color
                    cubes_vgroup.add(cube)
                    unique_colors.add(color)

        reasoning_trace.append(f"Colors used: {list(unique_colors)}")
        reasoning_trace.append("")

        max_height = config.frame_height * 0.6
        if cubes_vgroup.height > max_height:
            cubes_vgroup.scale(max_height / cubes_vgroup.height)
        cubes_vgroup.move_to(ORIGIN)
        
        if self.p_type == "project":
            self.set_camera_orientation(phi=70 * DEGREES, theta=-45 * DEGREES)
            self.begin_ambient_camera_rotation(rate=1)
        else:
            cubes_vgroup.rotate(45 * DEGREES, axis=UP)
            cubes_vgroup.rotate(20 * DEGREES, axis=RIGHT)
            cubes_vgroup.add_updater(
                lambda m, dt: m.rotate(1 * dt, axis=UP, about_point=ORIGIN)
            )
        self.play(Write(cubes_vgroup), run_time=1)
        self.wait(1)

        # Remove cubes logic
        heights = [[depth - 1 for j in range(cols)] for i in range(rows)]
        iters = 0
        cubes_to_remove = set()
        idxs_to_remove = set()
        done = False
        
        reasoning_trace.append("Cube removal process:")
        while iters < self.max_iters and not done:
            for row in range(rows):
                for col in range(cols):
                    old_height = heights[row][col]
                    if random.uniform(0, 1) < 0.5:
                        new_height = random.randint(0, heights[row][col])
                    else:
                        new_height = old_height

                    heights[row][col] = new_height
                    for layer in range(old_height, new_height - 1, -1):
                        cubes_to_remove.add(cube_list[row][col][layer])
                        idxs_to_remove.add((row, col, layer))
                        reasoning_trace.append(f"  Removed cube at ({row}, {col}, {layer})")
                        if len(cubes_to_remove) == self.n_removed:
                            done = True
                            break
                    if done:
                        break
                if done:
                    break
            iters += 1

        if not done:
            raise ValueError(f"Failed to find valid configuration in {iters} attempts")

        reasoning_trace.append(f"Total cubes removed: {len(idxs_to_remove)}")
        reasoning_trace.append("")

        # Animate cube removal
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

        # Calculate answer based on problem type
        unique_colors = list(unique_colors)
        color = random.choice(unique_colors)
        N = random.randint(1, 4)

        reasoning_trace.append("Calculating answer:")
        if self.p_type == "count":
            self.answer = self.total - self.n_removed
            reasoning_trace.append(f"Remaining cubes: {self.total} - {self.n_removed} = {self.answer}")
        elif self.p_type == "missing":
            self.answer = self.n_removed
            reasoning_trace.append(f"Missing cubes: {self.answer}")
        elif self.p_type == "surface_area":
            self.answer = self.surface_area(idxs_to_remove)
            reasoning_trace.append(f"Surface area calculation: {self.answer}")
        elif self.p_type == "exposed":
            self.answer = self.count_cubes_with_exposed_faces(idxs_to_remove, N)
            reasoning_trace.append(f"Cubes with {N} exposed faces: {self.answer}")
        elif self.p_type == "colors":
            self.answer = self.count_cube_colors(color, colors, idxs_to_remove)
            reasoning_trace.append(f"Visible {color} cubes: {self.answer}")
        elif self.p_type == "max_color":
            counts = [
                self.count_cube_colors(c, colors, idxs_to_remove) for c in unique_colors
            ]
            if counts.count(max(counts)) > 1:
                raise ValueError(
                    "Multiple max colors found, please regenerate problem"
                )
            self.answer = unique_colors[counts.index(max(counts))]
            reasoning_trace.append(f"Color counts: {dict(zip(unique_colors, counts))}")
            reasoning_trace.append(f"Most frequent color: {self.answer}")
        elif self.p_type == "project":
            self.answer = self.count_project(idxs_to_remove)
            reasoning_trace.append(f"Maximum projection faces: {self.answer}")
        elif self.p_type == "missing_shape":
            correct = idxs_to_remove
            all_idxs = [
                (i, j, k)
                for i in range(rows)
                for j in range(cols)
                for k in range(depth)
            ]
            avail_to_add = set(all_idxs) - correct

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
                    c.set_fill(VALID_COLORS[colors[i][j][k]], opacity=1)
                    c.set_stroke(VALID_COLORS[colors[i][j][k]], width=1)
                    c.shift(i * 0.2 * RIGHT + j * 0.2 * UP + k * 0.2 * OUT)
                    mini.add(c)
                mini.scale(0.8)
                mini.move_to(pos + DOWN * 2)  # Moved up from DOWN * 3
                mini.rotate(-45 * DEGREES, axis=UP)
                mini.rotate(20 * DEGREES, axis=RIGHT)
                mini.add_updater(lambda m, dt: m.rotate(1.57 * dt, axis=UP))

                label = Text(lbl).scale(0.2).next_to(mini, UP)
                option_groups.add(VGroup(mini, label))

            # Add option E
            option_e_text = Text("E. None of the above").scale(0.2)
            option_e_text.move_to(DOWN * 2.8)  # Position below the mini-diagrams
            option_groups.add(option_e_text)

            option_groups.scale_to_fit_height(config.frame_height * 0.4)
            option_groups.scale_to_fit_width(config.frame_width * 0.9)
            option_groups.move_to(DOWN * 1.2)  # Moved up from to_edge(DOWN)
            
            self.answer = labels[options.index(correct)]
            reasoning_trace.append(f"Generated {len(variants)} variant shapes for multiple choice")
            reasoning_trace.append(f"Correct answer is option: {self.answer}")
            
        elif self.p_type == "matching":
            all_idxs = [
                (i, j, k)
                for i in range(rows)
                for j in range(cols)
                for k in range(depth)
            ]
            correct = set(all_idxs) - idxs_to_remove
            avail_to_add = idxs_to_remove

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
                mini.move_to(pos + DOWN * 2)  # Moved up from DOWN * 3
                mini.rotate(-45 * DEGREES, axis=UP)
                mini.rotate(20 * DEGREES, axis=RIGHT)
                mini.add_updater(lambda m, dt: m.rotate(1.57 * dt, axis=UP))

                label = Text(lbl).scale(0.2).next_to(mini, UP)
                option_groups.add(VGroup(mini, label))

            # Add option E
            option_e_text = Text("E. None of the above").scale(0.2)
            option_e_text.move_to(DOWN * 2.8)  # Position below the mini-diagrams
            option_groups.add(option_e_text)

            option_groups.scale_to_fit_height(config.frame_height * 0.4)
            option_groups.scale_to_fit_width(config.frame_width * 0.9)
            option_groups.move_to(DOWN * 1.2)  # Moved up from to_edge(DOWN)
            
            self.answer = labels[options.index(correct)]
            reasoning_trace.append(f"Generated {len(variants)} variant shapes for multiple choice")
            reasoning_trace.append(f"Correct answer is option: {self.answer}")
        else:
            raise ValueError(f"Invalid problem type: {self.p_type}")

        self.wait(3)
        self.play(*[FadeOut(mob) for mob in self.mobjects if mob != bg])
        
        # Display question
        title_text = random.choice(self.cfg["text"][self.p_type])
        title_text = title_text.replace("<C>", color)
        title_text = title_text.replace("<N>", str(N))
        lines = title_text.split("\n")

        para = Paragraph(*lines, alignment="center", font_size=36, line_spacing=0.8)
        para.move_to(ORIGIN)
        if para.width > 0.9 * config.frame_width:
            para.scale_to_fit_width(config.frame_width * 0.9)
        self.add_fixed_in_frame_mobjects(para)
        self.play(Write(para), run_time=1.5)

        if self.p_type == "matching" or self.p_type == "missing_shape":
            self.play(para.animate.to_edge(UP, buff=0.2 * config.frame_height))
            self.wait(0.5)
            self.play(Write(option_groups, run_time=1))

        self.wait(3)
        
        self.question_text = f"Observe the following structure. {title_text.replace(chr(10), ' ')}"
        
        # Save files
        with open(f"solutions/cubes_{self.p_type}_max{self.max_size}_seed{self.seed}.txt", "w") as f:
            f.write(str(self.answer))
        
        with open(f"question_text/cubes_{self.p_type}_max{self.max_size}_seed{self.seed}.txt", "w") as f:
            f.write(self.question_text)
            
        # Save detailed reasoning trace
        reasoning_trace.append("")
        reasoning_trace.append(f"Final Answer: {self.answer}")
        with open(f"reasoning_traces/cubes_{self.p_type}_max{self.max_size}_seed{self.seed}.txt", "w") as f:
            f.write("\n".join(reasoning_trace))

# Generate the cube video
scene = Cubes()
scene.render()

# Move the output file with descriptive name
output = Path("manim_output/videos/1080p30/Cubes.mp4")
if output.exists():
    filename = f"cubes_{scene.p_type}_max{scene.max_size}_seed{scene.seed}.mp4"
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