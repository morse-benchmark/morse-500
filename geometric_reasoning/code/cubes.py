from manim import *
import random
import json
import math
import os

class Cubes(ThreeDScene):
    def __init__(
        self,
        p_type,
        grid_size,
        p_removed,
        cfg_path="../templates/prisms.json",
        max_iters=25,
    ):
        super().__init__()
        self.p_type = p_type
        self.grid_size = grid_size
        self.total = math.prod(grid_size)
        self.n_removed = int(self.total * p_removed)
        self.max_iters = max_iters
        with open(cfg_path, "r") as f:
            self.cfg = json.load(f)

    def surface_area(self, removed):
        rows, cols, layers = self.grid_size

        # 6 axis directions
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
            (1, 0, 0),
            (-1, 0, 0),
            (0, 1, 0),
            (0, -1, 0),
            (0, 0, 1),
            (0, 0, -1),
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
                # project onto YZ plane:
                for j in range(ny):
                    for k in range(nz):
                        # depending on sign, scan i from front to back
                        i_range = range(nx - 1, -1, -1) if sign > 0 else range(0, nx)
                        for i in i_range:
                            if (i, j, k) in removed:
                                continue
                            # check neighbor in the viewing direction
                            ii = i + sign
                            # if neighbor is out of bounds or removed, this face shows
                            if ii < 0 or ii >= nx or (ii, j, k) in removed:
                                cnt += 1
                            break
            else:  # axis == 'y'
                # project onto XZ plane:
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

        # return the largest of the four
        return max(counts.values())

    def count_cubes_with_exposed_faces(self, removed, n):
        """
        Count how many cubes (not removed) have exactly `n` faces exposed,
        where a face is exposed if it either lies on the outside of the grid
        or its neighboring cube in that direction is removed.

        Parameters:
        - grid_size: (X, Y, Z) dimensions of the cube grid.
        - removed: set of (i, j, k) coordinates for cubes that have been removed.
        - n: the exact number of faces exposed to count.

        Returns:
        - The number of cubes with exactly `n` exposed faces.
        """
        X, Y, Z = self.grid_size
        # The 6 face-neighbor offsets
        directions = [
            (1, 0, 0),
            (-1, 0, 0),
            (0, 1, 0),
            (0, -1, 0),
            (0, 0, 1),
            (0, 0, -1),
        ]

        count = 0
        # Iterate over every possible cube
        for i in range(X):
            for j in range(Y):
                for k in range(Z):
                    if (i, j, k) in removed:
                        continue  # skip missing cubes

                    exposed = 0
                    # Check each of the six faces
                    for dx, dy, dz in directions:
                        ni, nj, nk = i + dx, j + dy, k + dz
                        # Exposed if neighbor is outside the grid...
                        if not (0 <= ni < X and 0 <= nj < Y and 0 <= nk < Z):
                            exposed += 1
                        # ...or if that neighbor cube has been removed
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

        # Labels positioned relative to squares
        labels = VGroup(*[Text(nm, font_size=32) for nm in names])
        for sq, lbl in zip(squares, labels):
            lbl.next_to(sq, RIGHT, buff=1.2)

        # Arrows connecting squares to labels
        arrows = VGroup(
            *[
                Arrow(
                    start=sq.get_right(), end=lbl.get_left(), buff=0.05, stroke_width=4
                )
                for sq, lbl in zip(squares, labels)
            ]
        )

        # Title
        title = Text("Remember the following color names", font_size=40)
        title.to_edge(UP)
        all_mobjects = VGroup(squares, arrows, labels)
        all_mobjects.move_to(ORIGIN)
        # Animation sequence
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

        cubes_vgroup = VGroup()
        cube_list = [
            [[None for z in range(depth)] for y in range(cols)] for z in range(rows)
        ]
        colors = [
            [[None for z in range(depth)] for y in range(cols)] for z in range(rows)
        ]
        unique_colors = set()

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

        max_height = config.frame_height * 0.6  # 5% buffer top and bottom

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
        self.play(Write(cubes_vgroup), runtime=1)

        self.wait(1)

        heights = [[depth - 1 for j in range(cols)] for i in range(rows)]
        iters = 0
        cubes_to_remove = set()
        idxs_to_remove = set()
        done = False
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
                raise ValueError(
                    "Multiple max colors found, please regenerate problem"
                )
            self.answer = unique_colors[counts.index(max(counts))]
        elif self.p_type == "project":
            self.answer = self.count_project(idxs_to_remove)
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
                # random number to add and remove
                na = random.randint(
                    1, min(len(avail_to_add), max(1, len(correct))) // 4
                )
                nr = (
                    random.randint(1, min(len(correct), max(1, len(correct) - 1)) // 4)
                    if len(correct) > 1
                    else 1
                )
                to_add = set(random.sample(avail_to_add, na))
                to_remove = set(random.sample(correct, nr))
                variant = (correct - to_remove) | to_add
                if variant != correct and variant not in variants:
                    variants.append(variant)

            options = [correct] + variants
            random.shuffle(options)
            labels = ["A", "B", "C", "D"]

            # Draw each option as a mini-diagram at the bottom
            option_groups = VGroup()
            positions = [LEFT * 3, LEFT, RIGHT, RIGHT * 3]
            for pos, inds, lbl in zip(positions, options, labels):
                mini = VGroup()
                for i, j, k in inds:
                    c = Cube(side_length=0.2)
                    c.set_fill(colors[i][j][k], opacity=1)
                    c.set_stroke(colors[i][j][k], width=1)
                    c.shift(i * 0.2 * RIGHT + j * 0.2 * UP + k * 0.2 * OUT)
                    mini.add(c)
                mini.scale(0.8)
                mini.move_to(pos + DOWN * 3)
                mini.rotate(-45, axis=UP)
                mini.rotate(20, axis=RIGHT)
                mini.add_updater(lambda m, dt: m.rotate(1.57 * dt, axis=UP))

                label = Text(lbl).scale(0.2).next_to(mini, UP)
                option_groups.add(VGroup(mini, label))

            option_groups.scale_to_fit_height(config.frame_height * 0.4)
            option_groups.scale_to_fit_width(config.frame_width * 0.9)

            option_groups.to_edge(DOWN)
            self.answer = labels[options.index(correct)]
        elif self.p_type == "matching":
            self.wait(0.5)

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
                to_add = set(random.sample(avail_to_add, na))
                to_remove = set(random.sample(correct, nr))
                variant = (correct - to_remove) | to_add
                if variant != correct and variant not in variants:
                    variants.append(variant)

            options = [correct] + variants
            random.shuffle(options)
            labels = ["A", "B", "C", "D"]

            # Draw each option as a mini-diagram at the bottom
            option_groups = VGroup()
            positions = [LEFT * 3, LEFT, RIGHT, RIGHT * 3]
            for pos, inds, lbl in zip(positions, options, labels):
                mini = VGroup()
                for i, j, k in inds:
                    c = Cube(side_length=0.2)
                    c.set_fill(colors[i][j][k], opacity=0.8)
                    c.set_stroke(colors[i][j][k], width=1)
                    c.shift(i * 0.2 * RIGHT + j * 0.2 * UP + k * 0.2 * OUT)
                    mini.add(c)
                mini.scale(0.8)
                mini.move_to(pos + DOWN * 3)
                mini.rotate(-45, axis=UP)
                mini.rotate(20, axis=RIGHT)
                mini.add_updater(lambda m, dt: m.rotate(1.57 * dt, axis=UP))

                label = Text(lbl).scale(0.2).next_to(mini, UP)
                option_groups.add(VGroup(mini, label))

            option_groups.scale_to_fit_height(config.frame_height * 0.4)
            option_groups.scale_to_fit_width(config.frame_width * 0.9)

            option_groups.to_edge(DOWN)
            # self.play(Write(option_groups, run_time=1))
            self.answer = labels[options.index(correct)]
        else:
            raise ValueError("invalid problem type")

        self.wait(3)
        self.play(*[FadeOut(mob) for mob in self.mobjects if mob != bg])
        title_text = random.choice(self.cfg["text"][self.p_type])
        title_text = title_text.replace("<C>", color)
        title_text = title_text.replace("<N>", str(N))
        lines = title_text.split("\n")

        para = Paragraph(*lines, alignment="center", font_size=36, line_spacing=0.8)

        # 3. Move it to the center of the screen (ORIGIN).
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
        self.question_text = "Observe the following structure. {}".format(
            title_text.replace("\n", " ")
        )


def create_problem(grid_size, p_type, p_removed, path, file_name):
    config.output_file = f"{path}/questions/{file_name}"
    scene = Cubes(p_type, grid_size, p_removed)
    scene.render()
    with open(f"media/videos/1080p60/{path}/solutions/{file_name}.txt", "w") as f:
        f.write(str(scene.answer))
    with open(f"media/videos/1080p60/{path}/question_text/{file_name}.txt", "w") as f:
        f.write(scene.question_text)


if __name__ == "__main__":
    os.makedirs("media/videos/1080p60/cubes/questions", exist_ok=True)
    os.makedirs("media/videos/1080p60/cubes/solutions", exist_ok=True)
    os.makedirs("media/videos/1080p60/cubes/question_text", exist_ok=True)
    types = [
        "count",
        "missing",
        "surface_area",
        "exposed",
        "colors",
        "project",
        "missing_shape",
        "matching",
        "max_color",
    ]
    for p_type in types:
        create_problem((4, 4, 2), p_type, 0.4, "cubes", f"{p_type}_1")
        create_problem((5, 5, 3), p_type, 0.45, "cubes", f"{p_type}_2")
        create_problem((6, 6, 4), p_type, 0.5, "cubes", f"{p_type}_3")
