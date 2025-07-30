from manim import *
import random
import json 
import itertools
import numpy as np
import os 

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

# ------------------------------------------------------------
# Enumerate the 24 cube orientations as index permutations
# ------------------------------------------------------------

def enumerate_rotations():
    """Return a list of 24 permutations.  Each permutation p maps
    ORDER[i] (original face) → world axis i in ORDER after some rigid
    rotation.  Example: p[3] = 0 means original RIGHT face ends up on IN.
    """
    rots = []

    for up_name, up_vec in NAME_TO_VEC.items():
        for front_name, front_vec in NAME_TO_VEC.items():
            if front_name == up_name:
                continue
            if not np.isclose(np.dot(up_vec, front_vec), 0):
                continue  # must be perpendicular

            right_vec = np.cross(up_vec, front_vec)
            if np.linalg.norm(right_vec) < 0.5:
                continue
            right_name = VEC_TO_NAME[tuple(right_vec.astype(int))]

            mapping_names = {
                "UP": up_name,
                "DOWN": VEC_TO_NAME[tuple((-up_vec).astype(int))],
                "OUT": front_name,
                "IN": VEC_TO_NAME[tuple((-front_vec).astype(int))],
                "RIGHT": right_name,
                "LEFT": VEC_TO_NAME[tuple((-right_vec).astype(int))],
            }
            # Build permutation in canonical ORDER of axis names (strings)
            mapping = [NAME_TO_IDX[mapping_names[axis]] for axis in ORDER]
            if mapping not in rots:
                rots.append(mapping)
    return rots

ROTATIONS = enumerate_rotations() 

def build_cube_map(color_list, side=0.55):
    """Return a VGroup showing a simple T‑shaped cube net. The mapping of
    ORDER faces to positions:
           UP
    LEFT  OUT  RIGHT  IN
           DOWN
    """
    # grid offsets (x, y) in squares
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
    gap = 0.05 * side
    for name, (dx, dy) in offsets.items():
        idx = NAME_TO_IDX[name]
        sq = base.copy().set_fill(color_list[idx], 1).set_stroke(BLACK, 0.5)
        sq.shift(RIGHT * dx * (side + gap) + UP * dy * (side + gap))
        g.add(sq)
    return g

class Dice(ThreeDScene):
    def __init__(self, p_type, num_dice, n_roll=3, cfg_path = "../templates/dice.json"):
        super().__init__()
        self.p_type = p_type
        self.num_dice = num_dice
        self.n_roll = n_roll 

        self.VALID_COLORS = {
            BLUE: "blue", 
            RED: "red", 
            WHITE: "white",
            GREEN: "green", 
            YELLOW: "yellow",
            PURPLE: "purple",
        }
        with open(cfg_path, 'r') as f:
            self.cfg = json.load(f)
    def axis_to_idx_exact(self, v):
        AXES = [IN, OUT, LEFT, RIGHT, UP, DOWN] 
        for i, axis in enumerate(AXES):
            if np.array_equal(v, axis):              # element‑wise “==”
                return i
        raise ValueError("axis not found")
    def roll_to_face(self, cube, curr_face, new_face):
        animations = []
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
        if np.array_equal(curr_face, new_face):
            return animations
        axis = np.cross(curr_face, new_face)
        if np.all(axis == 0):          # opposite face (e.g. DOWN → UP)
            # any perpendicular axis works; choose RIGHT unless collinear
            axis = RIGHT if not np.array_equal(curr_face, RIGHT) else OUT
            angle = PI                 # 180°
        else:
            axis = axis / np.linalg.norm(axis)
            angle = PI/2               # 90°

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
        """Return colours after applying index‑mapping list of len 6."""
        return [colors[i] for i in mapping]
    def build_cube(self, color_list, side = 2.0):
        """Return a Manim cube coloured according to *color_list* of length 6
        in order [IN, OUT, LEFT, RIGHT, UP, DOWN]."""
        cube = Cube(side_length=side)
        for face, col in zip(cube, color_list):
            face.set_fill(col, 1).set_stroke(BLACK, 0.5)
        return cube
    def show_colors(self, colors, names):

        squares1 = VGroup(
            *[Square(1.0).set_fill(col, 1).set_stroke(width=0) for col in colors[:3]]
        )
        squares1.arrange(DOWN, buff=0.5, aligned_edge=LEFT).to_edge(LEFT, buff=2)

        # Labels positioned relative to squares
        labels1 = VGroup(*[Text(nm, font_size=32) for nm in names[:3]])
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
            *[Square(1.0).set_fill(col, 1).set_stroke(width=0) for col in colors[3:]]
        )
        squares2.arrange(DOWN, buff=0.4, aligned_edge=LEFT).to_edge(LEFT, buff=2)

        # Labels positioned relative to squares
        labels2 = VGroup(*[Text(nm, font_size=32) for nm in names[3:]])
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
        self.add_fixed_in_frame_mobjects(bg)
        if self.p_type not in ["match", "fold"]:
            self.show_colors(list(self.VALID_COLORS.keys()), list(self.VALID_COLORS.values()))
        
        if "roll" in self.p_type:
            prompt = Text(
                "After each roll, record the color on the top of each cube", color=WHITE, font_size=36
            ).move_to(ORIGIN)
            self.question_text = "After each roll, record the color on the top of each cube. "
        else:
            prompt = Text(
                "Observe the following cubes", color=WHITE, font_size=36
            ).move_to(ORIGIN)
            self.question_text = "Observe the following cubes. "

        self.play(FadeIn(prompt), run_time=0.5)
        self.wait(1.5)
        self.play(FadeOut(prompt), run_time=0.5)
        self.wait(1)
        self.set_camera_orientation(phi=70 * DEGREES, theta=-45 * DEGREES)
        cubes = VGroup()
        cube_colors = []
        for i in range(self.num_dice):
            c = Cube(side_length=1.5)

            # IN, LEFT, UP
            # IN, OUT, LEFT, RIGHT, UP, DOWN
            c_color = []
            for face in c:
                color = random.choice(list(self.VALID_COLORS.keys()))
                face.set_fill(color, 1).set_stroke(BLACK, 1)
                c_color.append(color)
            cube_colors.append(c_color)
            # pick a color cycling through the list:
            # position in a row:
            c.shift(RIGHT * i * 3)
            cubes.add(c)
        print(self.VALID_COLORS[cube_colors[0][0]], self.VALID_COLORS[cube_colors[0][2]], self.VALID_COLORS[cube_colors[0][4]])
        # center the whole row on screen
        cubes.move_to(ORIGIN)
        self.add(cubes)
        
        color = random.choice(list(self.VALID_COLORS.keys()))

        hidden = {}
        for c in cube_colors:
            hidden[c[0]] = hidden.get(c[0], 0) + 1
            hidden[c[2]] = hidden.get(c[0], 0) + 1
            hidden[c[4]] = hidden.get(c[0], 0) + 1

        choices = None
        if self.p_type == "fold" or self.p_type == "match":
            self.play(*[
                Rotate(
                        cube,
                        angle=2 * PI,
                        axis=[1,1,1],
                        about_point=cube.get_center(),
                    )
                    for cube in cubes
                ],
                run_time=5,
                rate_func=smooth
            )
        elif self.p_type == "hidden":
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
            self.wait()
            self.answer = hidden.get(color, 0)
        elif self.p_type == "max_hidden":
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
            self.wait()
            max_val = max(hidden.values())
            max_keys = [k for k, v in hidden.items() if v == max_val]

            if len(max_keys) > 1:
                raise ValueError(f"Multiple keys have the same max value: {max_keys}")
            self.answer = self.VALID_COLORS[max(hidden, key=hidden.get)]
        elif "roll" in self.p_type:
            curr_faces = [OUT for _ in range(self.num_dice)]
            rolls = {c: 0 for c in list(self.VALID_COLORS.keys())}
            for i in range(self.n_roll):
                final_faces = [random.choice([IN, OUT, LEFT, RIGHT, UP, DOWN]) for _ in range(self.num_dice)]
                animations = []
                for j in range(self.num_dice):
                    animations += self.roll_to_face(cubes[j], curr_faces[j], final_faces[j])
                self.play(*animations, run_time=1)
                self.wait(1)
                curr_faces = final_faces.copy()

                for j, face in enumerate(final_faces):
                    idx = self.axis_to_idx_exact(face)
                    face_color = cubes[j][idx].get_fill_color()
                    rolls[face_color] += 1
            
            if self.p_type == "roll":
                self.answer=self.VALID_COLORS[max(rolls, key=rolls.get)]
            else:
                self.answer = rolls[color]

        if self.p_type == "fold" or self.p_type=="match":
            self.set_camera_orientation(phi=0 * DEGREES, theta=-90 * DEGREES)
            cubes.rotate(-90*DEGREES, axis=RIGHT, about_point=ORIGIN)
            cubes.rotate(-45*DEGREES, axis=UP, about_point=ORIGIN)
            cubes.rotate(20*DEGREES, axis=RIGHT, about_point=ORIGIN)
            # self.play(
            #     cubes.animate.to_edge(UP, buff=0.1 * config.frame_height),
            #     run_time=1
            # )
            face_colors = cube_colors[0]
            legal_layouts = {tuple(self.apply_perm(face_colors, p)) for p in ROTATIONS}
            correct_layout = random.choice(tuple(legal_layouts))

            # Build three distractors: random permutations that are *not* legal
            distractors = []
            attempts = 0
            while len(distractors) < 3 and attempts < 3000:
                attempts += 1
                perm = random.sample(range(6), 6)
                cand = tuple(face_colors[i] for i in perm)
                if cand in legal_layouts or cand in distractors or cand == correct_layout:
                    continue  # skip legal or duplicate candidate
                distractors.append(cand)

            if len(distractors) < 3:
                raise ValueError(
                    "Could not generate 3 distinct impossible layouts; try with more varied colours.")

            options = [correct_layout] + distractors
            random.shuffle(options)
            correct_index = options.index(correct_layout)
            
            # 3) Build cube‑maps and labels (fixed in frame)
            choices = VGroup()
            for i, layout in enumerate(options):
                choice = VGroup()
                if self.p_type == "fold":
                    m = build_cube_map(list(layout))
                else:
                    m = self.build_cube(list(layout), side=1.5)
                    m.move_to(ORIGIN)
                    m.rotate(-90*DEGREES, axis=RIGHT, about_point=ORIGIN)
                    m.rotate(-45*DEGREES, axis=UP, about_point=ORIGIN)
                    m.rotate(20*DEGREES, axis=RIGHT, about_point=ORIGIN)
                choice.add(m)
                lbl = Text(chr(ord('A')+i), font_size=28)
                lbl.next_to(m, DOWN, buff=0.15)
                choice.add(lbl)
                choices.add(choice)
            # arrange maps in a row visually (screen space)
            choices.arrange(buff=1.2, aligned_edge=DOWN)
            choices.to_edge(DOWN, buff=0.4)

            # Keep them facing the camera regardless of 3‑D camera moves
            
            # self.play(Indicate(choices[correct_index]))
            self.answer = chr(ord('A')+correct_index)

        self.wait(1.5)
        self.play(*[FadeOut(mob) for mob in self.mobjects if mob != bg])
        title_text = random.choice(self.cfg["text"][self.p_type])
        title_text = title_text.replace("<C>", self.VALID_COLORS[color])
        lines = title_text.split('\n')
        para = Paragraph(
            *lines, alignment="center", font_size=36, line_spacing=0.8
        )

        # 3. Move it to the center of the screen (ORIGIN).
        para.move_to(ORIGIN)
        if para.width > 0.9*config.frame_width:
            para.scale_to_fit_width(config.frame_width * 0.9)
        self.add_fixed_in_frame_mobjects(para)
        self.play(Write(para), run_time=1.5)

        self.wait(0.5)
        if choices is not None:
            self.play(para.animate.to_edge(UP, buff=0.2*config.frame_height))
            if self.p_type == "fold":
                self.add_fixed_in_frame_mobjects(choices)
            self.play(FadeIn(choices, run_time=1.5))
            self.wait(0.5)
            if self.p_type == "match":
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
        self.wait()
        self.question_text += title_text.replace('\n', ' ')

def create_problem(p_type, num_dice, n_roll, path, file_name):
    config.output_file = f'{path}/questions/{file_name}'
    scene = Dice(p_type, num_dice, n_roll)
    scene.render()
    with open(f'media/videos/1080p60/{path}/solutions/{file_name}.txt', 'w') as f:
        f.write(str(scene.answer))  
    with open(f"media/videos/1080p60/{path}/question_text/{file_name}.txt", "w") as f:
        f.write(scene.question_text)

if __name__ == "__main__":
    random.seed(2)
    os.makedirs('media/videos/1080p60/cube_rotation/questions', exist_ok=True)
    os.makedirs('media/videos/1080p60/cube_rotation/solutions', exist_ok=True)
    os.makedirs("media/videos/1080p60/cube_rotation/question_text", exist_ok=True)
    types1 = [
        "hidden",
        "roll",
        "n_roll",
        "max_hidden",
    ]
    types2 = [
        # "match",
        # "fold"
    ]
    for p_type in types1:
        create_problem(p_type, 3, 5, 'cube_rotation', f'{p_type}_1')
        create_problem(p_type, 4, 7,'cube_rotation', f'{p_type}_2')
        create_problem(p_type, 5, 9, 'cube_rotation', f'{p_type}_3')
    for p_type in types2:
        create_problem(p_type, 1, 1, 'cube_rotation', f'{p_type}_1')
        create_problem(p_type, 1, 1,'cube_rotation', f'{p_type}_2')
        create_problem(p_type, 1, 1, 'cube_rotation', f'{p_type}_3')
