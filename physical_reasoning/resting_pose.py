from manim import *
import numpy as np
import random
from scipy.integrate import solve_ivp
import time
import os

DEFAULT_M_RANGE = (0.8, 1.5)
DEFAULT_K_RANGE = (3.0, 8.0)
DEFAULT_V0_RANGE = (5.0, 8.5) 


DAMPING_FACTOR_RANGE = (0.05, 0.12) 


X_OFFSET_RANGE = (-2.0, 2.0)

SIM_DURATION = 1000 
SIM_POINTS_PER_SEC = 50
ANIMATION_DURATION = 4 

SPRING_BUMPS = 8
SPRING_RADIUS = 0.12
MASS_SIDE_LENGTH = 0.6
WALL_HEIGHT = 0.5


SYSTEM_LABELS_3 = ["A", "B", "C"]
SYSTEM_LABELS_4 = ["A", "B", "C", "D"]
SYSTEM_LABELS_5 = ["A", "B", "C", "D", "E"]

SYSTEM_MASS_COLORS = [RED_C, BLUE_C, GREEN_C, YELLOW_C, PURPLE_C]
SYSTEM_SPRING_COLORS = [RED_D, BLUE_D, GREEN_D, YELLOW_D, PURPLE_D]


VERTICAL_SYSTEM_Y_POSITIONS_3 = [1.0, -0.5, -2.0]
VERTICAL_SYSTEM_Y_POSITIONS_4 = [1.5, 0.0, -1.5, -3.0]
VERTICAL_SYSTEM_Y_POSITIONS_5 = [1.8, 0.8, -0.2, -1.2, -2.2]


def oscillator_derivs(t, state_X, m, k, b):
    X, V = state_X
    aX = (-k * X - b * V) / m if m > 1e-6 else 0
    return [V, aX]

def simulate_oscillator_for_X(t_duration, initial_state_X, params_mkb):
    m, k, b = params_mkb
    t_span = [0, t_duration]
    n_points = int(t_duration * SIM_POINTS_PER_SEC) + 1
    t_eval = np.linspace(t_span[0], t_span[1], n_points)

    if not np.all(np.isfinite(initial_state_X)):
        initial_state_X = [0.0, 0.0]

    try:
        sol = solve_ivp(
            oscillator_derivs, t_span, initial_state_X, args=(m, k, b),
            t_eval=t_eval, rtol=1e-6, atol=1e-8, dense_output=False
        )
        if sol.status != 0:
            t_pts_sol = sol.t
            X_pts_sol = sol.y[0]
            V_pts_sol = sol.y[1] if sol.y.shape[0] > 1 else np.zeros_like(X_pts_sol)
            X_pts = np.interp(t_eval, t_pts_sol, X_pts_sol, left=initial_state_X[0], right=X_pts_sol[-1] if len(X_pts_sol)>0 else initial_state_X[0])
            V_pts = np.interp(t_eval, t_pts_sol, V_pts_sol, left=initial_state_X[1], right=V_pts_sol[-1] if len(V_pts_sol)>0 else initial_state_X[1])
            t_pts = t_eval
        else:
            t_pts = sol.t
            X_pts = sol.y[0]
            V_pts = sol.y[1]

        valid_indices = np.where(np.isfinite(X_pts) & np.isfinite(V_pts))[0]
        if len(valid_indices) < len(X_pts):
            if len(valid_indices) > 0:
                max_valid_idx = valid_indices[-1]
                t_pts = t_pts[:max_valid_idx+1]; X_pts = X_pts[:max_valid_idx+1]; V_pts = V_pts[:max_valid_idx+1]
            else:
                X0, V0 = initial_state_X
                return np.array([0, 1e-3]), np.array([X0, X0]), np.array([V0, V0])

        if len(t_pts) < 2:
            X0, V0 = initial_state_X
            return np.array([0, 1e-3]), np.array([X0, X0]), np.array([V0, V0])

        return t_pts, X_pts, V_pts
    except Exception as e:
        print(f"Error during solve_ivp for X: {e}")
        X0, V0 = initial_state_X
        return np.array([0, 1e-3]), np.array([X0, X0]), np.array([V0, V0])


def create_single_spring_visual(start_point, end_point, bumps=SPRING_BUMPS, radius=SPRING_RADIUS, color=WHITE, stroke_width=2):
    start=np.array(start_point); end=np.array(end_point)
    length=np.linalg.norm(end-start)
    direction_vec = end - start
    if length < 1e-6: return Line(start, end, color=color, stroke_width=stroke_width)
    direction = normalize(direction_vec)
    if abs(direction[0]) < 1e-6 and abs(direction[1]) < 1e-6 : perp_direction_for_bumps = RIGHT
    elif abs(direction[1]) > 0.999: perp_direction_for_bumps = RIGHT
    else: perp_direction_for_bumps = normalize(np.cross(direction, OUT))
    def spring_func(t): return start + t*direction_vec + perp_direction_for_bumps*radius*np.sin(bumps*TAU*t)
    num_segments = bumps * 20
    points = [spring_func(t) for t in np.linspace(0, 1, num_segments + 1)]
    return VMobject(color=color, stroke_width=stroke_width).set_points_as_corners(points)


class DampedOscillatorsRestingPositionQuiz(Scene):

    def generate_distinct_offsets(self, num_offsets, min_val, max_val, min_abs_separation=0.25):
        offsets = []
        abs_offsets = []
        attempts = 0
        max_attempts = 100
        while len(offsets) < num_offsets and attempts < max_attempts:
            attempts += 1
            new_offset = random.uniform(min_val, max_val)
            if abs(new_offset) < min_abs_separation / 2 and num_offsets > 1 :
                if random.random() < 0.5: continue
            new_abs_offset = abs(new_offset)
            too_close = False
            for existing_abs_offset in abs_offsets:
                if abs(new_abs_offset - existing_abs_offset) < min_abs_separation:
                    too_close = True; break
            if not too_close:
                for existing_offset in offsets:
                    if abs(new_offset - existing_offset) < min_abs_separation / 2:
                        too_close = True; break
                if not too_close:
                    offsets.append(new_offset); abs_offsets.append(new_abs_offset); attempts = 0
        if len(offsets) < num_offsets:
            print("Warning: Could not generate fully distinct offsets. Using potentially close values.")
            while len(offsets) < num_offsets:
                offset_val = random.uniform(min_val, max_val)
                if abs(offset_val) < 0.15 and num_offsets > 1:
                    offset_val = np.sign(offset_val) * 0.15 if offset_val != 0 else random.choice([-1,1])*0.15
                offsets.append(offset_val)
        return offsets


    def construct(self):
        num_systems = int(os.environ.get('MANIM_OBJECT_COUNT', 4))
        
        if num_systems not in [3, 4, 5]:
            print(f"Warning: object_count {num_systems} is not valid. Using default value 4.")
            num_systems = 4
        

        seed = int(time.time()); random.seed(seed); np.random.seed(seed)

        scene_title = Text(f"Damped Oscillators - Resting Positions ({num_systems} Systems)", font_size=32).to_edge(UP)

        max_possible_x = max(abs(val) for val in X_OFFSET_RANGE) + 0.5
        x_axis_bnd = min(np.ceil(max_possible_x * 1.2) + 0.5, 4.5)
        x_axis_bnd = max(x_axis_bnd, 2.5)
        x_tick_freq = 0.5 if x_axis_bnd <= 3.5 else 1.0

        x_axis = NumberLine(
            x_range=[-x_axis_bnd, x_axis_bnd, x_tick_freq],
            length=self.camera.frame_width - 2.5,
            color=WHITE, include_numbers=True,
            label_direction=DOWN, font_size=18, stroke_width=2
        ).move_to(UP * 2.5) 

        systems_data = []
        system_mobjects = VGroup()

        if num_systems == 3:
            SYSTEM_LABELS = SYSTEM_LABELS_3
            VERTICAL_SYSTEM_Y_POSITIONS = VERTICAL_SYSTEM_Y_POSITIONS_3
            current_mass_colors = SYSTEM_MASS_COLORS[:3]
            current_spring_colors = SYSTEM_SPRING_COLORS[:3]
        elif num_systems == 4:
            SYSTEM_LABELS = SYSTEM_LABELS_4
            VERTICAL_SYSTEM_Y_POSITIONS = VERTICAL_SYSTEM_Y_POSITIONS_4
            current_mass_colors = SYSTEM_MASS_COLORS[:4]
            current_spring_colors = SYSTEM_SPRING_COLORS[:4]
        elif num_systems == 5:
            SYSTEM_LABELS = SYSTEM_LABELS_5
            VERTICAL_SYSTEM_Y_POSITIONS = VERTICAL_SYSTEM_Y_POSITIONS_5
            current_mass_colors = SYSTEM_MASS_COLORS[:5]
            current_spring_colors = SYSTEM_SPRING_COLORS[:5]

        x_offsets = self.generate_distinct_offsets(num_systems, X_OFFSET_RANGE[0], X_OFFSET_RANGE[1])

        for i in range(num_systems):
            sys_label_text = SYSTEM_LABELS[i]
            mass_color = current_mass_colors[i]
            spring_color = current_spring_colors[i]
            y_pos = VERTICAL_SYSTEM_Y_POSITIONS[i]

            m = random.uniform(*DEFAULT_M_RANGE)
            k = random.uniform(*DEFAULT_K_RANGE)
            damping_factor = random.uniform(*DAMPING_FACTOR_RANGE) 
            b = damping_factor * 2 * np.sqrt(m * k)

            x_offset = x_offsets[i]
            initial_x_actual = 0.0
            X0 = initial_x_actual - x_offset
            V0 = random.uniform(*DEFAULT_V0_RANGE) * random.choice([-1, 1])
            if abs(X0) < 0.2 and abs(V0) < 0.8 :
                 V0 = np.sign(V0) * random.uniform(1.2, DEFAULT_V0_RANGE[1]) if V0 != 0 else random.choice([-1,1]) * random.uniform(1.2, DEFAULT_V0_RANGE[1])

            params_mkb = [m, k, b]
            initial_state_X = [X0, V0]

            print(f"--- System {sys_label_text} (Seed: {seed}) ---")
            print(f"Params: m={m:.2f}, k={k:.2f}, b={b:.2f} (zeta={damping_factor:.3f}), x_offset={x_offset:.2f}")
            print(f"Initial X: X0={X0:.2f}, V0={V0:.2f} (Actual x0={initial_x_actual:.2f})")

            system_group = VGroup()
            wall_x_coord = x_axis.n2p(-x_axis_bnd)[0] - 0.5
            wall = Line(UP * WALL_HEIGHT/2, DOWN * WALL_HEIGHT/2, color=GRAY_C, stroke_width=3)
            wall.move_to([wall_x_coord, y_pos, 0])
            spring_anchor_point = wall.get_center()
            mass_initial_point_on_axis = x_axis.n2p(initial_x_actual)
            mass_obj = Square(side_length=MASS_SIDE_LENGTH, color=mass_color, fill_opacity=0.8)
            mass_obj.move_to([mass_initial_point_on_axis[0], y_pos, 0])
            spring_obj = create_single_spring_visual(spring_anchor_point, mass_obj.get_left(), color=spring_color, bumps=SPRING_BUMPS, radius=SPRING_RADIUS)
            label = Tex(sys_label_text, font_size=24).next_to(wall, LEFT, buff=0.2)

            system_group.add(wall, spring_obj, mass_obj, label)
            system_mobjects.add(system_group)

            t_sim, X_sim, _ = simulate_oscillator_for_X(SIM_DURATION, initial_state_X, params_mkb)

            systems_data.append({
                "label_text": sys_label_text, "m": m, "k": k, "b": b, "x_offset": x_offset,
                "X0": X0, "V0": V0, "initial_x_actual": initial_x_actual,
                "mass_obj": mass_obj, "spring_obj": spring_obj, "spring_anchor": spring_anchor_point,
                "y_pos": y_pos,
                "t_sim": t_sim, "X_sim": X_sim,
                "anim_time": 0.0,
                "spring_color": spring_color,
                "damping_factor": damping_factor 
            })

        self.play(
            Write(scene_title),
            Create(x_axis),
            FadeIn(system_mobjects, lag_ratio=0.1), run_time=2.0
        )
        self.wait(0.5)

        for i in range(num_systems):
            sys_data = systems_data[i]
            def mass_updater_func(mobj, dt, system_index=i): 
                s_data = systems_data[system_index]
                s_data["anim_time"] += dt
                current_X = np.interp(min(s_data["anim_time"], s_data["t_sim"][-1]), s_data["t_sim"], s_data["X_sim"])
                actual_x_pos = current_X + s_data["x_offset"]

                actual_x_pos = np.clip(actual_x_pos, -3.5, 3.5)
                
                screen_x_coord = x_axis.n2p(actual_x_pos)[0]
                mobj.move_to([screen_x_coord, s_data["y_pos"], 0])
            sys_data["mass_obj"].add_updater(mass_updater_func)

            def spring_updater_func(mobj, system_index=i): 
                s_data = systems_data[system_index]
                current_X = np.interp(min(s_data["anim_time"], s_data["t_sim"][-1]), s_data["t_sim"], s_data["X_sim"])
                actual_x_pos = current_X + s_data["x_offset"]

                actual_x_pos = np.clip(actual_x_pos, -3.5, 3.5)
                
                screen_x_coord = x_axis.n2p(actual_x_pos)[0]
                mobj.become(create_single_spring_visual(s_data["spring_anchor"], [screen_x_coord - MASS_SIDE_LENGTH/2, s_data["y_pos"], 0], color=s_data["spring_color"], bumps=SPRING_BUMPS, radius=SPRING_RADIUS))
            sys_data["spring_obj"].add_updater(spring_updater_func)

        self.wait(ANIMATION_DURATION)

        for sys_data in systems_data:
            sys_data["mass_obj"].clear_updaters()
            sys_data["spring_obj"].clear_updaters()

            final_anim_time = sys_data["anim_time"]
            final_X_at_anim_end = np.interp(min(final_anim_time, sys_data["t_sim"][-1]), sys_data["t_sim"], sys_data["X_sim"])
            final_actual_x_pos_at_anim_end = final_X_at_anim_end + sys_data["x_offset"]

            final_actual_x_pos_at_anim_end = np.clip(final_actual_x_pos_at_anim_end, -3.5, 3.5)

            screen_x_coord = x_axis.n2p(final_actual_x_pos_at_anim_end)[0]
            sys_data["mass_obj"].move_to([screen_x_coord, sys_data["y_pos"], 0])
            sys_data["spring_obj"].become(create_single_spring_visual(sys_data["spring_anchor"], [screen_x_coord - MASS_SIDE_LENGTH/2, sys_data["y_pos"], 0], color=sys_data["spring_color"], bumps=SPRING_BUMPS, radius=SPRING_RADIUS))

        self.wait(0.5)

        ranking_data = []
        for sys_data in systems_data:
            distance_from_zero = abs(sys_data["x_offset"])
            ranking_data.append({"label": sys_data["label_text"], "distance": distance_from_zero, "x_offset": sys_data["x_offset"]})

        ranking_data.sort(key=lambda item: (item["distance"], item["x_offset"], item["label"]))
        ranked_labels = [item["label"] for item in ranking_data]
        final_ans_script = ", ".join(ranked_labels)

        quiz_title_text = Tex("Quiz: Resting Position Ranking", font_size=28).to_edge(UP, buff=0.2) 
        quiz_instruction1 = Tex("The systems A, B, C, D will eventually come to rest.", font_size=22)
        quiz_instruction2 = Tex("All systems started with their mass at the ruler's zero point (0.0).", font_size=22)
        quiz_instruction3 = Tex("Rank them by the distance of their final resting position", font_size=22)
        quiz_instruction4 = Tex("from the ruler's zero point, from CLOSEST to FURTHEST.", font_size=22)

        if num_systems == 3:
            quiz_instruction1 = Tex("The systems A, B, C will eventually come to rest.", font_size=22)
            quiz_question = Tex("List the labels (A, B, C) in the correct order.", font_size=25, color=YELLOW_C)
        elif num_systems == 4:
            quiz_instruction1 = Tex("The systems A, B, C, D will eventually come to rest.", font_size=22)
            quiz_question = Tex("List the labels (A, B, C, D) in the correct order.", font_size=25, color=YELLOW_C)
        elif num_systems == 5:
            quiz_instruction1 = Tex("The systems A, B, C, D, E will eventually come to rest.", font_size=22)
            quiz_question = Tex("List the labels (A, B, C, D, E) in the correct order.", font_size=25, color=YELLOW_C)


        quiz_items_for_display = VGroup(quiz_instruction1, quiz_instruction2, quiz_instruction3, quiz_instruction4, quiz_question).arrange(DOWN, buff=0.20)

        quiz_items_for_display.next_to(quiz_title_text, DOWN, buff=0.4)


        self.play(
            FadeOut(scene_title), FadeOut(system_mobjects), FadeOut(x_axis),
            Write(quiz_title_text), Write(quiz_items_for_display)
        )
        print(f"FINAL_ANSWER: {final_ans_script}")
        for item in ranking_data:
            print(f"System {item['label']}: x_offset = {item['x_offset']:.3f}, distance = {item['distance']:.3f}, zeta = {next(s['damping_factor'] for s in systems_data if s['label_text'] == item['label']):.3f}")
        self.wait(20)
