from manim import *
import numpy as np
import random
from scipy.integrate import solve_ivp
import time
import os


DEFAULT_N_VALUES = [0, 1, 3] 
DEFAULT_K_RANGE = (1.0, 5.0)
DEFAULT_M_FIXED = 1.0
DEFAULT_R0_RANGE = (1.5, 3.5)
DEFAULT_V0_RANGE = (0.8, 2.5)
DEFAULT_PHI0_RANGE_DEG = (30.0, 150.0) 


FORCE_EPSILON = 1e-4 
SIM_DURATION = 10.0 
SIM_POINTS_PER_SEC = 60 
INITIAL_ANIM_DURATION = 1.0 
CHOICE_PATH_MAX_POINTS = 150
MAX_R_CUTOFF = 15.0 
MIN_R_CUTOFF = 0.05 
PARTICLE_RADIUS = 0.08
ARROW_SCALE = 0.5
CHOICE_SCALE = 0.4 
TRACE_STROKE_WIDTH = 2.0
CENTER_DOT_RADIUS = 0.05


def central_force_derivs(t, state, m, k, n):
    x, y, vx, vy = state
    r_sq = x**2 + y**2
    r = np.sqrt(r_sq)
    ax = 0; ay = 0
    if r < MIN_R_CUTOFF: return [vx, vy, 0, 0] 

    if k != 0: 
      
        if n == 0: force_mag = -k 
        elif n == 1: force_mag = -k / (r + FORCE_EPSILON)
        elif n == 2: force_mag = -k / (r_sq + FORCE_EPSILON) 
        elif n == 3: force_mag = -k / (r_sq * r + FORCE_EPSILON) 
        else: force_mag = -k / (np.power(r,n) + FORCE_EPSILON if r > 1e-3 else 1e6)
        if m > 1e-6 and r > 1e-6:
            common_factor = force_mag / (r * m)
            ax = common_factor * x
            ay = common_factor * y
    return [vx, vy, ax, ay]

def bounds_event(t, state, *args):
    x, y, vx, vy = state
    r = np.sqrt(x**2 + y**2)
    val_far = MAX_R_CUTOFF - r   
    val_close = r - MIN_R_CUTOFF   
    return min(val_far, val_close)
bounds_event.terminal = True
bounds_event.direction = -1

def simulate_central_force(max_time, initial_state, params):
    m, k, n = params
    t_span = [0, max_time]
    n_points = int(max_time * SIM_POINTS_PER_SEC) + 1
    t_eval = np.linspace(t_span[0], t_span[1], n_points)
    term_reason_str = 'unknown_error'
    coords_list = []
    time_pts = np.array([0])
    origin_pt = np.array([initial_state[0], initial_state[1], 0.0])
    if np.any(np.isnan(origin_pt)): origin_pt = ORIGIN

    try:
        sol = solve_ivp(
            central_force_derivs, t_span, initial_state, args=(m, k, n),
            t_eval=t_eval, events=bounds_event, dense_output=False, 
            rtol=1e-7, atol=1e-9 
        )

        coords_raw = sol.y[:2].T 
        valid_indices = [i for i, pt in enumerate(coords_raw) if not np.any(np.isnan(pt)) and not np.any(np.isinf(pt))]
        coords_list = [np.array([coords_raw[i][0], coords_raw[i][1], 0.0]) for i in valid_indices] 
        time_pts = sol.t[valid_indices] if len(valid_indices) > 0 else np.array([0.0])

        # Determine termination reason
        if sol.status == 1 and sol.t_events and sol.t_events[0].size > 0 : 
            term_reason_str = f'event (bounds @ {sol.t_events[0][0]:.2f}s)'
        elif sol.status == 0: 
             term_reason_str = 'completed_duration'
        elif sol.status < 0: 
            term_reason_str = f'integration_error (status {sol.status})'
        else: 
             term_reason_str = f'finished_unknown_status ({sol.status})'

    except Exception as e:
        print(f"  Warning: solve_ivp failed! Error: {e}")
        term_reason_str = f'exception_in_solve_ivp ({type(e).__name__})'
        return [origin_pt, origin_pt], [0,0], term_reason_str

    if len(coords_list) < 2: 
        return [origin_pt, origin_pt], [0,0], term_reason_str 

    return coords_list, time_pts, term_reason_str

def simulate_multiple_objects(max_time, initial_states, params_list):
    """Simulate multiple objects with central force motion."""
    results = []
    for i, (initial_state, params) in enumerate(zip(initial_states, params_list)):
        coords, time_pts, term_reason = simulate_central_force(max_time, initial_state, params)
        results.append((coords, time_pts, term_reason))
    return results


def create_trajectory_path(coords, color=WHITE, stroke_width=TRACE_STROKE_WIDTH):
    path = VMobject(color=color, stroke_width=stroke_width)
    if coords is not None and hasattr(coords, '__len__') and len(coords) >= 2:
        coords_array = np.array(coords)
        if not np.any(np.isnan(coords_array)) and not np.any(np.isinf(coords_array)):
            try:
                path.set_points_as_corners(coords_array)
            except Exception as e: 
                print(f"Warning: Error setting path points for VMobject: {e}")
    return path

def create_choice_content(coords, path_color=WHITE):
    center_dot = Dot(ORIGIN, radius=CENTER_DOT_RADIUS*0.8, color=GRAY) 
    path = create_trajectory_path(coords, color=path_color)
    content = VGroup(center_dot)
    if path.has_points():
         content.add(path)
    else:
         error_text = Text("Path Error", font_size=10, color=RED).move_to(ORIGIN + RIGHT * 0.5)
         content.add(error_text)
    return content

def create_multi_object_choice_content(coords_list, colors):
    """Create choice content with multiple object trajectories."""
    center_dot = Dot(ORIGIN, radius=CENTER_DOT_RADIUS*0.8, color=GRAY)
    content = VGroup(center_dot)
    
    for i, coords in enumerate(coords_list):
        if coords is not None and len(coords) >= 2:
            path = create_trajectory_path(coords, color=colors[i], stroke_width=TRACE_STROKE_WIDTH)
            if path.has_points():
                content.add(path)
    
    return content

def adjust_ranges(difficulty): 
    difficulty = max(0, min(difficulty, 9))
    diff_factor = difficulty / 9.0 

    k_mid = np.mean(DEFAULT_K_RANGE)
    k_width_factor = 0.3 + 0.7 * diff_factor 
    k_half_width = (DEFAULT_K_RANGE[1] - DEFAULT_K_RANGE[0]) / 2 * k_width_factor
    k_range = (max(0.1, k_mid - k_half_width), k_mid + k_half_width) 

    r0_mid = np.mean(DEFAULT_R0_RANGE)
    r0_width_factor = 0.4 + 0.6 * diff_factor
    r0_half_width = (DEFAULT_R0_RANGE[1] - DEFAULT_R0_RANGE[0]) / 2 * r0_width_factor
    r0_min_val = max(MIN_R_CUTOFF * 4, DEFAULT_R0_RANGE[0] * (1.0 - 0.5 * diff_factor) )
    r0_range = (r0_min_val, r0_mid + r0_half_width)

    v0_mid = np.mean(DEFAULT_V0_RANGE)
    v0_width_factor = 0.3 + 1.2 * diff_factor 
    v0_half_width = (DEFAULT_V0_RANGE[1] - DEFAULT_V0_RANGE[0]) / 2 * v0_width_factor
    v0_min_val = max(0.05, DEFAULT_V0_RANGE[0] * (1.0 - 0.7 * diff_factor))
    v0_range = (v0_min_val, v0_mid + v0_half_width)

    min_angle_deg = max(5.0, 75.0 - 70.0 * diff_factor) 
    max_angle_deg = min(175.0, 105.0 + 70.0 * diff_factor)
    if min_angle_deg >= max_angle_deg: 
        phi0_range_deg_calc = ( (min_angle_deg+max_angle_deg)/2 - 5, (min_angle_deg+max_angle_deg)/2 + 5)
    else:
        phi0_range_deg_calc = (min_angle_deg, max_angle_deg)

    n_values_options = DEFAULT_N_VALUES 


    print(f"Adjusted Ranges (Diff={difficulty}): K={k_range}, R0={r0_range}, V0={v0_range}, Phi0_deg={phi0_range_deg_calc}")
    return n_values_options, k_range, r0_range, v0_range, phi0_range_deg_calc


class CentralForceQuiz(Scene):
    def construct(self):
        seed = int(os.environ.get('MANIM_SEED', time.time()))
        difficulty = int(os.environ.get('MANIM_DIFFICULTY', 5))
        object_count = int(os.environ.get('MANIM_OBJECT_COUNT', 1))
        
        if object_count not in [1, 2, 3]:
            print(f"Warning: object_count {object_count} is not valid. Using default value 1.")
            object_count = 1
        
        random.seed(seed); np.random.seed(seed)

        n_options_for_correct, k_range, r0_range, v0_range, phi0_range_deg = adjust_ranges(difficulty)
        m = DEFAULT_M_FIXED

        objects_params = []
        objects_initial_states = []
        object_colors = [RED, GREEN, BLUE]
        
        for i in range(object_count):
            n_correct = random.choice(n_options_for_correct + [2])
            k = random.uniform(*k_range)
            r0 = random.uniform(*r0_range)
            v0_mag = random.uniform(*v0_range)
            phi0_deg_magnitude = random.uniform(*phi0_range_deg)
            phi0_rad_magnitude = np.radians(phi0_deg_magnitude)
            theta0_rad = random.uniform(0, 2 * PI) + i * PI/3  
            theta0_deg = np.degrees(theta0_rad)
            pos0 = np.array([r0 * np.cos(theta0_rad), r0 * np.sin(theta0_rad), 0])

            if r0 > 1e-6: 
                vel0_radial_dir = normalize(pos0) 
                vel0_tangential_dir = np.array([-vel0_radial_dir[1], vel0_radial_dir[0], 0]) 
            else: 
                vel0_radial_dir = RIGHT; vel0_tangential_dir = UP; pos0 = ORIGIN

            tangential_sign = random.choice([-1, 1]) 

            vel0_radial_comp = v0_mag * np.cos(phi0_rad_magnitude) * vel0_radial_dir
            vel0_tangential_comp = tangential_sign * v0_mag * np.sin(phi0_rad_magnitude) * vel0_tangential_dir
            vel0 = vel0_radial_comp + vel0_tangential_comp

            if np.any(np.isnan(vel0)):
                print("Warning: Initial velocity calculation resulted in NaN. Using default tangential velocity.")
                fallback_tangential_dir = UP if r0 <= 1e-6 else vel0_tangential_dir
                vel0 = v0_mag * fallback_tangential_dir * tangential_sign

            params_correct = [m, k, n_correct]
            initial_state_correct = [pos0[0], pos0[1], vel0[0], vel0[1]]

            if np.any(np.isnan(initial_state_correct)):
                err_msg = Text(f"Error: NaN in initial state for object {i+1}", color=RED, font_size=20).move_to(ORIGIN)
                self.play(Write(err_msg)); self.wait(3); print(f"FINAL_ANSWER: ERROR_NAN_INIT"); return

            objects_params.append(params_correct)
            objects_initial_states.append(initial_state_correct)
            
            print(f"Object {i+1}: n={n_correct}, k={k:.3f}, r0={r0:.2f}, v0={v0_mag:.2f}, phi0={phi0_deg_magnitude:.1f}deg")

        print(f"--- Quiz Setup (Seed: {seed}, Difficulty: {difficulty}, Objects: {object_count}) ---")

        title = Text(f"Problem: Central Force Motion ({object_count} Object{'s' if object_count > 1 else ''})", font_size=30).to_edge(UP)
        

        center_dot = Dot(ORIGIN, radius=CENTER_DOT_RADIUS, color=WHITE)
        
        particles = []
        arrows = []
        param_texts = []
        
        for i in range(object_count):
            pos0 = np.array([objects_initial_states[i][0], objects_initial_states[i][1], 0])
            vel0 = np.array([objects_initial_states[i][2], objects_initial_states[i][3], 0])
            

            particle = Dot(point=pos0, radius=PARTICLE_RADIUS, color=object_colors[i])
            particles.append(particle)
            

            if np.linalg.norm(vel0) > 1e-4:
                arrow = Arrow(pos0, pos0 + vel0 * ARROW_SCALE, buff=PARTICLE_RADIUS, 
                            color=object_colors[i], stroke_width=3, max_tip_length_to_length_ratio=0.25)
                arrows.append(arrow)
            

            m, k, n = objects_params[i]
            r0 = np.linalg.norm(pos0)
            v0_mag = np.linalg.norm(vel0)
            phi0_deg = np.degrees(np.arccos(np.clip(np.dot(vel0, pos0) / (r0 * v0_mag), -1, 1)))
            
            param_text = VGroup(
                MathTex(f"\\text{{Object {i+1}:}}", font_size=20, color=object_colors[i]),
                MathTex(f"n = {n}, k \\approx {k:.1f}", font_size=16),
                MathTex(f"r_0 \\approx {r0:.1f}, v_0 \\approx {v0_mag:.1f}", font_size=16),
                MathTex(f"\\phi_0 \\approx {phi0_deg:.0f}^\\circ", font_size=16)
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
            param_texts.append(param_text)
        

        if object_count == 1:
            param_group = param_texts[0].to_corner(UR)
        elif object_count == 2:
            param_group = VGroup(
                param_texts[0].to_corner(UR), 
                param_texts[1].to_corner(UL)  
            )
        else: 
            param_group = VGroup(
                param_texts[0].to_corner(UR),  
                param_texts[1].to_corner(DR),
                param_texts[2].to_corner(UL)  
            )
        
        initial_setup_group = VGroup(title, center_dot, param_group)
        for particle in particles:
            initial_setup_group.add(particle)
        for arrow in arrows:
            initial_setup_group.add(arrow)
            
        self.add(initial_setup_group); self.wait(0.5)

        sim_duration_calc = SIM_DURATION
        results_correct = simulate_multiple_objects(sim_duration_calc, objects_initial_states, objects_params)

        if not results_correct or any(len(coords) < 2 for coords, _, _ in results_correct):
            err_msg = Text("Sim Error (Correct Path)", color=RED).move_to(ORIGIN)
            self.play(FadeOut(VGroup(*self.mobjects)), FadeIn(err_msg)); self.wait(2); print(f"FINAL_ANSWER: ERROR_SIM_CORRECT"); return

        initial_anim_duration_actual = INITIAL_ANIM_DURATION
        fade_out_mobjects_after_anim = VGroup(initial_setup_group)
        

        anim_particles = []
        anim_trails = []
        
        for i, (coords, time_pts, _) in enumerate(results_correct):
            if len(coords) >= 2:
                anim_end_index = np.searchsorted(time_pts, initial_anim_duration_actual, side='right')
                if anim_end_index < 2: anim_end_index = min(2, len(coords))
                anim_coords = coords[:anim_end_index]
                
                if len(anim_coords) >= 2:
                    particle_anim = particles[i].copy()
                    trail_anim = TracedPath(particle_anim.get_center, stroke_color=object_colors[i], 
                                          stroke_width=TRACE_STROKE_WIDTH, dissipating_time=0.5, stroke_opacity=[0,1])
                    
                    anim_path_obj = create_trajectory_path(anim_coords, color=object_colors[i], stroke_width=TRACE_STROKE_WIDTH)
                    
                    if anim_path_obj.has_points():
                        self.add(trail_anim, particle_anim)
                        anim_particles.append(particle_anim)
                        anim_trails.append(trail_anim)
                        
                        actual_sim_time_for_anim = time_pts[anim_end_index-1] if anim_end_index > 0 and anim_end_index <= len(time_pts) else 0
                        anim_play_time = min(initial_anim_duration_actual, actual_sim_time_for_anim) if actual_sim_time_for_anim > 1e-6 else initial_anim_duration_actual
                        
                        try:
                            if anim_play_time > 1e-4:
                                self.play(MoveAlongPath(particle_anim, anim_path_obj), run_time=anim_play_time, rate_func=linear)
                        except Exception as e:
                            print(f"Error during MoveAlongPath for object {i+1}: {e}")
        

        fade_out_mobjects_after_anim.add(*anim_particles, *anim_trails)

        question = Text(f"Which path corresponds to this motion?", font_size=28).to_edge(DOWN, buff=MED_LARGE_BUFF)
        self.play(Write(question), run_time=0.75)
        self.wait(0.5)
        fade_out_mobjects_after_anim.add(question)
        

        distractor_results_list = [] 
        num_needed_distractors = 3
        

        def add_distractor(sim_func, sim_args, param_desc, current_dist_results_list):
            if len(current_dist_results_list) >= num_needed_distractors: return False
            if any(desc == param_desc for _, desc in current_dist_results_list): return False

            results_d = sim_func(*sim_args)
            print(f"  Distractor attempt '{param_desc}': {len(results_d)} objects")
        
            if all(len(coords) >= 2 for coords, _, _ in results_d):
                current_dist_results_list.append((results_d, param_desc))
                return True
            return False

        possible_n_dist = [n_val for n_val in (DEFAULT_N_VALUES + [2]) if n_val not in [params[2] for params in objects_params]]
        random.shuffle(possible_n_dist)
        for n_d in possible_n_dist:
            params_different_n = [[m, k, n_d] for m, k, _ in objects_params]
            if add_distractor(simulate_multiple_objects, (sim_duration_calc, objects_initial_states, params_different_n), f"n={n_d}", distractor_results_list):
                if len(distractor_results_list) >= num_needed_distractors: break

        if any(k != 0 and abs(k) > 1e-3 for _, k, _ in objects_params):
            params_zero_k = [[m, 0, n] for m, _, n in objects_params]
            add_distractor(simulate_multiple_objects, (sim_duration_calc, objects_initial_states, params_zero_k), "k=0", distractor_results_list)

        k_factors = [0.5, 1.5, 2.0]
        for k_factor in k_factors:
            params_k_factor = [[m, k * k_factor, n] for m, k, n in objects_params]
            add_distractor(simulate_multiple_objects, (sim_duration_calc, objects_initial_states, params_k_factor), f"k_factor={k_factor}", distractor_results_list)
            if len(distractor_results_list) >= num_needed_distractors: break


        v0_factors = [0.5, 1.5, 2.0]
        for v0_factor in v0_factors:
            new_initial_states = []
            for i, initial_state in enumerate(objects_initial_states):
                pos0 = np.array([initial_state[0], initial_state[1], 0])
                vel0 = np.array([initial_state[2], initial_state[3], 0])
                new_vel0 = vel0 * v0_factor
                new_initial_states.append([pos0[0], pos0[1], new_vel0[0], new_vel0[1]])
            
            add_distractor(simulate_multiple_objects, (sim_duration_calc, new_initial_states, objects_params), f"v0_factor={v0_factor}", distractor_results_list)
            if len(distractor_results_list) >= num_needed_distractors: break

        if len(distractor_results_list) < num_needed_distractors:
            err_msg = Text(f"Error: Only {len(distractor_results_list)}/{num_needed_distractors} distractors generated.", color=RED, font_size=20).move_to(ORIGIN)
            self.play(FadeOut(fade_out_mobjects_after_anim, shift=DOWN*0.5), FadeIn(err_msg)); self.wait(3)
            print(f"FINAL_ANSWER: ERROR_DISTRACTORS_LOW_COUNT"); return

        max_pts_for_choice = CHOICE_PATH_MAX_POINTS
        def get_choice_coords_segment(coords):
             if coords is None or len(coords) < 2: return None
             pts_to_show = min(len(coords), max_pts_for_choice)
             return coords[:pts_to_show] if pts_to_show >= 2 else None

        final_coords_correct_for_choice = [get_choice_coords_segment(coords) for coords, _, _ in results_correct]

        final_coords_distractors_for_choice = []
        for results_list, _ in distractor_results_list:
            distractor_coords = [get_choice_coords_segment(coords) for coords, _, _ in results_list]
            final_coords_distractors_for_choice.append(distractor_coords)

        all_choice_coords_for_display = [final_coords_correct_for_choice] + final_coords_distractors_for_choice
        path_colors = [YELLOW, BLUE, RED, GREEN, ORANGE, PURPLE] 

        choices_vgroups = []
        valid_choices_count = 0
        
        for i, current_choice_coords_list in enumerate(all_choice_coords_for_display):
            content_mobject = None
            if all(coords is not None and len(coords) >= 2 for coords in current_choice_coords_list):
                content_mobject = create_multi_object_choice_content(current_choice_coords_list, object_colors[:object_count])
                has_actual_paths = any(isinstance(mobj, VMobject) and mobj.has_points() and mobj is not content_mobject.submobjects[0] for mobj in content_mobject.submobjects)
                if has_actual_paths:
                    choices_vgroups.append(content_mobject)
                    valid_choices_count += 1
                else:
                    choices_vgroups.append(Text(f"Path Error {i}", font_size=15, color=RED))
            else:
                choices_vgroups.append(Text(f"Sim Error {i}", font_size=15, color=RED))

        if valid_choices_count < 4:
            err_msg = Text(f"Error: Only {valid_choices_count}/4 valid choice visuals generated.", color=RED, font_size=20).move_to(ORIGIN)
            self.play(FadeOut(fade_out_mobjects_after_anim, shift=DOWN*0.5), FadeIn(err_msg)); self.wait(3)
            print(f"FINAL_ANSWER: ERROR_CHOICES_VISUAL_LOW_COUNT"); return

        indexed_choices_to_shuffle = list(enumerate(choices_vgroups))
        random.shuffle(indexed_choices_to_shuffle)
        
        final_choice_display_group = VGroup()
        final_correct_answer_label_char = "ERROR"
        choice_labels_chars = ["A", "B", "C", "D", "E"]

        for i, (original_idx, choice_content_mobj) in enumerate(indexed_choices_to_shuffle):
            if i >= len(choice_labels_chars) - 1: break 

            label_char = choice_labels_chars[i]
            label_mobj = Tex(f"{label_char}", font_size=36)  
            
            if not isinstance(choice_content_mobj, Mobject):
                 choice_content_mobj = Text("Content Error", font_size=15, color=RED)
            
            choice_content_mobj.scale(CHOICE_SCALE)
            label_mobj.next_to(choice_content_mobj, DOWN, buff=SMALL_BUFF)
            
            single_choice_with_label_vg = VGroup(choice_content_mobj, label_mobj)
            final_choice_display_group.add(single_choice_with_label_vg)
            
            if original_idx == 0:
                final_correct_answer_label_char = label_char
        
        none_option = Text("None of the above", font_size=52, color=GRAY).scale(CHOICE_SCALE)
        none_label = Tex("E", font_size=36, color=GRAY)  
        none_label.next_to(none_option, DOWN, buff=SMALL_BUFF)
        none_choice_vg = VGroup(none_option, none_label)
        final_choice_display_group.add(none_choice_vg)
        
        if len(final_choice_display_group) == 5:  
             top_row = VGroup(*final_choice_display_group[:2]).arrange(RIGHT, buff=LARGE_BUFF*2.5)
             middle_row = VGroup(*final_choice_display_group[2:4]).arrange(RIGHT, buff=LARGE_BUFF*2.5)
             bottom_row = final_choice_display_group[4]
             

             top_row.move_to(ORIGIN + UP*2.0)
             middle_row.move_to(ORIGIN)  
             bottom_row.move_to(ORIGIN + DOWN*1.5)
             
             final_choice_display_group = VGroup(top_row, middle_row, bottom_row)
             
             self.play(FadeOut(fade_out_mobjects_after_anim, shift=DOWN*0.5), run_time=0.75)
             self.play(FadeIn(final_choice_display_group, shift=UP*0.5), run_time=0.75)
        else:
             err_msg = Text("Error arranging choices.", color=RED).move_to(ORIGIN)
             self.play(FadeOut(fade_out_mobjects_after_anim, shift=DOWN*0.5), Write(err_msg)); self.wait(2)
             print(f"FINAL_ANSWER: ERROR_CHOICES_ARRANGE_FINAL"); return

        if final_correct_answer_label_char != "ERROR":
             print(f"FINAL_ANSWER: {final_correct_answer_label_char}")
        else:
             print("Error: Could not determine correct answer label after shuffling.")
             print(f"FINAL_ANSWER: ERROR_LABEL_INDEXING")
        self.wait(1)
