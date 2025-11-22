
from manim import *
import numpy as np
import random
from scipy.integrate import solve_ivp
import time
import os
import shutil
from pathlib import Path

# Setup directories
Path("questions").mkdir(exist_ok=True)
Path("solutions").mkdir(exist_ok=True)
Path("question_text").mkdir(exist_ok=True)
Path("reasoning_traces").mkdir(exist_ok=True)

DEFAULT_M_RANGE = (0.5, 2.0)
DEFAULT_K_RANGE = (2.0, 10.0)
DEFAULT_B_RANGE = (0.1, 2.0)  
DEFAULT_V0_RANGE = (-3.0, 3.0)
DEFAULT_X0_RANGE = (-1.0, 1.0)

SIM_DURATION = 8.0
SIM_POINTS_PER_SEC = 50
PREVIEW_ANIM_DURATION = 2.0
GRAPH_X_RANGE = (0, SIM_DURATION)

SPRING_BUMPS = 8
SPRING_RADIUS = 0.12
MASS_SIDE_LENGTH = 0.6
WALL_WIDTH = 0.3
LABEL_SCALE_FACTOR = 0.7
CHOICE_SCALE = 0.4
TRACE_STROKE_WIDTH = 2.5


def oscillator_derivs(t, state, m, k, b):
    """ODE derivatives for x and v: [dx/dt, dv/dt]"""
    x, v = state
    ax = (-k * x - b * v) / m if m > 1e-6 else 0
    return [v, ax]

def simulate_oscillator(t_duration, initial_state, params):
    """Solves the ODE using solve_ivp. Returns t, x, v arrays."""
    try:
        m, k, b = params
        t_span = [0, t_duration]
        n_points = max(10, int(t_duration * SIM_POINTS_PER_SEC) + 1)
        t_eval = np.linspace(t_span[0], t_span[1], n_points)

        sol = solve_ivp(
            oscillator_derivs,
            t_span,
            initial_state,
            args=(m, k, b),
            t_eval=t_eval,
            method='RK45',
            rtol=1e-6,
            atol=1e-8
        )
        if not sol.success:
            print(f"Warning: ODE solver failed! Status: {sol.status}, Message: {sol.message}")
            return np.array([0.0, 1e-3]), np.array([initial_state[0]]*2), np.array([initial_state[1]]*2)
        if len(sol.t) < 2:
            print("Warning: Simulation returned < 2 time points.")
            return np.array([0.0, 1e-3]), np.array([initial_state[0]]*2), np.array([initial_state[1]]*2)
        return sol.t, sol.y[0], sol.y[1]
    except Exception as e:
        print(f"Error during solve_ivp: {e}")
        return np.array([0.0, 1e-3]), np.array([initial_state[0]]*2), np.array([initial_state[1]]*2)

def simulate_multiple_oscillators(t_duration, initial_states, params_list):
    """Simulate multiple oscillators simultaneously."""
    results = []
    for i, (initial_state, params) in enumerate(zip(initial_states, params_list)):
        t, x, v = simulate_oscillator(t_duration, initial_state, params)
        results.append((t, x, v))
    return results

def create_spring(start_point, end_point, bumps=SPRING_BUMPS, radius=SPRING_RADIUS, color=WHITE, stroke_width=2):
    """Creates a spring visual representation."""
    try:
        start = np.array(start_point)
        end = np.array(end_point)
        length = np.linalg.norm(end - start)
        
        if length < 1e-6:
            return Line(start, end, color=color, stroke_width=stroke_width)
        
        direction = normalize(end - start)
        if abs(direction[0]) < 1e-6 and abs(direction[1]) < 1e-6:
            perp_direction = RIGHT
        elif abs(direction[1]) > 0.999:
            perp_direction = RIGHT
        else:
            perp_direction = normalize(np.cross(direction, OUT))
        
        def spring_func(t):
                return start + t * (end - start) + perp_direction * radius * np.sin(bumps * TAU * t)
            
        num_segments = bumps * 8
        points = [spring_func(t) for t in np.linspace(0, 1, num_segments + 1)]
        points = [p for p in points if np.all(np.isfinite(p))]
        if len(points) < 2:
            return Line(start, end, color=color, stroke_width=stroke_width)
        return VMobject(color=color, stroke_width=stroke_width).set_points_as_corners(points)
    except Exception as e:
        print(f"Error creating spring: {e}")
        return None

def create_oscillator_setup(y_positions, colors):
    """Creates the visual setup for multiple oscillators in vertical arrangement."""
    setup_group = VGroup()

    walls = []
    springs = []
    masses = []
    
    for i, (y_pos, color) in enumerate(zip(y_positions, colors)):
        wall_x = -3.0
        wall = Rectangle(width=WALL_WIDTH, height=1.0, color=GRAY_C, fill_opacity=0.8)
        wall.move_to([wall_x, y_pos, 0])
        walls.append(wall)

        mass = Square(side_length=MASS_SIDE_LENGTH, color=color, fill_opacity=0.8)
        mass.move_to([0, y_pos, 0])
        masses.append(mass)

        spring = create_spring(wall.get_right(), mass.get_left(), color=WHITE)
        springs.append(spring)
    
    setup_group.add(*walls, *masses, *springs)
    return setup_group, walls, masses, springs

def create_position_graph(t_pts, x_pts, x_range, y_range, graph_color=WHITE):
    """Creates an Axes object with the plotted x(t) data."""
    if len(t_pts) < 2 or len(x_pts) < 2:
        print("ERROR: Invalid data for graph")
        axes = Axes(x_range=x_range, y_range=[-1, 1], x_length=4.0, y_length=2.5, axis_config={"include_tip": False})
        labels = axes.get_axis_labels(x_label=MathTex("t", font_size=20), y_label=MathTex("x", font_size=20))
        return VGroup(axes, labels)

    y_min_eff, y_max_eff = y_range
    y_min_eff = max(y_min_eff, -3.5)
    y_max_eff = min(y_max_eff, 3.5)
    
    if y_min_eff >= y_max_eff:
        y_min_eff, y_max_eff = -2.0, 2.0
    
    y_step = abs(y_max_eff - y_min_eff) / 4.0
    if y_step < 1e-3: y_step = 0.5
    x_step = abs(x_range[1] - x_range[0]) / 4.0
    if x_step < 1e-3: x_step = 1.0
    
    try:
        axes = Axes(
            x_range=[x_range[0], x_range[1], x_step],
            y_range=[y_min_eff, y_max_eff, y_step],
            x_length=4.0,
            y_length=2.5,
            axis_config={"include_numbers": True, "include_tip": False, "stroke_width": 1.5, "decimal_number_config": {"num_decimal_places": 1}},
            tips=False
        )
        labels = axes.get_axis_labels(x_label=MathTex("t", font_size=20), y_label=MathTex("x", font_size=20))
        
        valid_indices = np.isfinite(t_pts) & np.isfinite(x_pts)
        if not np.all(valid_indices):
            t_pts_clean = t_pts[valid_indices]
            x_pts_clean = x_pts[valid_indices]
        else:
            t_pts_clean = t_pts
            x_pts_clean = x_pts

        x_pts_clean = np.clip(x_pts_clean, y_min_eff, y_max_eff)
        
        if len(t_pts_clean) < 2:
            graph_group = VGroup(axes, labels)
        else:
            graph = axes.plot_line_graph(x_values=t_pts_clean, y_values=x_pts_clean, line_color=graph_color, add_vertex_dots=False, stroke_width=TRACE_STROKE_WIDTH)
            graph_group = VGroup(axes, graph["line_graph"], labels)
    
    except Exception as e:
        print(f"ERROR creating graph object: {e}")
        axes = Axes(x_range=x_range, y_range=[y_min_eff, y_max_eff, y_step], x_length=4.0, y_length=2.5, axis_config={"include_tip": False})
        labels = axes.get_axis_labels(x_label=MathTex("t", font_size=20), y_label=MathTex("x", font_size=20))
        graph_group = VGroup(axes, labels)
    
    return graph_group

def create_multi_oscillator_graph(t_pts_list, x_pts_list, x_range, y_range, colors):
    """Creates a graph with multiple position curves for different oscillators."""
    if not t_pts_list or not x_pts_list or len(t_pts_list) != len(x_pts_list):
        print("ERROR: Invalid data for multi-oscillator graph")
        return create_position_graph([0, 1], [0, 0], x_range, y_range)

    y_min_eff, y_max_eff = y_range
    y_min_eff = max(y_min_eff, -3.5)
    y_max_eff = min(y_max_eff, 3.5)
    
    if y_min_eff >= y_max_eff:
        y_min_eff, y_max_eff = -2.0, 2.0
    
    y_step = abs(y_max_eff - y_min_eff) / 4.0
    if y_step < 1e-3: y_step = 0.5
    x_step = abs(x_range[1] - x_range[0]) / 4.0
    if x_step < 1e-3: x_step = 1.0
    
    axes = Axes(
        x_range=[x_range[0], x_range[1], x_step],
        y_range=[y_min_eff, y_max_eff, y_step],
        x_length=4.0,
        y_length=2.5,
        axis_config={"include_numbers": True, "include_tip": False, "stroke_width": 1.5, "decimal_number_config": {"num_decimal_places": 1}},
        tips=False
    )
    labels = axes.get_axis_labels(x_label=MathTex("t", font_size=20), y_label=MathTex("x", font_size=20))
    
    graph_group = VGroup(axes, labels)
    
    for i, (t_pts, x_pts) in enumerate(zip(t_pts_list, x_pts_list)):
        if len(t_pts) >= 2 and len(x_pts) >= 2:
            try:
                x_pts_clipped = np.clip(x_pts, y_min_eff, y_max_eff)
                
                graph = axes.plot_line_graph(
                    x_values=t_pts,
                    y_values=x_pts_clipped,
                    line_color=colors[i],
                    add_vertex_dots=False,
                    stroke_width=TRACE_STROKE_WIDTH
                )
                graph_group.add(graph["line_graph"])
            except Exception as e:
                print(f"Error plotting oscillator {i}: {e}")
    
    return graph_group

def adjust_oscillator_parameters(difficulty):
    """Adjusts parameter ranges based on difficulty level (0-9)."""
    difficulty = max(0, min(difficulty, 9))
    diff_factor = difficulty / 9.0

    m_min = DEFAULT_M_RANGE[0]
    m_max = DEFAULT_M_RANGE[0] + (DEFAULT_M_RANGE[1] - DEFAULT_M_RANGE[0]) * (0.6 + 0.4 * diff_factor)
    adj_m_range = (m_min, m_max)

    k_min = DEFAULT_K_RANGE[0]
    k_max = DEFAULT_K_RANGE[0] + (DEFAULT_K_RANGE[1] - DEFAULT_K_RANGE[0]) * (0.5 + 0.5 * diff_factor)
    adj_k_range = (k_min, k_max)

    b_min = DEFAULT_B_RANGE[0]
    b_max = DEFAULT_B_RANGE[0] + (DEFAULT_B_RANGE[1] - DEFAULT_B_RANGE[0]) * (0.3 + 0.7 * diff_factor)
    adj_b_range = (b_min, b_max)

    v0_min = DEFAULT_V0_RANGE[0]
    v0_max = DEFAULT_V0_RANGE[0] + (DEFAULT_V0_RANGE[1] - DEFAULT_V0_RANGE[0]) * (0.7 + 0.3 * diff_factor)
    adj_v0_range = (v0_min, v0_max)
    
    print(f"Adjusted Ranges (Diff={difficulty}):")
    print(f"  Mass: {adj_m_range}")
    print(f"  Spring constant: {adj_k_range}")
    print(f"  Damping: {adj_b_range}")
    print(f"  Initial velocity: {adj_v0_range}")
    
    return adj_m_range, adj_k_range, adj_b_range, adj_v0_range

class DampedOscillatorQuizRefactored(Scene):
    anim_time = 0.0

    def construct(self):
        seed = int(os.environ.get('MANIM_SEED', time.time()))
        difficulty = int(os.environ.get('MANIM_DIFFICULTY', 5))
        object_count = int(os.environ.get('MANIM_OBJECT_COUNT', 1))

        if object_count not in [1, 2, 3]:
            print(f"Warning: object_count {object_count} is not valid. Using default value 1.")
            object_count = 1

        random.seed(seed)
        np.random.seed(seed)

        # Store for reasoning trace
        self.seed = seed
        self.difficulty = difficulty
        self.object_count = object_count
        self.oscillator_events = []
        

        adj_m_range, adj_k_range, adj_b_range, adj_v0_range = adjust_oscillator_parameters(difficulty)

        oscillators_params = []
        oscillators_initial_states = []
        object_colors = [RED, GREEN, BLUE]
        
        base_positions = [2.0, 0.0, -2.0]  
        
        for i in range(object_count):
            m = random.uniform(*adj_m_range)
            k = random.uniform(*adj_k_range)
            b = random.uniform(*adj_b_range)

            x0 = random.uniform(*DEFAULT_X0_RANGE)
            v0 = random.uniform(*adj_v0_range)

            oscillators_params.append([m, k, b])
            oscillators_initial_states.append([x0, v0])

            # Record oscillator event for reasoning trace
            self.oscillator_events.append({
                "oscillator_id": i + 1,
                "mass": m,
                "spring_constant": k,
                "damping_coefficient": b,
                "initial_position": x0,
                "initial_velocity": v0,
                "color": object_colors[i].name if hasattr(object_colors[i], "name") else str(object_colors[i])
            })

            print(f"Oscillator {i+1}: m={m:.2f}, k={k:.2f}, b={b:.2f}, x0={x0:.2f}, v0={v0:.2f}")
        
        print(f"--- Damped Oscillator Setup (Seed: {seed}, Difficulty: {difficulty}, Objects: {object_count}) ---")
        print(f"--------------------")

        title = Text(f"Problem: {object_count} Damped Oscillator(s)", font_size=32).to_edge(UP)

        y_positions = [base_positions[i] for i in range(object_count)]
        colors = object_colors[:object_count]
        setup_group, walls, masses, springs = create_oscillator_setup(y_positions, colors)

        param_texts = []
        for i in range(object_count):
            m, k, b = oscillators_params[i]
            x0, v0 = oscillators_initial_states[i]
            param_text = VGroup(
                MathTex(f"\\text{{Oscillator {i+1}:}}", font_size=24, color=colors[i]),
                MathTex(f"m = {m:.2f}\\, \\text{{kg}}", font_size=20),
                MathTex(f"k = {k:.2f}\\, \\text{{N/m}}", font_size=20),
                MathTex(f"b = {b:.2f}\\, \\text{{N s/m}}", font_size=20),
                MathTex(f"x_0 = {x0:.2f}\\, \\text{{m}}", font_size=20),
                MathTex(f"v_0 = {v0:.2f}\\, \\text{{m/s}}", font_size=20)
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
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
        
        initial_static_group = VGroup(title, setup_group, param_group)
        self.add(initial_static_group)
        self.wait(1)

        sim_duration_total = SIM_DURATION
        print("Calculating Correct Trajectories for Animation...")

        results_correct = simulate_multiple_oscillators(sim_duration_total, oscillators_initial_states, oscillators_params)

        if not results_correct or any(len(t) < 2 for t, x, v in results_correct):
            print("ERROR: Simulation failed to produce sufficient points.")
            self.play(Write(Text("Simulation Error!", color=RED).move_to(ORIGIN)))
            self.wait(2)
            return

        self.anim_time = 0.0
        mass_anims = [mass.copy() for mass in masses]

        def multi_oscillator_updater(mobj, dt, osc_index):
            try:
                self.anim_time += dt
                t_correct, x_correct, v_correct = results_correct[osc_index]
                current_x = np.interp(self.anim_time, t_correct, x_correct)

                x_min = -3.5
                x_max = 3.5
                current_x = np.clip(current_x, x_min, x_max)
                
                mobj.move_to([current_x, y_positions[osc_index], 0])  
            except Exception as e:
                print(f"Error in mass updater {osc_index}: {e}")

        for i, mass_anim in enumerate(mass_anims):
            mass_anim.add_updater(lambda mobj, dt, idx=i: multi_oscillator_updater(mobj, dt, idx))
            self.add(mass_anim)

        spring_anims = [spring.copy() for spring in springs]
        for i, spring_anim in enumerate(spring_anims):
            def spring_updater(mobj, dt, idx=i):
                try:
                    t_correct, x_correct, v_correct = results_correct[idx]
                    current_x = np.interp(self.anim_time, t_correct, x_correct)
                    
                    x_min = -3.5
                    x_max = 3.5
                    current_x = np.clip(current_x, x_min, x_max)
                    
                    mass_pos = [current_x, y_positions[idx], 0]
                    wall_pos = [-3.0, y_positions[idx], 0]
                    spring_end_x = current_x - MASS_SIDE_LENGTH/2
                    spring_end_pos = [spring_end_x, y_positions[idx], 0]
                    new_spring = create_spring(wall_pos, spring_end_pos, color=WHITE)
                    if new_spring is not None:
                        mobj.become(new_spring)
                except Exception as e:
                    print(f"Error in spring updater {idx}: {e}")
            
            spring_anim.add_updater(spring_updater)
            self.add(spring_anim)

        for mass in masses:
            self.remove(mass)
        for spring in springs:
            self.remove(spring)
        
        print(f"  Playing {PREVIEW_ANIM_DURATION:.1f}s preview animation...")
        self.wait(PREVIEW_ANIM_DURATION)

        for mass_anim in mass_anims:
            mass_anim.clear_updaters()
        for spring_anim in spring_anims:
            spring_anim.clear_updaters()

        for spring in springs:
            self.remove(spring)
        for mass in masses:
            self.remove(mass)

        self.wait(0.5)

        question = Text("Which graph best represents the positions x(t)?", font_size=32).to_edge(DOWN, buff=MED_SMALL_BUFF)
        self.play(Write(question))
        self.wait(0.5)

        fade_out_group = VGroup(title, param_group, question, *walls, *mass_anims, *spring_anims)

        print("Calculating Distractor Trajectories...")

        params_no_damping = []
        for params in oscillators_params:
            m, k, b = params
            params_no_damping.append([m, k, 0.0])
        results_no_damping = simulate_multiple_oscillators(sim_duration_total, oscillators_initial_states, params_no_damping)

        params_strong_damping = []
        for params in oscillators_params:
            m, k, b = params
            params_strong_damping.append([m, k, b * 3.0])
        results_strong_damping = simulate_multiple_oscillators(sim_duration_total, oscillators_initial_states, params_strong_damping)

        params_different_k = []
        for i, params in enumerate(oscillators_params):
            m, k, b = params
            k_new = k * (0.5 + 0.5 * i) 
            params_different_k.append([m, k_new, b])
        results_different_k = simulate_multiple_oscillators(sim_duration_total, oscillators_initial_states, params_different_k)

        params_different_m = []
        for i, params in enumerate(oscillators_params):
            m, k, b = params
            m_new = m * (0.7 + 0.6 * i) 
            params_different_m.append([m_new, k, b])
        results_different_m = simulate_multiple_oscillators(sim_duration_total, oscillators_initial_states, params_different_m)

        params_no_spring = []
        for params in oscillators_params:
            m, k, b = params
            params_no_spring.append([m, 0.0, b])
        results_no_spring = simulate_multiple_oscillators(sim_duration_total, oscillators_initial_states, params_no_spring)

        all_xs = []
        all_results = [results_correct, results_no_damping, results_strong_damping, results_different_k, results_different_m, results_no_spring]
        
        for results in all_results:
            for t, x, v in results:
                if len(x) > 0:
                    all_xs.extend(x)
        
        if not all_xs:
            print("ERROR: No valid position data from simulations for graphing.")
            self.play(Write(Text("Graphing Error!", color=RED).move_to(ORIGIN)))
            self.wait(2)
            return
        
        min_x_actual = min(all_xs)
        max_x_actual = max(all_xs)

        y_buffer = abs(max_x_actual - min_x_actual) * 0.1 + 0.2
        y_min_graph = min_x_actual - y_buffer
        y_max_graph = max_x_actual + y_buffer
        if y_min_graph >= y_max_graph:
            y_min_graph -= 0.5
            y_max_graph += 0.5
        
        graph_y_range = [y_min_graph, y_max_graph]
        print(f"Graph Y Range (Position): [{graph_y_range[0]:.2f}, {graph_y_range[1]:.2f}]")

        def extract_data(results):
            t_list = [t for t, x, v in results]
            x_list = [x for t, x, v in results]
            return t_list, x_list
        
        t_correct_list, x_correct_list = extract_data(results_correct)
        t_no_damping_list, x_no_damping_list = extract_data(results_no_damping)
        t_strong_damping_list, x_strong_damping_list = extract_data(results_strong_damping)
        t_different_k_list, x_different_k_list = extract_data(results_different_k)
        t_different_m_list, x_different_m_list = extract_data(results_different_m)
        t_no_spring_list, x_no_spring_list = extract_data(results_no_spring)

        graph_correct = create_multi_oscillator_graph(t_correct_list, x_correct_list, GRAPH_X_RANGE, graph_y_range, colors[:object_count])
        graph_no_damping = create_multi_oscillator_graph(t_no_damping_list, x_no_damping_list, GRAPH_X_RANGE, graph_y_range, colors[:object_count])
        graph_strong_damping = create_multi_oscillator_graph(t_strong_damping_list, x_strong_damping_list, GRAPH_X_RANGE, graph_y_range, colors[:object_count])
        graph_different_k = create_multi_oscillator_graph(t_different_k_list, x_different_k_list, GRAPH_X_RANGE, graph_y_range, colors[:object_count])
        graph_different_m = create_multi_oscillator_graph(t_different_m_list, x_different_m_list, GRAPH_X_RANGE, graph_y_range, colors[:object_count])
        graph_no_spring = create_multi_oscillator_graph(t_no_spring_list, x_no_spring_list, GRAPH_X_RANGE, graph_y_range, colors[:object_count])

        all_choices = [graph_correct, graph_no_damping, graph_strong_damping, graph_different_k, graph_different_m, graph_no_spring]
        choice_descriptions = [
            "Correct (All forces)",
            "No Damping",
            "Strong Damping",
            "Different Spring Constants",
            "Different Masses",
            "No Spring Force"
        ]

        correct_index = 0
        other_indices = list(range(1, len(all_choices)))
        random.shuffle(other_indices)
        selected_indices = [correct_index] + other_indices[:3]
        
        choices = [all_choices[i] for i in selected_indices]
        selected_descriptions = [choice_descriptions[i] for i in selected_indices]

        indexed_choices = list(enumerate(choices))
        random.shuffle(indexed_choices)
        choice_labels_text = ["A", "B", "C", "D", "E"]
        choice_group = VGroup()
        new_correct_index = -1
        
        for i, (original_index, graph_vgroup) in enumerate(indexed_choices):
            if graph_vgroup is None or not isinstance(graph_vgroup, VGroup):
                print(f"ERROR: Choice {i} received invalid graph object. Creating placeholder.")
                graph_vgroup = Text("Graph Error", font_size=16, color=RED).scale(CHOICE_SCALE)
            
            graph_vgroup.scale(CHOICE_SCALE)
            label = Tex(f"{choice_labels_text[i]}", font_size=36).next_to(graph_vgroup, DOWN, buff=SMALL_BUFF * 0.8)
            choice_vg = VGroup(graph_vgroup, label)
            choice_group.add(choice_vg)
            
            if original_index == 0:  
                new_correct_index = i
            print(f"Choice {choice_labels_text[i]}: {selected_descriptions[original_index]}")

        none_option = Text("None of the above", font_size=52, color=GRAY).scale(CHOICE_SCALE)
        none_label = Tex("E", font_size=36, color=GRAY)  
        none_label.next_to(none_option, DOWN, buff=SMALL_BUFF)
        none_choice_vg = VGroup(none_option, none_label)
        choice_group.add(none_choice_vg)

        if len(choice_group) == 5:
            top_row = VGroup(choice_group[0], choice_group[1]).arrange(RIGHT, buff=LARGE_BUFF*2.5)
            middle_row = VGroup(choice_group[2], choice_group[3]).arrange(RIGHT, buff=LARGE_BUFF*2.5)
            bottom_row = choice_group[4] 

            top_row.move_to(UP*2.0)
            middle_row.move_to(ORIGIN)
            bottom_row.move_to(DOWN*1.5)
            
            final_choice_display_group = VGroup(top_row, middle_row, bottom_row)
            
            self.play(FadeOut(fade_out_group))
            self.play(FadeIn(final_choice_display_group))
        else:
            print("ERROR: Could not generate all 5 choice VGroups.")
            self.play(FadeOut(fade_out_group), Write(Text("Error generating choices.", color=RED)))
            self.wait(2)
            return

        correct_label = "ERROR"
        if new_correct_index != -1 and new_correct_index < len(choice_labels_text):
            correct_label = choice_labels_text[new_correct_index]
            print(f"\nFINAL_ANSWER: {correct_label}")
        else:
            print("\nError: Could not determine correct answer index after shuffling.")
            print(f"FINAL_ANSWER: ERROR_INDEX")
            error_text_reveal = Text("Internal Error finding correct answer!", color=RED, font_size=24).to_edge(BOTTOM)
            self.play(Write(error_text_reveal))

        self.wait(5)

        # Store answer and additional info for reasoning trace
        self.correct_label = correct_label
        self.selected_descriptions = selected_descriptions
        self.selected_indices = selected_indices
        self.results_correct = results_correct

        # Save solution
        with open(f"solutions/damped_shm_quiz_d{self.difficulty}_n{self.object_count}_seed{self.seed}.txt", "w") as f:
            f.write(correct_label)

        # Save question text
        question_text_content = "Which graph best represents the positions x(t)?\nReturn the answer as a single letter (A, B, C, D, or E)."
        with open(f"question_text/damped_shm_quiz_d{self.difficulty}_n{self.object_count}_seed{self.seed}.txt", "w") as f:
            f.write(question_text_content)

        # Generate and save reasoning trace
        reasoning_trace = self.generate_reasoning_trace()
        with open(f"reasoning_traces/damped_shm_quiz_d{self.difficulty}_n{self.object_count}_seed{self.seed}.txt", "w") as f:
            f.write(reasoning_trace)

    def generate_reasoning_trace(self):
        """Generate a detailed reasoning trace describing the damped oscillator scene chronologically"""

        trace = []
        trace.append("=== DAMPED HARMONIC OSCILLATOR PROBLEM ===\n")

        trace.append(f"Difficulty Level: {self.difficulty}")
        trace.append(f"Number of Oscillators: {self.object_count}\n")

        trace.append("\n=== INITIAL SETUP ===\n")

        # Describe each oscillator's parameters
        for event in self.oscillator_events:
            trace.append(f"Oscillator {event['oscillator_id']} ({event['color']} color):")
            trace.append(f"  - Mass (m): {event['mass']:.2f} kg")
            trace.append(f"  - Spring Constant (k): {event['spring_constant']:.2f} N/m")
            trace.append(f"  - Damping Coefficient (b): {event['damping_coefficient']:.2f} N·s/m")
            trace.append(f"  - Initial Position (x₀): {event['initial_position']:.2f} m")
            trace.append(f"  - Initial Velocity (v₀): {event['initial_velocity']:.2f} m/s")
            trace.append("")

        trace.append("\n=== VIDEO CHRONOLOGY ===\n")

        trace.append("Phase 1: Parameter Display (0.0s - 1.0s)")
        trace.append("  - The video begins by showing the oscillator setup")
        trace.append("  - Physical parameters are displayed for each oscillator")
        trace.append(f"  - {self.object_count} mass{'es' if self.object_count > 1 else ''} attached to spring{'s' if self.object_count > 1 else ''} and wall{'s' if self.object_count > 1 else ''}")
        trace.append("")

        trace.append(f"Phase 2: Oscillation Animation (1.0s - {1.0 + PREVIEW_ANIM_DURATION:.1f}s)")
        trace.append(f"  - The oscillator{'s' if self.object_count > 1 else ''} begin{'s' if self.object_count == 1 else ''} to move")
        for i, event in enumerate(self.oscillator_events):
            if event['initial_velocity'] > 0:
                direction = "positive (right)"
            elif event['initial_velocity'] < 0:
                direction = "negative (left)"
            else:
                direction = "rest"
            trace.append(f"  - Oscillator {event['oscillator_id']}: Starts at x={event['initial_position']:.2f}m with velocity in {direction} direction")

        trace.append("  - The spring force pulls the mass back toward equilibrium (x=0)")
        trace.append("  - The damping force opposes motion, gradually reducing amplitude")
        trace.append("  - The mass oscillates with decreasing amplitude over time")
        trace.append("")

        trace.append(f"Phase 3: Question Display ({1.0 + PREVIEW_ANIM_DURATION:.1f}s - end)")
        trace.append("  - The animation pauses and the question appears")
        trace.append("  - Multiple choice graphs (A, B, C, D, E) are displayed")
        trace.append("  - Each graph shows a different position vs. time trajectory")
        trace.append("")

        trace.append("\n=== PHYSICAL ANALYSIS ===\n")

        trace.append("The motion of a damped harmonic oscillator follows the differential equation:")
        trace.append("  m·(d²x/dt²) + b·(dx/dt) + k·x = 0\n")

        trace.append("Key physical behaviors:")
        trace.append("1. Spring Force: F_spring = -k·x (restoring force proportional to displacement)")
        trace.append("2. Damping Force: F_damping = -b·v (opposes motion, proportional to velocity)")
        trace.append("3. Net acceleration: a = (-k·x - b·v) / m\n")

        for i, event in enumerate(self.oscillator_events):
            m = event['mass']
            k = event['spring_constant']
            b = event['damping_coefficient']

            # Calculate damping characteristics
            critical_damping = 2 * np.sqrt(k * m)
            damping_ratio = b / critical_damping
            natural_frequency = np.sqrt(k / m)

            trace.append(f"Oscillator {event['oscillator_id']} Analysis:")
            trace.append(f"  - Natural frequency: ω₀ = √(k/m) = {natural_frequency:.2f} rad/s")
            trace.append(f"  - Critical damping coefficient: b_crit = 2√(km) = {critical_damping:.2f} N·s/m")
            trace.append(f"  - Damping ratio: ζ = b/b_crit = {damping_ratio:.2f}")

            if damping_ratio < 1:
                trace.append(f"  - System is UNDERDAMPED: oscillates with exponentially decaying amplitude")
                damped_frequency = natural_frequency * np.sqrt(1 - damping_ratio**2)
                trace.append(f"  - Damped frequency: ω_d = {damped_frequency:.2f} rad/s")
            elif damping_ratio == 1:
                trace.append(f"  - System is CRITICALLY DAMPED: returns to equilibrium without oscillating")
            else:
                trace.append(f"  - System is OVERDAMPED: slowly returns to equilibrium without oscillating")
            trace.append("")

        trace.append("\n=== EVALUATING THE CHOICES ===\n")

        trace.append("The question asks which graph best represents x(t). Let me analyze each choice:\n")

        choice_labels = ["A", "B", "C", "D", "E"]
        for i, desc in enumerate(self.selected_descriptions):
            trace.append(f"Choice {choice_labels[i]}: {desc}")

            if desc == "Correct (All forces)":
                trace.append("  - Includes both spring force (-kx) and damping force (-bv)")
                trace.append("  - Shows oscillation with exponentially decaying amplitude")
                trace.append("  - Amplitude decreases smoothly over time")
                trace.append("  - Frequency remains constant (damped frequency)")
                trace.append("  ✓ This matches the physics of a damped harmonic oscillator")
            elif desc == "No Damping":
                trace.append("  - Only includes spring force, ignores damping (b=0)")
                trace.append("  - Shows perfect sinusoidal oscillation with constant amplitude")
                trace.append("  - Amplitude never decreases")
                trace.append("  ✗ Incorrect: Real system has damping that reduces amplitude")
            elif desc == "Strong Damping":
                trace.append("  - Uses damping coefficient 3× larger than actual value")
                trace.append("  - Amplitude decays much faster than the actual system")
                trace.append("  - May approach critical or overdamping")
                trace.append("  ✗ Incorrect: Damping is too strong compared to actual parameters")
            elif desc == "Different Spring Constants":
                trace.append("  - Uses modified spring constants (incorrect k values)")
                trace.append("  - Changes the natural frequency of oscillation")
                trace.append("  - Period and frequency don't match the actual system")
                trace.append("  ✗ Incorrect: Spring constant is different from given value")
            elif desc == "Different Masses":
                trace.append("  - Uses modified masses (incorrect m values)")
                trace.append("  - Changes both frequency and damping behavior")
                trace.append("  - System dynamics don't match actual parameters")
                trace.append("  ✗ Incorrect: Mass is different from given value")
            elif desc == "No Spring Force":
                trace.append("  - Ignores spring force (k=0)")
                trace.append("  - No restoring force, so no oscillation occurs")
                trace.append("  - Position would decay to zero without oscillating")
                trace.append("  ✗ Incorrect: Actual system has spring force causing oscillation")
            trace.append("")

        trace.append("\n=== REASONING PROCESS ===\n")

        trace.append("Step 1: Identify the physical system")
        trace.append(f"  - I observed {self.object_count} damped harmonic oscillator{'s' if self.object_count > 1 else ''}")
        trace.append("  - Each has mass, spring, and damping components\n")

        trace.append("Step 2: Determine expected behavior")
        trace.append("  - With damping present (b > 0), amplitude must decrease over time")
        trace.append("  - With spring force present (k > 0), system must oscillate")
        trace.append("  - The combination produces oscillation with exponentially decaying envelope\n")

        trace.append("Step 3: Match simulation to graph")
        trace.append("  - The correct graph must show:")
        trace.append("    • Oscillatory motion (peaks and troughs)")
        trace.append("    • Decreasing amplitude over time")
        trace.append("    • Frequency determined by √(k/m) and damping")
        if self.object_count > 1:
            trace.append(f"    • {self.object_count} distinct curves (one per oscillator)")
        trace.append("")

        trace.append("Step 4: Eliminate incorrect choices")
        trace.append("  - No damping: Wrong because amplitude stays constant")
        trace.append("  - Strong damping: Wrong because decay rate is too fast")
        trace.append("  - Different k or m: Wrong because frequency/period is incorrect")
        trace.append("  - No spring: Wrong because there's no oscillation")
        trace.append("")

        trace.append("\n=== FINAL ANSWER ===\n")

        trace.append(f"Based on the physical analysis and matching the observed behavior to the graphs,")
        trace.append(f"the correct answer is: {self.correct_label}\n")

        trace.append(f"This graph correctly represents the position x(t) for a damped harmonic oscillator")
        trace.append(f"with the given parameters (m={self.oscillator_events[0]['mass']:.2f} kg, ")
        trace.append(f"k={self.oscillator_events[0]['spring_constant']:.2f} N/m, ")
        trace.append(f"b={self.oscillator_events[0]['damping_coefficient']:.2f} N·s/m).")

        if self.object_count > 1:
            trace.append(f"\nAll {self.object_count} oscillators are correctly represented with their")
            trace.append(f"respective parameters and initial conditions.")

        return "\n".join(trace)
