
from manim import *
import numpy as np
import random
from scipy.integrate import solve_ivp
import time
import os

RHO_OBJ_BASE_RANGE = (1200, 5000)  
RHO_FLUID_BASE_RANGE = (800, 1100) 
RADIUS_BASE_RANGE = (0.05, 0.15)
B_BASE_RANGE = (0.5, 5.0)    
DEFAULT_G = 9.8
Y0 = 2.5 
V0 = 0.0 
SIM_DURATION = 4.0
SIM_POINTS_PER_SEC = 50
PREVIEW_ANIM_DURATION = 1.0 
GRAPH_X_RANGE = (0, SIM_DURATION)
FLUID_BOX_WIDTH = 4.0
FLUID_BOX_HEIGHT = 5.0
OBJECT_DISPLAY_RADIUS = 0.15 
GROUND_Y = -3.5
LABEL_SCALE_FACTOR = 0.65
CHOICE_SCALE = 0.45
TRACE_STROKE_WIDTH = 2.5


def fluid_fall_derivs(t, y_state, m, V, rho_fluid, g, b):
    """ ODE derivatives for y and vy: [dy/dt, dvy/dt] """
    y, vy = y_state
    Fg = m * g            
    Fb = rho_fluid * V * g 
    Fd = b * vy            
    ay = (rho_fluid * V * g - m * g - b * vy) / m if m > 1e-6 else 0
    return [vy, ay]

def simulate_fluid_fall(t_duration, initial_state_y_vy, params):
    """ Solves the ODE using solve_ivp. Returns t, y, vy arrays. """
    m, V, rho_fluid, g, b = params
    t_span = [0, t_duration]
    n_points = max(10, int(t_duration * SIM_POINTS_PER_SEC) + 1)
    t_eval = np.linspace(t_span[0], t_span[1], n_points)
    sol = solve_ivp(
        fluid_fall_derivs,
        t_span,
        initial_state_y_vy, 
        args=(m, V, rho_fluid, g, b),
        t_eval=t_eval,
        method='RK45', 
        rtol=1e-6,
        atol=1e-8
    )
    if not sol.success:
        print(f"Warning: ODE solver failed! Status: {sol.status}, Message: {sol.message}")
        return np.array([0.0, 1e-3]), np.array([initial_state_y_vy[0]]*2), np.array([initial_state_y_vy[1]]*2)
    if len(sol.t) < 2:
        print("Warning: Simulation returned < 2 time points.")
        return np.array([0.0, 1e-3]), np.array([initial_state_y_vy[0]]*2), np.array([initial_state_y_vy[1]]*2)
    return sol.t, sol.y[0], sol.y[1] 

def simulate_multiple_objects(t_duration, initial_states, params_list):
    """Simulate multiple objects falling simultaneously."""
    results = []
    for i, (initial_state, params) in enumerate(zip(initial_states, params_list)):
        t, y, v = simulate_fluid_fall(t_duration, initial_state, params)
        results.append((t, y, v))
    return results


def create_fluid_box(y_center=0, width=FLUID_BOX_WIDTH, height=FLUID_BOX_HEIGHT):
    """Creates the visual representation of the fluid container."""
    box = Rectangle(width=width, height=height, color=BLUE, fill_opacity=0.2, stroke_width=2)
    box.move_to(y_center * UP)

    surface_y = box.get_top()[1] - 0.05
    surface = Line(box.get_corner(UL) + DOWN*0.05, box.get_corner(UR)+ DOWN*0.05, color=BLUE_A, stroke_width=3)
    return VGroup(box, surface)

def create_ground(y_level=GROUND_Y):
    """Creates the ground line."""
    return Line(config.frame_x_radius*LEFT + y_level*UP, config.frame_x_radius*RIGHT + y_level*UP, color=GRAY, stroke_width=2)

def create_vel_graph(t_pts, v_pts, x_range, y_range, graph_color=WHITE):
    """Creates an Axes object with the plotted v(t) data."""
    print(f"--- Creating Graph ---")
    print(f"  Input: t_pts len={len(t_pts)}, v_pts len={len(v_pts)}, y_range={y_range}")
    t_pts = np.array(t_pts); v_pts = np.array(v_pts)

    if len(t_pts)<2 or len(v_pts)<2 or len(t_pts)!=len(v_pts) or np.isnan(t_pts).any() or np.isnan(v_pts).any() or np.isinf(t_pts).any() or np.isinf(v_pts).any():
        print(f"  ERROR: Invalid data for graph (len<2, mismatch, NaN, or Inf). Creating empty axes.")
        print(f"    t[:5]={t_pts[:5]}, v[:5]={v_pts[:5]}")
        axes = Axes(x_range=x_range, y_range=[-1, 1] if len(y_range)<2 or y_range[0]>=y_range[1] else y_range, x_length=4.0, y_length=2.5, axis_config={"include_tip": False})
        labels = axes.get_axis_labels(x_label=MathTex("t",font_size=20), y_label=MathTex("v_y",font_size=20))
        return VGroup(axes, labels)

    y_min_eff, y_max_eff = y_range

    if y_min_eff >= y_max_eff:
        print(f"  Warn: Invalid y_range {y_range}. Setting to default [-5, 1].")
        y_min_eff, y_max_eff = -5.0, 1.0

    y_step = abs(y_max_eff - y_min_eff) / 4.0
    if y_step < 1e-3: y_step = 0.1 
    x_step = abs(x_range[1]-x_range[0]) / 4.0
    if x_step < 1e-3: x_step = 0.5

    print(f"  Using axes ranges: x=[{x_range[0]},{x_range[1]},{x_step:.2f}], y=[{y_min_eff:.2f},{y_max_eff:.2f},{y_step:.2f}]")
    print(f"  Plotting data: t[:5]={t_pts[:5]}, v[:5]={v_pts[:5]}")

    try:
        axes = Axes(
            x_range=[x_range[0], x_range[1], x_step],
            y_range=[y_min_eff, y_max_eff, y_step],
            x_length=4.0,
            y_length=2.5,
            axis_config={"include_numbers": True, "include_tip": False, "stroke_width": 1.5, "decimal_number_config": {"num_decimal_places": 1}},
            tips=False
        )
        labels = axes.get_axis_labels(x_label=MathTex("t",font_size=20), y_label=MathTex("v_y",font_size=20))

        valid_indices = np.isfinite(t_pts) & np.isfinite(v_pts)
        if not np.all(valid_indices):
             print("  Warn: Filtering non-finite values before plotting.")
             t_pts_clean = t_pts[valid_indices]
             v_pts_clean = v_pts[valid_indices]
        else:
             t_pts_clean = t_pts
             v_pts_clean = v_pts

        if len(t_pts_clean) < 2:
             print("  ERROR: Not enough valid points left after filtering to plot.")
             graph_group = VGroup(axes, labels)
        else:
             print(f"  Calling axes.plot with {len(t_pts_clean)} points...")
             graph = axes.plot_line_graph(x_values=t_pts_clean, y_values=v_pts_clean, line_color=graph_color, add_vertex_dots=False, stroke_width=TRACE_STROKE_WIDTH)
             print(f"  axes.plot_line_graph result: {graph}")
             graph_group = VGroup(axes, graph["line_graph"], labels) 
             print(f"  Graph VGroup created successfully.")

    except Exception as e:
         print(f"  ERROR creating graph object: {e}") 
         axes = Axes(x_range=x_range, y_range=[y_min_eff,y_max_eff,y_step], x_length=4.0, y_length=2.5, axis_config={"include_tip": False})
         labels = axes.get_axis_labels(x_label=MathTex("t",font_size=20),y_label=MathTex("v_y",font_size=20))
         graph_group = VGroup(axes, labels)

    print(f"--- Finished Graph Creation ---")
    return graph_group

def create_multi_object_graph(t_pts_list, v_pts_list, x_range, y_range, colors):
    """Creates a graph with multiple velocity curves for different objects."""
    if not t_pts_list or not v_pts_list or len(t_pts_list) != len(v_pts_list):
        print("ERROR: Invalid data for multi-object graph")
        return create_vel_graph([0, 1], [0, 0], x_range, y_range)

    y_min_eff, y_max_eff = y_range
    if y_min_eff >= y_max_eff:
        y_min_eff, y_max_eff = -5.0, 1.0
    
    y_step = abs(y_max_eff - y_min_eff) / 4.0
    if y_step < 1e-3: y_step = 0.1
    x_step = abs(x_range[1]-x_range[0]) / 4.0
    if x_step < 1e-3: x_step = 0.5
    
    axes = Axes(
        x_range=[x_range[0], x_range[1], x_step],
        y_range=[y_min_eff, y_max_eff, y_step],
        x_length=4.0,
        y_length=2.5,
        axis_config={"include_numbers": True, "include_tip": False, "stroke_width": 1.5, "decimal_number_config": {"num_decimal_places": 1}},
        tips=False
    )
    labels = axes.get_axis_labels(x_label=MathTex("t",font_size=20), y_label=MathTex("v_y",font_size=20))
    
    graph_group = VGroup(axes, labels)
    for i, (t_pts, v_pts) in enumerate(zip(t_pts_list, v_pts_list)):
        if len(t_pts) >= 2 and len(v_pts) >= 2:
            try:
                graph = axes.plot_line_graph(
                    x_values=t_pts, 
                    y_values=v_pts, 
                    line_color=colors[i], 
                    add_vertex_dots=False, 
                    stroke_width=TRACE_STROKE_WIDTH
                )
                graph_group.add(graph["line_graph"])
            except Exception as e:
                print(f"Error plotting object {i}: {e}")
    
    return graph_group

def adjust_fluid_fall_parameters(difficulty):
    """Adjusts parameter ranges based on difficulty level (0-9)."""
    difficulty = max(0, min(difficulty, 9))
    diff_factor = difficulty / 9.0 

    rho_f_min = RHO_FLUID_BASE_RANGE[0]
    rho_f_max = RHO_FLUID_BASE_RANGE[0] + (RHO_FLUID_BASE_RANGE[1] - RHO_FLUID_BASE_RANGE[0]) * (0.7 + 0.3 * diff_factor)
    adj_rho_fluid_range = (rho_f_min, rho_f_max)

    rho_o_min = rho_f_max * (1.05 + 0.5 * diff_factor) 
    rho_o_max = RHO_OBJ_BASE_RANGE[0] + (RHO_OBJ_BASE_RANGE[1] - RHO_OBJ_BASE_RANGE[0]) * (0.4 + 0.6 * diff_factor)
    adj_rho_obj_range = (rho_o_min, rho_o_max)

    r_min = RADIUS_BASE_RANGE[0]
    r_max = RADIUS_BASE_RANGE[0] + (RADIUS_BASE_RANGE[1] - RADIUS_BASE_RANGE[0]) * (0.6 + 0.4 * diff_factor)
    adj_radius_range = (r_min, r_max)

    b_min = B_BASE_RANGE[0]
    b_max = B_BASE_RANGE[0] + (B_BASE_RANGE[1] - B_BASE_RANGE[0]) * (0.5 + 0.5 * diff_factor)
    adj_b_range = (b_min, b_max)

    print(f"Adjusted Ranges (Diff={difficulty}):")
    print(f"  Rho Fluid: {adj_rho_fluid_range}")
    print(f"  Rho Object: {adj_rho_obj_range}")
    print(f"  Radius: {adj_radius_range}")
    print(f"  Drag (b): {adj_b_range}")

    return adj_rho_obj_range, adj_rho_fluid_range, adj_radius_range, adj_b_range


class FluidFallQuizRefactored(Scene):
    anim_time = 0.0 

    def construct(self):
        seed = int(os.environ.get('MANIM_SEED', time.time()))
        difficulty = int(os.environ.get('MANIM_DIFFICULTY', 5))
        object_count = int(os.environ.get('MANIM_OBJECT_COUNT', 1))

        if object_count not in [1, 2, 3]:
            print(f"Warning: object_count {object_count} is not valid. Using default value 1.")
            object_count = 1
        
        random.seed(seed); np.random.seed(seed)

        adj_rho_obj_range, adj_rho_fluid_range, adj_radius_range, adj_b_range = adjust_fluid_fall_parameters(difficulty)


        rho_fluid = random.uniform(*adj_rho_fluid_range)
        

        objects_params = []
        objects_initial_states = []
        object_colors = [RED, GREEN, BLUE]
        
        for i in range(object_count):
            rho_obj = random.uniform(*adj_rho_obj_range)
            loop_count = 0
            while rho_obj <= rho_fluid and loop_count < 20: 
                print(f"  Regen rho_obj_{i+1} ({rho_obj:.1f}) <= rho_fluid ({rho_fluid:.1f}). Loop: {loop_count}")
                rho_obj = random.uniform(*adj_rho_obj_range)
                loop_count += 1
            if rho_obj <= rho_fluid:
                print(f"ERROR: Failed to generate rho_obj_{i+1} > rho_fluid. Using defaults.")
                rho_obj = rho_fluid * 1.2

            r = random.uniform(*adj_radius_range)
            b = random.uniform(*adj_b_range)
            g = DEFAULT_G

            y0 = Y0 + i * 0.3  
            v0 = V0


            V = (4/3) * PI * r**3 
            m = rho_obj * V    
            Fb = rho_fluid * V * g 
            Fg = m * g           
            v_terminal = (rho_fluid * V * g - m * g) / b if abs(b) > 1e-9 else -np.inf

            objects_params.append([m, V, rho_fluid, g, b])
            objects_initial_states.append([y0, v0])
            
            print(f"Object {i+1}: rho_obj={rho_obj:.1f}, r={r:.3f}, b={b:.2f}, v_T={v_terminal:.2f}")

        print(f"--- Fluid Fall Setup (Seed: {seed}, Difficulty: {difficulty}, Objects: {object_count}) ---")
        print(f"Fluid density: {rho_fluid:.1f}")
        print(f"--------------------")

        params_correct = objects_params
        

        params_no_drag = []
        for params in objects_params:
            m, V, rho_fluid, g, b = params
            params_no_drag.append([m, V, rho_fluid, g, 0.0])

        params_no_buoyancy = []
        for params in objects_params:
            m, V, rho_fluid, g, b = params
            params_no_buoyancy.append([m, V, 0.0, g, b])

        params_different_drag = []
        for i, params in enumerate(objects_params):
            m, V, rho_fluid, g, b = params
            b_new = b * (0.5 + 0.5 * i) 
            params_different_drag.append([m, V, rho_fluid, g, b_new])
        
        params_stronger_drag = []
        for params in objects_params:
            m, V, rho_fluid, g, b = params
            params_stronger_drag.append([m, V, rho_fluid, g, b * 2.0])
        
        params_weaker_drag = []
        for params in objects_params:
            m, V, rho_fluid, g, b = params
            params_weaker_drag.append([m, V, rho_fluid, g, b * 0.3])

        title = Text(f"Problem: {object_count} Object(s) Falling in Viscous Fluid", font_size=32).to_edge(UP)
        fluid_box = create_fluid_box(y_center=(GROUND_Y + Y0)/2)
        ground = create_ground(GROUND_Y)
        objects = []
        param_texts = []
        for i in range(object_count):
            y0_obj = objects_initial_states[i][0]
            obj = Dot(point=UP * y0_obj, radius=OBJECT_DISPLAY_RADIUS, color=object_colors[i], fill_opacity=1)
            objects.append(obj)
            
            m, V, rho_fluid, g, b = objects_params[i]
            rho_obj = m / V
            r = (3 * V / (4 * PI))**(1/3)
            param_text = VGroup(
                MathTex(f"\\text{{Object {i+1}:}}", font_size=24, color=object_colors[i]),
                MathTex(f"\\rho_{{obj}} = {rho_obj:.0f}\\, \\text{{kg/m}}^3", font_size=20),
                MathTex(f"r = {r*100:.1f}\\, \\text{{cm}}", font_size=20),
                MathTex(f"b = {b:.2f}\\, \\text{{N s/m}}", font_size=20)
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

        initial_static_group = VGroup(title, fluid_box, ground, param_group)
        self.add(initial_static_group)
        for obj in objects:
            self.add(obj)
        self.wait(1)

        sim_duration_total = SIM_DURATION
        print("Calculating Correct Trajectories for Animation...")
        
        results_correct = simulate_multiple_objects(sim_duration_total, objects_initial_states, params_correct)
        
        if not results_correct or any(len(t) < 2 for t, y, v in results_correct):
            print("ERROR: Simulation failed to produce sufficient points.")
            self.play(Write(Text("Simulation Error!", color=RED).move_to(ORIGIN)))
            self.wait(2)
            return

        self.anim_time = 0.0
        obj_anims = [obj.copy() for obj in objects]

        fluid_box_bottom_y = fluid_box[0].get_bottom()[1]
        min_y_visual = max(GROUND_Y + OBJECT_DISPLAY_RADIUS, fluid_box_bottom_y + OBJECT_DISPLAY_RADIUS)
        
        def multi_object_updater(mobj, dt, obj_index):
            self.anim_time += dt
            t_correct, y_correct, v_correct = results_correct[obj_index]
            current_y = np.interp(self.anim_time, t_correct, y_correct)
            current_y = max(current_y, min_y_visual)
            mobj.move_to(UP * current_y)

        for i, obj_anim in enumerate(obj_anims):
            obj_anim.add_updater(lambda mobj, dt, idx=i: multi_object_updater(mobj, dt, idx))
            self.add(obj_anim)


        for obj in objects:
            self.remove(obj)

        print(f"  Playing {PREVIEW_ANIM_DURATION:.1f}s preview animation...")
        self.wait(PREVIEW_ANIM_DURATION)
        

        for obj_anim in obj_anims:
            obj_anim.clear_updaters()
        self.wait(0.5)


        question = Text("Which graph best represents the vertical velocities v_y(t)?", font_size=32).to_edge(DOWN, buff=MED_SMALL_BUFF)
        self.play(Write(question))
        self.wait(0.5)

        fade_out_group = VGroup(initial_static_group, question, *obj_anims)

        print("Calculating Distractor Trajectories...")
        results_no_drag = simulate_multiple_objects(sim_duration_total, objects_initial_states, params_no_drag)
        results_no_buoyancy = simulate_multiple_objects(sim_duration_total, objects_initial_states, params_no_buoyancy)
        results_different_drag = simulate_multiple_objects(sim_duration_total, objects_initial_states, params_different_drag)
        results_stronger_drag = simulate_multiple_objects(sim_duration_total, objects_initial_states, params_stronger_drag)
        results_weaker_drag = simulate_multiple_objects(sim_duration_total, objects_initial_states, params_weaker_drag)

        all_vs = []
        all_results = [results_correct, results_no_drag, results_no_buoyancy, results_different_drag, results_stronger_drag, results_weaker_drag]
        
        for results in all_results:
            for t, y, v in results:
                if len(v) > 0:
                    all_vs.extend(v)
        
        if not all_vs:
            print("ERROR: No valid velocity data from simulations for graphing.")
            self.play(Write(Text("Graphing Error!", color=RED).move_to(ORIGIN)))
            self.wait(2)
            return
            
        min_v_actual = min(all_vs)
        max_v_actual = max(all_vs)

        y_buffer = abs(max_v_actual - min_v_actual) * 0.1 + 0.2
        y_min_graph = min_v_actual - y_buffer
        y_max_graph = max_v_actual + y_buffer
        if y_min_graph >= y_max_graph:
            y_min_graph -= 0.5
            y_max_graph += 0.5

        graph_y_range = [y_min_graph, y_max_graph]
        print(f"Graph Y Range (Velocity): [{graph_y_range[0]:.2f}, {graph_y_range[1]:.2f}]")

        def extract_data(results):
            t_list = [t for t, y, v in results]
            v_list = [v for t, y, v in results]
            return t_list, v_list

        t_correct_list, v_correct_list = extract_data(results_correct)
        t_no_drag_list, v_no_drag_list = extract_data(results_no_drag)
        t_no_buoyancy_list, v_no_buoyancy_list = extract_data(results_no_buoyancy)
        t_different_drag_list, v_different_drag_list = extract_data(results_different_drag)
        t_stronger_drag_list, v_stronger_drag_list = extract_data(results_stronger_drag)
        t_weaker_drag_list, v_weaker_drag_list = extract_data(results_weaker_drag)


        graph_correct = create_multi_object_graph(t_correct_list, v_correct_list, GRAPH_X_RANGE, graph_y_range, object_colors[:object_count])
        graph_no_drag = create_multi_object_graph(t_no_drag_list, v_no_drag_list, GRAPH_X_RANGE, graph_y_range, object_colors[:object_count])
        graph_no_buoyancy = create_multi_object_graph(t_no_buoyancy_list, v_no_buoyancy_list, GRAPH_X_RANGE, graph_y_range, object_colors[:object_count])
        graph_different_drag = create_multi_object_graph(t_different_drag_list, v_different_drag_list, GRAPH_X_RANGE, graph_y_range, object_colors[:object_count])
        graph_stronger_drag = create_multi_object_graph(t_stronger_drag_list, v_stronger_drag_list, GRAPH_X_RANGE, graph_y_range, object_colors[:object_count])
        graph_weaker_drag = create_multi_object_graph(t_weaker_drag_list, v_weaker_drag_list, GRAPH_X_RANGE, graph_y_range, object_colors[:object_count])

        all_choices = [graph_correct, graph_no_drag, graph_no_buoyancy, graph_different_drag, graph_stronger_drag, graph_weaker_drag]
        choice_descriptions = [
            "Correct (All forces)",
            "No Drag",
            "No Buoyancy", 
            "Different Drag",
            "Stronger Drag",
            "Weaker Drag"
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
