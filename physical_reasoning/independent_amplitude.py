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
DEFAULT_V0_RANGE = (0.5, 3.0)
DEFAULT_X0 = 0.0 

SIM_DURATION = 10
SIM_POINTS_PER_SEC = 50
ANIMATION_DURATION = 10 

SPRING_BUMPS = 10
SPRING_RADIUS = 0.15
MASS_SIDE_LENGTH = 0.7 
WALL_HEIGHT = 0.6 

SYSTEM_LABELS_2 = ["System 1", "System 2"]
SYSTEM_LABELS_3 = ["System 1", "System 2", "System 3"]
SYSTEM_LABELS_4 = ["System 1", "System 2", "System 3", "System 4"]

SYSTEM_MASS_COLORS = [BLUE_C, GREEN_C, RED_C, YELLOW_C]
SYSTEM_SPRING_COLORS = [BLUE_D, GREEN_D, RED_D, YELLOW_D]

VERTICAL_SYSTEM_Y_POSITIONS_2 = [DOWN * 2.0, DOWN * 5.0]
VERTICAL_SYSTEM_Y_POSITIONS_3 = [DOWN * 1.5, DOWN * 3.0, DOWN * 5.5]
VERTICAL_SYSTEM_Y_POSITIONS_4 = [DOWN * 1.0, DOWN * 2.0, DOWN * 3.5, DOWN * 5.5]

def oscillator_derivs(t, state, m, k, b):
    x, v = state
    ax = (-k * x - b * v) / m if m > 1e-6 else 0
    return [v, ax]

def simulate_oscillator(t_duration, initial_state, params):
    m, k, b = params
    t_span = [0, t_duration]
    n_points = int(t_duration * SIM_POINTS_PER_SEC) + 1
    t_eval = np.linspace(t_span[0], t_span[1], n_points)

    if not np.all(np.isfinite(initial_state)):
        initial_state = [0.0, 0.0]
    try:
        sol = solve_ivp(
            oscillator_derivs, t_span, initial_state, args=(m, k, b),
            t_eval=t_eval, rtol=1e-6, atol=1e-8, dense_output=False
        )
        if sol.status != 0:
            t_pts_sol = sol.t; x_pts_sol = sol.y[0]
            v_pts_sol = sol.y[1] if sol.y.shape[0] > 1 else (np.gradient(x_pts_sol, t_pts_sol, edge_order=2) if len(t_pts_sol) > 1 else np.zeros_like(t_pts_sol))
            x_pts = np.interp(t_eval, t_pts_sol, x_pts_sol)
            v_pts = np.interp(t_eval, t_pts_sol, v_pts_sol)
            t_pts = t_eval
        else:
            t_pts = sol.t; x_pts = sol.y[0]; v_pts = sol.y[1]

        valid_indices = np.where(np.isfinite(x_pts) & np.isfinite(v_pts))[0]
        if len(valid_indices) < len(x_pts):
            if len(valid_indices) > 0:
                max_valid_idx = valid_indices[-1]
                t_pts = t_pts[:max_valid_idx+1]; x_pts = x_pts[:max_valid_idx+1]; v_pts = v_pts[:max_valid_idx+1]
            else:
                t0 = t_eval[0] if len(t_eval) > 0 else 0; x0_val, v0_val = initial_state
                return np.array([t0, t0 + 1e-3]), np.array([x0_val, x0_val]), np.array([v0_val, v0_val])
        if len(t_pts) < 2:
            t0 = t_eval[0] if len(t_eval) > 0 else 0; x0_val, v0_val = initial_state
            return np.array([t0, t0 + 1e-3]), np.array([x0_val, x0_val]), np.array([v0_val, v0_val])
        return t_pts, x_pts, v_pts
    except Exception as e:
        print(f"Error during solve_ivp: {e}")
        t0 = t_eval[0] if len(t_eval) > 0 else 0; x0_val, v0_val = initial_state
        return np.array([t0, t0 + 1e-3]), np.array([x0_val, x0_val]), np.array([v0_val, v0_val])

def create_single_spring_visual(start_point, end_point, bumps=SPRING_BUMPS, radius=SPRING_RADIUS, color=WHITE, stroke_width=2):
    start=np.array(start_point); end=np.array(end_point)
    length=np.linalg.norm(end-start)
    direction_vec = end - start

    if length < 1e-6: return Line(start, end, color=color, stroke_width=stroke_width)
    
    direction = normalize(direction_vec)
    
    if abs(direction[0]) < 1e-6 and abs(direction[1]) < 1e-6 : 
        perp_direction_for_bumps = RIGHT 
    elif abs(direction[1]) > 0.999: 
        perp_direction_for_bumps = RIGHT 
    else: 
        perp_direction_for_bumps = normalize(np.cross(direction, OUT))

    def spring_func(t):
        return start + t*direction_vec + perp_direction_for_bumps*radius*np.sin(bumps*TAU*t)
    
    num_segments = bumps * 20
    points = [spring_func(t) for t in np.linspace(0, 1, num_segments + 1)]
    return VMobject(color=color, stroke_width=stroke_width).set_points_as_corners(points)


class IndependentOscillatorsAmplitudeQuiz(Scene):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Set random seed for reproducibility
        self.seed = int(time.time())
        random.seed(self.seed)
        np.random.seed(self.seed)

        # Parameters
        self.num_systems = int(os.environ.get('MANIM_OBJECT_COUNT', 2))
        if self.num_systems not in [2, 3, 4]:
            print(f"Warning: object_count {self.num_systems} is not valid. Using default value 2.")
            self.num_systems = 2

        # Store for reasoning trace
        self.oscillation_events = []
        self.systems_data = []
        self.amplitude_data = []

    def construct(self):
        num_systems = self.num_systems
        seed = self.seed

        self.camera.background_color = DARK_GRAY

        if num_systems == 2:
            SYSTEM_LABELS = SYSTEM_LABELS_2
            VERTICAL_SYSTEM_Y_POSITIONS = VERTICAL_SYSTEM_Y_POSITIONS_2
            current_mass_colors = SYSTEM_MASS_COLORS[:2]
            current_spring_colors = SYSTEM_SPRING_COLORS[:2]
        elif num_systems == 3:
            SYSTEM_LABELS = SYSTEM_LABELS_3
            VERTICAL_SYSTEM_Y_POSITIONS = VERTICAL_SYSTEM_Y_POSITIONS_3
            current_mass_colors = SYSTEM_MASS_COLORS[:3]
            current_spring_colors = SYSTEM_SPRING_COLORS[:3]
        elif num_systems == 4:
            SYSTEM_LABELS = SYSTEM_LABELS_4
            VERTICAL_SYSTEM_Y_POSITIONS = VERTICAL_SYSTEM_Y_POSITIONS_4
            current_mass_colors = SYSTEM_MASS_COLORS[:4]
            current_spring_colors = SYSTEM_SPRING_COLORS[:4]

        self.systems_data = []
        for i in range(num_systems):
            m = random.uniform(*DEFAULT_M_RANGE)
            k = random.uniform(*DEFAULT_K_RANGE)
            v0 = random.uniform(*DEFAULT_V0_RANGE) * random.choice([-1, 1])
            x0 = DEFAULT_X0
            b = 0.0

            if i > 0:
                while any(abs(abs(v0) - abs(sys_data['v0'])) < 0.5 and np.sign(v0) == np.sign(sys_data['v0']) for sys_data in self.systems_data):
                    v0 = random.uniform(*DEFAULT_V0_RANGE) * random.choice([-1, 1])

            self.systems_data.append({
                'm': m, 'k': k, 'v0': v0, 'x0': x0, 'b': b,
                'params': [m, k, b],
                'initial_state': [x0, v0],
                'label': SYSTEM_LABELS[i],
                'mass_color': current_mass_colors[i],
                'spring_color': current_spring_colors[i],
                'vertical_offset': VERTICAL_SYSTEM_Y_POSITIONS[i]
            })

            print(f"--- {SYSTEM_LABELS[i]} (Seed: {seed}) ---")
            print(f"Params: m{i+1}={m:.2f}, k{i+1}={k:.2f}, x0{i+1}={x0:.2f}, v0{i+1}={v0:.2f}")

        systems_data = self.systems_data

        scene_title = Text(f"{num_systems} Independent Oscillators", font_size=36).to_edge(UP)

        max_amp_est = 0.5
        for sys_data in systems_data:
            omega_val = np.sqrt(sys_data['k']/sys_data['m']) if sys_data['m'] > 1e-6 and sys_data['k'] > 1e-6 else 1.0
            amp_est = abs(sys_data['v0']/omega_val) if omega_val > 1e-6 else (abs(sys_data['x0']) if abs(sys_data['x0'])>0 else 0.1)
            max_amp_est = max(max_amp_est, amp_est)
        
        x_axis_bnd = min(np.ceil(max_amp_est * 1.2) + 0.5, 6.0)
        x_axis_bnd = max(x_axis_bnd, 1.0)
        x_tick_freq = 0.5 if x_axis_bnd <= 3.0 else 1.0
        if x_axis_bnd > 4.5: x_tick_freq = 1.0

        x_axis = NumberLine(
            x_range=[-x_axis_bnd, x_axis_bnd, x_tick_freq], 
            length=self.camera.frame_width - 2.0, 
            color=WHITE, include_numbers=True, 
            label_direction=DOWN, font_size=20, stroke_width=2
        ).move_to(ORIGIN)

        x_axis.move_to(UP * 2.5)

        system_mobjects = []
        for i, sys_data in enumerate(systems_data):
            wall_x_coord = x_axis.n2p(-x_axis_bnd)[0]
            wall = Line([wall_x_coord, WALL_HEIGHT/2, 0], [wall_x_coord, -WALL_HEIGHT/2, 0], 
                       color=GRAY_C, stroke_width=3).next_to(x_axis.n2p(-x_axis_bnd), RIGHT, buff=0).shift(sys_data['vertical_offset'])
            spring_anchor = wall.get_center()
            mass_obj = Square(side_length=MASS_SIDE_LENGTH, color=sys_data['mass_color'], fill_opacity=0.8).move_to(x_axis.n2p(sys_data['x0']) + sys_data['vertical_offset'])
            spring_obj = create_single_spring_visual(spring_anchor, mass_obj.get_left(), color=sys_data['spring_color'])
            label = Tex(sys_data['label'], font_size=20).next_to(wall, UP if i < num_systems//2 else DOWN, buff=0.1)
            
            system_mobjects.append({
                'wall': wall,
                'mass_obj': mass_obj,
                'spring_obj': spring_obj,
                'label': label,
                'spring_anchor': spring_anchor,
                'sys_data': sys_data
            })

        eq_line_length_factor = 0.5 
        eq_line = DashedLine(
            x_axis.n2p(0) + UP * eq_line_length_factor,
            x_axis.n2p(0) + DOWN * eq_line_length_factor,
            color=YELLOW_A, stroke_width=1.5
        )

        self.play(
            FadeIn(scene_title), Create(x_axis), Create(eq_line),
            *[FadeIn(m['wall']) for m in system_mobjects],
            *[FadeIn(m['mass_obj']) for m in system_mobjects],
            *[Create(m['spring_obj']) for m in system_mobjects],
            *[Write(m['label']) for m in system_mobjects]
        )
        self.wait(0.5)

        t_sim_data = {}
        for i, sys_data in enumerate(systems_data):
            t_sim, x_sim, v_sim = simulate_oscillator(SIM_DURATION, sys_data['initial_state'], sys_data['params'])
            t_sim_data[i] = {'t': t_sim, 'x': x_sim, 'v': v_sim}

            # Record oscillation event for reasoning trace
            self.oscillation_events.append({
                'system': sys_data['label'],
                'mass': sys_data['m'],
                'spring_constant': sys_data['k'],
                'initial_position': sys_data['x0'],
                'initial_velocity': sys_data['v0'],
                'simulation_time': t_sim,
                'positions': x_sim,
                'velocities': v_sim
            })

        if any(len(t_sim_data[i]['t']) < 2 or not (np.any(np.isfinite(t_sim_data[i]['x'])) and np.any(np.isfinite(t_sim_data[i]['v']))) for i in range(num_systems)):
            self.play(FadeOut(VGroup(*self.mobjects_without_background)), FadeIn(Text("Simulation Error!",color=RED))); self.wait(3); print("FINAL_ANSWER: ERROR_SIM"); return
        
        self.anim_time_data = {}
        for i, sys_data in enumerate(systems_data):
            self.anim_time_data[i] = 0.0
            
            def mass_updater_func(mobj, dt, system_index=i):
                self.anim_time_data[system_index] += dt
                current_x = np.interp(min(self.anim_time_data[system_index], t_sim_data[system_index]['t'][-1]), t_sim_data[system_index]['t'], t_sim_data[system_index]['x'])
                mobj.move_to(x_axis.n2p(current_x) + systems_data[system_index]['vertical_offset'])
            
            system_mobjects[i]['mass_obj'].add_updater(lambda mobj, dt, idx=i: mass_updater_func(mobj, dt, idx))

            def spring_updater_func(mob, system_index=i):
                current_x = np.interp(min(self.anim_time_data[system_index], t_sim_data[system_index]['t'][-1]), t_sim_data[system_index]['t'], t_sim_data[system_index]['x'])
                mob.become(create_single_spring_visual(system_mobjects[system_index]['spring_anchor'], x_axis.n2p(current_x) + systems_data[system_index]['vertical_offset'], color=systems_data[system_index]['spring_color']))
            
            system_mobjects[i]['spring_obj'].add_updater(lambda mob, idx=i: spring_updater_func(mob, idx))
        
        eff_anim_dur = min(ANIMATION_DURATION, max(t_sim_data[i]['t'][-1] if len(t_sim_data[i]['t'])>0 else 0 for i in range(num_systems)))
        if eff_anim_dur > 1e-4: self.wait(eff_anim_dur)
        
        for i, sys_data in enumerate(systems_data):
            system_mobjects[i]['mass_obj'].clear_updaters()
            system_mobjects[i]['spring_obj'].clear_updaters()
        
        final_x_data = {}
        for i, sys_data in enumerate(systems_data):
            final_x_data[i] = np.interp(min(self.anim_time_data[i], t_sim_data[i]['t'][-1]), t_sim_data[i]['t'], t_sim_data[i]['x']) if len(t_sim_data[i]['t']) > 0 else sys_data['x0']

        for i, sys_data in enumerate(systems_data):
            system_mobjects[i]['mass_obj'].move_to(x_axis.n2p(final_x_data[i]) + sys_data['vertical_offset'])
            system_mobjects[i]['spring_obj'].become(create_single_spring_visual(system_mobjects[i]['spring_anchor'], x_axis.n2p(final_x_data[i]) + sys_data['vertical_offset'], color=sys_data['spring_color']))
        self.wait(0.2)


        if any(sys_data['m'] < 1e-6 or sys_data['k'] < 1e-3 for sys_data in systems_data):
            q_text_mobj = Text("Error: m or k values too small.", color=RED)
            final_ans_script = "ERROR_MK_TOO_SMALL"
        else:
            self.amplitude_data = []
            for i, sys_data in enumerate(systems_data):
                omega_val = np.sqrt(sys_data['k'] / sys_data['m'])
                if abs(omega_val) < 1e-6:
                    amp_val = float('inf') if abs(sys_data['v0']) > 1e-6 else 0.0
                else:
                    amp_val = abs(sys_data['v0'] / omega_val)
                self.amplitude_data.append(amp_val)

            if any(amp == float('inf') for amp in self.amplitude_data) or any(abs(amp) < 1e-6 for amp in self.amplitude_data):
                q_text_mobj = Text("Error: Amplitude calculation issue (A is zero or infinite).", color=RED, font_size=18)
                final_ans_script = "ERROR_AMP_CALC"
            else:
                if num_systems == 2:
                    amplitude_ratio = self.amplitude_data[0] / self.amplitude_data[1]
                    question_text = r"\text{What is the ratio of the amplitudes } \frac{A_1}{A_2}\text{?} \text{ return answer in 2 decimal points}"
                    self.ratio_indices = (0, 1)
                elif num_systems == 3:
                    amplitude_ratio = self.amplitude_data[0] / self.amplitude_data[2]  # A1/A3
                    question_text = r"\text{What is the ratio of the amplitudes } \frac{A_1}{A_3}\text{?} \text{ return answer in 2 decimal points}"
                    self.ratio_indices = (0, 2)
                elif num_systems == 4:
                    amplitude_ratio = self.amplitude_data[0] / self.amplitude_data[3]  # A1/A4
                    question_text = r"\text{What is the ratio of the amplitudes } \frac{A_1}{A_4}\text{?} \text{ return answer in 2 decimal points}"
                    self.ratio_indices = (0, 3)

                self.amplitude_ratio = amplitude_ratio
                final_ans_script = f"{amplitude_ratio:.2f}"

                quiz_title = Tex("Quiz", font_size=32).to_edge(UP)

                param_info_lines = []
                for i, sys_data in enumerate(systems_data):
                    param_info_lines.append(Tex(f"System {i+1}: $m_{i+1}={sys_data['m']:.2f}, k_{i+1}={sys_data['k']:.2f}, v_{{0,{i+1}}}={sys_data['v0']:.2f}$ (initial $x_0=0$)", font_size=20))
                param_info_group = VGroup(*param_info_lines).arrange(DOWN, buff=0.15).next_to(quiz_title, DOWN, buff=0.3)

                question_line = MathTex(question_text, font_size=30, color=YELLOW_C)
                question_line.next_to(param_info_group, DOWN, buff=0.4)

                q_text_mobj = VGroup(question_line)
                q_text_mobj.move_to(ORIGIN)

        self.play(
            FadeOut(scene_title), FadeOut(x_axis), FadeOut(eq_line),
            *[FadeOut(m['wall']) for m in system_mobjects],
            *[FadeOut(m['spring_obj']) for m in system_mobjects],
            *[FadeOut(m['mass_obj']) for m in system_mobjects],
            *[FadeOut(m['label']) for m in system_mobjects]
            , Write(q_text_mobj)
        )
        print(f"FINAL_ANSWER: {final_ans_script}")
        self.wait(15)

        # Save solution
        with open(
            f"solutions/independent_amplitude_n{num_systems}_seed{seed}.txt", "w"
        ) as f:
            f.write(final_ans_script)

        # Save question text
        if num_systems == 2:
            question_text_content = "What is the ratio of the amplitudes A1/A2? Return answer in 2 decimal points."
        elif num_systems == 3:
            question_text_content = "What is the ratio of the amplitudes A1/A3? Return answer in 2 decimal points."
        elif num_systems == 4:
            question_text_content = "What is the ratio of the amplitudes A1/A4? Return answer in 2 decimal points."

        question_text_content += "\n\nParameters:\n"
        for i, sys_data in enumerate(systems_data):
            question_text_content += f"System {i+1}: m{i+1}={sys_data['m']:.2f}, k{i+1}={sys_data['k']:.2f}, v0{i+1}={sys_data['v0']:.2f} (initial x0=0)\n"

        with open(
            f"question_text/independent_amplitude_n{num_systems}_seed{seed}.txt", "w"
        ) as f:
            f.write(question_text_content)

        # Generate and save reasoning trace
        if final_ans_script not in ["ERROR_MK_TOO_SMALL", "ERROR_AMP_CALC", "ERROR_SIM"]:
            reasoning_trace = self.generate_reasoning_trace()
            with open(
                f"reasoning_traces/independent_amplitude_n{num_systems}_seed{seed}.txt", "w"
            ) as f:
                f.write(reasoning_trace)

    def generate_reasoning_trace(self):
        """Generate a detailed reasoning trace describing the oscillation scene chronologically"""

        trace = []
        trace.append("=== SCENE SETUP ===\n")

        trace.append(f"The video shows {len(self.systems_data)} independent mass-spring oscillator systems.")
        trace.append("Each system consists of a mass attached to a spring fixed to a wall.")
        trace.append("The systems are vertically separated and do not interact with each other.")
        trace.append("All systems start at equilibrium position (x0 = 0) with different initial velocities.\n")

        trace.append("\n=== SYSTEM PARAMETERS ===\n")

        for i, sys_data in enumerate(self.systems_data):
            trace.append(f"{sys_data['label']}:")
            trace.append(f"  - Mass (m{i+1}): {sys_data['m']:.2f} kg")
            trace.append(f"  - Spring constant (k{i+1}): {sys_data['k']:.2f} N/m")
            trace.append(f"  - Initial position (x0{i+1}): {sys_data['x0']:.2f} m")
            trace.append(f"  - Initial velocity (v0{i+1}): {sys_data['v0']:.2f} m/s")
            trace.append("")

        trace.append("\n=== CHRONOLOGICAL OBSERVATION ===\n")

        trace.append("At t = 0s:")
        trace.append("  - All masses are released from equilibrium position with their initial velocities")
        trace.append("  - The masses begin to oscillate independently\n")

        trace.append("During oscillation (t = 0s to 10s):")
        for i, event in enumerate(self.oscillation_events):
            direction = "right" if event['initial_velocity'] > 0 else "left"
            trace.append(f"  - {event['system']} oscillates with initial motion to the {direction}")
            if len(event['positions']) > 0:
                max_pos = np.max(np.abs(event['positions']))
                trace.append(f"    Maximum displacement observed: approximately {max_pos:.2f} m")
        trace.append("")

        trace.append("\n=== AMPLITUDE CALCULATION ===\n")

        trace.append("For a simple harmonic oscillator starting from equilibrium (x0 = 0) with initial velocity v0,")
        trace.append("the amplitude A is given by: A = |v0| / ω")
        trace.append("where ω = sqrt(k/m) is the angular frequency.\n")

        trace.append("Calculating amplitude for each system:")
        for i, sys_data in enumerate(self.systems_data):
            omega = np.sqrt(sys_data['k'] / sys_data['m'])
            amplitude = self.amplitude_data[i]
            trace.append(f"\n{sys_data['label']}:")
            trace.append(f"  - ω{i+1} = sqrt(k{i+1}/m{i+1}) = sqrt({sys_data['k']:.2f}/{sys_data['m']:.2f}) = {omega:.4f} rad/s")
            trace.append(f"  - A{i+1} = |v0{i+1}| / ω{i+1} = |{sys_data['v0']:.2f}| / {omega:.4f} = {amplitude:.4f} m")

        trace.append("\n\n=== AMPLITUDE RATIO CALCULATION ===\n")

        idx1, idx2 = self.ratio_indices
        sys1 = self.systems_data[idx1]
        sys2 = self.systems_data[idx2]
        amp1 = self.amplitude_data[idx1]
        amp2 = self.amplitude_data[idx2]

        trace.append(f"The question asks for the ratio A{idx1+1}/A{idx2+1}:")
        trace.append(f"  - A{idx1+1} = {amp1:.4f} m")
        trace.append(f"  - A{idx2+1} = {amp2:.4f} m")
        trace.append(f"  - A{idx1+1}/A{idx2+1} = {amp1:.4f} / {amp2:.4f} = {self.amplitude_ratio:.4f}")
        trace.append("")

        trace.append("\n=== PHYSICAL INTERPRETATION ===\n")

        if self.amplitude_ratio > 1:
            trace.append(f"{sys1['label']} has a larger amplitude than {sys2['label']}.")
        elif self.amplitude_ratio < 1:
            trace.append(f"{sys2['label']} has a larger amplitude than {sys1['label']}.")
        else:
            trace.append(f"Both systems have equal amplitudes.")

        trace.append(f"\nThe amplitude ratio depends on the initial velocities and the angular frequencies.")
        trace.append(f"A system with higher |v0| or lower ω (due to smaller k/m ratio) will have larger amplitude.")
        trace.append("")

        trace.append("\n=== FINAL ANSWER ===\n")

        trace.append(f"The ratio A{idx1+1}/A{idx2+1} = {self.amplitude_ratio:.2f}")
        trace.append("")

        trace.append("\n=== REASONING SUMMARY ===\n")

        trace.append(f"I observed {len(self.systems_data)} independent mass-spring oscillators in the video. ")
        trace.append(f"Each system was characterized by its mass, spring constant, and initial velocity. ")
        trace.append(f"Using the physical relationship A = |v0| / ω where ω = sqrt(k/m), I calculated ")
        trace.append(f"the amplitude of each system. The ratio A{idx1+1}/A{idx2+1} was then computed as {self.amplitude_ratio:.2f}.")

        return "\n".join(trace)