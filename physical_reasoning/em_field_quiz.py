from manim import *
import numpy as np
import random
import time
import os
import shutil
from pathlib import Path

# Setup directories
Path("questions").mkdir(exist_ok=True)
Path("solutions").mkdir(exist_ok=True)
Path("question_text").mkdir(exist_ok=True)
Path("reasoning_traces").mkdir(exist_ok=True)

DEFAULT_M_RANGE = (0.5, 3.0)
DEFAULT_Q_MAG_RANGE = (0.5, 2.0)
DEFAULT_V_MAG_RANGE = (1.0, 4.0)
DEFAULT_B_Z_RANGE = (0.5, 3.0)
DEFAULT_POS1 = np.array([-2.5, 0.2, 0.0])
DEFAULT_POS2 = np.array([2.5, -0.2, 0.0])
DEFAULT_POS3 = np.array([0.0, 2.0, 0.0]) 
PARTICLE_RADIUS = 0.12
DEFAULT_PHYSICS_DT = 1/120.0

DEFAULT_SIM_DURATION_CHOICE = 3.0
DEFAULT_SIM_POINTS_CHOICE = 150
DEFAULT_CHOICE_PATH_DURATION = 3.0
PREVIEW_ANIM_DURATION = 1.0

ARROW_SCALE = 0.5
LABEL_SCALE_FACTOR = 0.7
CHOICE_SCALE = 0.3
TRACE_STROKE_WIDTH = 2.0
CHOICE_FONT_SIZE = 30

def create_magnetic_field_background(B_z, z_index=-10):
    if abs(B_z) < 0.1: return VGroup()
    symbol = MathTex("\\times", font_size=36) if B_z > 0 else Dot(radius=0.03)
    symbol.set_opacity(0.4).set_color(GRAY)
    rows=int(config.frame_height)+2; cols=int(config.frame_width)+2
    field_marks=VGroup(*[symbol.copy().move_to(x*RIGHT+y*UP) for x in np.arange(-cols/2,cols/2,1.0) for y in np.arange(-rows/2,rows/2,1.0)])
    field_marks.set_z_index(z_index); return field_marks

def simulate_em_collision_multi_particles(t_duration, n_points_out, initial_states, params, dt_physics,
                                        apply_b_field=True, apply_collision=True, flip_force_q=None,
                                        force_scale_factor=1.0, collision_type="elastic", add_drag=False):
    """
    다중 입자 시뮬레이션 함수
    initial_states: [pos1, v1, pos2, v2, pos3, v3, ...] 형태
    params: [q1, m1, r1, q2, m2, r2, q3, m3, r3, ..., B_field] 형태
    """
    n_particles = len(initial_states) // 2
    positions = [np.copy(initial_states[i*2]) for i in range(n_particles)]
    velocities = [np.copy(initial_states[i*2+1]) for i in range(n_particles)]

    charges = [params[i*3] for i in range(n_particles)]
    masses = [max(params[i*3+1], 1e-6) for i in range(n_particles)]
    radii = [params[i*3+2] for i in range(n_particles)]
    B_field = params[-1]
    
    B_field_vec = B_field if apply_b_field else np.array([0.,0.,0.])
    num_steps = int(t_duration / dt_physics)
    save_interval = max(1, num_steps // n_points_out) if n_points_out > 0 else num_steps + 1
    

    coords_lists = [[np.array(pos)] for pos in positions]
    time_points = [0.0]
    current_time = 0.0
    
    for i in range(num_steps):

        for j in range(n_particles):
            force = charges[j] * np.cross(velocities[j], B_field_vec) * force_scale_factor
            if flip_force_q == j + 1:  
                force *= -1

            if add_drag:
                drag_force = -0.3 * velocities[j] * np.linalg.norm(velocities[j]) 
                force += drag_force
            
            velocities[j] += (force / masses[j]) * dt_physics
            positions[j] += velocities[j] * dt_physics

        if apply_collision and n_particles > 1:
            for j in range(n_particles):
                for k in range(j+1, n_particles):
                    r_vec = positions[k] - positions[j]
                    r_mag_sq = np.dot(r_vec, r_vec)
                    min_dist_sq = (radii[j] + radii[k])**2
                    
                    if r_mag_sq <= min_dist_sq and r_mag_sq > 1e-9:
                        r_mag = np.sqrt(r_mag_sq)
                        normal_vec = r_vec / r_mag
                        normal_vec_2d = normal_vec[:2]
                        tangent_vec_2d = np.array([-normal_vec_2d[1], normal_vec_2d[0]])
                        
                        v1_2d = velocities[j][:2]
                        v2_2d = velocities[k][:2]
                        
                        v1n = np.dot(v1_2d, normal_vec_2d)
                        v1t = np.dot(v1_2d, tangent_vec_2d)
                        v2n = np.dot(v2_2d, normal_vec_2d)
                        v2t = np.dot(v2_2d, tangent_vec_2d)
                        
                        m1, m2 = masses[j], masses[k]
                        
                        if collision_type == "elastic":
                            if abs(m1+m2) < 1e-9:
                                new_v1n, new_v2n = v1n, v2n
                            elif abs(m1-m2) < 1e-6:
                                new_v1n, new_v2n = v2n, v1n
                            else:
                                new_v1n = ((m1-m2)*v1n + 2*m2*v2n) / (m1+m2)
                                new_v2n = (2*m1*v1n + (m2-m1)*v2n) / (m1+m2)
                        elif collision_type == "inelastic":
                            if abs(m1+m2) < 1e-9:
                                new_v1n, new_v2n = v1n, v2n
                            else:
                                new_v1n = new_v2n = (m1*v1n + m2*v2n) / (m1+m2) * 0.5
                        elif collision_type == "sticky":
                            if abs(m1+m2) < 1e-9:
                                new_v1n, new_v2n = v1n * 0.1, v2n * 0.1  
                            else:
                                new_v1n = ((m1-m2)*v1n + 2*m2*v2n) / (m1+m2) * 0.3  
                                new_v2n = (2*m1*v1n + (m2-m1)*v2n) / (m1+m2) * 0.3 
                        
                        new_v1_2d = (new_v1n * normal_vec_2d) + (v1t * tangent_vec_2d)
                        new_v2_2d = (new_v2n * normal_vec_2d) + (v2t * tangent_vec_2d)
                        
                        velocities[j] = np.array([new_v1_2d[0], new_v1_2d[1], velocities[j][2]])
                        velocities[k] = np.array([new_v2_2d[0], new_v2_2d[1], velocities[k][2]])

                        overlap = (radii[j] + radii[k]) - r_mag
                        if overlap > 0:
                            correction = overlap * normal_vec * 0.51
                            positions[j] -= correction
                            positions[k] += correction
        
        current_time += dt_physics
        
        if (i + 1) % save_interval == 0 or i == num_steps - 1:
            for j in range(n_particles):
                coords_lists[j].append(np.array(positions[j]))
            time_points.append(current_time)
    
    final_len = len(time_points)
    coords_out = [np.array(coords_list[:final_len]) for coords_list in coords_lists]
    time_points_out = np.array(time_points)
    
    return coords_out, time_points_out

def simulate_em_collision(t_duration, n_points_out, initial_state, params, dt_physics,
                           apply_b_field=True, apply_collision=True, flip_force_q=None):

    pos1, v1, pos2, v2 = [np.copy(arr) for arr in initial_state]
    q1, m1_raw, r1, q2, m2_raw, r2, B_field = params
    m1 = max(m1_raw, 1e-6); m2 = max(m2_raw, 1e-6)
    B_field_vec = B_field if apply_b_field else np.array([0.,0.,0.])
    num_steps = int(t_duration / dt_physics); save_interval = max(1, num_steps // n_points_out) if n_points_out > 0 else num_steps + 1
    coords1_list=[np.array(pos1)]; coords2_list=[np.array(pos2)]; time_points=[0.0]; current_time=0.0
    for i in range(num_steps):
        force1=q1*np.cross(v1, B_field_vec); force2=q2*np.cross(v2, B_field_vec)
        if flip_force_q == 1: force1 *= -1
        if flip_force_q == 2: force2 *= -1
        v1 += (force1/m1) * dt_physics; v2 += (force2/m2) * dt_physics
        pos1 += v1 * dt_physics; pos2 += v2 * dt_physics
        if apply_collision:
            r_vec = pos2 - pos1; r_mag_sq = np.dot(r_vec, r_vec); min_dist_sq = (r1 + r2)**2
            if r_mag_sq <= min_dist_sq and r_mag_sq > 1e-9:
                r_mag = np.sqrt(r_mag_sq); normal_vec = r_vec / r_mag
                normal_vec_2d = normal_vec[:2]; tangent_vec_2d = np.array([-normal_vec_2d[1], normal_vec_2d[0]])
                v1_2d = v1[:2]; v2_2d = v2[:2]
                v1n = np.dot(v1_2d, normal_vec_2d); v1t = np.dot(v1_2d, tangent_vec_2d)
                v2n = np.dot(v2_2d, normal_vec_2d); v2t = np.dot(v2_2d, tangent_vec_2d)
                if abs(m1+m2) < 1e-9: new_v1n = v1n; new_v2n = v2n
                elif abs(m1-m2) < 1e-6: new_v1n = v2n; new_v2n = v1n
                else:
                     new_v1n = ((m1-m2)*v1n + 2*m2*v2n) / (m1+m2); new_v2n = (2*m1*v1n + (m2-m1)*v2n) / (m1+m2)
                new_v1_2d = (new_v1n * normal_vec_2d) + (v1t * tangent_vec_2d); new_v2_2d = (new_v2n * normal_vec_2d) + (v2t * tangent_vec_2d)
                v1 = np.array([new_v1_2d[0], new_v1_2d[1], v1[2]]); v2 = np.array([new_v2_2d[0], new_v2_2d[1], v2[2]])
                overlap = (r1 + r2) - r_mag
                if overlap > 0: correction = overlap * normal_vec * 0.51; pos1 -= correction; pos2 += correction
        current_time += dt_physics
        if (i + 1) % save_interval == 0 or i == num_steps - 1:
            coords1_list.append(np.array(pos1)); coords2_list.append(np.array(pos2)); time_points.append(current_time)

    final_len = len(time_points)
    coords1_out = np.array(coords1_list[:final_len])
    coords2_out = np.array(coords2_list[:final_len])
    time_points_out = np.array(time_points)
    return coords1_out, coords2_out, time_points_out

def create_trajectory_pair(coords1, coords2, color1, color2, stroke_width=TRACE_STROKE_WIDTH):
    path1=VMobject(color=color1,stroke_width=stroke_width); path2=VMobject(color=color2,stroke_width=stroke_width)
    if coords1 is not None and len(coords1)>=2: path1.set_points_as_corners(coords1)
    if coords2 is not None and len(coords2)>=2: path2.set_points_as_corners(coords2)
    dot1=Dot(coords1[0],radius=0.02,color=color1) if coords1 is not None and len(coords1)>0 else Dot(ORIGIN,radius=0.02)
    dot2=Dot(coords2[0],radius=0.02,color=color2) if coords2 is not None and len(coords2)>0 else Dot(ORIGIN,radius=0.02)
    elements=VGroup()
    if path1.has_points(): elements.add(path1)
    if path2.has_points(): elements.add(path2)
    if elements.submobjects:
        if dot1: elements.add(dot1)
        if dot2: elements.add(dot2)
    else:
        elements.add(Text("No Path", font_size=10, color=GREY))
    return elements

def create_trajectory_multi_particles(coords_list, colors, stroke_width=TRACE_STROKE_WIDTH):

    elements = VGroup()
    
    for i, coords in enumerate(coords_list):
        if coords is not None and len(coords) >= 2:
            path = VMobject(color=colors[i], stroke_width=stroke_width)
            path.set_points_as_corners(coords)
            elements.add(path)

            dot = Dot(coords[0], radius=0.02, color=colors[i])
            elements.add(dot)
    
    if not elements.submobjects:
        elements.add(Text("No Path", font_size=10, color=GREY))
    
    return elements

def adjust_em_parameters(difficulty):
    difficulty = max(0, min(difficulty, 9))
    diff_factor = difficulty / 9.0
    b_z_min = DEFAULT_B_Z_RANGE[0] * 1.1
    b_z_max = b_z_min + (DEFAULT_B_Z_RANGE[1] - b_z_min) * (0.6 + 0.4 * diff_factor)
    b_z_range = (b_z_min, b_z_max)
    v_mag_min = DEFAULT_V_MAG_RANGE[0] * 1.1
    v_mag_max = v_mag_min + (DEFAULT_V_MAG_RANGE[1] - v_mag_min) * (0.8 + 0.2 * diff_factor)
    v_mag_range = (v_mag_min, v_mag_max)
    prob_opposite_charge = max(0.1, 0.9 - 0.4 * diff_factor)
    prob_equal_mass = max(0.1, 0.8 - 0.6 * diff_factor)
    m_min = DEFAULT_M_RANGE[0]; m_max = m_min + (DEFAULT_M_RANGE[1] - m_min) * (0.6 + 0.4 * diff_factor)
    m_range = (m_min, m_max)
    q_mag_min = DEFAULT_Q_MAG_RANGE[0]; q_mag_max = q_mag_min + (DEFAULT_Q_MAG_RANGE[1] - q_mag_min) * (0.6 + 0.4 * diff_factor)
    q_mag_range = (q_mag_min, q_mag_max)
    print(f"Adjusted Ranges (Diff={difficulty}):")
    print(f"  Bz: {b_z_range}, Vmag: {v_mag_range}, P(opp q): {prob_opposite_charge:.2f}, P(eq m): {prob_equal_mass:.2f}")
    print(f"  m: {m_range}, q_mag: {q_mag_range}")
    return b_z_range, v_mag_range, prob_opposite_charge, prob_equal_mass, m_range, q_mag_range

class EMCollisionTrajectoryQuizRefactoredDiverse(Scene):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Set seed for reproducibility
        self.seed = int(os.environ.get('MANIM_SEED', time.time()))
        self.difficulty = int(os.environ.get('MANIM_DIFFICULTY', 5))
        self.particle_count = int(os.environ.get('MANIM_PARTICLE_COUNT', 2))

        if self.particle_count not in [1, 2, 3]:
            print(f"Warning: particle_count {self.particle_count} is not valid. Using default value 2.")
            self.particle_count = 2

        random.seed(self.seed)
        np.random.seed(self.seed)

        # Store for reasoning trace
        self.simulation_events = []
        self.particle_params = []
        self.field_params = {}
        self.correct_answer = ""

    def construct(self):
        seed = self.seed
        difficulty = self.difficulty
        particle_count = self.particle_count
        
        b_z_range, v_mag_range, prob_opposite, prob_equal_mass, m_range, q_mag_range = adjust_em_parameters(difficulty)

        masses = []
        charges = []
        positions = []
        velocities = []
        colors = []

        base_positions = [DEFAULT_POS1, DEFAULT_POS2, DEFAULT_POS3]
        
        for i in range(particle_count):

            m = random.uniform(*m_range)
            m = max(0.1, m)
            masses.append(m)
            
            q_mag = random.uniform(*q_mag_range)
            if i == 0:
                q_sign = 1 if random.random() < 0.5 else -1
            else:
                q_sign = -q_sign if random.random() < prob_opposite else q_sign
            q = q_sign * q_mag
            if abs(q) < 1e-3: q = np.sign(q + 1e-9) * 0.1
            charges.append(q)

            pos = base_positions[i] + np.array([random.uniform(-0.1,0.1), random.uniform(-0.1,0.1), 0])
            positions.append(pos)

            v_mag = random.uniform(*v_mag_range)
            if i == 0:
                angle_offset = random.uniform(-PI/12, PI/12)
                v = v_mag * np.array([np.cos(angle_offset), np.sin(angle_offset), 0])
            elif i == 1:
                angle_offset = random.uniform(-PI/12, PI/12)
                v = v_mag * np.array([-np.cos(angle_offset), -np.sin(angle_offset), 0])
            else:  # i == 2
                angle_offset = random.uniform(-PI/12, PI/12)
                v = v_mag * np.array([np.sin(angle_offset), -np.cos(angle_offset), 0])
            velocities.append(v)

            color = RED if q > 0 else BLUE
            colors.append(color)

        B_z = random.uniform(*b_z_range)
        if abs(B_z) < 0.2: B_z = np.sign(B_z + 1e-9) * 0.2
        if random.random() < 0.5: B_z *= -1
        B_field = np.array([0.,0.,B_z])

        # Store field parameters for reasoning trace
        self.field_params = {
            'B_z': B_z,
            'B_field': B_field,
            'direction': 'into page' if B_z > 0 else 'out of page'
        }
        
        physics_dt = DEFAULT_PHYSICS_DT
        sim_duration = DEFAULT_SIM_DURATION_CHOICE
        sim_points = DEFAULT_SIM_POINTS_CHOICE
        choice_path_duration = min(DEFAULT_CHOICE_PATH_DURATION, sim_duration)
        
        print(f"--- EM Quiz Setup (Seed: {seed}, Difficulty: {difficulty}, Particles: {particle_count}) ---")
        for i in range(particle_count):
            print(f"Particle {i+1}: q={charges[i]:.2f}, m={masses[i]:.2f}, v=[{velocities[i][0]:.1f},{velocities[i][1]:.1f}]")
            # Store particle parameters for reasoning trace
            self.particle_params.append({
                'particle_num': i + 1,
                'charge': charges[i],
                'mass': masses[i],
                'initial_velocity': velocities[i],
                'initial_position': positions[i],
                'color': 'red' if charges[i] > 0 else 'blue'
            })
        print(f"Field: B_z={B_z:.2f}")
        print(f"--------------------")

        params = []
        for i in range(particle_count):
            params.extend([charges[i], masses[i], PARTICLE_RADIUS])
        params.append(B_field)

        initial_states = []
        for i in range(particle_count):
            initial_states.extend([positions[i], velocities[i]])

        title = Text(f"Problem: Motion of {particle_count} Particle(s) in Magnetic Field", font_size=36).to_edge(UP)
        background_field = create_magnetic_field_background(B_z)
        b_label = MathTex(f"B = {B_z:.1f} \\hat{{k}}", font_size=30).to_corner(DR)

        particles = []
        labels = []
        arrows = []
        
        for i in range(particle_count):
            particle = Dot(point=positions[i], radius=PARTICLE_RADIUS, color=colors[i], fill_opacity=1)
            particles.append(particle)
            
            label = MathTex(f"q_{i+1}={charges[i]:.1f}, m_{i+1}={masses[i]:.1f}", font_size=20)
            label.next_to(particle, UR, buff=SMALL_BUFF*0.5)
            labels.append(label)
            
            arrow = Arrow(particle.get_center(), particle.get_center() + velocities[i] * ARROW_SCALE, 
                         buff=PARTICLE_RADIUS, color=WHITE, stroke_width=4, max_tip_length_to_length_ratio=0.3)
            arrows.append(arrow)
        
        initial_static_group = VGroup(title, background_field, b_label)
        self.add(initial_static_group)
        for i in range(particle_count):
            self.add(particles[i], labels[i], arrows[i])
        self.wait(1)

        print("Calculating Correct Trajectory (B Field + Elastic Collision)...")
        coords_correct, time_pts = simulate_em_collision_multi_particles(sim_duration, sim_points, initial_states, params, physics_dt)

        # Store simulation event for reasoning trace
        self.simulation_events.append({
            'type': 'correct_trajectory',
            'description': 'Correct motion with B field and elastic collision',
            'coordinates': coords_correct,
            'time_points': time_pts,
            'sim_duration': sim_duration
        })
        
        print("Calculating Distractor 1 (No B Field, Elastic Collision)...")
        coords_d1, _ = simulate_em_collision_multi_particles(sim_duration, sim_points, initial_states, params, physics_dt, apply_b_field=False, apply_collision=True)
        
        print("Calculating Distractor 2 (No Forces - Straight Lines)...")
        coords_d2, _ = simulate_em_collision_multi_particles(sim_duration, sim_points, initial_states, params, physics_dt, apply_b_field=False, apply_collision=False)
        
        print("Calculating Distractor 3 (Flipped Force q2)...")
        flip_particle = min(2, particle_count)  
        coords_d3, _ = simulate_em_collision_multi_particles(sim_duration, sim_points, initial_states, params, physics_dt, apply_b_field=True, apply_collision=True, flip_force_q=flip_particle)
        
        print("Calculating Distractor 4 (Inelastic Collision)...")
        coords_d4, _ = simulate_em_collision_multi_particles(sim_duration, sim_points, initial_states, params, physics_dt, apply_b_field=True, apply_collision=True, collision_type="inelastic")
        
        print("Calculating Distractor 5 (Sticky Collision)...")
        coords_d5, _ = simulate_em_collision_multi_particles(sim_duration, sim_points, initial_states, params, physics_dt, apply_b_field=True, apply_collision=True, collision_type="sticky")
        
        print("Calculating Distractor 6 (Reduced Force + Drag)...")
        coords_d6, _ = simulate_em_collision_multi_particles(sim_duration, sim_points, initial_states, params, physics_dt, apply_b_field=True, apply_collision=True, force_scale_factor=0.3, add_drag=True)  # 더 극적인 감소
        
        print("Calculating Distractor 7 (Stronger B Field)...")

        params_stronger_b = params.copy()
        params_stronger_b[-1] = B_field * 3.0
        coords_d7, _ = simulate_em_collision_multi_particles(sim_duration, sim_points, initial_states, params_stronger_b, physics_dt, apply_b_field=True, apply_collision=True)
        
        print("Calculating Distractor 8 (Opposite B Field)...")

        params_opposite_b = params.copy()
        params_opposite_b[-1] = -B_field  
        coords_d8, _ = simulate_em_collision_multi_particles(sim_duration, sim_points, initial_states, params_opposite_b, physics_dt, apply_b_field=True, apply_collision=True)
        
        print("Calculating Distractor 9 (Heavy Drag)...")
        coords_d9, _ = simulate_em_collision_multi_particles(sim_duration, sim_points, initial_states, params, physics_dt, apply_b_field=True, apply_collision=True, add_drag=True, force_scale_factor=0.6)  # 더 극적인 드래그
        
        print("Calculating Distractor 10 (No Collision, B Field Only)...")
        coords_d10, _ = simulate_em_collision_multi_particles(sim_duration, sim_points, initial_states, params, physics_dt, apply_b_field=True, apply_collision=False)

        preview_index = np.searchsorted(time_pts, PREVIEW_ANIM_DURATION, side='right')
        preview_index = max(2, preview_index)
        preview_index = min(preview_index, min(len(coords) for coords in coords_correct))
        
        coords_preview = [coords[:preview_index] for coords in coords_correct]
        paths_preview = [VMobject().set_points_as_corners(coords) for coords in coords_preview]

        print(f"  Playing {PREVIEW_ANIM_DURATION:.1f}s preview animation...")
        if preview_index > 1 and all(path.has_points() for path in paths_preview):
            animations = []
            for i in range(particle_count):
                animations.append(MoveAlongPath(particles[i], paths_preview[i]))
            for arrow in arrows:
                animations.append(FadeOut(arrow))
            for label in labels:
                animations.append(FadeOut(label))
            
            self.play(*animations, run_time=PREVIEW_ANIM_DURATION, rate_func=linear)
        else:
            print("  Warning: Not enough points for preview animation.")
            animations = []
            for arrow in arrows:
                animations.append(FadeOut(arrow))
            for label in labels:
                animations.append(FadeOut(label))
            self.play(*animations, run_time=0.5)
            self.wait(PREVIEW_ANIM_DURATION - 0.5)
        self.wait(0.5)

        question = Text("Which set of paths best represents the motion?", font_size=32).to_edge(DOWN)
        self.play(Write(question))
        self.wait(0.5)

        n_points_choice = np.searchsorted(time_pts, choice_path_duration, side='right')
        n_points_choice = max(2, n_points_choice)

        all_coords = [coords_correct, coords_d1, coords_d2, coords_d3, coords_d4, coords_d5, coords_d6, coords_d7, coords_d8, coords_d9, coords_d10]
        min_len = float('inf')
        for coords_set in all_coords:
            for coords in coords_set:
                min_len = min(min_len, len(coords))
        
        n_points_choice = min(n_points_choice, min_len)
        print(f"Using {n_points_choice} points for choice paths (up to t={time_pts[min(n_points_choice-1, len(time_pts)-1)]:.2f}s)")

        correct_choice = create_trajectory_multi_particles([coords[:n_points_choice] for coords in coords_correct], colors)
        distractor1 = create_trajectory_multi_particles([coords[:n_points_choice] for coords in coords_d1], colors)
        distractor2 = create_trajectory_multi_particles([coords[:n_points_choice] for coords in coords_d2], colors)
        distractor3 = create_trajectory_multi_particles([coords[:n_points_choice] for coords in coords_d3], colors)
        distractor4 = create_trajectory_multi_particles([coords[:n_points_choice] for coords in coords_d4], colors)
        distractor5 = create_trajectory_multi_particles([coords[:n_points_choice] for coords in coords_d5], colors)
        distractor6 = create_trajectory_multi_particles([coords[:n_points_choice] for coords in coords_d6], colors)
        distractor7 = create_trajectory_multi_particles([coords[:n_points_choice] for coords in coords_d7], colors)
        distractor8 = create_trajectory_multi_particles([coords[:n_points_choice] for coords in coords_d8], colors)
        distractor9 = create_trajectory_multi_particles([coords[:n_points_choice] for coords in coords_d9], colors)
        distractor10 = create_trajectory_multi_particles([coords[:n_points_choice] for coords in coords_d10], colors)

        all_choices = [correct_choice, distractor1, distractor2, distractor3, distractor4, distractor5, distractor6, distractor7, distractor8, distractor9, distractor10]
        choice_descriptions = [
            "Correct (B Field + Elastic)",
            "No B Field",
            "No Forces",
            "Flipped Force",
            "Inelastic Collision",
            "Sticky Collision", 
            "Reduced Force + Drag",
            "Stronger B Field",
            "Opposite B Field",
            "Heavy Drag",
            "No Collision, B Field Only"
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
        
        for i, (original_index, choice_vgroup) in enumerate(indexed_choices):
            choice_vgroup.scale(CHOICE_SCALE)
            label = Tex(f"{choice_labels_text[i]}", font_size=36).next_to(choice_vgroup, DOWN, buff=SMALL_BUFF * 1.5)
            choice_vg = VGroup(choice_vgroup, label)
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
            
            self.play(FadeOut(initial_static_group), FadeOut(question))
            for particle in particles:
                self.play(FadeOut(particle))
            self.play(FadeIn(final_choice_display_group))
        else:
            print("ERROR: Could not generate all 5 choice VGroups.")
            self.play(FadeOut(initial_static_group), FadeOut(question), Write(Text("Error generating choices.", color=RED)))
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

        # Store the correct answer
        self.correct_answer = correct_label

        # Store choice information for reasoning trace
        self.choice_info = {
            'correct_label': correct_label,
            'correct_index': new_correct_index,
            'all_choices': selected_descriptions,
            'choice_labels': choice_labels_text[:len(indexed_choices)]
        }

        self.wait(5)

        # Save solution
        with open(f"solutions/em_field_quiz_p{self.particle_count}_d{self.difficulty}_seed{self.seed}.txt", "w") as f:
            f.write(correct_label)

        # Save question text
        question_text_content = (
            f"Problem: Motion of {particle_count} Particle(s) in Magnetic Field\n\n"
            f"A magnetic field B = {B_z:.1f} k is present.\n"
            f"Particles have the following properties:\n"
        )
        for i, param in enumerate(self.particle_params):
            question_text_content += f"  Particle {i+1}: charge q{i+1}={param['charge']:.1f}, mass m{i+1}={param['mass']:.1f}\n"
        question_text_content += "\nWhich set of paths best represents the motion?\n"
        question_text_content += "Choose from options A, B, C, D, or E.\n"

        with open(f"question_text/em_field_quiz_p{self.particle_count}_d{self.difficulty}_seed{self.seed}.txt", "w") as f:
            f.write(question_text_content)

        # Generate and save reasoning trace
        reasoning_trace = self.generate_reasoning_trace()
        with open(f"reasoning_traces/em_field_quiz_p{self.particle_count}_d{self.difficulty}_seed{self.seed}.txt", "w") as f:
            f.write(reasoning_trace)

    def generate_reasoning_trace(self):
        """Generate a detailed reasoning trace describing the electromagnetic field problem"""

        trace = []
        trace.append("=== ELECTROMAGNETIC FIELD PROBLEM ANALYSIS ===\n")

        # Describe the setup
        trace.append("=== PROBLEM SETUP ===\n")
        trace.append(f"Number of particles: {self.particle_count}")
        trace.append(f"Difficulty level: {self.difficulty}")
        trace.append(f"Random seed: {self.seed}")
        trace.append("")

        # Describe the magnetic field
        trace.append("\n=== MAGNETIC FIELD CONFIGURATION ===\n")
        B_z = self.field_params['B_z']
        direction = self.field_params['direction']
        trace.append(f"Magnetic field strength: B_z = {B_z:.2f} T")
        trace.append(f"Field direction: {direction} (z-direction)")
        trace.append(f"Field vector: B = {B_z:.2f} k")
        trace.append("")

        # Describe each particle's initial conditions
        trace.append("\n=== PARTICLE INITIAL CONDITIONS ===\n")
        for param in self.particle_params:
            trace.append(f"Particle {param['particle_num']}:")
            trace.append(f"  - Charge: q{param['particle_num']} = {param['charge']:.2f} C")
            trace.append(f"  - Mass: m{param['particle_num']} = {param['mass']:.2f} kg")
            trace.append(f"  - Initial velocity: v{param['particle_num']} = [{param['initial_velocity'][0]:.2f}, {param['initial_velocity'][1]:.2f}, {param['initial_velocity'][2]:.2f}] m/s")
            trace.append(f"  - Initial position: [{param['initial_position'][0]:.2f}, {param['initial_position'][1]:.2f}, {param['initial_position'][2]:.2f}]")
            trace.append(f"  - Charge type: {param['color']} (positive charge)" if param['charge'] > 0 else f"  - Charge type: {param['color']} (negative charge)")
            trace.append("")

        # Explain the physics
        trace.append("\n=== PHYSICAL PRINCIPLES ===\n")
        trace.append("When charged particles move through a magnetic field, they experience the Lorentz force:")
        trace.append("  F = q(v × B)")
        trace.append("")
        trace.append("Key observations:")
        trace.append("1. The Lorentz force is perpendicular to both velocity and magnetic field")
        trace.append("2. This causes particles to move in curved paths (circular or helical motion)")
        trace.append(f"3. With B pointing {direction}, particles will curve based on their charge sign")
        trace.append("4. Positive charges curve in one direction, negative charges curve in the opposite direction")
        trace.append("5. The radius of curvature depends on mass, charge, velocity, and field strength: r = mv/(qB)")
        trace.append("")

        # Describe the motion
        trace.append("\n=== MOTION ANALYSIS ===\n")
        trace.append("Expected motion for each particle:")
        for param in self.particle_params:
            q = param['charge']
            m = param['mass']
            v_mag = np.linalg.norm(param['initial_velocity'])
            if abs(B_z) > 0.01:
                radius = (m * v_mag) / (abs(q * B_z))
                trace.append(f"Particle {param['particle_num']}:")
                trace.append(f"  - Radius of curvature: r ≈ {radius:.2f} units")
                if (q > 0 and B_z > 0) or (q < 0 and B_z < 0):
                    trace.append(f"  - Curves counterclockwise (when viewed from above)")
                else:
                    trace.append(f"  - Curves clockwise (when viewed from above)")
            trace.append("")

        # Describe collision physics (if multiple particles)
        if self.particle_count > 1:
            trace.append("\n=== COLLISION DYNAMICS ===\n")
            trace.append("When particles collide, elastic collision physics applies:")
            trace.append("1. Conservation of momentum: m1*v1 + m2*v2 = m1*v1' + m2*v2'")
            trace.append("2. Conservation of kinetic energy (elastic collision)")
            trace.append("3. The collision changes velocity directions but particles continue to experience Lorentz force")
            trace.append("4. Post-collision trajectories are curved paths from the new velocities")
            trace.append("")

        # Describe the video chronology
        trace.append("\n=== VIDEO CHRONOLOGY ===\n")
        trace.append("The video shows the following sequence:")
        trace.append(f"1. Initial setup: {self.particle_count} particle(s) with initial velocities shown as arrows")
        trace.append(f"2. Preview animation: Particles move for {PREVIEW_ANIM_DURATION:.1f} seconds showing the start of their trajectories")
        trace.append(f"3. Question display: Asked to identify which trajectory matches the physical laws")
        trace.append(f"4. Multiple choice options: Showing different possible trajectories")
        trace.append("")

        # Explain the distractors
        trace.append("\n=== UNDERSTANDING THE CHOICES ===\n")
        trace.append("The problem presents multiple trajectory options:")
        if hasattr(self, 'choice_info'):
            for i, desc in enumerate(self.choice_info['all_choices']):
                label = self.choice_info['choice_labels'][i] if i < len(self.choice_info['choice_labels']) else str(i)
                trace.append(f"  - Option {label}: {desc}")
        trace.append("")
        trace.append("Common incorrect trajectories might show:")
        trace.append("  - Straight lines (ignoring magnetic force)")
        trace.append("  - Wrong curvature direction (sign error in Lorentz force)")
        trace.append("  - Incorrect collision behavior (inelastic or sticky collisions)")
        trace.append("  - Wrong field strength effects (too strong or too weak curvature)")
        trace.append("")

        # Reasoning process
        trace.append("\n=== REASONING PROCESS ===\n")
        trace.append("To solve this problem, I need to:")
        trace.append("1. Calculate the expected trajectory using F = q(v × B)")
        trace.append("2. Consider the direction of the Lorentz force for each particle's charge")
        trace.append("3. Account for elastic collisions if particles interact")
        trace.append("4. Compare the preview animation with the calculated trajectories")
        trace.append("5. Identify which choice matches the correct physics")
        trace.append("")

        # Final answer
        trace.append("\n=== FINAL ANSWER ===\n")
        trace.append(f"The correct answer is: {self.correct_answer}")
        trace.append("")
        trace.append("This trajectory correctly represents:")
        trace.append("- Lorentz force causing curved paths")
        trace.append(f"- Proper curvature direction for the {direction} magnetic field")
        trace.append("- Correct particle interactions (elastic collisions)")
        trace.append("- Conservation of energy and momentum throughout the motion")
        trace.append("")

        trace.append("\n=== SUMMARY ===\n")
        trace.append(f"This problem tests understanding of charged particle motion in magnetic fields. ")
        trace.append(f"With {self.particle_count} particle(s) in a magnetic field of {B_z:.2f} T, ")
        trace.append(f"the particles experience Lorentz forces causing curved trajectories. ")
        if self.particle_count > 1:
            trace.append(f"Elastic collisions between particles add complexity to the motion. ")
        trace.append(f"By applying electromagnetic theory and collision physics, ")
        trace.append(f"the correct answer is option {self.correct_answer}.")

        return "\n".join(trace)
