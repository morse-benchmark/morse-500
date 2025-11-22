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

# --- Configuration Constants ---
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

# --- Manim Global Config ---
config.media_dir = "manim_output"
config.verbosity = "WARNING"
config.pixel_height = 1080
config.pixel_width = 1920
config.frame_rate = 30
config.preview = False


def oscillator_derivs(t, state_X, m, k, b):
    """Differential equation for damped harmonic oscillator."""
    X, V = state_X
    aX = (-k * X - b * V) / m if m > 1e-6 else 0
    return [V, aX]


def simulate_oscillator_for_X(t_duration, initial_state_X, params_mkb):
    """
    Simulates the oscillator.
    Returns (t_pts, X_pts, V_pts) where X is displacement from equilibrium.
    """
    m, k, b = params_mkb
    t_span = [0, t_duration]
    n_points = int(t_duration * SIM_POINTS_PER_SEC) + 1
    t_eval = np.linspace(t_span[0], t_span[1], n_points)

    if not np.all(np.isfinite(initial_state_X)):
        initial_state_X = [0.0, 0.0]

    try:
        sol = solve_ivp(
            oscillator_derivs,
            t_span,
            initial_state_X,
            args=(m, k, b),
            t_eval=t_eval,
            rtol=1e-6,
            atol=1e-8,
            dense_output=False,
        )

        t_pts = sol.t
        if sol.status != 0 or len(t_pts) < 2:
            # Fallback for solver failure
            X0, V0 = initial_state_X
            return np.array([0, 1e-3]), np.array([X0, X0]), np.array([V0, V0])

        X_pts = sol.y[0]
        V_pts = sol.y[1]

        # Filter out potential NaNs/Infs
        valid_indices = np.where(np.isfinite(X_pts) & np.isfinite(V_pts))[0]
        if len(valid_indices) < len(X_pts):
            if len(valid_indices) > 0:
                max_valid_idx = valid_indices[-1]
                t_pts = t_pts[: max_valid_idx + 1]
                X_pts = X_pts[: max_valid_idx + 1]
                V_pts = V_pts[: max_valid_idx + 1]
            else:
                X0, V0 = initial_state_X
                return np.array([0, 1e-3]), np.array([X0, X0]), np.array([V0, V0])

        return t_pts, X_pts, V_pts
    except Exception as e:
        print(f"Error during solve_ivp for X: {e}")
        X0, V0 = initial_state_X
        return np.array([0, 1e-3]), np.array([X0, X0]), np.array([V0, V0])


def create_single_spring_visual(
    start_point,
    end_point,
    bumps=SPRING_BUMPS,
    radius=SPRING_RADIUS,
    color=WHITE,
    stroke_width=2,
):
    """Generates a zig-zag spring mobject between two points."""
    start = np.array(start_point)
    end = np.array(end_point)
    length = np.linalg.norm(end - start)
    direction_vec = end - start

    if length < 1e-6:
        return Line(start, end, color=color, stroke_width=stroke_width)

    direction = normalize(direction_vec)

    # Determine perpendicular direction for the zig-zag
    if abs(direction[0]) < 1e-6 and abs(direction[1]) < 1e-6:
        perp_direction_for_bumps = RIGHT
    elif abs(direction[1]) > 0.999:
        perp_direction_for_bumps = RIGHT
    else:
        perp_direction_for_bumps = normalize(np.cross(direction, OUT))

    def spring_func(t):
        return (
            start
            + t * direction_vec
            + perp_direction_for_bumps * radius * np.sin(bumps * TAU * t)
        )

    num_segments = bumps * 20
    points = [spring_func(t) for t in np.linspace(0, 1, num_segments + 1)]
    return VMobject(color=color, stroke_width=stroke_width).set_points_as_corners(
        points
    )


class DampedOscillatorsRestingPositionQuiz(Scene):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Set random seed for reproducibility
        self.seed = int(time.time())
        random.seed(self.seed)
        np.random.seed(self.seed)

        # Store for reasoning trace
        self.system_events = []
        self.num_systems = int(os.environ.get("MANIM_OBJECT_COUNT", 4))

    def generate_distinct_offsets(
        self, num_offsets, min_val, max_val, min_abs_separation=0.25
    ):
        offsets = []
        abs_offsets = []
        attempts = 0
        max_attempts = 100

        while len(offsets) < num_offsets and attempts < max_attempts:
            attempts += 1
            new_offset = random.uniform(min_val, max_val)

            # Avoid too close to zero if needed (optional logic from original)
            if abs(new_offset) < min_abs_separation / 2 and num_offsets > 1:
                if random.random() < 0.5:
                    continue

            new_abs_offset = abs(new_offset)
            too_close = False

            # Check against absolute offsets (for distinct difficulty)
            for existing_abs_offset in abs_offsets:
                if abs(new_abs_offset - existing_abs_offset) < min_abs_separation:
                    too_close = True
                    break

            if not too_close:
                # Check against actual offsets
                for existing_offset in offsets:
                    if abs(new_offset - existing_offset) < min_abs_separation / 2:
                        too_close = True
                        break

                if not too_close:
                    offsets.append(new_offset)
                    abs_offsets.append(new_abs_offset)
                    attempts = 0

        # Fallback if distinct generation failed
        if len(offsets) < num_offsets:
            print(
                "Warning: Could not generate fully distinct offsets. Using potentially close values."
            )
            while len(offsets) < num_offsets:
                offset_val = random.uniform(min_val, max_val)
                # Ensure not exactly zero to avoid division issues elsewhere if any
                if abs(offset_val) < 0.15 and num_offsets > 1:
                    offset_val = (
                        np.sign(offset_val) * 0.15
                        if offset_val != 0
                        else random.choice([-1, 1]) * 0.15
                    )
                offsets.append(offset_val)

        return offsets

    def construct(self):
        num_systems = self.num_systems

        # Validate number of systems
        if num_systems not in [3, 4, 5]:
            print(
                f"Warning: object_count {num_systems} is not valid. Using default value 4."
            )
            num_systems = 4
            self.num_systems = 4

        scene_title = Text(
            f"Damped Oscillators - Resting Positions ({num_systems} Systems)",
            font_size=32,
        ).to_edge(UP)

        # Setup Axis
        max_possible_x = max(abs(val) for val in X_OFFSET_RANGE) + 0.5
        x_axis_bnd = min(np.ceil(max_possible_x * 1.2) + 0.5, 4.5)
        x_axis_bnd = max(x_axis_bnd, 2.5)
        x_tick_freq = 0.5 if x_axis_bnd <= 3.5 else 1.0

        x_axis = NumberLine(
            x_range=[-x_axis_bnd, x_axis_bnd, x_tick_freq],
            length=self.camera.frame_width - 2.5,
            color=WHITE,
            include_numbers=True,
            label_direction=DOWN,
            font_size=18,
            stroke_width=2,
        ).move_to(UP * 2.5)

        systems_data = []
        system_mobjects = VGroup()

        # Select Labels and Colors
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

        x_offsets = self.generate_distinct_offsets(
            num_systems, X_OFFSET_RANGE[0], X_OFFSET_RANGE[1]
        )

        # Initialize Systems
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
            # All masses start at x=0 on the ruler
            initial_x_actual = 0.0
            # Simulation variable X is displacement from equilibrium
            X0 = initial_x_actual - x_offset

            V0 = random.uniform(*DEFAULT_V0_RANGE) * random.choice([-1, 1])
            if abs(X0) < 0.2 and abs(V0) < 0.8:
                V0 = (
                    np.sign(V0) * random.uniform(1.2, DEFAULT_V0_RANGE[1])
                    if V0 != 0
                    else random.choice([-1, 1])
                    * random.uniform(1.2, DEFAULT_V0_RANGE[1])
                )

            params_mkb = [m, k, b]
            initial_state_X = [X0, V0]

            print(
                f"--- System {sys_label_text} (Seed: {self.seed}) ---"
            )  # FIXED: Used self.seed
            print(
                f"Params: m={m:.2f}, k={k:.2f}, b={b:.2f} (zeta={damping_factor:.3f}), x_offset={x_offset:.2f}"
            )
            print(
                f"Initial X: X0={X0:.2f}, V0={V0:.2f} (Actual x0={initial_x_actual:.2f})"
            )

            # Visual Elements
            system_group = VGroup()
            wall_x_coord = x_axis.n2p(-x_axis_bnd)[0] - 0.5
            wall = Line(
                UP * WALL_HEIGHT / 2,
                DOWN * WALL_HEIGHT / 2,
                color=GRAY_C,
                stroke_width=3,
            )
            wall.move_to([wall_x_coord, y_pos, 0])

            spring_anchor_point = wall.get_center()
            mass_initial_point_on_axis = x_axis.n2p(initial_x_actual)

            mass_obj = Square(
                side_length=MASS_SIDE_LENGTH, color=mass_color, fill_opacity=0.8
            )
            mass_obj.move_to([mass_initial_point_on_axis[0], y_pos, 0])

            spring_obj = create_single_spring_visual(
                spring_anchor_point,
                mass_obj.get_left(),
                color=spring_color,
                bumps=SPRING_BUMPS,
                radius=SPRING_RADIUS,
            )
            label = Tex(sys_label_text, font_size=24).next_to(wall, LEFT, buff=0.2)

            system_group.add(wall, spring_obj, mass_obj, label)
            system_mobjects.add(system_group)

            t_sim, X_sim, _ = simulate_oscillator_for_X(
                SIM_DURATION, initial_state_X, params_mkb
            )

            systems_data.append(
                {
                    "label_text": sys_label_text,
                    "m": m,
                    "k": k,
                    "b": b,
                    "x_offset": x_offset,
                    "X0": X0,
                    "V0": V0,
                    "initial_x_actual": initial_x_actual,
                    "mass_obj": mass_obj,
                    "spring_obj": spring_obj,
                    "spring_anchor": spring_anchor_point,
                    "y_pos": y_pos,
                    "t_sim": t_sim,
                    "X_sim": X_sim,
                    "anim_time": 0.0,
                    "spring_color": spring_color,
                    "damping_factor": damping_factor,
                }
            )

            # Record event for reasoning trace
            self.system_events.append(
                {
                    "label": sys_label_text,
                    "mass": m,
                    "spring_constant": k,
                    "damping_coefficient": b,
                    "damping_factor": damping_factor,
                    "x_offset": x_offset,
                    "initial_displacement": X0,
                    "initial_velocity": V0,
                    "distance_from_zero": abs(x_offset),
                }
            )

        self.play(
            Write(scene_title),
            Create(x_axis),
            FadeIn(system_mobjects, lag_ratio=0.1),
            run_time=2.0,
        )
        self.wait(0.5)

        # Animation Loop
        for i in range(num_systems):
            sys_data = systems_data[i]

            # Note: We must use default argument system_index=i to capture the value in the loop
            def mass_updater_func(mobj, dt, system_index=i):
                s_data = systems_data[system_index]
                s_data["anim_time"] += dt
                current_X = np.interp(
                    min(s_data["anim_time"], s_data["t_sim"][-1]),
                    s_data["t_sim"],
                    s_data["X_sim"],
                )
                actual_x_pos = current_X + s_data["x_offset"]
                actual_x_pos = np.clip(actual_x_pos, -3.5, 3.5)

                screen_x_coord = x_axis.n2p(actual_x_pos)[0]
                mobj.move_to([screen_x_coord, s_data["y_pos"], 0])

            sys_data["mass_obj"].add_updater(mass_updater_func)

            def spring_updater_func(mobj, system_index=i):
                s_data = systems_data[system_index]
                # Calculate position based on mass logic (redundant calc but safe for independent updater)
                current_X = np.interp(
                    min(s_data["anim_time"], s_data["t_sim"][-1]),
                    s_data["t_sim"],
                    s_data["X_sim"],
                )
                actual_x_pos = current_X + s_data["x_offset"]
                actual_x_pos = np.clip(actual_x_pos, -3.5, 3.5)

                screen_x_coord = x_axis.n2p(actual_x_pos)[0]
                mobj.become(
                    create_single_spring_visual(
                        s_data["spring_anchor"],
                        [screen_x_coord - MASS_SIDE_LENGTH / 2, s_data["y_pos"], 0],
                        color=s_data["spring_color"],
                        bumps=SPRING_BUMPS,
                        radius=SPRING_RADIUS,
                    )
                )

            sys_data["spring_obj"].add_updater(spring_updater_func)

        self.wait(ANIMATION_DURATION)

        # Clean up updaters and snap to final state
        for sys_data in systems_data:
            sys_data["mass_obj"].clear_updaters()
            sys_data["spring_obj"].clear_updaters()

            final_anim_time = sys_data["anim_time"]
            final_X_at_anim_end = np.interp(
                min(final_anim_time, sys_data["t_sim"][-1]),
                sys_data["t_sim"],
                sys_data["X_sim"],
            )
            final_actual_x_pos_at_anim_end = final_X_at_anim_end + sys_data["x_offset"]
            final_actual_x_pos_at_anim_end = np.clip(
                final_actual_x_pos_at_anim_end, -3.5, 3.5
            )

            screen_x_coord = x_axis.n2p(final_actual_x_pos_at_anim_end)[0]
            sys_data["mass_obj"].move_to([screen_x_coord, sys_data["y_pos"], 0])
            sys_data["spring_obj"].become(
                create_single_spring_visual(
                    sys_data["spring_anchor"],
                    [screen_x_coord - MASS_SIDE_LENGTH / 2, sys_data["y_pos"], 0],
                    color=sys_data["spring_color"],
                    bumps=SPRING_BUMPS,
                    radius=SPRING_RADIUS,
                )
            )

        self.wait(0.5)

        # Generate Ranking Data
        ranking_data = []
        for sys_data in systems_data:
            distance_from_zero = abs(sys_data["x_offset"])
            ranking_data.append(
                {
                    "label": sys_data["label_text"],
                    "distance": distance_from_zero,
                    "x_offset": sys_data["x_offset"],
                }
            )

        ranking_data.sort(
            key=lambda item: (item["distance"], item["x_offset"], item["label"])
        )
        ranked_labels = [item["label"] for item in ranking_data]
        final_ans_script = ", ".join(ranked_labels)

        # Quiz UI
        quiz_title_text = Tex("Quiz: Resting Position Ranking", font_size=28).to_edge(
            UP, buff=0.2
        )
        quiz_instruction2 = Tex(
            "All systems started with their mass at the ruler's zero point (0.0).",
            font_size=22,
        )
        quiz_instruction3 = Tex(
            "Rank them by the distance of their final resting position", font_size=22
        )
        quiz_instruction4 = Tex(
            "from the ruler's zero point, from CLOSEST to FURTHEST.", font_size=22
        )

        if num_systems == 3:
            quiz_instruction1 = Tex(
                "The systems A, B, C will eventually come to rest.", font_size=22
            )
            quiz_question = Tex(
                "List the labels (A, B, C) in the correct order.",
                font_size=25,
                color=YELLOW_C,
            )
        elif num_systems == 4:
            quiz_instruction1 = Tex(
                "The systems A, B, C, D will eventually come to rest.", font_size=22
            )
            quiz_question = Tex(
                "List the labels (A, B, C, D) in the correct order.",
                font_size=25,
                color=YELLOW_C,
            )
        elif num_systems == 5:
            quiz_instruction1 = Tex(
                "The systems A, B, C, D, E will eventually come to rest.", font_size=22
            )
            quiz_question = Tex(
                "List the labels (A, B, C, D, E) in the correct order.",
                font_size=25,
                color=YELLOW_C,
            )

        quiz_items_for_display = VGroup(
            quiz_instruction1,
            quiz_instruction2,
            quiz_instruction3,
            quiz_instruction4,
            quiz_question,
        ).arrange(DOWN, buff=0.20)
        quiz_items_for_display.next_to(quiz_title_text, DOWN, buff=0.4)

        self.play(
            FadeOut(scene_title),
            FadeOut(system_mobjects),
            FadeOut(x_axis),
            Write(quiz_title_text),
            Write(quiz_items_for_display),
        )

        print(f"FINAL_ANSWER: {final_ans_script}")
        for item in ranking_data:
            # Locate original data to print debug info
            orig_data = next(
                s for s in systems_data if s["label_text"] == item["label"]
            )
            print(
                f"System {item['label']}: x_offset = {item['x_offset']:.3f}, distance = {item['distance']:.3f}, zeta = {orig_data['damping_factor']:.3f}"
            )

        self.wait(20)

        # Save solution
        with open(
            f"solutions/resting_pose_n{num_systems}_seed{self.seed}.txt", "w"
        ) as f:
            f.write(final_ans_script)

        # Save question text
        systems_list = ", ".join(SYSTEM_LABELS[:num_systems])
        question_text_content = (
            f"The systems {systems_list} will eventually come to rest.\n"
            "All systems started with their mass at the ruler's zero point (0.0).\n"
            "Rank them by the distance of their final resting position "
            "from the ruler's zero point, from CLOSEST to FURTHEST.\n"
            f"List the labels ({systems_list}) in the correct order."
        )
        with open(
            f"question_text/resting_pose_n{num_systems}_seed{self.seed}.txt", "w"
        ) as f:
            f.write(question_text_content)

        # Generate and save reasoning trace
        reasoning_trace = self.generate_reasoning_trace(ranking_data)
        with open(
            f"reasoning_traces/resting_pose_n{num_systems}_seed{self.seed}.txt", "w"
        ) as f:
            f.write(reasoning_trace)

    def generate_reasoning_trace(self, ranking_data):
        """Generate a detailed reasoning trace describing the scene chronologically"""

        trace = []
        trace.append("=== SCENE DESCRIPTION ===\n")
        trace.append(
            f"This video demonstrates {self.num_systems} damped spring-mass oscillator systems, "
            f"labeled {', '.join([event['label'] for event in self.system_events])}.\n"
        )
        trace.append(
            "Each system consists of a mass attached to a spring, connected to a fixed wall. "
            "All masses start at the ruler's zero point (0.0) but have different initial velocities "
            "and physical properties.\n\n"
        )

        trace.append("=== INITIAL SETUP ===\n")
        for event in self.system_events:
            trace.append(f"System {event['label']}:")
            trace.append(f"  - Mass (m): {event['mass']:.2f} kg")
            trace.append(f"  - Spring constant (k): {event['spring_constant']:.2f} N/m")
            trace.append(
                f"  - Damping coefficient (b): {event['damping_coefficient']:.2f} N·s/m"
            )
            trace.append(f"  - Damping factor (zeta): {event['damping_factor']:.3f}")
            trace.append(f"  - Initial position: 0.0 (ruler zero point)")
            trace.append(f"  - Initial velocity: {event['initial_velocity']:.2f} m/s")
            trace.append(f"  - Hidden equilibrium offset: {event['x_offset']:.2f} m")
            trace.append("")

        trace.append("\n=== SYSTEM MOTION ===\n")
        trace.append(
            f"The animation shows all {self.num_systems} systems oscillating simultaneously for "
            f"{ANIMATION_DURATION} seconds.\n"
        )
        trace.append(
            "Each system undergoes damped harmonic motion:\n"
            "  1. The mass is released from the zero point with an initial velocity\n"
            "  2. The spring force pulls/pushes the mass toward its natural equilibrium position\n"
            "  3. Damping dissipates energy, causing oscillations to decay over time\n"
            "  4. Eventually, each system settles at its resting position\n\n"
        )

        trace.append("=== KEY INSIGHT: RESTING POSITION ===\n")
        trace.append(
            "The crucial insight is that each spring has a HIDDEN equilibrium offset (x_offset). "
            "This offset represents where the spring's natural equilibrium position is located "
            "relative to the ruler's zero point.\n\n"
        )
        trace.append(
            "When the mass is released from position 0.0:\n"
            "  - If x_offset > 0: The spring pulls the mass to the RIGHT (positive direction)\n"
            "  - If x_offset < 0: The spring pulls the mass to the LEFT (negative direction)\n"
            "  - If x_offset = 0: The mass would return to the zero point\n\n"
        )
        trace.append(
            "Regardless of mass, spring constant, damping, or initial velocity, "
            "the final resting position equals the x_offset value.\n\n"
        )

        trace.append("\n=== ANALYZING RESTING POSITIONS ===\n")
        trace.append("Each system's resting position (x_offset):\n")
        for event in self.system_events:
            trace.append(
                f"  System {event['label']}: x_offset = {event['x_offset']:.3f} m "
                f"(distance from zero = {event['distance_from_zero']:.3f} m)"
            )
        trace.append("")

        trace.append("\n=== RANKING PROCESS ===\n")
        trace.append(
            "To rank systems from CLOSEST to FURTHEST from the zero point, "
            "I need to sort by the absolute value of x_offset (distance from zero):\n"
        )

        for i, item in enumerate(ranking_data, 1):
            trace.append(
                f"  {i}. System {item['label']}: distance = {item['distance']:.3f} m "
                f"(x_offset = {item['x_offset']:.3f} m)"
            )
        trace.append("")

        trace.append("\n=== FINAL ANSWER ===\n")
        answer_labels = [item["label"] for item in ranking_data]
        trace.append(f"Ranking from CLOSEST to FURTHEST from the ruler's zero point:\n")
        trace.append(", ".join(answer_labels))
        trace.append("")

        trace.append("\n=== REASONING SUMMARY ===\n")
        trace.append(
            f"I observed {self.num_systems} damped spring-mass oscillator systems starting from "
            "the ruler's zero point. Each system has a hidden equilibrium offset that determines "
            "its final resting position. By identifying the x_offset for each system and calculating "
            "the distance from zero (absolute value), I ranked them from closest to furthest. "
        )
        trace.append(
            f"The physical properties (mass, spring constant, damping) affect HOW the system "
            f"oscillates and HOW FAST it settles, but the FINAL resting position is solely "
            f"determined by the x_offset parameter. "
        )
        trace.append(f"The final ranking is: {', '.join(answer_labels)}.")
        return "\n".join(trace)


if __name__ == "__main__":
    # Generate the resting pose video
    scene = DampedOscillatorsRestingPositionQuiz()
    scene.render()

    # Dynamically locate the output file
    output_path = None
    if scene.renderer.file_writer.movie_file_path:
        output_path = Path(scene.renderer.file_writer.movie_file_path)

    # Fallback search if Manim didn't report the path directly (rare but possible in some configs)
    if not output_path or not output_path.exists():
        print("Searching for output file...")
        videos_dir = Path("manim_output/videos")
        if videos_dir.exists():
            # Find the most recently modified mp4 file
            files = list(videos_dir.rglob("*.mp4"))
            if files:
                output_path = max(files, key=os.path.getmtime)

    # Move and Rename
    if output_path and output_path.exists():
        filename = f"resting_pose_n{scene.num_systems}_seed{scene.seed}.mp4"
        destination = Path("questions") / filename
        print(f"Moving {output_path} to {destination}")
        shutil.move(str(output_path), str(destination))
    else:
        print("Error: Could not locate the generated video file.")

    # Cleanup
    if os.path.exists("manim_output") and os.path.isdir("manim_output"):
        shutil.rmtree("manim_output", ignore_errors=True)
