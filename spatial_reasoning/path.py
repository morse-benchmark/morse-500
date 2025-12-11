from manim import *
import random
import numpy as np
import os
import shutil
from pathlib import Path

# ============================================================================
# Setup directories for output files
# ============================================================================
Path("questions").mkdir(exist_ok=True)          # Video files
Path("solutions").mkdir(exist_ok=True)          # Answer text files
Path("question_text").mkdir(exist_ok=True)      # Question text files
Path("reasoning_traces").mkdir(exist_ok=True)   # Step-by-step reasoning

# ============================================================================
# Manim configuration
# ============================================================================
config.media_dir = "manim_output"
config.verbosity = "WARNING"
config.pixel_height = 1080
config.pixel_width = 1920
config.frame_rate = 30
config.preview = False

def ordinal(n):
    """
    Convert number to ordinal string (1st, 2nd, 3rd, etc.).
    
    Args:
        n: Integer to convert
        
    Returns:
        String like "1st", "2nd", "3rd", "4th", etc.
    """
    if 11 <= (n % 100) <= 13:
        return f"{n}th"
    return f"{n}{  {1:'st', 2:'nd', 3:'rd'}.get(n%10, 'th')  }"

class Paths(Scene):
    """
    A scene that generates a path-following puzzle:
    - Shows an arrow moving through colored shapes
    - User must answer questions about order, distance, or time
    - Generates question video, solution, and detailed reasoning trace
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # ====================================================================
        # Random seed for reproducibility
        # ====================================================================
        # Each video gets a unique seed but same seed produces same video
        self.seed = random.randint(1000, 9999)
        
        # ====================================================================
        # Parameters from environment variables (with defaults)
        # ====================================================================
        # Problem type determines what question is asked
        self.p_type = os.getenv("P_TYPE", "order")
        # Number of shapes in the scene
        self.num_shapes = int(os.getenv("NUM_SHAPES", 4))
        # Max attempts to place shapes without overlap
        self.max_placement_tries = int(os.getenv("MAX_PLACEMENT_TRIES", 250))
        
        # ====================================================================
        # Scene visual parameters
        # ====================================================================
        # Size multiplier for all shapes
        self.SHAPE_SCALE = 0.1
        # Empty space around edges (as fraction of frame height)
        self.MARGIN = config.frame_height * 0.2
        # Minimum animation time between shapes
        self.min_time = 0.5
        # Multiplier for creating varied animation times
        self.time_step = 2

        # ====================================================================
        # Color palette and name mapping
        # ====================================================================
        # Maps Manim color objects to their string names
        self.VALID_COLORS = {
            BLUE: "blue", 
            RED: "red", 
            ORANGE: "orange",
            GREEN: "green", 
            YELLOW: "yellow",
            PURPLE: "purple",
            WHITE: "white",
        }

        # ====================================================================
        # Question templates for different problem types
        # ====================================================================
        # <N> will be replaced with ordinal number (1st, 2nd, etc.)
        self.cfg = {
            "text": {
                "order": [
                    "What was the color of the <N> shape to be visited?\nAnswer with only the color name."
                ],
                "max_dist": [
                    "Which color shape was the furthest from\nthe previous shape in the path?\nAnswer with only the color name."
                ],
                "min_dist": [
                    "Which color shape was the closest to\nthe previous shape in the path?\nAnswer with only the color name."
                ],
                "min_time": [
                    "Which color shape took the shortest amount\nof time to reach from the previous shape?\nAnswer with only the color name."
                ],
                "max_time": [
                    "Which color shape took the longest amount\nof time to reach from the previous shape?\nAnswer with only the color name."
                ]
            }
        }
        
        # ====================================================================
        # Initialize tracking for reasoning trace
        # ====================================================================
        # This will store scene events with timestamps for detailed reasoning
        self.scene_events = []
        
    def log_event(self, description):
        """
        Log a scene event with video timestamp.
        Uses Manim's internal renderer.time which tracks actual video playback time,
        not wall-clock execution time.
        
        Args:
            description: String describing what happened in the scene
        """
        # Get current video time from Manim's renderer
        # self.renderer.time tracks the cumulative duration of all animations/waits
        current_time = self.renderer.time
        
        self.scene_events.append({
            'time': current_time,
            'description': description
        })

    def format_time(self, seconds):
        """
        Format seconds as M:SS for display in reasoning trace.
        
        Args:
            seconds: Time in seconds (float)
            
        Returns:
            String formatted as "M:SS" (e.g., "2:37")
        """
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}:{secs:02d}"

    def _non_overlapping_position(self, new_shape, existing, x_min, x_max, y_min, y_max):
        """
        Find a random position for a shape that doesn't overlap with existing shapes.
        
        Uses random sampling with collision detection. Tries up to max_placement_tries
        times before giving up.
        
        Args:
            new_shape: The shape to position
            existing: VGroup of already-placed shapes
            x_min, x_max: Horizontal bounds for placement
            y_min, y_max: Vertical bounds for placement
            
        Returns:
            True if successful placement found, False otherwise
        """
        for attempt in range(self.max_placement_tries):
            # Try a random position within bounds
            new_shape.move_to([
                random.uniform(x_min, x_max),
                random.uniform(y_min, y_max),
                0,
            ])
            
            # Check if this position overlaps with any existing shape
            # Shapes must be separated by at least 55% of their combined width
            if all(
                np.linalg.norm(new_shape.get_center() - s.get_center())
                > (new_shape.width + s.width) * 0.55
                for s in existing
            ):
                return True  # Found valid position
                
        return False  # Failed to find position after max attempts

    def _random_point_outside_shapes(self, shapes, x_min, x_max, y_min, y_max):
        """
        Find a random point that doesn't overlap with any shapes.
        
        This is used for the arrow's starting position.
        
        Args:
            shapes: VGroup of shapes to avoid
            x_min, x_max: Horizontal bounds
            y_min, y_max: Vertical bounds
            
        Returns:
            Numpy array [x, y, 0] representing the point
        """
        attempts = 0
        while attempts < 100:  # Prevent infinite loop
            attempts += 1
            p = np.array([
                random.uniform(x_min, x_max),
                random.uniform(y_min, y_max),
                0,
            ])
            # Point must be at least 60% of shape width away from all shapes
            if all(np.linalg.norm(p - s.get_center()) > s.width * 0.6 for s in shapes):
                return p
                
        # Fallback if no good position found (shouldn't happen often)
        return np.array([0, 0, 0])

    def show_colors(self, colors, names):
        """
        Display color legend at the beginning of the video.
        Shows colored squares with arrows pointing to their names in two columns.
        
        This helps viewers learn the color-to-name mapping before the puzzle begins.
        
        Args:
            colors: List of Manim color objects
            names: List of color name strings (parallel to colors)
        """
        # ====================================================================
        # Create first column (first 4 colors)
        # ====================================================================
        squares1 = VGroup(
            *[Square(0.5).set_fill(col, 1).set_stroke(width=0) for col in colors[:4]]
        )
        squares1.arrange(DOWN, buff=0.3, aligned_edge=LEFT).to_edge(LEFT, buff=2)

        # Create labels positioned to the right of squares
        labels1 = VGroup(*[Text(nm, font_size=20) for nm in names[:4]])
        for sq, lbl in zip(squares1, labels1):
            lbl.next_to(sq, RIGHT, buff=0.8)

        # Create arrows connecting squares to labels
        arrows1 = VGroup(
            *[
                Arrow(
                    start=sq.get_right(), end=lbl.get_left(), buff=0.05, stroke_width=2
                )
                for sq, lbl in zip(squares1, labels1)
            ]
        )

        # ====================================================================
        # Create second column (remaining colors)
        # ====================================================================
        squares2 = VGroup(
            *[Square(0.5).set_fill(col, 1).set_stroke(width=0) for col in colors[4:]]
        )
        squares2.arrange(DOWN, buff=0.3, aligned_edge=LEFT).to_edge(LEFT, buff=2)

        labels2 = VGroup(*[Text(nm, font_size=20) for nm in names[4:]])
        for sq, lbl in zip(squares2, labels2):
            lbl.next_to(sq, RIGHT, buff=0.8)

        arrows2 = VGroup(
            *[
                Arrow(
                    start=sq.get_right(), end=lbl.get_left(), buff=0.05, stroke_width=2
                )
                for sq, lbl in zip(squares2, labels2)
            ]
        )

        # ====================================================================
        # Arrange both columns side by side
        # ====================================================================
        left = VGroup(squares1, arrows1, labels1)
        right = VGroup(squares2, arrows2, labels2)
        both = VGroup(left, right)
        both.arrange(buff=1.7, aligned_edge=UP)
        both.move_to(ORIGIN)

        # ====================================================================
        # Add title and animate
        # ====================================================================
        title = Text("Remember the following color names", font_size=32)
        title.to_edge(UP)

        # Animation sequence: title -> squares -> arrows -> labels
        self.log_event("Title appears: 'Remember the following color names'")
        self.play(Write(title))
        
        self.play(
            Succession(
                FadeIn(squares1, squares2),
                AnimationGroup(*([GrowArrow(ar) for ar in arrows1]+[GrowArrow(ar) for ar in arrows2]), lag_ratio=0.1),
                FadeIn(labels1, labels2)
            ),
            run_time=1.5
        )
        self.wait(2)
        
        # Log which colors were shown
        color_list = ", ".join(names)
        self.log_event(f"Color legend displayed with {len(colors)} colors: {color_list}")
        
        # Clean up legend
        self.play(FadeOut(title, both))
        self.wait(0.5)
        self.log_event("Color legend fades out")

    def construct(self):
        """
        Main scene construction method.
        This is called by Manim to build and render the entire scene.
        """
        # ====================================================================
        # Use the seed set in __init__ for reproducible randomness
        # ====================================================================
        random.seed(self.seed)
        
        # ====================================================================
        # Create animated gradient background
        # ====================================================================
        bg = (
            Rectangle(height=config.frame_height, width=config.frame_width)
            .set_color(
                color_gradient([random_bright_color(), random_bright_color()], 5)
            )
            .set_opacity(0.6)
            .set_z_index(-2)  # Ensure it stays in background
        )
        self.add(bg)
        
        # ====================================================================
        # Show color legend
        # ====================================================================
        self.show_colors(list(self.VALID_COLORS.keys()), list(self.VALID_COLORS.values()))
        
        # ====================================================================
        # Display instruction prompt
        # ====================================================================
        prompt = Text(
            "Observe the trajectory of the arrow", color=WHITE, font_size=36
        ).move_to(ORIGIN)
        
        self.log_event("Instruction prompt appears")
        self.play(FadeIn(prompt), run_time=0.5)
        self.wait(1.5)
        self.play(FadeOut(prompt), run_time=0.5)
        self.wait(1)
        self.log_event("Instruction prompt fades out")
        
        # ====================================================================
        # Calculate frame bounds (inner safe zone with margins)
        # ====================================================================
        x_min = -config.frame_width / 2 + self.MARGIN
        x_max =  config.frame_width / 2 - self.MARGIN
        y_min = -config.frame_height / 2 + self.MARGIN
        y_max =  config.frame_height / 2 - self.MARGIN

        # ====================================================================
        # Create random non-overlapping shapes
        # ====================================================================
        shapes = VGroup()
        shape_colors = list(self.VALID_COLORS.keys())
        random.shuffle(shape_colors)  # Randomize color assignment
        
        # Store shape details for reasoning trace
        self.shape_details = []
        
        for i in range(self.num_shapes):
            # Randomly choose shape type
            shape_cls = random.choice([
                Circle,
                Square,
                Triangle,
                lambda: RegularPolygon(n=random.randint(5, 8)),  # Pentagon to octagon
            ])
            shp = shape_cls()
            
            # Assign color (cycle through colors if more shapes than colors)
            color = shape_colors[i % len(shape_colors)]
            shp.set_fill(opacity=0).set_stroke(width=4, color=color)
            shp.scale(self.SHAPE_SCALE)

            # Try to find non-overlapping position
            if self._non_overlapping_position(shp, shapes, x_min, x_max, y_min, y_max):
                shapes.add(shp)
                # Store details for reasoning trace
                self.shape_details.append({
                    'number': i + 1,
                    'type': type(shp).__name__,
                    'color': self.VALID_COLORS[color],
                    'position': shp.get_center().copy(),
                    'placed': True
                })
            else:
                # Failed to place shape (shouldn't happen often)
                self.shape_details.append({
                    'number': i + 1,
                    'type': type(shp).__name__,
                    'color': self.VALID_COLORS[color],
                    'position': None,
                    'placed': False
                })

        # ====================================================================
        # Scale shapes to fit screen nicely
        # ====================================================================
        # Calculate available space (frame size minus buffers)
        frame_width = config.frame_width
        frame_height = config.frame_height
        h_prop, v_prop = 0.1, 0.2  # Buffer proportions
        h_buf = frame_width * h_prop
        v_buf = frame_height * v_prop
        max_w = frame_width - 2*h_buf
        max_h = frame_height - 2*v_buf
        
        # Scale to fit within available space
        scale = min(max_w / shapes.width, max_h / shapes.height)
        shapes.scale(scale).move_to(ORIGIN)
        
        # Reverse individual shape scaling to maintain stroke width
        for shape in shapes:
            shape.scale(1/scale)

        # Update positions in shape_details after scaling
        for i, shape in enumerate(shapes):
            if i < len(self.shape_details):
                self.shape_details[i]['position'] = shape.get_center().copy()

        # ====================================================================
        # Animate shape creation
        # ====================================================================
        self.log_event(f"Shapes begin appearing ({self.num_shapes} total)")
        self.play(LaggedStart(*[Create(s) for s in shapes], lag_ratio=0.15))
        self.wait(0.3)
        self.log_event(f"All {self.num_shapes} shapes are now visible")

        # ====================================================================
        # Determine start point and visit order
        # ====================================================================
        # Arrow starts from a random point outside all shapes
        start_pt = self._random_point_outside_shapes(shapes, x_min, x_max, y_min, y_max)
        
        # Randomly shuffle which order to visit shapes
        visit_order = random.sample(list(shapes), len(shapes))
        
        # Store visit order for reasoning trace
        self.visit_order_details = []
        for i, shape in enumerate(visit_order):
            color_name = self.VALID_COLORS[shape.stroke_color]
            shape_type = type(shape).__name__
            self.visit_order_details.append({
                'position': i + 1,
                'color': color_name,
                'type': shape_type,
                'center': shape.get_center().copy()
            })

        # ====================================================================
        # Create smooth path through shape centers
        # ====================================================================
        # Path goes: start_pt -> shape1 center -> shape2 center -> ... -> shapeN center
        control_pts = [start_pt] + [s.get_center() for s in visit_order]
        path = VMobject().set_points_smoothly(control_pts)
        path.set_stroke(opacity=0)  # Invisible - only used for animation
        self.add(path)

        # ====================================================================
        # Create arrowhead (no visible shaft)
        # ====================================================================
        arrow_len = 0.9  # Length from tip to tail
        arrow = Arrow(
            start_pt - RIGHT * arrow_len,
            start_pt,
            buff=0,
            color=WHITE,
            max_tip_length_to_length_ratio=0.2,
            stroke_width=0,  # No visible shaft
        )
        
        # Make only the tip visible
        if hasattr(arrow, "body"):
            arrow.body.set_stroke(width=0)
        if hasattr(arrow, "tip"):
            arrow.tip.set_stroke(width=0)
            arrow.tip.set_fill(WHITE, 1)
        arrow.set_z_index(2)  # Draw on top of everything
        self.add(arrow)

        # ====================================================================
        # Create trail that follows the arrow
        # ====================================================================
        # Trail fades out after 0.7 seconds (dissipating effect)
        trail = TracedPath(
            arrow.get_end,
            dissipating_time=0.7,
            stroke_color=WHITE,
            stroke_width=2
        )
        self.add(trail)

        # ====================================================================
        # Setup arrow orientation updater
        # ====================================================================
        # This makes the arrow point in the direction it's moving
        tracker = ValueTracker(0)  # Goes from 0 to 1 along the path

        def orient_arrow(mob):
            """
            Update arrow orientation to point along path tangent.
            
            Uses the path's tangent vector at the current position to determine
            which direction the arrow should point.
            """
            alpha = tracker.get_value()  # Current position along path (0 to 1)
            pos = path.point_from_proportion(alpha)

            # Calculate tangent vector using finite differences
            delta = 1e-3  # Small step for numerical derivative
            
            if alpha < delta:
                # Near start: use forward difference
                q = path.point_from_proportion(alpha + delta)
                tangent = q - pos
            elif alpha > 1 - delta:
                # Near end: use backward difference
                q = path.point_from_proportion(alpha - delta)
                tangent = pos - q
            else:
                # Middle: use central difference (more accurate)
                q1 = path.point_from_proportion(alpha + delta)
                q0 = path.point_from_proportion(alpha - delta)
                tangent = q1 - q0

            # Handle edge case of zero tangent
            if np.linalg.norm(tangent) == 0:
                tangent = RIGHT * 0.001

            # Orient arrow to point in tangent direction
            unit = tangent / np.linalg.norm(tangent)
            tail = pos - unit * arrow_len
            mob.put_start_and_end_on(tail, pos)

        # Attach the updater so arrow stays oriented as it moves
        arrow.add_updater(orient_arrow)

        # ====================================================================
        # Generate variable-duration hops between shapes
        # ====================================================================
        # Create different animation speeds for each segment
        hops = len(visit_order)
        
        # First hop (start to first shape) is always 1.9s
        # Remaining hops use exponentially increasing times, then shuffle
        durations = [1.9] + [self.min_time*(self.time_step**i) for i in range(hops-1)]
        random.shuffle(durations)  # Randomize which hop gets which duration
        
        # Each hop covers 1/hops of the total path
        alpha_step = 1 / hops

        # Store durations for reasoning trace
        self.durations = durations

        # ====================================================================
        # Animate arrow movement along path
        # ====================================================================
        self.log_event(f"Arrow begins moving from start point toward first shape")
        
        for i, dur in enumerate(durations):
            # Move tracker from current position to next position
            target_alpha = (i + 1) * alpha_step
            
            # Log before animation starts
            if i == 0:
                target_color = self.VALID_COLORS[visit_order[i].stroke_color]
                self.log_event(f"Arrow moving to first shape ({target_color}) - duration: {dur:.2f}s")
            else:
                prev_color = self.VALID_COLORS[visit_order[i-1].stroke_color]
                target_color = self.VALID_COLORS[visit_order[i].stroke_color]
                self.log_event(f"Arrow moving from {prev_color} to {target_color} - duration: {dur:.2f}s")
            
            self.play(
                tracker.animate.set_value(target_alpha),
                run_time=dur,
                rate_func=linear,  # Constant speed
            )
            
            # Log after animation completes
            if i == 0:
                self.log_event(f"Arrow reaches first shape ({target_color})")
            else:
                self.log_event(f"Arrow reaches {target_color} shape")

        # ====================================================================
        # Ensure arrow ends exactly at final shape center
        # ====================================================================
        tracker.set_value(1)
        orient_arrow(arrow)  # Manual final orientation update
        arrow.remove_updater(orient_arrow)  # Stop automatic updates
        
        self.log_event("Arrow completes its journey through all shapes")
        
        # ====================================================================
        # Calculate distances and prepare answer data
        # ====================================================================
        # Calculate Euclidean distance between consecutive shape centers
        centers = np.array([poly.get_center() for poly in list(visit_order)])
        deltas = centers[1:] - centers[:-1]  # Vector differences
        dists = list(np.linalg.norm(deltas, axis=1))  # Euclidean distances
        
        # Get colors of visited shapes (for answer calculation)
        shape_colors = [poly.stroke_color for poly in list(visit_order)]
        
        # Store for reasoning trace
        self.distances = dists
        self.shape_colors_visited = [self.VALID_COLORS[c] for c in shape_colors]
        
        # ====================================================================
        # Calculate answer based on problem type
        # ====================================================================
        N = random.randint(1, self.num_shapes)  # Random position for "order" questions
        
        if self.p_type == "order": 
            # Which shape was visited Nth?
            self.answer = self.VALID_COLORS[shape_colors[N-1]]
            self.answer_detail = f"The {ordinal(N)} shape visited was {self.answer}"
            
        elif self.p_type == "min_dist":
            # Which shape was closest to its predecessor?
            idx = dists.index(min(dists))
            self.answer = self.VALID_COLORS[shape_colors[idx+1]]
            self.answer_detail = f"Closest shape (min distance {min(dists):.2f}): {self.answer}"
            
        elif self.p_type == "max_dist":
            # Which shape was furthest from its predecessor?
            idx = dists.index(max(dists))
            self.answer = self.VALID_COLORS[shape_colors[idx+1]]
            self.answer_detail = f"Furthest shape (max distance {max(dists):.2f}): {self.answer}"
            
        elif self.p_type == "min_time":
            # Which shape took shortest time to reach?
            idx = durations[1:].index(min(durations[1:]))
            self.answer = self.VALID_COLORS[shape_colors[idx]]
            self.answer_detail = f"Shortest time (min {min(durations[1:]):.2f}s): {self.answer}"
            
        elif self.p_type == "max_time":
            # Which shape took longest time to reach?
            idx = durations[1:].index(max(durations[1:]))
            self.answer = self.VALID_COLORS[shape_colors[idx]]
            self.answer_detail = f"Longest time (max {max(durations[1:]):.2f}s): {self.answer}"
        
        # Store N for reasoning trace
        self.ordinal_N = N

        # ====================================================================
        # Transition to question display
        # ====================================================================
        self.wait(2)
        self.log_event("All objects begin fading out")
        self.play(*[FadeOut(mob) for mob in self.mobjects if mob!=bg])
        self.log_event("Scene cleared except background")
        
        # ====================================================================
        # Display the question
        # ====================================================================
        title_text = random.choice(self.cfg["text"][self.p_type])
        title_text = title_text.replace("<N>", ordinal(N))
        lines = title_text.split('\n')
        
        para = Paragraph(
            *lines, alignment="center", font_size=36, line_spacing=0.8
        )
        para.move_to(ORIGIN)
        
        # Scale if too wide for screen
        if para.width > 0.9*config.frame_width:
            para.scale_to_fit_width(config.frame_width * 0.9)
        
        self.log_event("Question text appears on screen")
        self.play(Write(para), run_time=1.5)
        self.wait(3)
        self.log_event("Question remains displayed for viewer to read")

        # Create question text (for output file)
        self.question_text = f"Observe the trajectory of the arrow. {title_text.replace(chr(10), ' ')}"
        
        # ====================================================================
        # Build comprehensive reasoning trace
        # ====================================================================
        self.build_reasoning_trace()
        
        # ====================================================================
        # Save output files
        # ====================================================================
        # Solution file (just the answer)
        solution_filename = f"solutions/path_{self.p_type}_shapes{self.num_shapes}_seed{self.seed}.txt"
        with open(solution_filename, "w") as f:
            f.write(str(self.answer))
        
        # Question text file
        question_filename = f"question_text/path_{self.p_type}_shapes{self.num_shapes}_seed{self.seed}.txt"
        with open(question_filename, "w") as f:
            f.write(self.question_text)
            
        # Detailed reasoning trace file
        trace_filename = f"reasoning_traces/path_{self.p_type}_shapes{self.num_shapes}_seed{self.seed}.txt"
        with open(trace_filename, "w") as f:
            f.write("\n".join(self.reasoning_trace))

    def build_reasoning_trace(self):
        """
        Build a comprehensive, step-by-step reasoning trace.
        This explains how to solve the puzzle systematically.
        
        Similar structure to CubeRollScene for consistency.
        """
        self.reasoning_trace = []
        
        # ====================================================================
        # Introduction with question
        # ====================================================================
        self.reasoning_trace.append(f"**Question:** {self.question_text}")
        self.reasoning_trace.append("")
        self.reasoning_trace.append("Let's solve this step by step.")
        self.reasoning_trace.append("")
        
        # ====================================================================
        # Scene description with timestamps
        # ====================================================================
        self.reasoning_trace.append("### Scene Description")
        self.reasoning_trace.append("")
        
        for event in self.scene_events:
            time_str = self.format_time(event['time'])
            self.reasoning_trace.append(f"At {time_str}, {event['description']}")
        
        self.reasoning_trace.append("")
        
        # ====================================================================
        # Step 1: Understand the shapes
        # ====================================================================
        self.reasoning_trace.append("### Step 1: Understand the shapes in the scene")
        self.reasoning_trace.append(f"The scene contains **{self.num_shapes} colored shapes** positioned randomly:")
        self.reasoning_trace.append("")
        
        for detail in self.shape_details:
            if detail['placed']:
                pos = detail['position']
                self.reasoning_trace.append(
                    f"- Shape {detail['number']}: {detail['color']} {detail['type']} "
                    f"at position ({pos[0]:.2f}, {pos[1]:.2f})"
                )
            else:
                self.reasoning_trace.append(
                    f"- Shape {detail['number']}: Failed to place (overlapping)"
                )
        
        self.reasoning_trace.append("")
        
        # ====================================================================
        # Step 2: Track the arrow's path
        # ====================================================================
        self.reasoning_trace.append("### Step 2: Track the arrow's path")
        self.reasoning_trace.append(
            f"The arrow visits all {len(self.visit_order_details)} shapes in a specific order. "
            "The visit sequence is:"
        )
        self.reasoning_trace.append("")
        
        for detail in self.visit_order_details:
            center = detail['center']
            self.reasoning_trace.append(
                f"{detail['position']}. **{detail['color']}** {detail['type']} "
                f"at ({center[0]:.2f}, {center[1]:.2f})"
            )
        
        self.reasoning_trace.append("")
        
        # ====================================================================
        # Step 3: Analyze distances between consecutive shapes
        # ====================================================================
        self.reasoning_trace.append("### Step 3: Analyze distances between consecutive shapes")
        self.reasoning_trace.append(
            "The Euclidean distance between each pair of consecutive shapes is:"
        )
        self.reasoning_trace.append("")
        
        for i in range(len(self.distances)):
            from_color = self.shape_colors_visited[i]
            to_color = self.shape_colors_visited[i + 1]
            dist = self.distances[i]
            self.reasoning_trace.append(
                f"- From **{from_color}** to **{to_color}**: {dist:.2f} units"
            )
        
        self.reasoning_trace.append("")
        
        # Add summary statistics
        if self.distances:
            self.reasoning_trace.append(f"- **Minimum distance**: {min(self.distances):.2f} units")
            self.reasoning_trace.append(f"- **Maximum distance**: {max(self.distances):.2f} units")
            self.reasoning_trace.append(f"- **Average distance**: {np.mean(self.distances):.2f} units")
            self.reasoning_trace.append("")
        
        # ====================================================================
        # Step 4: Analyze animation timing
        # ====================================================================
        self.reasoning_trace.append("### Step 4: Analyze animation timing")
        self.reasoning_trace.append(
            "The time taken for the arrow to travel between shapes varies:"
        )
        self.reasoning_trace.append("")
        
        # First segment (start to first shape)
        self.reasoning_trace.append(
            f"- From **start** to **{self.shape_colors_visited[0]}**: {self.durations[0]:.2f} seconds"
        )
        
        # Subsequent segments
        for i in range(1, len(self.durations)):
            from_color = self.shape_colors_visited[i - 1]
            to_color = self.shape_colors_visited[i]
            duration = self.durations[i]
            self.reasoning_trace.append(
                f"- From **{from_color}** to **{to_color}**: {duration:.2f} seconds"
            )
        
        self.reasoning_trace.append("")
        
        # Add timing summary
        if len(self.durations) > 1:
            self.reasoning_trace.append(f"- **Shortest time** (excluding start): {min(self.durations[1:]):.2f} seconds")
            self.reasoning_trace.append(f"- **Longest time** (excluding start): {max(self.durations[1:]):.2f} seconds")
            self.reasoning_trace.append("")
        
        # ====================================================================
        # Step 5: Solve based on problem type
        # ====================================================================
        self.reasoning_trace.append("### Step 5: Determine the answer")
        
        if self.p_type == "order":
            self.reasoning_trace.append(
                f"The question asks for the color of the **{ordinal(self.ordinal_N)}** shape visited."
            )
            self.reasoning_trace.append("")
            self.reasoning_trace.append("Looking at the visit sequence:")
            for i, detail in enumerate(self.visit_order_details[:self.ordinal_N + 1]):
                marker = " ← **This is the answer**" if i == self.ordinal_N - 1 else ""
                self.reasoning_trace.append(f"{detail['position']}. {detail['color']}{marker}")
            self.reasoning_trace.append("")
            self.reasoning_trace.append(
                f"The {ordinal(self.ordinal_N)} shape visited was **{self.answer}**."
            )
            
        elif self.p_type == "min_dist":
            min_dist = min(self.distances)
            idx = self.distances.index(min_dist)
            from_color = self.shape_colors_visited[idx]
            to_color = self.shape_colors_visited[idx + 1]
            
            self.reasoning_trace.append(
                "The question asks which shape was **closest** to its previous shape."
            )
            self.reasoning_trace.append("")
            self.reasoning_trace.append(
                f"The minimum distance is **{min_dist:.2f} units**, which occurs when moving "
                f"from **{from_color}** to **{to_color}**."
            )
            
        elif self.p_type == "max_dist":
            max_dist = max(self.distances)
            idx = self.distances.index(max_dist)
            from_color = self.shape_colors_visited[idx]
            to_color = self.shape_colors_visited[idx + 1]
            
            self.reasoning_trace.append(
                "The question asks which shape was **furthest** from its previous shape."
            )
            self.reasoning_trace.append("")
            self.reasoning_trace.append(
                f"The maximum distance is **{max_dist:.2f} units**, which occurs when moving "
                f"from **{from_color}** to **{to_color}**."
            )
            
        elif self.p_type == "min_time":
            min_time = min(self.durations[1:])
            idx = self.durations[1:].index(min_time)
            from_color = self.shape_colors_visited[idx]
            to_color = self.shape_colors_visited[idx + 1]
            
            self.reasoning_trace.append(
                "The question asks which shape took the **shortest time** to reach."
            )
            self.reasoning_trace.append("")
            self.reasoning_trace.append(
                f"The minimum time (excluding the start) is **{min_time:.2f} seconds**, "
                f"which occurs when moving from **{from_color}** to **{to_color}**."
            )
            
        elif self.p_type == "max_time":
            max_time = max(self.durations[1:])
            idx = self.durations[1:].index(max_time)
            from_color = self.shape_colors_visited[idx]
            to_color = self.shape_colors_visited[idx + 1]
            
            self.reasoning_trace.append(
                "The question asks which shape took the **longest time** to reach."
            )
            self.reasoning_trace.append("")
            self.reasoning_trace.append(
                f"The maximum time (excluding the start) is **{max_time:.2f} seconds**, "
                f"which occurs when moving from **{from_color}** to **{to_color}**."
            )
        
        self.reasoning_trace.append("")
        
        # ====================================================================
        # Final answer
        # ====================================================================
        self.reasoning_trace.append("### Final Answer")
        self.reasoning_trace.append(f"The answer is **{self.answer}**.")
        self.reasoning_trace.append("")
        self.reasoning_trace.append(f"\\boxed{{{self.answer}}}")

# ============================================================================
# Main execution
# ============================================================================
if __name__ == "__main__":
    # Generate the path video
    scene = Paths()
    scene.render()

    # ========================================================================
    # Move output file to questions directory with descriptive name
    # ========================================================================
    output = Path("manim_output/videos/1080p30/Paths.mp4")
    if output.exists():
        filename = f"path_{scene.p_type}_shapes{scene.num_shapes}_seed{scene.seed}.mp4"
        shutil.move(str(output), f"questions/{filename}")
        print(f"✓ Video saved: questions/{filename}")
        print(f"✓ Solution saved: solutions/path_{scene.p_type}_shapes{scene.num_shapes}_seed{scene.seed}.txt")
        print(f"✓ Question saved: question_text/path_{scene.p_type}_shapes{scene.num_shapes}_seed{scene.seed}.txt")
        print(f"✓ Reasoning saved: reasoning_traces/path_{scene.p_type}_shapes{scene.num_shapes}_seed{scene.seed}.txt")
    else:
        # Debug: Print what files actually exist
        print("ERROR: Expected output file not found!")
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
        print("✓ Cleaned up temporary files")