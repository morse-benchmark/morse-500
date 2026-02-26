from manim import *
import random
import math
import os
import shutil
from pathlib import Path

# ============================================================================
# Setup directories for output files
# ============================================================================
Path("questions").mkdir(exist_ok=True)  # Video files
Path("solutions").mkdir(exist_ok=True)  # Answer text files
Path("question_text").mkdir(exist_ok=True)  # Question text files
Path("reasoning_traces").mkdir(exist_ok=True)  # Step-by-step reasoning

# ============================================================================
# Manim configuration
# ============================================================================
config.media_dir = "manim_output"
config.verbosity = "WARNING"
config.pixel_height = 1080
config.pixel_width = 1920
config.frame_rate = 30
config.preview = False


# NUM_BOUNCES [1-5] range
# NUM_BOUNCES=3 python3 bounceball.py
class bounce_ball(Scene):
    """
    A 2D scene that generates a ball bouncing puzzle:
    - Shows a ball traveling and bouncing off walls
    - User must determine timing information relative to a SPECIFIC bounce
    - Generates question video, solution, and reasoning trace
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Set random seed for reproducibility
        self.seed = random.randint(1000, 9999)
        random.seed(self.seed)

        # Parameters - difficulty controls number of bounces
        self.num_bounces = int(os.getenv("NUM_BOUNCES", 3))  # Default to 3 for variety

        # Track timing for reasoning trace using Manim's internal video time
        self.scene_events = []

    def log_event(self, description):
        """Log a scene event with video timestamp."""
        current_time = self.renderer.time
        self.scene_events.append({"time": current_time, "description": description})

    def format_time(self, seconds):
        """Format seconds as M:SS."""
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}:{secs:02d}"

    def get_ordinal(self, n):
        """Helper to convert int to ordinal string (1st, 2nd, 3rd)."""
        if 11 <= (n % 100) <= 13:
            suffix = "th"
        else:
            suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
        return f"{n}{suffix}"

    def construct(self):
        # ====================================================================
        # Constrain bounces to reasonable bounds [1-5]
        # ====================================================================
        bounces = max(1, min(self.num_bounces, 5))

        # ====================================================================
        # Ball colors
        # ====================================================================
        ball_colors = [BLUE, RED, GREEN, YELLOW, PURPLE, ORANGE]
        ball = Circle(radius=0.3, color=ball_colors[0], fill_opacity=1)

        # ====================================================================
        # Create wall positions
        # ====================================================================
        wall_positions = [RIGHT * 3, LEFT * 3, RIGHT * 2, LEFT * 2, RIGHT * 1]
        walls = []

        # ====================================================================
        # Set initial ball position
        # ====================================================================
        if bounces > 0:
            first_wall_x = wall_positions[0][0]
            if first_wall_x > 0:
                ball.move_to(LEFT * 5)
            else:
                ball.move_to(RIGHT * 5)

        self.add(ball)
        self.log_event(
            f"Scene begins with a ball at ({ball.get_center()[0]:.1f}, {ball.get_center()[1]:.1f})"
        )

        # ====================================================================
        # Generate random travel times for each segment
        # ====================================================================
        # Need bounces+1 segments
        travel_times = []
        for i in range(bounces + 1):
            travel_times.append(random.uniform(1.5, 4.0))

        current_time = 0

        # ====================================================================
        # Animate ball movement with bounces
        # ====================================================================
        for bounce_idx in range(bounces):
            # Create wall
            wall = Rectangle(width=0.2, height=2, color=GREY, fill_opacity=1)
            wall.move_to(wall_positions[bounce_idx])
            walls.append(wall)
            self.add(wall)
            self.log_event(f"Wall #{bounce_idx + 1} appears")

            # Calculate movement
            wall_x = wall_positions[bounce_idx][0]
            ball_current_x = ball.get_center()[0]

            if wall_x > ball_current_x:
                move_distance = wall_x - ball_current_x - 0.5
                move_direction = RIGHT * move_distance
                direction_text = "right"
            else:
                move_distance = ball_current_x - wall_x - 0.5
                move_direction = LEFT * move_distance
                direction_text = "left"

            # Animate travel
            travel_time = travel_times[bounce_idx]

            self.log_event(
                f"Ball travels {direction_text} toward wall #{bounce_idx + 1} ({travel_time:.1f}s)"
            )
            self.play(ball.animate.shift(move_direction), run_time=travel_time)
            current_time += travel_time

            self.log_event(f"Ball hits wall #{bounce_idx + 1}")

        # ====================================================================
        # Final movement after last bounce
        # ====================================================================
        if bounces > 0:
            last_wall_x = wall_positions[bounces - 1][0]
            if last_wall_x > 0:
                final_move = LEFT * 4
                final_direction_text = "left"
            else:
                final_move = RIGHT * 4
                final_direction_text = "right"
        else:
            final_move = RIGHT * 8
            final_direction_text = "right"

        final_travel_time = travel_times[-1]

        if bounces > 0:
            self.log_event(
                f"Ball travels {final_direction_text} away from wall #{bounces} ({final_travel_time:.1f}s)"
            )
        else:
            self.log_event(
                f"Ball travels {final_direction_text} across screen ({final_travel_time:.1f}s)"
            )

        self.play(ball.animate.shift(final_move), run_time=final_travel_time)
        current_time += final_travel_time
        self.log_event(f"Ball's movement concludes")

        self.wait(0.5)

        # ====================================================================
        # Generate question and answer based on random specific hit
        # ====================================================================

        # Default fallback for 0 bounces
        selected_question = "What was the total travel time?"
        answer = sum(travel_times)
        target_hit_idx = 0
        mode = "total"

        if bounces > 0:
            # Pick a random hit index (1 to bounces)
            target_hit_idx = random.randint(1, bounces)
            ordinal_str = self.get_ordinal(target_hit_idx)

            # Determine times relative to this specific hit
            # The hit occurs AFTER segment index (target_hit_idx - 1)
            # Time BEFORE: sum(travel_times[0] ... travel_times[target_hit_idx-1])
            time_before_target = sum(travel_times[:target_hit_idx])

            # Time AFTER: sum(travel_times[target_hit_idx] ... end)
            time_after_target = sum(travel_times[target_hit_idx:])

            # Randomly choose "before" or "after"
            if random.choice([True, False]):
                mode = "before"
                selected_question = f"How many seconds did the ball travel BEFORE hitting the {ordinal_str} object?"
                answer = time_before_target
            else:
                mode = "after"
                selected_question = f"How many seconds did the ball travel AFTER hitting the {ordinal_str} object?"
                answer = time_after_target

        # Store info for reasoning trace
        self.selected_question = selected_question
        self.answer = answer
        self.bounces = bounces
        self.travel_times = travel_times
        self.target_hit_idx = target_hit_idx
        self.mode = mode

        # ====================================================================
        # Display question screen
        # ====================================================================
        self.clear()
        self.wait(0.2)

        question_lines = [
            selected_question,
            "",
            "Answer in seconds (rounded to 1 decimal place)",
        ]

        question_texts = []
        line_height = 0.8
        start_y = 2.0

        for i, line in enumerate(question_lines):
            if line:
                text = Text(line, font_size=28, weight=BOLD if i == 0 else NORMAL)
                if text.width > 0.9 * config.frame_width:
                    text.scale_to_fit_width(config.frame_width * 0.9)
                text.move_to(UP * (start_y - i * line_height))
                question_texts.append(text)

        self.play(*[Write(text) for text in question_texts])
        self.wait(3)

        # ====================================================================
        # Output Generation
        # ====================================================================
        formatted_answer = f"{answer:.1f}"

        with open(
            f"solutions/bounce_ball_n{self.num_bounces}_seed{self.seed}.txt", "w"
        ) as f:
            f.write(formatted_answer)

        question_text_content = (
            f"{selected_question}\nAnswer in seconds (rounded to 1 decimal place)"
        )
        with open(
            f"question_text/bounce_ball_n{self.num_bounces}_seed{self.seed}.txt", "w"
        ) as f:
            f.write(question_text_content)

        self.build_reasoning_trace()

        with open(
            f"reasoning_traces/bounce_ball_n{self.num_bounces}_seed{self.seed}.txt", "w"
        ) as f:
            f.write("\n".join(self.reasoning_trace))

    def build_reasoning_trace(self):
        """Build reasoning trace explaining the specific selected hit."""
        self.reasoning_trace = []

        self.reasoning_trace.append(f"**Question:** {self.selected_question}")
        self.reasoning_trace.append("")
        self.reasoning_trace.append("Let's solve this step by step.")
        self.reasoning_trace.append("")

        # Scene Description
        self.reasoning_trace.append("### Scene Description")
        self.reasoning_trace.append("Chronological events in the video:")
        for event in self.scene_events:
            time_str = self.format_time(event["time"])
            self.reasoning_trace.append(f"- At {time_str}, {event['description']}")
        self.reasoning_trace.append("")

        # Step 1
        self.reasoning_trace.append("### Step 1: Identify the segments")
        self.reasoning_trace.append(
            f"The ball makes {self.bounces} bounces, creating {len(self.travel_times)} distinct travel segments."
        )
        self.reasoning_trace.append("")

        for i, t in enumerate(self.travel_times):
            label = (
                "Before 1st wall"
                if i == 0
                else (
                    f"Between wall {i} and {i+1}"
                    if i < len(self.travel_times) - 1
                    else f"After wall {self.bounces}"
                )
            )
            self.reasoning_trace.append(f"Segment {i+1} ({label}): {t:.1f}s")
        self.reasoning_trace.append("")

        # Step 2 & 3 Combined for specific logic
        self.reasoning_trace.append("### Step 2: Calculate based on the specific hit")

        if self.mode == "total":
            self.reasoning_trace.append("The question asks for total time.")
            self.reasoning_trace.append(
                f"Summing all segments: {sum(self.travel_times):.1f}s"
            )
        else:
            ordinal = self.get_ordinal(self.target_hit_idx)
            self.reasoning_trace.append(
                f"The question identifies the **{ordinal} hit**."
            )

            # Explain which segments matter
            if self.mode == "before":
                self.reasoning_trace.append(
                    f"We need to sum all time segments occurring **before** hit #{self.target_hit_idx}."
                )
                self.reasoning_trace.append(
                    f"This implies summing segments 1 through {self.target_hit_idx}."
                )

                calculation_str = " + ".join(
                    [f"{t:.1f}" for t in self.travel_times[: self.target_hit_idx]]
                )
                self.reasoning_trace.append(f"Calculation: {calculation_str}")
                self.reasoning_trace.append(
                    f"Total time before {ordinal} hit: **{self.answer:.1f}s**"
                )

            elif self.mode == "after":
                self.reasoning_trace.append(
                    f"We need to sum all time segments occurring **after** hit #{self.target_hit_idx}."
                )
                self.reasoning_trace.append(
                    f"This implies summing segments from {self.target_hit_idx + 1} to the end."
                )

                calculation_str = " + ".join(
                    [f"{t:.1f}" for t in self.travel_times[self.target_hit_idx :]]
                )
                self.reasoning_trace.append(f"Calculation: {calculation_str}")
                self.reasoning_trace.append(
                    f"Total time after {ordinal} hit: **{self.answer:.1f}s**"
                )

        self.reasoning_trace.append("")
        self.reasoning_trace.append("### Final Answer")
        self.reasoning_trace.append(f"\\boxed{{{self.answer:.1f}}}")


# ============================================================================
# Main execution
# ============================================================================
if __name__ == "__main__":
    scene = bounce_ball()
    scene.render()

    # Move output file
    output = Path("manim_output/videos/1080p30/bounce_ball.mp4")
    if output.exists():
        filename = f"bounce_ball_n{scene.num_bounces}_seed{scene.seed}.mp4"
        shutil.move(str(output), f"questions/{filename}")
        print(f"✓ Video saved: questions/{filename}")

    # Cleanup
    if os.path.exists("manim_output"):
        shutil.rmtree("manim_output")
