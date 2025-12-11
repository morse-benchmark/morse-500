from manim import *
import random
import math
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

# NUM_BOUNCES [1-5] range
# NUM_BOUNCES=1 python3 bounceball.py
class bounce_ball(Scene):
    """
    A 2D scene that generates a ball bouncing puzzle:
    - Shows a ball traveling and bouncing off walls
    - User must determine timing information (before hit, after hit, or total)
    - Generates question video, solution, and reasoning trace
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Set random seed for reproducibility
        self.seed = random.randint(1000, 9999)
        random.seed(self.seed)

        # Parameters - difficulty controls number of bounces
        self.num_bounces = int(os.getenv("NUM_BOUNCES", 2))

        # Track timing for reasoning trace using Manim's internal video time
        # This will be set when construct() is called and renderer is available
        self.scene_events = []

    def log_event(self, description):
        """
        Log a scene event with video timestamp.
        Uses Manim's internal renderer.time which tracks actual video playback time,
        not wall-clock execution time.

        Args:
            description: Human-readable description of the event
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

    def construct(self):
        """
        Main scene construction method.
        This is called by Manim to build and render the entire scene.
        """
        # ====================================================================
        # Constrain bounces to reasonable bounds [1-5]
        # ====================================================================
        bounces = max(1, min(self.num_bounces, 5))

        # ====================================================================
        # Ball colors for different states (currently using single color)
        # ====================================================================
        ball_colors = [BLUE, RED, GREEN, YELLOW, PURPLE, ORANGE]

        # ====================================================================
        # Create ball object
        # ====================================================================
        ball = Circle(radius=0.3, color=ball_colors[0], fill_opacity=1)

        # ====================================================================
        # Create wall positions based on number of bounces
        # ====================================================================
        # Walls alternate between right and left sides
        wall_positions = [RIGHT * 3, LEFT * 3, RIGHT * 2, LEFT * 2, RIGHT * 1]
        walls = []

        # ====================================================================
        # Set initial ball position based on first wall location
        # ====================================================================
        # Ball starts on opposite side from first wall
        if bounces > 0:
            first_wall_x = wall_positions[0][0]  # Get x coordinate of first wall
            if first_wall_x > 0:  # Wall on right
                ball.move_to(LEFT * 5)
            else:  # Wall on left
                ball.move_to(RIGHT * 5)

        # Add ball to scene (appears at time 0)
        self.add(ball)
        self.log_event(f"Scene begins with a {ball_colors[0].name if hasattr(ball_colors[0], 'name') else 'blue'} ball at position ({ball.get_center()[0]:.1f}, {ball.get_center()[1]:.1f})")

        # ====================================================================
        # Generate random travel times for each segment
        # ====================================================================
        # Need bounces+1 segments: before each wall and after last wall
        travel_times = []
        for i in range(bounces + 1):
            travel_times.append(random.uniform(1.5, 4.0))

        # ====================================================================
        # Initialize timing trackers
        # ====================================================================
        time_before_hit = 0  # Time until first wall collision
        time_after_hit = 0   # Time after first wall collision
        current_time = 0     # Running total of elapsed video time

        # ====================================================================
        # Animate ball movement with bounces
        # ====================================================================
        for bounce_idx in range(bounces):
            # ================================================================
            # Create wall for this bounce
            # ================================================================
            wall = Rectangle(width=0.2, height=2, color=GREY, fill_opacity=1)
            wall.move_to(wall_positions[bounce_idx])
            walls.append(wall)
            self.add(wall)
            self.log_event(f"Wall #{bounce_idx + 1} appears at x = {wall_positions[bounce_idx][0]:.1f}")

            # ================================================================
            # Calculate movement direction and distance to wall
            # ================================================================
            wall_x = wall_positions[bounce_idx][0]
            ball_current_x = ball.get_center()[0]

            if wall_x > ball_current_x:  # Moving right to wall
                move_distance = wall_x - ball_current_x - 0.5  # Stop just before wall
                move_direction = RIGHT * move_distance
                direction_text = "right"
            else:  # Moving left to wall
                move_distance = ball_current_x - wall_x - 0.5  # Stop just before wall
                move_direction = LEFT * move_distance
                direction_text = "left"

            # ================================================================
            # Animate ball traveling to wall
            # ================================================================
            travel_time = travel_times[bounce_idx]
            start_time = current_time

            # Log event BEFORE animation starts
            wall_side = "right" if wall_x > 0 else "left"
            self.log_event(f"Ball begins traveling {direction_text} toward wall #{bounce_idx + 1} (duration: {travel_time:.1f}s, distance: {abs(move_distance):.1f} units)")

            self.play(ball.animate.shift(move_direction), run_time=travel_time)

            # Update current time after animation completes
            current_time += travel_time

            # Log event AFTER animation completes
            self.log_event(f"Ball reaches wall #{bounce_idx + 1} and bounces back")

            # ================================================================
            # Track time before first hit
            # ================================================================
            if bounce_idx == 0:
                time_before_hit = travel_time

        # ====================================================================
        # Final movement after last bounce (or only movement if no bounces)
        # ====================================================================
        if bounces > 0:
            # Move away from last wall
            last_wall_x = wall_positions[bounces-1][0]
            ball_x = ball.get_center()[0]

            if last_wall_x > 0:  # Last wall on right, move left
                final_move = LEFT * 4
                final_direction_text = "left"
            else:  # Last wall on left, move right
                final_move = RIGHT * 4
                final_direction_text = "right"
        else:
            # No bounces, just move across screen
            final_move = RIGHT * 8
            final_direction_text = "right"

        final_travel_time = travel_times[-1]
        start_time = current_time

        # Log event BEFORE final animation starts
        if bounces > 0:
            self.log_event(f"Ball begins traveling {final_direction_text} away from wall #{bounces} (duration: {final_travel_time:.1f}s, distance: {abs(final_move[0]):.1f} units)")
        else:
            self.log_event(f"Ball begins traveling {final_direction_text} across screen (duration: {final_travel_time:.1f}s, no walls)")

        self.play(ball.animate.shift(final_move), run_time=final_travel_time)

        # Update current time after animation completes
        current_time += final_travel_time

        # Log event AFTER final animation completes
        self.log_event(f"Ball's movement concludes")

        # ====================================================================
        # Calculate time after hitting (sum of all times after first hit)
        # ====================================================================
        if bounces > 0:
            time_after_hit = sum(travel_times[1:])
        else:
            time_after_hit = 0  # No hits occurred

        self.wait(0.5)

        # ====================================================================
        # Generate question and answer
        # ====================================================================
        question_types = [
            ("How many seconds did the ball travel before hitting an object?", time_before_hit),
            ("How many seconds did the ball travel after hitting an object?", time_after_hit),
            ("What was the total travel time of the ball?", sum(travel_times))
        ]

        # Select a random question type
        selected_question, answer = random.choice(question_types)

        # Handle edge cases: if no bounces, only ask about total time
        if bounces == 0 and "hitting" in selected_question:
            selected_question = "What was the total travel time of the ball?"
            answer = sum(travel_times)

        # Store question info for reasoning trace
        self.selected_question = selected_question
        self.answer = answer
        self.bounces = bounces
        self.travel_times = travel_times
        self.time_before_hit = time_before_hit
        self.time_after_hit = time_after_hit

        # ====================================================================
        # Display question screen
        # ====================================================================
        self.clear()
        self.wait(0.2)
        self.log_event("All objects fade out, question screen appears")

        # Create question text
        question_lines = [
            selected_question,
            "",
            "Answer in seconds (rounded to 1 decimal place)"
        ]

        # Display question
        question_texts = []
        line_height = 0.8
        start_y = 2.0

        for i, line in enumerate(question_lines):
            if line:  # Skip empty lines for spacing
                text = Text(line, font_size=28, weight=BOLD if i == 0 else NORMAL)
                text.move_to(UP * (start_y - i * line_height))
                question_texts.append(text)

        self.play(*[Write(text) for text in question_texts])
        self.wait(3)
        self.log_event("Question displayed and remains on screen")

        # ====================================================================
        # Format answer and save output files
        # ====================================================================
        formatted_answer = f"{answer:.1f}"

        # Save solution file (just the answer)
        with open(f"solutions/bounce_ball_n{self.num_bounces}_seed{self.seed}.txt", "w") as f:
            f.write(formatted_answer)

        # Save question text file
        question_text_content = (
            f"{selected_question}\n"
            "Answer in seconds (rounded to 1 decimal place)"
        )
        with open(f"question_text/bounce_ball_n{self.num_bounces}_seed{self.seed}.txt", "w") as f:
            f.write(question_text_content)

        # ====================================================================
        # Generate comprehensive reasoning trace
        # ====================================================================
        self.build_reasoning_trace()

        # Save reasoning trace file
        with open(f"reasoning_traces/bounce_ball_n{self.num_bounces}_seed{self.seed}.txt", "w") as f:
            f.write("\n".join(self.reasoning_trace))

    def build_reasoning_trace(self):
        """
        Build a comprehensive, step-by-step reasoning trace.
        This explains how to solve the puzzle systematically.
        Follows the same structure as cube_path.py for consistency.
        """
        self.reasoning_trace = []

        # ====================================================================
        # Introduction
        # ====================================================================
        self.reasoning_trace.append(f"**Question:** {self.selected_question}")
        self.reasoning_trace.append("")
        self.reasoning_trace.append("Let's solve this step by step.")
        self.reasoning_trace.append("")

        # ====================================================================
        # Scene description with timestamps
        # ====================================================================
        self.reasoning_trace.append("### Scene Description")
        self.reasoning_trace.append("")
        self.reasoning_trace.append("The video shows a ball moving horizontally and bouncing off walls. Here's what happens in chronological order:")
        self.reasoning_trace.append("")

        for event in self.scene_events:
            time_str = self.format_time(event['time'])
            self.reasoning_trace.append(f"At {time_str}, {event['description']}")

        self.reasoning_trace.append("")

        # ====================================================================
        # Step 1: Understand the problem
        # ====================================================================
        self.reasoning_trace.append("### Step 1: Understand the problem")
        if self.bounces > 0:
            self.reasoning_trace.append(f"The ball travels and bounces off **{self.bounces} wall(s)** during its journey.")
            self.reasoning_trace.append(f"This creates **{self.bounces + 1} distinct segments** of movement:")
            self.reasoning_trace.append(f"- {self.bounces} segment(s) leading up to and between walls")
            self.reasoning_trace.append(f"- 1 final segment after the last bounce")
        else:
            self.reasoning_trace.append("The ball travels across the screen without hitting any walls.")
            self.reasoning_trace.append("There is only **1 segment** of continuous movement.")
        self.reasoning_trace.append("")

        # ====================================================================
        # Step 2: Identify the travel times
        # ====================================================================
        self.reasoning_trace.append("### Step 2: Identify the travel times for each segment")
        self.reasoning_trace.append("")
        self.reasoning_trace.append("By carefully observing the video, we can measure the duration of each segment:")
        self.reasoning_trace.append("")

        for i, t in enumerate(self.travel_times):
            if i == 0:
                if self.bounces > 0:
                    self.reasoning_trace.append(f"**Segment {i + 1}** (before first wall): {t:.1f}s")
                else:
                    self.reasoning_trace.append(f"**Segment {i + 1}** (entire journey): {t:.1f}s")
            elif i == len(self.travel_times) - 1:
                self.reasoning_trace.append(f"**Segment {i + 1}** (after last bounce): {t:.1f}s")
            else:
                self.reasoning_trace.append(f"**Segment {i + 1}** (between walls): {t:.1f}s")

        self.reasoning_trace.append("")

        # ====================================================================
        # Step 3: Calculate the requested value
        # ====================================================================
        self.reasoning_trace.append("### Step 3: Calculate the requested value")
        self.reasoning_trace.append("")

        if "before hitting" in self.selected_question:
            # Time before first hit
            self.reasoning_trace.append("**The question asks for time BEFORE hitting an object.**")
            self.reasoning_trace.append("")
            self.reasoning_trace.append(f"This is simply the duration of the first segment, before the ball reaches the first wall:")
            self.reasoning_trace.append(f"- Time before hitting = **{self.time_before_hit:.1f}s**")

        elif "after hitting" in self.selected_question:
            # Time after first hit
            self.reasoning_trace.append("**The question asks for time AFTER hitting an object.**")
            self.reasoning_trace.append("")
            self.reasoning_trace.append(f"This includes all segments after the first wall collision.")

            if len(self.travel_times) > 1:
                after_segments = [f"{t:.1f}s" for t in self.travel_times[1:]]
                segments_sum = " + ".join(after_segments)
                self.reasoning_trace.append(f"- Time after hitting = {segments_sum}")
                self.reasoning_trace.append(f"- Time after hitting = **{self.time_after_hit:.1f}s**")
            else:
                self.reasoning_trace.append(f"- Time after hitting = **{self.time_after_hit:.1f}s**")

        else:
            # Total time
            self.reasoning_trace.append("**The question asks for total travel time.**")
            self.reasoning_trace.append("")
            self.reasoning_trace.append(f"This is the sum of all segments:")

            all_segments = " + ".join([f"{t:.1f}s" for t in self.travel_times])
            total = sum(self.travel_times)
            self.reasoning_trace.append(f"- Total time = {all_segments}")
            self.reasoning_trace.append(f"- Total time = **{total:.1f}s**")

        self.reasoning_trace.append("")

        # ====================================================================
        # Step 4: Verify the answer
        # ====================================================================
        self.reasoning_trace.append("### Step 4: Verify the answer")
        self.reasoning_trace.append("")

        # Show all three values for context
        self.reasoning_trace.append("Let's verify by calculating all three timing values:")
        self.reasoning_trace.append("")
        self.reasoning_trace.append(f"1. **Time before hitting**: {self.time_before_hit:.1f}s")
        self.reasoning_trace.append(f"2. **Time after hitting**: {self.time_after_hit:.1f}s")
        self.reasoning_trace.append(f"3. **Total time**: {sum(self.travel_times):.1f}s")
        self.reasoning_trace.append("")

        # Sanity check: time_before + time_after should equal total
        if self.bounces > 0:
            calculated_total = self.time_before_hit + self.time_after_hit
            actual_total = sum(self.travel_times)
            self.reasoning_trace.append(f"Verification: {self.time_before_hit:.1f}s + {self.time_after_hit:.1f}s = {calculated_total:.1f}s ✓")

        self.reasoning_trace.append("")

        # ====================================================================
        # Final answer
        # ====================================================================
        self.reasoning_trace.append("### Final Answer")
        self.reasoning_trace.append("")
        self.reasoning_trace.append(f"Based on our analysis, the answer to \"{self.selected_question}\" is:")
        self.reasoning_trace.append("")
        self.reasoning_trace.append(f"\\boxed{{{self.answer:.1f}}}")


# ============================================================================
# Main execution
# ============================================================================
if __name__ == "__main__":
    # Generate the ball hits wall video
    scene = bounce_ball()
    scene.render()

    # ========================================================================
    # Move output file to questions directory with descriptive name
    # ========================================================================
    output = Path("manim_output/videos/1080p30/bounce_ball.mp4")
    if output.exists():
        filename = f"bounce_ball_n{scene.num_bounces}_seed{scene.seed}.mp4"
        shutil.move(str(output), f"questions/{filename}")
        print(f"✓ Video saved: questions/{filename}")
    else:
        # Debug: Print what files actually exist
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
