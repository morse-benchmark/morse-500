from manim import *
import random
import math
import os
import shutil
from pathlib import Path

# Setup directories
Path("questions").mkdir(exist_ok=True)
Path("solutions").mkdir(exist_ok=True)
Path("question_text").mkdir(exist_ok=True)
Path("reasoning_traces").mkdir(exist_ok=True)

config.media_dir = "manim_output"
config.verbosity = "WARNING"
config.pixel_height = 1080
config.pixel_width = 1920
config.frame_rate = 30
config.preview = False

# NUM_BOUNCES [1-5] range
# NUM_BOUNCES=1 python3 bounceball.py
class BallHitsWall(Scene):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Set random seed for reproducibility
        self.seed = random.randint(1000, 9999)
        random.seed(self.seed)

        # Parameters - difficulty controls number of bounces
        self.num_bounces = int(os.getenv("NUM_BOUNCES", 2))

        # Initialize reasoning trace
        self.reasoning_trace = []
        self.reasoning_trace.append(f"Problem: Ball hitting walls")
        self.reasoning_trace.append(f"Number of bounces: {self.num_bounces}")
        self.reasoning_trace.append(f"Random Seed: {self.seed}")
        self.reasoning_trace.append("")

    def construct(self):
        # Constrain count to reasonable bounds
        bounces = max(1, min(self.num_bounces, 5))

        self.reasoning_trace.append(f"Actual bounces (constrained): {bounces}")

        # Ball colors for different states
        ball_colors = [BLUE, RED, GREEN, YELLOW, PURPLE, ORANGE]

        # Create ball
        ball = Circle(radius=0.3, color=ball_colors[0], fill_opacity=1)

        # Create walls at different positions
        wall_positions = [RIGHT * 3, LEFT * 3, RIGHT * 2, LEFT * 2, RIGHT * 1]
        walls = []

        # Set initial ball position based on first wall
        if bounces > 0:
            first_wall_x = wall_positions[0][0]  # Get x coordinate of first wall
            if first_wall_x > 0:  # Wall on right
                ball.move_to(LEFT * 5)
            else:  # Wall on left
                ball.move_to(RIGHT * 5)

        self.add(ball)

        # Generate random travel times
        travel_times = []
        for i in range(bounces + 1):  # One more segment than bounces
            travel_times.append(random.uniform(1.5, 4.0))

        self.reasoning_trace.append("Generated travel times:")
        for i, t in enumerate(travel_times):
            self.reasoning_trace.append(f"  Segment {i}: {t:.1f}s")
        self.reasoning_trace.append("")

        # Track total times for questions
        time_before_hit = 0
        time_after_hit = 0
        current_time = 0

        # Add initial scene description
        ball_start_x = ball.get_center()[0]
        ball_color = ball_colors[0].name if hasattr(ball_colors[0], 'name') else "blue"
        if ball_start_x < 0:
            ball_position = f"on the left side at x = {ball_start_x:.1f}"
        else:
            ball_position = f"on the right side at x = {ball_start_x:.1f}"

        self.reasoning_trace.append("=== CHRONOLOGICAL SCENE DESCRIPTION ===")
        self.reasoning_trace.append("")
        self.reasoning_trace.append(f"At time 0.0s, the scene begins with a ball positioned {ball_position}. The ball has a circular shape with a radius of 0.3 units.")

        # Animate ball movement with bounces
        for bounce_idx in range(bounces):
            # Create wall for this bounce
            wall = Rectangle(width=0.2, height=2, color=GREY, fill_opacity=1)
            wall.move_to(wall_positions[bounce_idx])
            walls.append(wall)
            self.add(wall)

            # Calculate movement direction and distance
            wall_x = wall_positions[bounce_idx][0]
            ball_current_x = ball.get_center()[0]

            if wall_x > ball_current_x:  # Moving right to wall
                move_distance = wall_x - ball_current_x - 0.5  # Stop just before wall
                move_direction = RIGHT * move_distance
                direction_text = "rightward"
            else:  # Moving left to wall
                move_distance = ball_current_x - wall_x - 0.5  # Stop just before wall
                move_direction = LEFT * move_distance
                direction_text = "leftward"

            # Move ball to wall
            travel_time = travel_times[bounce_idx]

            # Add chronological description for this movement
            start_time = current_time
            end_time = current_time + travel_time
            wall_side = "right" if wall_x > 0 else "left"

            if bounce_idx == 0:
                self.reasoning_trace.append("")
                self.reasoning_trace.append(f"From {start_time:.1f}s to {end_time:.1f}s (duration: {travel_time:.1f}s), the ball travels {direction_text} toward a wall positioned on the {wall_side} side at x = {wall_x:.1f}. The ball covers a distance of approximately {abs(move_distance):.1f} units during this segment.")
            else:
                self.reasoning_trace.append(f"From {start_time:.1f}s to {end_time:.1f}s (duration: {travel_time:.1f}s), the ball travels {direction_text} toward the next wall positioned on the {wall_side} side at x = {wall_x:.1f}. This is bounce #{bounce_idx + 1}.")

            self.play(ball.animate.shift(move_direction), run_time=travel_time)

            # Add description of reaching the wall
            self.reasoning_trace.append(f"At {end_time:.1f}s, the ball reaches the wall and bounces back.")

            # Track time before first hit
            if bounce_idx == 0:
                time_before_hit = travel_time
            current_time += travel_time

        # Final movement after last bounce (or only movement if no bounces)
        if bounces > 0:
            # Move away from last wall
            last_wall_x = wall_positions[bounces-1][0]
            ball_x = ball.get_center()[0]

            if last_wall_x > 0:  # Last wall on right, move left
                final_move = LEFT * 4
                final_direction_text = "leftward"
            else:  # Last wall on left, move right
                final_move = RIGHT * 4
                final_direction_text = "rightward"
        else:
            # No bounces, just move across screen
            final_move = RIGHT * 8
            final_direction_text = "rightward"

        final_travel_time = travel_times[-1]
        start_time = current_time
        end_time = current_time + final_travel_time

        # Add chronological description for final movement
        self.reasoning_trace.append("")
        if bounces > 0:
            self.reasoning_trace.append(f"From {start_time:.1f}s to {end_time:.1f}s (duration: {final_travel_time:.1f}s), after bouncing off the last wall, the ball travels {final_direction_text} away from the wall. The ball moves approximately {abs(final_move[0]):.1f} units during this final segment.")
        else:
            self.reasoning_trace.append(f"From {start_time:.1f}s to {end_time:.1f}s (duration: {final_travel_time:.1f}s), the ball travels {final_direction_text} across the screen without hitting any walls.")

        self.play(ball.animate.shift(final_move), run_time=final_travel_time)

        self.reasoning_trace.append(f"At {end_time:.1f}s, the ball's movement concludes.")

        # Calculate time after hitting (sum of all times after first hit)
        if bounces > 0:
            time_after_hit = sum(travel_times[1:])
        else:
            time_after_hit = 0  # No hits occurred

        self.wait(0.5)
        
        # Generate question and answer
        question_types = [
            ("How many seconds did the ball travel before hitting an object?", time_before_hit),
            ("How many seconds did the ball travel after hitting an object?", time_after_hit),
            ("What was the total travel time of the ball?", sum(travel_times))
        ]

        # Select a random question type
        selected_question, answer = random.choice(question_types)

        # Add reasoning section that derives the answer from observations
        self.reasoning_trace.append("")
        self.reasoning_trace.append("=== REASONING AND ANSWER DERIVATION ===")
        self.reasoning_trace.append("")
        self.reasoning_trace.append("After observing all movements in the scene, we can now calculate the key timing information:")
        self.reasoning_trace.append("")

        # Provide detailed reasoning based on observations
        if bounces > 0:
            self.reasoning_trace.append(f"Time before first hit: The ball traveled for {time_before_hit:.1f}s from the start (0.0s) until it reached the first wall at {time_before_hit:.1f}s.")
            self.reasoning_trace.append("")

            # Calculate and explain time after hitting
            after_segments = []
            cumulative = time_before_hit
            for i in range(1, len(travel_times)):
                seg_time = travel_times[i]
                after_segments.append(f"{seg_time:.1f}s")
                cumulative += seg_time

            if len(after_segments) > 1:
                segments_sum = " + ".join(after_segments)
                self.reasoning_trace.append(f"Time after hitting: After the first bounce, the ball continued traveling through {len(after_segments)} additional segment(s). The durations were: {segments_sum} = {time_after_hit:.1f}s total.")
            else:
                self.reasoning_trace.append(f"Time after hitting: After the first bounce, the ball traveled for {time_after_hit:.1f}s in the final segment.")
        else:
            self.reasoning_trace.append(f"The ball did not hit any walls during its movement. It simply traveled across the screen for {sum(travel_times):.1f}s.")

        self.reasoning_trace.append("")
        all_segments = " + ".join([f"{t:.1f}s" for t in travel_times])
        self.reasoning_trace.append(f"Total travel time: Adding all movement segments together: {all_segments} = {sum(travel_times):.1f}s.")
        self.reasoning_trace.append("")
        self.reasoning_trace.append(f"Question asked: {selected_question}")
        self.reasoning_trace.append(f"Answer: {answer:.1f}s")
        
        # Handle edge cases
        if bounces == 0 and "hitting" in selected_question:
            # If no bounces, ask about total time instead
            selected_question = "What was the total travel time of the ball?"
            answer = sum(travel_times)
        
        # Create question text
        question_lines = [
            selected_question,
            "",
            "Answer in seconds (rounded to 1 decimal place)"
        ]
        
        self.clear()
        self.wait(0.2)
        
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
        
        # Format answer to 1 decimal place
        formatted_answer = f"{answer:.1f}"
        
        # Save solution and question text
        with open(f"solutions/ballhitswall_n{self.num_bounces}_seed{self.seed}.txt", "w") as f:
            f.write(formatted_answer)
        
        question_text_content = (
            f"{selected_question}\n"
            "Answer in seconds (rounded to 1 decimal place)"
        )
        with open(f"question_text/ballhitswall_n{self.num_bounces}_seed{self.seed}.txt", "w") as f:
            f.write(question_text_content)

        # Save detailed reasoning trace
        self.reasoning_trace.append("")
        self.reasoning_trace.append(f"Final Answer: {formatted_answer}")
        with open(f"reasoning_traces/ballhitswall_n{self.num_bounces}_seed{self.seed}.txt", "w") as f:
            f.write("\n".join(self.reasoning_trace))


if __name__ == "__main__":
    # Generate the ball hits wall video
    scene = BallHitsWall()
    scene.render()

    # Move the output file with descriptive name
    output = Path("manim_output/videos/1080p30/BallHitsWall.mp4")
    if output.exists():
        filename = f"ballhitswall_n{scene.num_bounces}_seed{scene.seed}.mp4"
        shutil.move(str(output), f"questions/{filename}")
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
