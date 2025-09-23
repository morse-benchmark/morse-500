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

    def construct(self):
        # Constrain count to reasonable bounds
        bounces = max(1, min(self.num_bounces, 5))
        
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
        
        # Track total times for questions
        time_before_hit = 0
        time_after_hit = 0
        current_time = 0
        
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
            else:  # Moving left to wall
                move_distance = ball_current_x - wall_x - 0.5  # Stop just before wall
                move_direction = LEFT * move_distance
            
            # Move ball to wall
            travel_time = travel_times[bounce_idx]
            self.play(ball.animate.shift(move_direction), run_time=travel_time)
            
            # Track time before first hit
            if bounce_idx == 0:
                time_before_hit = travel_time
            current_time += travel_time
            
            # Bounce effect - change color and slight squish
            bounce_color = ball_colors[(bounce_idx + 1) % len(ball_colors)]
            self.play(
                ball.animate.scale([1.2, 0.8, 1]).set_fill(bounce_color), 
                run_time=0.3
            )
            self.play(ball.animate.scale([1/1.2, 1/0.8, 1]), run_time=0.2)
            
            self.wait(0.2)
        
        # Final movement after last bounce (or only movement if no bounces)
        if bounces > 0:
            # Move away from last wall
            last_wall_x = wall_positions[bounces-1][0]
            ball_x = ball.get_center()[0]
            
            if last_wall_x > 0:  # Last wall on right, move left
                final_move = LEFT * 4
            else:  # Last wall on left, move right
                final_move = RIGHT * 4
        else:
            # No bounces, just move across screen
            final_move = RIGHT * 8
            
        final_travel_time = travel_times[-1]
        self.play(ball.animate.shift(final_move), run_time=final_travel_time)
        
        # Calculate time after hitting (sum of all times after first hit)
        if bounces > 0:
            time_after_hit = sum(travel_times[1:]) + final_travel_time
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
