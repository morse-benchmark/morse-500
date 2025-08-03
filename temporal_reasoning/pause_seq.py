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

class pause_seq(Scene):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Set random seed for reproducibility
        self.seed = random.randint(1000, 9999)
        random.seed(self.seed)
        
        # Parameters
        self.difficulty = int(os.getenv("DIFFICULTY", 1))
        
    def construct(self):
        # Number of sequences based on difficulty (3-7 sequences)
        num_sequences = self.difficulty + 2
        
        # Generate sequences based on difficulty
        seqs = []
        for i in range(num_sequences):
            # Each sequence has 5 numbers, range scales with difficulty
            sequence = [random.randint(1, 100 * self.difficulty) for _ in range(5)]
            seqs.append(sequence)
        
        # Display all sequences with pauses between them
        for seq_idx, seq in enumerate(seqs):
            # Display each number in the sequence one by one
            for num_idx, num in enumerate(seq):
                # Create and display single number
                num_text = Text(str(num), font_size=48, color=WHITE)         
                self.play(FadeIn(num_text), run_time=0.2)
                self.wait(0.1)  # Brief pause between numbers
                self.play(FadeOut(num_text, run_time=0.2))

            # Clear the sequence
            # current_numbers = [self.mobjects[-5+i] for i in range(5)]  # Get last 5 text objects
            # self.play(*[FadeOut(num) for num in current_numbers], run_time=0.3)
            
            # Pause between sequences (except after the last one)
            if seq_idx < len(seqs) - 1:
                self.wait(2)  # Pause duration
        
        # Wait before showing question
        self.wait(1)
        
        # Generate question based on difficulty
        ordinal_numbers = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh"]
        
        if self.difficulty == 1:
            # Fixed: between first and second pause
            start_pause = 1
            end_pause = 2
        else:
            # Randomized based on available pauses
            max_pause = num_sequences - 1  # Number of pauses is sequences - 1
            
            # Generate valid combinations
            valid_combinations = []
            for start in range(1, max_pause):
                for end in range(start + 1, max_pause + 1):
                    valid_combinations.append((start, end))
            
            # Choose random combination
            start_pause, end_pause = random.choice(valid_combinations)
        
        # Calculate the answer (sum of sequences between the specified pauses)
        # If we have: [Seq0] PAUSE1 [Seq1] PAUSE2 [Seq2] PAUSE3 [Seq3]...
        # "Between first and second pause" means Seq1 (index 1)
        # "Between second and third pause" means Seq2 (index 2)
        # So sequences between pause X and pause Y are at indices start_pause to end_pause-1
        sequences_to_sum = seqs[start_pause:end_pause]
        answer = sum(sum(seq) for seq in sequences_to_sum)
        
        # Create question text
        start_ordinal = ordinal_numbers[start_pause - 1]
        end_ordinal = ordinal_numbers[end_pause - 1]
        
        question_text = f"What is the sum of the sequences that appeared between \nthe {start_ordinal} and {end_ordinal} pause?"
        question = Text(question_text, font_size=24, weight=BOLD).move_to(UP * 2)
        
        # Display question
        self.play(FadeIn(question), run_time=0.8)
        self.wait(0.5)
        
        # Show additional information
        info_text = Text(
            f"Each sequence contained 5 numbers.",
            font_size=24,
            color=GRAY
        ).move_to(UP * 1)
        
        self.play(FadeIn(info_text), run_time=0.5)
        self.wait(2)
        
        # Show answer instruction
        note_text = Text(
            f"Return the answer as a number.",
            font_size=32,
            color=YELLOW
        ).move_to(DOWN * 2)
        
        self.play(FadeIn(note_text, shift=UP*0.3), run_time=0.8)
        self.wait(3)
        
        # Save solution and question text
        question_for_file = f"What is the sum of the sequences that appeared between the {start_ordinal} and {end_ordinal} pause?\nReturn the answer as a number."
        
        with open(f"solutions/pause_seq_d{self.difficulty}_seed{self.seed}.txt", "w") as f:
            f.write(str(answer))
        with open(f"question_text/pause_seq_d{self.difficulty}_seed{self.seed}.txt", "w") as f:
            f.write(question_for_file)
        
        # Debug info for verification
        if os.getenv("DEBUG", "false").lower() == "true":
            print(f"Sequences: {seqs}")
            print(f"Between pause {start_pause} and {end_pause}: {sequences_to_sum}")
            print(f"Answer: {answer}")

# Generate pause sequence video
scene = pause_seq()
scene.render()

# Move the output file with descriptive name
# Try multiple possible output paths
possible_paths = [
    Path("manim_output/videos/pause_seq/1080p30/pause_seq.mp4"),
    Path("manim_output/videos/1080p30/pause_seq.mp4"),
    Path("manim_output/videos/1080p30/1080p30/pause_seq.mp4")
]

output_found = False
for output_path in possible_paths:
    if output_path.exists():
        filename = f"pause_seq_d{scene.difficulty}_seed{scene.seed}.mp4"
        shutil.move(str(output_path), f"questions/{filename}")
        output_found = True
        print(f"Video moved from {output_path} to questions/{filename}")
        break

if not output_found:
    # Debug: Print what files actually exist
    videos_dir = Path("manim_output/videos")
    if videos_dir.exists():
        print(f"Available folders in videos/: {list(videos_dir.iterdir())}")
        for folder in videos_dir.iterdir():
            if folder.is_dir():
                print(f"Contents of {folder}: {list(folder.iterdir())}")
                for subfolder in folder.iterdir():
                    if subfolder.is_dir():
                        print(f"Files in {subfolder}: {list(subfolder.iterdir())}")
    else:
        print("manim_output/videos directory doesn't exist")

# Final cleanup
if os.path.exists("manim_output"):
    shutil.rmtree("manim_output")