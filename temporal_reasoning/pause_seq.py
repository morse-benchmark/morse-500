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

class pause_seq(Scene):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Set random seed for reproducibility
        self.seed = random.randint(1000, 9999)
        random.seed(self.seed)

        # Parameters
        self.difficulty = int(os.getenv("DIFFICULTY", 1))

        # Initialize reasoning trace
        self.reasoning_trace = []
        self.reasoning_trace.append(f"Problem: Pause sequence summation")
        self.reasoning_trace.append(f"Difficulty: {self.difficulty}")
        self.reasoning_trace.append(f"Random Seed: {self.seed}")
        self.reasoning_trace.append("")
        
    def construct(self):
        # Number of sequences based on difficulty (3-7 sequences)
        num_sequences = self.difficulty + 2
        
        # Generate sequences based on difficulty
        seqs = []
        for i in range(num_sequences):
            # Each sequence has 5 numbers, range scales with difficulty
            sequence = [random.randint(1, 100 * self.difficulty) for _ in range(5)]
            seqs.append(sequence)

        self.reasoning_trace.append(f"Number of sequences: {num_sequences}")
        self.reasoning_trace.append("Sequences:")
        for i, seq in enumerate(seqs):
            self.reasoning_trace.append(f"  Sequence {i}: {seq}")

        # Add chronological scene description
        self.reasoning_trace.append("")
        self.reasoning_trace.append("=== CHRONOLOGICAL REASONING TRACE ===")
        self.reasoning_trace.append("")
        self.reasoning_trace.append("Scene begins: The video starts with a blank screen. I'm about to see a series of numbers displayed one at a time, organized into sequences with pauses between them.")
        self.reasoning_trace.append("")

        # Track cumulative time for chronological trace
        current_time = 0.0
        pause_count = 0

        # Display all sequences with pauses between them
        for seq_idx, seq in enumerate(seqs):
            # Add chronological description for the start of this sequence
            if seq_idx == 0:
                self.reasoning_trace.append(f"At t={current_time:.1f}s: The first sequence begins. I see numbers appearing one after another.")
            else:
                self.reasoning_trace.append(f"At t={current_time:.1f}s: After the pause, sequence {seq_idx + 1} begins. Numbers start appearing again.")

            # Display each number in the sequence one by one
            for num_idx, num in enumerate(seq):
                # Add chronological description for each number
                self.reasoning_trace.append(f"  t={current_time:.1f}s: Number {num_idx + 1} of this sequence appears on screen: {num}")

                # Create and display single number
                num_text = Text(str(num), font_size=48, color=WHITE)
                self.play(FadeIn(num_text), run_time=0.2)
                current_time += 0.2

                self.wait(0.1)  # Brief pause between numbers
                current_time += 0.1

                self.reasoning_trace.append(f"  t={current_time:.1f}s: The number fades out.")
                self.play(FadeOut(num_text, run_time=0.2))
                current_time += 0.2

            # Clear the sequence
            # current_numbers = [self.mobjects[-5+i] for i in range(5)]  # Get last 5 text objects
            # self.play(*[FadeOut(num) for num in current_numbers], run_time=0.3)

            # Pause between sequences (except after the last one)
            if seq_idx < len(seqs) - 1:
                pause_count += 1
                ordinal = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh"][pause_count - 1]
                self.reasoning_trace.append(f"At t={current_time:.1f}s: The {ordinal} PAUSE begins. The screen is blank - no numbers are showing.")
                self.wait(2)  # Pause duration
                current_time += 2.0
                self.reasoning_trace.append(f"At t={current_time:.1f}s: The {ordinal} pause ends.")
        
        # Add summary after all events
        self.reasoning_trace.append("")
        self.reasoning_trace.append(f"At t={current_time:.1f}s: All sequences have finished displaying. The video showed {num_sequences} sequences with {pause_count} pauses between them.")

        # Wait before showing question
        self.wait(1)
        current_time += 1.0

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

        self.reasoning_trace.append("")
        self.reasoning_trace.append(f"At t={current_time:.1f}s: The question appears on screen asking for the sum of sequences between the {start_ordinal} and {end_ordinal} pause.")
        self.reasoning_trace.append("")
        self.reasoning_trace.append("=== DERIVING THE ANSWER ===")
        self.reasoning_trace.append("")
        self.reasoning_trace.append(f"Now I need to think about what appeared between the {start_ordinal} and {end_ordinal} pause.")
        self.reasoning_trace.append("")

        # Add narrative reasoning about the structure
        self.reasoning_trace.append("Let me recall the structure of what I saw:")
        for i in range(num_sequences):
            if i == 0:
                self.reasoning_trace.append(f"- Sequence 1 appeared at the start (before any pauses)")
            else:
                if i < num_sequences - 1:
                    pause_before = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh"][i - 1]
                    self.reasoning_trace.append(f"- Then the {pause_before} pause occurred")
                    self.reasoning_trace.append(f"- Sequence {i + 1} appeared after the {pause_before} pause")
                else:
                    pause_before = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh"][i - 1]
                    self.reasoning_trace.append(f"- Then the {pause_before} pause occurred")
                    self.reasoning_trace.append(f"- Sequence {i + 1} appeared last (after all pauses)")

        self.reasoning_trace.append("")
        self.reasoning_trace.append(f"The question asks for sequences BETWEEN the {start_ordinal} and {end_ordinal} pause.")
        self.reasoning_trace.append(f"This means I need sequences that appeared AFTER the {start_ordinal} pause and BEFORE the {end_ordinal} pause.")
        self.reasoning_trace.append("")

        # Identify which sequences to sum
        sequences_description = []
        for i in range(start_pause, end_pause):
            sequences_description.append(f"Sequence {i + 1}")

        if len(sequences_description) == 1:
            self.reasoning_trace.append(f"Only {sequences_description[0]} appeared between these two pauses.")
        else:
            sequences_list = ", ".join(sequences_description[:-1]) + f", and {sequences_description[-1]}"
            self.reasoning_trace.append(f"The sequences that appeared between these pauses are: {sequences_list}.")

        self.reasoning_trace.append("")
        self.reasoning_trace.append("Now let me calculate the sum of all numbers in these sequences:")
        self.reasoning_trace.append("")

        # Show the calculation step by step
        running_total = 0
        for idx, seq_idx in enumerate(range(start_pause, end_pause)):
            seq = seqs[seq_idx]
            seq_sum = sum(seq)
            running_total += seq_sum
            numbers_str = " + ".join(str(n) for n in seq)
            self.reasoning_trace.append(f"Sequence {seq_idx + 1}: {numbers_str} = {seq_sum}")
            self.reasoning_trace.append(f"  Running total: {running_total}")

        self.reasoning_trace.append("")
        self.reasoning_trace.append(f"Therefore, the sum of all numbers in the sequences between the {start_ordinal} and {end_ordinal} pause is: {answer}")

        self.reasoning_trace.append("")
        self.reasoning_trace.append(f"Question: Between {start_ordinal} and {end_ordinal} pause")
        self.reasoning_trace.append(f"Sequences to sum (indices {start_pause} to {end_pause-1}): {sequences_to_sum}")
        self.reasoning_trace.append(f"Answer: {answer}")

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

        # Save detailed reasoning trace
        self.reasoning_trace.append("")
        self.reasoning_trace.append(f"Final Answer: {answer}")
        with open(f"reasoning_traces/pause_seq_d{self.difficulty}_seed{self.seed}.txt", "w") as f:
            f.write("\n".join(self.reasoning_trace))
        
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