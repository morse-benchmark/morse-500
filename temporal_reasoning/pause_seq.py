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


class pause_seq(Scene):
    """
    A scene that generates a pause sequence summation puzzle:
    - Shows multiple sequences of numbers with pauses between them
    - User must sum all numbers between two specific pauses
    - Generates question video, solution, and reasoning trace
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Set random seed for reproducibility of sequence generation
        self.seed = random.randint(1000, 9999)
        random.seed(self.seed)

        self.num_sequences = int(os.getenv("NUM_SEQUENCES", 3))  # e.g. 3–7

        # Initialize reasoning trace storage
        self.reasoning_trace = []

        # Track scene events with timestamps for chronological reasoning
        # This will be populated during construct() when renderer is available
        self.scene_events = []

    def log_event(self, description):
        """
        Log a scene event with video timestamp.
        Uses Manim's internal renderer.time which tracks actual video playback time,
        not wall-clock execution time.
        """
        # Get current video time from Manim's renderer
        # self.renderer.time tracks the cumulative duration of all animations/waits
        current_time = self.renderer.time

        self.scene_events.append({"time": current_time, "description": description})

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
        # Generate sequences based on difficulty
        # ====================================================================
        # Number of sequences based on difficulty (3-7 sequences)
        num_sequences = self.num_sequences

        seqs = []
        max_val = 50 * num_sequences
        for i in range(num_sequences):
            sequence = [random.randint(1, max_val) for _ in range(5)]
            seqs.append(sequence)

        # Store sequence information for later use in reasoning trace
        self.seqs = seqs
        self.pause_count = num_sequences - 1  # Number of pauses = sequences - 1

        # ====================================================================
        # Display all sequences with pauses between them
        # ====================================================================
        self.log_event("Video begins with blank screen")

        for seq_idx, seq in enumerate(seqs):
            # Log sequence start
            if seq_idx == 0:
                self.log_event(f"First sequence begins")
            else:
                self.log_event(f"Sequence {seq_idx + 1} begins (after pause {seq_idx})")

            # Display each number in the sequence one by one
            for num_idx, num in enumerate(seq):
                # Create and display single number
                num_text = Text(str(num), font_size=48, color=WHITE)

                self.log_event(
                    f"Number {num} appears (sequence {seq_idx + 1}, position {num_idx + 1})"
                )
                self.play(FadeIn(num_text), run_time=0.2)
                self.wait(0.1)  # Brief pause to read the number

                self.log_event(f"Number {num} fades out")
                self.play(FadeOut(num_text), run_time=0.2)

            # Pause between sequences (except after the last one)
            if seq_idx < len(seqs) - 1:
                ordinal = [
                    "first",
                    "second",
                    "third",
                    "fourth",
                    "fifth",
                    "sixth",
                    "seventh",
                    "eighth",
                    "ninth",
                    "tenth",
                    "eleventh",
                    "twelfth",
                ][seq_idx]
                self.log_event(f"The {ordinal} PAUSE begins (screen is blank)")
                self.wait(2)  # Pause duration
                self.log_event(f"The {ordinal} PAUSE ends")

        self.log_event(f"All {num_sequences} sequences have been displayed")

        # Wait before showing question
        self.wait(1)

        # ====================================================================
        # Generate question based on difficulty
        # ====================================================================
        ordinal_numbers = [
            "first",
            "second",
            "third",
            "fourth",
            "fifth",
            "sixth",
            "seventh",
        ]

        # Randomized based on available pauses
        max_pause = num_sequences - 1  # Number of pauses is sequences - 1

        # Generate valid combinations
        valid_combinations = []
        for start in range(1, max_pause):
            for end in range(start + 1, max_pause + 1):
                valid_combinations.append((start, end))

        # Choose random combination
        start_pause, end_pause = random.choice(valid_combinations)

        # Store for reasoning trace
        self.start_pause = start_pause
        self.end_pause = end_pause

        # ====================================================================
        # Calculate the answer
        # ====================================================================
        # If we have: [Seq0] PAUSE1 [Seq1] PAUSE2 [Seq2] PAUSE3 [Seq3]...
        # "Between first and second pause" means Seq1 (index 1)
        # "Between second and third pause" means Seq2 (index 2)
        # So sequences between pause X and pause Y are at indices start_pause to end_pause-1
        sequences_to_sum = seqs[start_pause:end_pause]
        answer = sum(sum(seq) for seq in sequences_to_sum)

        # Store for later use
        self.sequences_to_sum = sequences_to_sum
        self.answer = answer

        # Create question text
        start_ordinal = ordinal_numbers[start_pause - 1]
        end_ordinal = ordinal_numbers[end_pause - 1]
        self.start_ordinal = start_ordinal
        self.end_ordinal = end_ordinal

        # ====================================================================
        # Display the question
        # ====================================================================
        question_text = f"What is the sum of the sequences that appeared between \nthe {start_ordinal} and {end_ordinal} pause?"
        question = Text(question_text, font_size=24, weight=BOLD).move_to(UP * 2)

        self.log_event(
            f"Question appears: Sum sequences between {start_ordinal} and {end_ordinal} pause"
        )
        self.play(FadeIn(question), run_time=0.8)
        self.wait(0.5)

        # Show additional information
        info_text = Text(
            f"Each sequence contained 5 numbers.", font_size=24, color=GRAY
        ).move_to(UP * 1)

        self.log_event("Additional hint displayed: Each sequence has 5 numbers")
        self.play(FadeIn(info_text), run_time=0.5)
        self.wait(2)

        # Show answer instruction
        note_text = Text(
            f"Return the answer as a number.", font_size=32, color=YELLOW
        ).move_to(DOWN * 2)

        self.log_event("Instruction appears: Return answer as a number")
        self.play(FadeIn(note_text, shift=UP * 0.3), run_time=0.8)
        self.wait(3)

        self.log_event("Video ends")

        # ====================================================================
        # Generate comprehensive reasoning trace
        # ====================================================================
        self.build_reasoning_trace()

        # ====================================================================
        # Save output files
        # ====================================================================
        # Save solution and question text
        question_for_file = f"What is the sum of the sequences that appeared between the {start_ordinal} and {end_ordinal} pause?\nReturn the answer as a number."

        with open(
            f"solutions/pause_seq_n{self.num_sequences}_seed{self.seed}.txt", "w"
        ) as f:
            f.write(str(answer))
        with open(
            f"question_text/pause_seq_n{self.num_sequences}_seed{self.seed}.txt", "w"
        ) as f:
            f.write(question_for_file)

        # Save detailed reasoning trace
        with open(
            f"reasoning_traces/pause_seq_n{self.num_sequences}_seed{self.seed}.txt", "w"
        ) as f:
            f.write("\n".join(self.reasoning_trace))

        # Debug info for verification
        if os.getenv("DEBUG", "false").lower() == "true":
            print(f"Sequences: {seqs}")
            print(f"Between pause {start_pause} and {end_pause}: {sequences_to_sum}")
            print(f"Answer: {answer}")

    def build_reasoning_trace(self):
        """
        Build a comprehensive, step-by-step reasoning trace.
        This explains how to solve the puzzle systematically.
        """
        self.reasoning_trace = []

        # ====================================================================
        # Introduction
        # ====================================================================
        self.reasoning_trace.append(
            "**Question:** What is the sum of the sequences that appeared between"
        )
        self.reasoning_trace.append(
            f"the {self.start_ordinal} and {self.end_ordinal} pause?"
        )
        self.reasoning_trace.append("")
        self.reasoning_trace.append("Let's solve this step by step.")
        self.reasoning_trace.append("")

        # ====================================================================
        # Scene description with timestamps
        # ====================================================================
        self.reasoning_trace.append("### Scene Description")
        self.reasoning_trace.append("")
        self.reasoning_trace.append(
            "The video displays a series of number sequences with pauses between them."
        )
        self.reasoning_trace.append("Here's what happens chronologically:")
        self.reasoning_trace.append("")

        for event in self.scene_events:
            time_str = self.format_time(event["time"])
            self.reasoning_trace.append(f"At {time_str}: {event['description']}")

        self.reasoning_trace.append("")

        # ====================================================================
        # Step 1: Understanding the structure
        # ====================================================================
        self.reasoning_trace.append("### Step 1: Understand the sequence structure")
        self.reasoning_trace.append("")
        self.reasoning_trace.append(
            f"The video showed **{self.num_sequences} sequences** of numbers,"
        )
        self.reasoning_trace.append(f"separated by **{self.pause_count} pauses**.")
        self.reasoning_trace.append("")
        self.reasoning_trace.append("The structure looks like this:")
        self.reasoning_trace.append("")

        ordinal_names = [
            "first",
            "second",
            "third",
            "fourth",
            "fifth",
            "sixth",
            "seventh",
            "eighth",
            "ninth",
            "tenth",
            "eleventh",
            "twelfth",
        ]
        for i in range(self.num_sequences):
            if i == 0:
                self.reasoning_trace.append(f"- Sequence 1: Five numbers appeared")
            else:
                pause_before = ordinal_names[i - 1]
                self.reasoning_trace.append(
                    f"- **{pause_before.capitalize()} PAUSE** (2 seconds of blank screen)"
                )
                self.reasoning_trace.append(
                    f"- Sequence {i + 1}: Five numbers appeared"
                )

        self.reasoning_trace.append("")

        # ====================================================================
        # Step 2: Identify the actual sequences
        # ====================================================================
        self.reasoning_trace.append("### Step 2: Identify what numbers were shown")
        self.reasoning_trace.append("")
        self.reasoning_trace.append("Let me list all the sequences that appeared:")
        self.reasoning_trace.append("")

        for i, seq in enumerate(self.seqs):
            seq_str = ", ".join(str(n) for n in seq)
            self.reasoning_trace.append(f"**Sequence {i + 1}:** {seq_str}")

        self.reasoning_trace.append("")

        # ====================================================================
        # Step 3: Determine which sequences to sum
        # ====================================================================
        self.reasoning_trace.append("### Step 3: Determine which sequences to sum")
        self.reasoning_trace.append("")
        self.reasoning_trace.append(
            f"The question asks for sequences **between the {self.start_ordinal}"
        )
        self.reasoning_trace.append(f"and {self.end_ordinal} pause**.")
        self.reasoning_trace.append("")
        self.reasoning_trace.append("This means:")
        self.reasoning_trace.append(
            f"- We need sequences that appeared **AFTER** the {self.start_ordinal} pause"
        )
        self.reasoning_trace.append(f"- AND **BEFORE** the {self.end_ordinal} pause")
        self.reasoning_trace.append("")

        # Identify which sequences fall in this range
        sequences_description = []
        for i in range(self.start_pause, self.end_pause):
            sequences_description.append(f"Sequence {i + 1}")

        if len(sequences_description) == 1:
            self.reasoning_trace.append(
                f"Only **{sequences_description[0]}** appeared between these two pauses."
            )
        else:
            sequences_list = (
                ", ".join(sequences_description[:-1])
                + f", and {sequences_description[-1]}"
            )
            self.reasoning_trace.append(
                f"The sequences that fall in this range are: **{sequences_list}**."
            )

        self.reasoning_trace.append("")

        # ====================================================================
        # Step 4: Calculate the sum
        # ====================================================================
        self.reasoning_trace.append("### Step 4: Calculate the sum")
        self.reasoning_trace.append("")
        self.reasoning_trace.append(
            "Now I'll sum all the numbers from these sequences:"
        )
        self.reasoning_trace.append("")

        # Show the calculation step by step
        running_total = 0
        for idx, seq_idx in enumerate(range(self.start_pause, self.end_pause)):
            seq = self.seqs[seq_idx]
            seq_sum = sum(seq)
            running_total += seq_sum
            numbers_str = " + ".join(str(n) for n in seq)
            self.reasoning_trace.append(
                f"**Sequence {seq_idx + 1}:** {numbers_str} = {seq_sum}"
            )
            if idx == 0 and len(sequences_description) > 1:
                self.reasoning_trace.append(f"  Subtotal: {running_total}")
            elif idx > 0:
                self.reasoning_trace.append(f"  Subtotal: {running_total}")

        self.reasoning_trace.append("")
        self.reasoning_trace.append(f"**Total sum:** {self.answer}")
        self.reasoning_trace.append("")

        # ====================================================================
        # Final answer
        # ====================================================================
        self.reasoning_trace.append("### Final Answer")
        self.reasoning_trace.append("")
        self.reasoning_trace.append(
            f"The sum of all numbers in the sequences between the {self.start_ordinal}"
        )
        self.reasoning_trace.append(
            f"and {self.end_ordinal} pause is **{self.answer}**."
        )
        self.reasoning_trace.append("")
        self.reasoning_trace.append(f"\\boxed{{{self.answer}}}")


# ============================================================================
# Main execution
# ============================================================================
# Generate pause sequence video
scene = pause_seq()
scene.render()

# ============================================================================
# Move output file to questions directory with descriptive name
# ============================================================================
# Try multiple possible output paths (Manim's output location varies)
possible_paths = [
    Path("manim_output/videos/pause_seq/1080p30/pause_seq.mp4"),
    Path("manim_output/videos/1080p30/pause_seq.mp4"),
    Path("manim_output/videos/1080p30/1080p30/pause_seq.mp4"),
]

output_found = False
for output_path in possible_paths:
    if output_path.exists():
        filename = f"pause_seq_n{scene.num_sequences}_seed{scene.seed}.mp4"
        shutil.move(str(output_path), f"questions/{filename}")
        output_found = True
        print(f"✓ Video saved: questions/{filename}")
        break

if not output_found:
    # Debug: Print what files actually exist to help diagnose issues
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
