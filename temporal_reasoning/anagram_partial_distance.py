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

class anagram_distance(Scene):
    """
    A scene that generates an anagram shuffle puzzle:
    - Shows a word being shuffled multiple times
    - User must determine the position difference of a specific letter
      between two shuffle states
    - Generates question video, solution, and reasoning trace
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Set random seed for reproducibility
        self.seed = random.randint(1000, 9999)

        # Parameters from environment variables (with defaults)
        self.num_shuffles = int(os.getenv("NUM_SHUFFLES", 10))

        # Initialize reasoning trace storage
        self.reasoning_trace = []

        # Track timing for reasoning trace using Manim's internal video time
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
        # Use the seed set in __init__ for reproducible randomness
        random.seed(self.seed)

        # ====================================================================
        # Word selection and target letter determination
        # ====================================================================
        word_options = [
            "GOVERNMENT", "MARYLAND", "COMPUTER", "TERRAPIN",
            "KEYBOARD", "MOUNTAIN", "PRINCESS", "UMBRELLA",
            "BUTTERFLY", "CHOCOLATE", "WONDERFUL", "BEAUTIFUL"
        ]

        # Choose random word
        self.word = random.choice(word_options)

        # Find letters that appear exactly once (to ensure unambiguous tracking)
        def get_unique_letters(word):
            """Find letters that appear exactly once in the word."""
            letter_counts = {}
            for letter in word:
                letter_counts[letter] = letter_counts.get(letter, 0) + 1
            return [letter for letter, count in letter_counts.items() if count == 1]

        unique_letters = get_unique_letters(self.word)
        print(f"Unique letters in '{self.word}': {unique_letters}")

        # Choose random target letter from unique letters
        if unique_letters:
            self.target_letter = random.choice(unique_letters)
        else:
            # Fallback if no unique letters (shouldn't happen with good word choices)
            self.target_letter = self.word[0]

        # Choose random starting and ending positions for the question
        # These represent which shuffle states we'll compare (0 = initial state)
        self.random_start_position = random.randint(0, self.num_shuffles)
        self.random_end_position = random.randint(0, self.num_shuffles)

        # ====================================================================
        # Display question
        # ====================================================================
        question_lines = [
            f"What is the difference in position of the letter '{self.target_letter}'",
            f"between after shuffle {self.random_start_position} and after shuffle {self.random_end_position}?",
            "",
            "Return the answer as a number."
        ]

        question_texts = []
        line_height = 0.6
        start_y = 3.0

        for i, line in enumerate(question_lines):
            if line:  # Skip empty lines for spacing
                text = Text(line, font_size=32, weight=BOLD if i < 2 else NORMAL)
                text.move_to(UP * (start_y - i * line_height))
                question_texts.append(text)

        # Animate question appearance
        self.log_event(f"Question appears asking for position difference of letter '{self.target_letter}' between shuffle {self.random_start_position} and shuffle {self.random_end_position}")
        self.play(*[FadeIn(text, shift=DOWN*0.2) for text in question_texts], run_time=1.0)
        self.wait(1.0)
        self.log_event("Question display complete")

        # ====================================================================
        # Create initial word display
        # ====================================================================
        letter_spacing = 0.8
        word_width = (len(self.word) - 1) * letter_spacing
        start_x = -word_width / 2

        # Create letter objects with consistent styling
        letters = []
        for i, char in enumerate(self.word):
            letter = Text(char, font_size=48, weight=BOLD, color=WHITE)
            letter.move_to([start_x + i * letter_spacing, 0, 0])
            letters.append(letter)

        # Display the initial word
        self.log_event(f"Initial word '{self.word}' appears with {len(self.word)} letters")
        self.play(*[FadeIn(letter, scale=0.8) for letter in letters], run_time=1.0)
        self.wait(1.5)
        self.log_event(f"Initial word display complete, target letter '{self.target_letter}' is at position {self.word.find(self.target_letter)}")

        # ====================================================================
        # Track movements and prepare for shuffles
        # ====================================================================
        positions = []  # Store positions at each shuffle
        current_position = self.word.find(self.target_letter)
        positions.append(current_position)  # Position at shuffle 0 (initial)

        # Store detailed shuffle information for reasoning trace
        self.shuffle_details = []
        self.shuffle_details.append({
            'shuffle_num': 0,
            'word': self.word,
            'position': current_position
        })

        # ====================================================================
        # Perform shuffles with animations
        # ====================================================================
        for shuffle_num in range(self.num_shuffles):
            # Create new arrangement
            letter_indices = list(range(len(self.word)))
            random.shuffle(letter_indices)

            # Find new position of target letter
            new_word = ''.join([letters[i].text for i in letter_indices])
            new_position = new_word.find(self.target_letter)
            positions.append(new_position)  # Store position at this shuffle

            # Record detailed shuffle information
            self.shuffle_details.append({
                'shuffle_num': shuffle_num + 1,
                'word': new_word,
                'position': new_position
            })

            # Animate the rearrangement
            self.log_event(f"Shuffle {shuffle_num + 1} begins")
            animations = []
            for i, original_index in enumerate(letter_indices):
                new_x = start_x + i * letter_spacing
                animations.append(letters[original_index].animate.move_to([new_x, 0, 0]))

            self.play(*animations, run_time=0.8)
            self.log_event(f"Shuffle {shuffle_num + 1} complete: word is now '{new_word}', letter '{self.target_letter}' at position {new_position}")

            # Update tracking
            current_position = new_position
            letters = [letters[i] for i in letter_indices]  # Reorder letters list

            self.wait(0.5)

        # ====================================================================
        # Calculate the answer
        # ====================================================================
        start_pos = positions[self.random_start_position]
        end_pos = positions[self.random_end_position]
        self.answer = end_pos - start_pos

        self.log_event(f"All shuffles complete, calculating answer")
        self.wait(2)
        self.log_event(f"Video ends")

        # ====================================================================
        # Store question text for output file
        # ====================================================================
        self.question_text = (
            f"What is the difference in position of the letter '{self.target_letter}' "
            f"between after shuffle {self.random_start_position} and after shuffle {self.random_end_position}? "
            "Return the answer as a number."
        )

        # ====================================================================
        # Generate reasoning trace
        # ====================================================================
        self.build_reasoning_trace()

        # ====================================================================
        # Save output files
        # ====================================================================
        # Solution file (just the answer - absolute value)
        with open(f"solutions/anagram_partial_diff_n{self.num_shuffles}_seed{self.seed}.txt", "w") as f:
            f.write(str(abs(self.answer)))

        # Question text file
        with open(f"question_text/anagram_partial_diff_n{self.num_shuffles}_seed{self.seed}.txt", "w") as f:
            f.write(self.question_text)

        # Reasoning trace file
        with open(f"reasoning_traces/anagram_partial_diff_n{self.num_shuffles}_seed{self.seed}.txt", "w") as f:
            f.write("\n".join(self.reasoning_trace))

    def build_reasoning_trace(self):
        """
        Build a comprehensive, step-by-step reasoning trace.
        This explains how to solve the puzzle systematically.
        """
        self.reasoning_trace = []

        # ====================================================================
        # Introduction
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
        # Step 1: Initial setup
        # ====================================================================
        self.reasoning_trace.append("### Step 1: Understand the initial setup")
        self.reasoning_trace.append(f"The initial word is **'{self.word}'** with {len(self.word)} letters.")
        self.reasoning_trace.append(f"We are tracking the letter **'{self.target_letter}'**.")
        self.reasoning_trace.append(f"Initially (at shuffle 0), the letter '{self.target_letter}' is at position **{self.shuffle_details[0]['position']}**.")
        self.reasoning_trace.append("")

        # ====================================================================
        # Step 2: Track all shuffles
        # ====================================================================
        self.reasoning_trace.append("### Step 2: Track the position through all shuffles")
        self.reasoning_trace.append("")

        # Show first few shuffles in detail
        num_detailed = min(3, len(self.shuffle_details) - 1)
        if num_detailed > 0:
            self.reasoning_trace.append(f"Let's examine the first {num_detailed} shuffle(s) in detail:")
            self.reasoning_trace.append("")
            for detail in self.shuffle_details[1:num_detailed + 1]:
                self.reasoning_trace.append(f"**After shuffle {detail['shuffle_num']}:**")
                self.reasoning_trace.append(f"  - Word: {detail['word']}")
                self.reasoning_trace.append(f"  - Position of '{self.target_letter}': {detail['position']}")
                self.reasoning_trace.append("")

        # Show summary table of all positions
        self.reasoning_trace.append("Complete position tracking for all shuffles:")
        self.reasoning_trace.append("")
        self.reasoning_trace.append("| Shuffle # | Word | Position of '{}' |".format(self.target_letter))
        self.reasoning_trace.append("|-----------|------|------------------|")
        for detail in self.shuffle_details:
            self.reasoning_trace.append(f"| {detail['shuffle_num']} | {detail['word']} | {detail['position']} |")
        self.reasoning_trace.append("")

        # ====================================================================
        # Step 3: Extract the relevant positions
        # ====================================================================
        self.reasoning_trace.append("### Step 3: Identify the positions we need to compare")
        start_detail = self.shuffle_details[self.random_start_position]
        end_detail = self.shuffle_details[self.random_end_position]

        self.reasoning_trace.append(f"The question asks for the difference between shuffle {self.random_start_position} and shuffle {self.random_end_position}.")
        self.reasoning_trace.append("")
        self.reasoning_trace.append(f"**At shuffle {self.random_start_position}:**")
        self.reasoning_trace.append(f"  - Word: {start_detail['word']}")
        self.reasoning_trace.append(f"  - Position of '{self.target_letter}': {start_detail['position']}")
        self.reasoning_trace.append("")
        self.reasoning_trace.append(f"**At shuffle {self.random_end_position}:**")
        self.reasoning_trace.append(f"  - Word: {end_detail['word']}")
        self.reasoning_trace.append(f"  - Position of '{self.target_letter}': {end_detail['position']}")
        self.reasoning_trace.append("")

        # ====================================================================
        # Step 4: Calculate the difference
        # ====================================================================
        self.reasoning_trace.append("### Step 4: Calculate the difference")
        self.reasoning_trace.append(f"Difference = Position at shuffle {self.random_end_position} - Position at shuffle {self.random_start_position}")
        self.reasoning_trace.append(f"Difference = {end_detail['position']} - {start_detail['position']} = {self.answer}")
        self.reasoning_trace.append("")
        self.reasoning_trace.append(f"The absolute value of the difference is **{abs(self.answer)}**.")
        self.reasoning_trace.append("")

        # ====================================================================
        # Final answer
        # ====================================================================
        self.reasoning_trace.append("### Final Answer")
        self.reasoning_trace.append(f"The difference in position of the letter '{self.target_letter}' between shuffle {self.random_start_position} and shuffle {self.random_end_position} is **{abs(self.answer)}**.")
        self.reasoning_trace.append("")
        self.reasoning_trace.append(f"\\boxed{{{abs(self.answer)}}}")


# ============================================================================
# Main execution
# ============================================================================
if __name__ == "__main__":
    # Generate the anagram video
    scene = anagram_distance()
    scene.render()

    # ========================================================================
    # Move output file to questions directory with descriptive name
    # ========================================================================
    output = Path("manim_output/videos/1080p30/anagram_distance.mp4")
    if output.exists():
        filename = f"anagram_partial_diff_n{scene.num_shuffles}_seed{scene.seed}.mp4"
        shutil.move(str(output), f"questions/{filename}")
        print(f"✓ Video saved: questions/{filename}")
        print(f"✓ Solution saved: solutions/anagram_partial_diff_n{scene.num_shuffles}_seed{scene.seed}.txt")
        print(f"✓ Question saved: question_text/anagram_partial_diff_n{scene.num_shuffles}_seed{scene.seed}.txt")
        print(f"✓ Reasoning trace saved: reasoning_traces/anagram_partial_diff_n{scene.num_shuffles}_seed{scene.seed}.txt")
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