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
    A scene that generates an anagram position tracking puzzle:
    - Shows a word being shuffled multiple times
    - User must determine the position of a specific letter after a given shuffle
    - Generates question video, solution, and reasoning trace
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Set random seed for reproducibility
        self.seed = random.randint(1000, 9999)
        random.seed(self.seed)

        # Parameters from environment variables (with defaults)
        self.num_shuffles = int(os.getenv("NUM_SHUFFLES", 10))

        # Initialize reasoning trace storage
        self.reasoning_trace = []

        # Track timing for reasoning trace using Manim's internal video time
        # This will be set when construct() is called and renderer is available
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

        # ====================================================================
        # Select word and target letter
        # ====================================================================
        # Words to choose from (all have unique letters for tracking)
        word_options = [
            "GOVERNMENT", "MARYLAND", "COMPUTER", "TERRAPIN",
            "KEYBOARD", "MOUNTAIN", "PRINCESS", "UMBRELLA",
            "BUTTERFLY", "CHOCOLATE", "WONDERFUL", "BEAUTIFUL"
        ]

        # Choose random word
        self.word = random.choice(word_options)

        # Find letters that appear exactly once (easier to track)
        def get_unique_letters(word):
            """Return list of letters that appear exactly once in the word."""
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

        # Choose random shuffle number to ask about (0 = initial, 1-num_shuffles = after each shuffle)
        self.query_shuffle = random.randint(0, self.num_shuffles)

        # ====================================================================
        # Display question
        # ====================================================================
        # Create multi-line question text
        question_lines = [
            f"What is the position of the letter '{self.target_letter}'",
            f"after shuffle {self.query_shuffle}?",
            "",
            "Return the answer as a number (0-indexed)."
        ]

        # Create text objects for each line
        question_texts = []
        line_height = 0.6
        start_y = 3.0

        for i, line in enumerate(question_lines):
            if line:  # Skip empty lines for spacing
                text = Text(line, font_size=32, weight=BOLD if i < 2 else NORMAL)
                text.move_to(UP * (start_y - i * line_height))
                question_texts.append(text)

        # Animate question appearance
        self.log_event(f"Question appears: 'What is the position of the letter {self.target_letter} after shuffle {self.query_shuffle}?'")
        self.play(*[FadeIn(text, shift=DOWN*0.2) for text in question_texts], run_time=1.0)
        self.wait(1.0)
        self.log_event("Question displayed on screen")

        # ====================================================================
        # Create and display initial word
        # ====================================================================
        # Calculate positioning for centered word display
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
        self.log_event(f"Initial word '{self.word}' appears")
        self.play(*[FadeIn(letter, scale=0.8) for letter in letters], run_time=1.0)
        self.wait(1.5)
        self.log_event(f"Initial word displayed with letter '{self.target_letter}' at position {self.word.find(self.target_letter)}")

        # ====================================================================
        # Track positions for all shuffles
        # ====================================================================
        # Store positions at each shuffle (index 0 = initial, 1-n = after each shuffle)
        positions = []
        current_position = self.word.find(self.target_letter)
        positions.append(current_position)  # Position at shuffle 0 (initial)

        # Store detailed shuffle information for reasoning trace
        shuffle_details = []
        shuffle_details.append({
            'shuffle_num': 0,
            'word': self.word,
            'position': current_position,
            'is_initial': True
        })

        # ====================================================================
        # Perform shuffles with animations
        # ====================================================================
        for shuffle_num in range(self.num_shuffles):
            # Create new arrangement by shuffling letter indices
            letter_indices = list(range(len(self.word)))
            random.shuffle(letter_indices)

            # Find new position of target letter in shuffled word
            new_word = ''.join([letters[i].text for i in letter_indices])
            new_position = new_word.find(self.target_letter)
            positions.append(new_position)  # Store position at this shuffle

            # Record shuffle details for reasoning trace
            shuffle_details.append({
                'shuffle_num': shuffle_num + 1,
                'word': new_word,
                'position': new_position,
                'is_initial': False
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
            letters = [letters[i] for i in letter_indices]  # Reorder letters list to match visual

            self.wait(0.5)

        # ====================================================================
        # Determine final answer
        # ====================================================================
        # The answer is the position after the queried shuffle
        self.answer = positions[self.query_shuffle]
        self.shuffle_details = shuffle_details

        self.wait(2)
        self.log_event("All shuffles complete")

        # ====================================================================
        # Generate reasoning trace
        # ====================================================================
        self.build_reasoning_trace()

        # ====================================================================
        # Save output files
        # ====================================================================
        # Solution file (just the answer)
        with open(f"solutions/anagram_position_n{self.num_shuffles}_seed{self.seed}.txt", "w") as f:
            f.write(str(self.answer))

        # Question text file
        question_text_content = (
            f"What is the position of the letter '{self.target_letter}' after shuffle {self.query_shuffle}?\n"
            "Return the answer as a number (0-indexed)."
        )
        with open(f"question_text/anagram_position_n{self.num_shuffles}_seed{self.seed}.txt", "w") as f:
            f.write(question_text_content)

        # Reasoning trace file
        with open(f"reasoning_traces/anagram_position_n{self.num_shuffles}_seed{self.seed}.txt", "w") as f:
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
        self.reasoning_trace.append(f"**Question:** What is the position of the letter '{self.target_letter}' after shuffle {self.query_shuffle}?")
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
        # Step 1: Initial configuration
        # ====================================================================
        self.reasoning_trace.append("### Step 1: Understand the initial configuration")
        initial_detail = self.shuffle_details[0]
        self.reasoning_trace.append(f"At the beginning, we have the word: **{initial_detail['word']}**")
        self.reasoning_trace.append(f"We need to track the letter **'{self.target_letter}'**")
        self.reasoning_trace.append(f"Initial position of '{self.target_letter}': position **{initial_detail['position']}** (0-indexed)")
        self.reasoning_trace.append("")

        # ====================================================================
        # Step 2: Understanding the task
        # ====================================================================
        self.reasoning_trace.append("### Step 2: Understand the task")
        self.reasoning_trace.append(f"The word will be shuffled **{self.num_shuffles} times** in total.")
        self.reasoning_trace.append(f"We need to find the position of '{self.target_letter}' after shuffle **{self.query_shuffle}**.")
        if self.query_shuffle == 0:
            self.reasoning_trace.append("Note: Shuffle 0 means the initial state (before any shuffling).")
        self.reasoning_trace.append("")

        # ====================================================================
        # Step 3: Track through shuffles
        # ====================================================================
        self.reasoning_trace.append("### Step 3: Track the letter through shuffles")
        self.reasoning_trace.append("")

        # Show initial state
        self.reasoning_trace.append(f"**Shuffle 0 (Initial):**")
        self.reasoning_trace.append(f"  - Word: {initial_detail['word']}")
        self.reasoning_trace.append(f"  - Position of '{self.target_letter}': {initial_detail['position']}")
        self.reasoning_trace.append("")

        # Show all shuffles (or a selection if there are many)
        if len(self.shuffle_details) <= 12:  # Show all if 11 shuffles or fewer
            for detail in self.shuffle_details[1:]:
                self.reasoning_trace.append(f"**Shuffle {detail['shuffle_num']}:**")
                self.reasoning_trace.append(f"  - Word: {detail['word']}")
                self.reasoning_trace.append(f"  - Position of '{self.target_letter}': {detail['position']}")
                self.reasoning_trace.append("")
        else:
            # Show first 3, middle few, and last 3
            first_few = self.shuffle_details[1:4]
            middle_idx = len(self.shuffle_details) // 2
            middle_few = self.shuffle_details[middle_idx-1:middle_idx+2]
            last_few = self.shuffle_details[-3:]

            for detail in first_few:
                self.reasoning_trace.append(f"**Shuffle {detail['shuffle_num']}:**")
                self.reasoning_trace.append(f"  - Word: {detail['word']}")
                self.reasoning_trace.append(f"  - Position of '{self.target_letter}': {detail['position']}")
                self.reasoning_trace.append("")

            self.reasoning_trace.append("... (intermediate shuffles) ...")
            self.reasoning_trace.append("")

            for detail in middle_few:
                self.reasoning_trace.append(f"**Shuffle {detail['shuffle_num']}:**")
                self.reasoning_trace.append(f"  - Word: {detail['word']}")
                self.reasoning_trace.append(f"  - Position of '{self.target_letter}': {detail['position']}")
                self.reasoning_trace.append("")

            self.reasoning_trace.append("... (more shuffles) ...")
            self.reasoning_trace.append("")

            for detail in last_few:
                self.reasoning_trace.append(f"**Shuffle {detail['shuffle_num']}:**")
                self.reasoning_trace.append(f"  - Word: {detail['word']}")
                self.reasoning_trace.append(f"  - Position of '{self.target_letter}': {detail['position']}")
                self.reasoning_trace.append("")

        # ====================================================================
        # Step 4: Find the answer
        # ====================================================================
        self.reasoning_trace.append("### Step 4: Determine the answer")
        queried_detail = self.shuffle_details[self.query_shuffle]
        self.reasoning_trace.append(f"The question asks for the position after shuffle **{self.query_shuffle}**.")
        self.reasoning_trace.append(f"Looking at shuffle {self.query_shuffle}:")
        self.reasoning_trace.append(f"  - Word: {queried_detail['word']}")
        self.reasoning_trace.append(f"  - Position of '{self.target_letter}': **{queried_detail['position']}**")
        self.reasoning_trace.append("")

        # ====================================================================
        # Final answer
        # ====================================================================
        self.reasoning_trace.append("### Final Answer")
        self.reasoning_trace.append(f"The position of the letter '{self.target_letter}' after shuffle {self.query_shuffle} is **{self.answer}**.")
        self.reasoning_trace.append("")
        self.reasoning_trace.append(f"\\boxed{{{self.answer}}}")

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
        filename = f"anagram_position_n{scene.num_shuffles}_seed{scene.seed}.mp4"
        shutil.move(str(output), f"questions/{filename}")
        print(f"✓ Video saved: questions/{filename}")
        print(f"✓ Solution saved: solutions/anagram_position_n{scene.num_shuffles}_seed{scene.seed}.txt")
        print(f"✓ Question text saved: question_text/anagram_position_n{scene.num_shuffles}_seed{scene.seed}.txt")
        print(f"✓ Reasoning trace saved: reasoning_traces/anagram_position_n{scene.num_shuffles}_seed{scene.seed}.txt")
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

    # Final cleanup - remove temporary manim output directory
    if os.path.exists("manim_output"):
        shutil.rmtree("manim_output")