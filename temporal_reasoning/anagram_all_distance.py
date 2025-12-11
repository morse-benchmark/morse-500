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
    A scene that generates an anagram distance puzzle:
    - Shows a word being shuffled multiple times
    - User must calculate the sum of distances a target letter moved
    - Generates question video, solution, and detailed reasoning trace
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Set random seed for reproducibility of word selection and shuffles
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
        # Use the seed set in __init__ for reproducible randomness
        random.seed(self.seed)

        # ====================================================================
        # Choose word and target letter
        # ====================================================================
        # Words to choose from - all have unique letters for clearer puzzles
        word_options = [
            "GOVERNMENT", "MARYLAND", "COMPUTER", "TERRAPIN",
            "KEYBOARD", "MOUNTAIN", "PRINCESS", "UMBRELLA",
            "BUTTERFLY", "CHOCOLATE", "WONDERFUL", "BEAUTIFUL"
        ]

        # Choose random word
        self.word = random.choice(word_options)

        # Find letters that appear exactly once (to ensure unique tracking)
        def get_unique_letters(word):
            """Count letter occurrences and return those appearing exactly once."""
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

        # ====================================================================
        # Display the question
        # ====================================================================
        # Create multi-line question text for better readability
        question_lines = [
            f"What is the sum of the distance of all shifts",
            f"that happened to the letter '{self.target_letter}' across all word moves?",
            "",
            "Return the answer as a number."
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
        self.log_event(f"Question appears asking about letter '{self.target_letter}' movement")
        self.play(*[FadeIn(text, shift=DOWN*0.2) for text in question_texts], run_time=1.0)
        self.wait(1.0)
        self.log_event("Question display complete")

        # ====================================================================
        # Create and display initial word
        # ====================================================================
        letter_spacing = 0.8
        word_width = (len(self.word) - 1) * letter_spacing
        start_x = -word_width / 2

        # Create letter objects with consistent styling
        # Each letter is a separate Text object so we can animate them independently
        letters = []
        for i, char in enumerate(self.word):
            letter = Text(char, font_size=48, weight=BOLD, color=WHITE)
            letter.move_to([start_x + i * letter_spacing, 0, 0])
            letters.append(letter)

        # Display the initial word
        self.log_event(f"Initial word '{self.word}' appears with {len(self.word)} letters")
        self.play(*[FadeIn(letter, scale=0.8) for letter in letters], run_time=1.0)
        self.wait(1.5)
        self.log_event(f"Initial word display complete")

        # ====================================================================
        # Track movements and prepare for shuffles
        # ====================================================================
        # Initialize answer counter
        answer = 0

        # Find initial position of target letter (0-indexed)
        last_position = self.word.find(self.target_letter)

        # Store detailed movement information for reasoning trace
        self.shuffle_details = []

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

            # Calculate distance moved (absolute difference in positions)
            distance = abs(new_position - last_position)
            answer += distance

            # Store shuffle details for reasoning trace
            self.shuffle_details.append({
                'shuffle_num': shuffle_num + 1,
                'old_word': ''.join([letter.text for letter in letters]),
                'new_word': new_word,
                'old_position': last_position,
                'new_position': new_position,
                'distance': distance,
                'running_total': answer
            })

            # Animate the rearrangement
            # Each letter moves to its new position simultaneously
            animations = []
            for i, original_index in enumerate(letter_indices):
                new_x = start_x + i * letter_spacing
                animations.append(letters[original_index].animate.move_to([new_x, 0, 0]))

            self.log_event(f"Shuffle {shuffle_num + 1} begins: '{self.target_letter}' moves from position {last_position} to {new_position}")
            self.play(*animations, run_time=0.8)
            self.log_event(f"Shuffle {shuffle_num + 1} complete: distance = {distance}, running total = {answer}")

            # Update tracking
            last_position = new_position
            letters = [letters[i] for i in letter_indices]  # Reorder letters list to match new positions

            self.wait(0.5)

        # Store final answer
        self.answer = answer

        self.wait(2)
        self.log_event("Animation sequence complete")

        # ====================================================================
        # Generate reasoning trace
        # ====================================================================
        self.build_reasoning_trace()

        # ====================================================================
        # Save output files
        # ====================================================================
        # Solution file (just the answer)
        with open(f"solutions/anagram_all_diff_n{self.num_shuffles}_seed{self.seed}.txt", "w") as f:
            f.write(str(self.answer))

        # Question text file
        question_text_content = (
            f"What is the sum of the distance of all shifts that happened to the letter '{self.target_letter}' across all word moves?\n"
            "Return the answer as a number."
        )
        with open(f"question_text/anagram_all_diff_n{self.num_shuffles}_seed{self.seed}.txt", "w") as f:
            f.write(question_text_content)

        # Reasoning trace file
        with open(f"reasoning_traces/anagram_all_diff_n{self.num_shuffles}_seed{self.seed}.txt", "w") as f:
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
        self.reasoning_trace.append(f"**Question:** What is the sum of the distance of all shifts that happened to the letter '{self.target_letter}' across all word moves?")
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
        self.reasoning_trace.append(f"We start with the word: **{self.word}**")
        self.reasoning_trace.append(f"The word has {len(self.word)} letters.")
        self.reasoning_trace.append(f"We need to track the letter **'{self.target_letter}'** throughout all the shuffles.")
        self.reasoning_trace.append("")

        # Find initial position
        initial_position = self.word.find(self.target_letter)
        self.reasoning_trace.append(f"Initially, the letter '{self.target_letter}' is at position **{initial_position}** (using 0-based indexing).")
        self.reasoning_trace.append("")

        # ====================================================================
        # Step 2: Track each shuffle
        # ====================================================================
        self.reasoning_trace.append("### Step 2: Track the letter's movement through each shuffle")
        self.reasoning_trace.append(f"The word will be shuffled **{self.num_shuffles} times**. For each shuffle, we:")
        self.reasoning_trace.append("1. Note the new arrangement of letters")
        self.reasoning_trace.append("2. Find the new position of our target letter")
        self.reasoning_trace.append("3. Calculate the distance it moved (absolute difference in positions)")
        self.reasoning_trace.append("4. Add this distance to our running total")
        self.reasoning_trace.append("")

        # Show first few shuffles in detail
        num_detailed = min(5, len(self.shuffle_details))
        self.reasoning_trace.append(f"Here are the first {num_detailed} shuffle(s) in detail:")
        self.reasoning_trace.append("")

        for i in range(num_detailed):
            detail = self.shuffle_details[i]
            self.reasoning_trace.append(f"**Shuffle {detail['shuffle_num']}:**")
            self.reasoning_trace.append(f"  - Previous word: {detail['old_word']}")
            self.reasoning_trace.append(f"  - New word: {detail['new_word']}")
            self.reasoning_trace.append(f"  - Letter '{self.target_letter}' moved from position {detail['old_position']} to position {detail['new_position']}")
            self.reasoning_trace.append(f"  - Distance: |{detail['new_position']} - {detail['old_position']}| = {detail['distance']}")
            self.reasoning_trace.append(f"  - Running total: {detail['running_total']}")
            self.reasoning_trace.append("")

        # Show summary for remaining shuffles if there are many
        if len(self.shuffle_details) > num_detailed:
            self.reasoning_trace.append(f"Continuing with the remaining {len(self.shuffle_details) - num_detailed} shuffles:")
            self.reasoning_trace.append("")

            # Show checkpoints
            if len(self.shuffle_details) > 10:
                # Show middle and end for long sequences
                checkpoints = [len(self.shuffle_details) // 2, -1]
                for idx in checkpoints:
                    detail = self.shuffle_details[idx]
                    self.reasoning_trace.append(f"**Shuffle {detail['shuffle_num']}:**")
                    self.reasoning_trace.append(f"  - New word: {detail['new_word']}")
                    self.reasoning_trace.append(f"  - Letter '{self.target_letter}' at position {detail['new_position']}, distance = {detail['distance']}")
                    self.reasoning_trace.append(f"  - Running total: {detail['running_total']}")
                    self.reasoning_trace.append("")
            else:
                # Show all remaining for shorter sequences
                for i in range(num_detailed, len(self.shuffle_details)):
                    detail = self.shuffle_details[i]
                    self.reasoning_trace.append(f"**Shuffle {detail['shuffle_num']}:** New word: {detail['new_word']}, '{self.target_letter}' at position {detail['new_position']}, distance = {detail['distance']}, total = {detail['running_total']}")
                self.reasoning_trace.append("")

        # ====================================================================
        # Step 3: Calculate final answer
        # ====================================================================
        self.reasoning_trace.append("### Step 3: Calculate the final answer")
        self.reasoning_trace.append(f"After all {self.num_shuffles} shuffles, we sum up all the distances:")
        self.reasoning_trace.append("")

        # Show the sum calculation
        distance_list = [str(detail['distance']) for detail in self.shuffle_details]
        if len(distance_list) <= 15:
            self.reasoning_trace.append(f"Total distance = {' + '.join(distance_list)}")
        else:
            # Show first few and last few
            first_few = ' + '.join(distance_list[:5])
            last_few = ' + '.join(distance_list[-5:])
            self.reasoning_trace.append(f"Total distance = {first_few} + ... + {last_few}")

        self.reasoning_trace.append(f"Total distance = **{self.answer}**")
        self.reasoning_trace.append("")

        # ====================================================================
        # Final answer
        # ====================================================================
        self.reasoning_trace.append("### Final Answer")
        self.reasoning_trace.append(f"The sum of the distance of all shifts that happened to the letter '{self.target_letter}' is **{self.answer}**.")
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
        filename = f"anagram_all_diff_n{scene.num_shuffles}_seed{scene.seed}.mp4"
        shutil.move(str(output), f"questions/{filename}")
        print(f"✓ Video saved: questions/{filename}")
        print(f"✓ Solution saved: solutions/anagram_all_diff_n{scene.num_shuffles}_seed{scene.seed}.txt")
        print(f"✓ Question saved: question_text/anagram_all_diff_n{scene.num_shuffles}_seed{scene.seed}.txt")
        print(f"✓ Reasoning trace saved: reasoning_traces/anagram_all_diff_n{scene.num_shuffles}_seed{scene.seed}.txt")
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