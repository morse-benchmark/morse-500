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
    A scene that generates an anagram shuffle counting puzzle:
    - Shows a word being shuffled multiple times
    - User must count the number of shuffles performed
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
        # Choose random word from word pool
        # ====================================================================
        word_options = [
            "GOVERNMENT", "MARYLAND", "COMPUTER", "TERRAPIN",
            "KEYBOARD", "MOUNTAIN", "PRINCESS", "UMBRELLA",
            "BUTTERFLY", "CHOCOLATE", "WONDERFUL", "BEAUTIFUL"
        ]

        # Choose random word
        self.word = random.choice(word_options)

        # Find letters that appear exactly once (for internal tracking)
        def get_unique_letters(word):
            """
            Identify letters that appear exactly once in the word.
            This helps us track specific letters through shuffles.
            """
            letter_counts = {}
            for letter in word:
                letter_counts[letter] = letter_counts.get(letter, 0) + 1
            return [letter for letter, count in letter_counts.items() if count == 1]

        unique_letters = get_unique_letters(self.word)
        print(f"Unique letters in '{self.word}': {unique_letters}")

        # Choose random target letter from unique letters (for tracking purposes)
        if unique_letters:
            self.target_letter = random.choice(unique_letters)
        else:
            # Fallback if no unique letters (shouldn't happen with good word choices)
            self.target_letter = self.word[0]

        # ====================================================================
        # Display the question
        # ====================================================================
        question_lines = [
            f"How many shuffles were performed on the word?",
            "",
            "Return the answer as a number."
        ]

        # Create question text objects
        question_texts = []
        line_height = 0.6
        start_y = 3.0

        for i, line in enumerate(question_lines):
            if line:  # Skip empty lines for spacing
                text = Text(line, font_size=32, weight=BOLD if i < 1 else NORMAL)
                text.move_to(UP * (start_y - i * line_height))
                question_texts.append(text)

        # Animate question appearance
        self.log_event("Question appears: 'How many shuffles were performed on the word?'")
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
        letters = []
        for i, char in enumerate(self.word):
            letter = Text(char, font_size=48, weight=BOLD, color=WHITE)
            letter.move_to([start_x + i * letter_spacing, 0, 0])
            letters.append(letter)

        # Display the initial word
        self.log_event(f"Initial word '{self.word}' appears on screen")
        self.play(*[FadeIn(letter, scale=0.8) for letter in letters], run_time=1.0)
        self.wait(1.5)
        self.log_event(f"Initial word display complete")

        # Store shuffle details for reasoning trace
        self.shuffle_details = []
        self.shuffle_details.append({
            'shuffle_num': 0,
            'word_arrangement': self.word,
            'is_initial': True
        })

        # ====================================================================
        # Perform shuffles with animations
        # ====================================================================
        for shuffle_num in range(self.num_shuffles):
            # Create new arrangement by shuffling indices
            letter_indices = list(range(len(self.word)))
            random.shuffle(letter_indices)

            # Find new word arrangement
            new_word = ''.join([letters[i].text for i in letter_indices])

            # Log event before animation
            self.log_event(f"Shuffle {shuffle_num + 1} begins")

            # Animate the rearrangement
            animations = []
            for i, original_index in enumerate(letter_indices):
                new_x = start_x + i * letter_spacing
                animations.append(letters[original_index].animate.move_to([new_x, 0, 0]))

            self.play(*animations, run_time=0.8)

            # Log event after animation completes
            self.log_event(f"Shuffle {shuffle_num + 1} completes - new arrangement: {new_word}")

            # Update tracking - reorder letters list to match new visual order
            letters = [letters[i] for i in letter_indices]

            # Store shuffle details for reasoning trace
            self.shuffle_details.append({
                'shuffle_num': shuffle_num + 1,
                'word_arrangement': new_word,
                'is_initial': False
            })

            self.wait(0.5)

        # ====================================================================
        # Determine the answer
        # ====================================================================
        # The answer is simply the number of shuffles performed
        self.answer = self.num_shuffles

        self.wait(2)
        self.log_event("All shuffles complete, video ending")

        # ====================================================================
        # Generate reasoning trace
        # ====================================================================
        self.build_reasoning_trace()

        # ====================================================================
        # Save output files
        # ====================================================================
        # Solution file (just the answer)
        with open(f"solutions/anagram_num_shuffles_n{self.num_shuffles}_seed{self.seed}.txt", "w") as f:
            f.write(str(self.answer))

        # Question text file
        question_text_content = (
            f"How many shuffles were performed on the word?\n"
            "Return the answer as a number."
        )
        with open(f"question_text/anagram_num_shuffles_n{self.num_shuffles}_seed{self.seed}.txt", "w") as f:
            f.write(question_text_content)

        # Reasoning trace file
        with open(f"reasoning_traces/anagram_num_shuffles_n{self.num_shuffles}_seed{self.seed}.txt", "w") as f:
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
        self.reasoning_trace.append("**Question:** How many shuffles were performed on the word?")
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
        # Step 1: Understand the problem
        # ====================================================================
        self.reasoning_trace.append("### Step 1: Understand the problem")
        self.reasoning_trace.append(f"We need to count how many times the letters of the word are shuffled (rearranged).")
        self.reasoning_trace.append(f"The initial word shown is: **{self.shuffle_details[0]['word_arrangement']}**")
        self.reasoning_trace.append("")

        # ====================================================================
        # Step 2: Track each shuffle
        # ====================================================================
        self.reasoning_trace.append("### Step 2: Track each shuffle")
        self.reasoning_trace.append(f"Let's observe each shuffle that occurs:")
        self.reasoning_trace.append("")

        # Show initial state
        self.reasoning_trace.append(f"**Initial state:** {self.shuffle_details[0]['word_arrangement']}")
        self.reasoning_trace.append("")

        # Show each shuffle
        for detail in self.shuffle_details[1:]:
            self.reasoning_trace.append(f"**Shuffle {detail['shuffle_num']}:** → {detail['word_arrangement']}")

        self.reasoning_trace.append("")

        # ====================================================================
        # Step 3: Count the shuffles
        # ====================================================================
        self.reasoning_trace.append("### Step 3: Count the shuffles")
        self.reasoning_trace.append(f"By counting each rearrangement of the letters, we can determine:")
        self.reasoning_trace.append(f"- Initial arrangement: 1 (starting state)")
        self.reasoning_trace.append(f"- Number of shuffles performed: **{self.num_shuffles}**")
        self.reasoning_trace.append("")

        # ====================================================================
        # Final answer
        # ====================================================================
        self.reasoning_trace.append("### Final Answer")
        self.reasoning_trace.append(f"The total number of shuffles performed on the word is **{self.answer}**.")
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
        filename = f"anagram_num_shuffles_n{scene.num_shuffles}_seed{scene.seed}.mp4"
        shutil.move(str(output), f"questions/{filename}")
        print(f"✓ Video saved: questions/{filename}")
        print(f"✓ Solution saved: solutions/anagram_num_shuffles_n{scene.num_shuffles}_seed{scene.seed}.txt")
        print(f"✓ Question text saved: question_text/anagram_num_shuffles_n{scene.num_shuffles}_seed{scene.seed}.txt")
        print(f"✓ Reasoning trace saved: reasoning_traces/anagram_num_shuffles_n{scene.num_shuffles}_seed{scene.seed}.txt")
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