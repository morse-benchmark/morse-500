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

class anagram_distance(Scene):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Set random seed for reproducibility
        self.seed = random.randint(1000, 9999)
        random.seed(self.seed)
        # Parameters
        self.num_shuffles = int(os.getenv("NUM_SHUFFLES", 10))

    def construct(self):
        
        # Words to choose from
        word_options = [
            "GOVERNMENT", "MARYLAND", "COMPUTER", "TERRAPIN",
            "KEYBOARD", "MOUNTAIN", "PRINCESS", "UMBRELLA",
            "BUTTERFLY", "CHOCOLATE", "WONDERFUL", "BEAUTIFUL"
        ]
        
        # Choose random word
        self.word = random.choice(word_options)
        
        # Find letters that appear exactly once
        def get_unique_letters(word):
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
        
        # Create multi-line question text
        question_lines = [
            f"What is the sum of the distance of all shifts",
            f"that happened to the letter '{self.target_letter}' across all word moves?",
            "",
            "Return the answer as a number."
        ]
        
        # Display question
        question_texts = []
        line_height = 0.6
        start_y = 3.0
        
        for i, line in enumerate(question_lines):
            if line:  # Skip empty lines for spacing
                text = Text(line, font_size=32, weight=BOLD if i < 2 else NORMAL)
                text.move_to(UP * (start_y - i * line_height))
                question_texts.append(text)
        
        # Animate question appearance
        self.play(*[FadeIn(text, shift=DOWN*0.2) for text in question_texts], run_time=1.0)
        self.wait(1.0)
        
        # Create initial word display
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
        self.play(*[FadeIn(letter, scale=0.8) for letter in letters], run_time=1.0)
        self.wait(1.5)
        
        # Track movements and reasoning
        answer = 0
        last_position = self.word.find(self.target_letter)
        reasoning_trace = []
        reasoning_trace.append(f"Initial word: {self.word}")
        reasoning_trace.append(f"Target letter: '{self.target_letter}'")
        reasoning_trace.append(f"Initial position of '{self.target_letter}': {last_position}")
        reasoning_trace.append("")
        
        # Perform shuffles with animations
        for shuffle_num in range(self.num_shuffles):
            # Create new arrangement
            letter_indices = list(range(len(self.word)))
            random.shuffle(letter_indices)
            
            # Find new position of target letter
            new_word = ''.join([letters[i].text for i in letter_indices])
            new_position = new_word.find(self.target_letter)
            
            # Calculate distance moved
            distance = abs(new_position - last_position)
            answer += distance
            
            # Record reasoning
            reasoning_trace.append(f"Shuffle {shuffle_num + 1}:")
            reasoning_trace.append(f"  New word: {new_word}")
            reasoning_trace.append(f"  '{self.target_letter}' moved from position {last_position} to {new_position}")
            reasoning_trace.append(f"  Distance: |{new_position} - {last_position}| = {distance}")
            reasoning_trace.append(f"  Running total: {answer}")
            reasoning_trace.append("")
            
            # Animate the rearrangement
            animations = []
            for i, original_index in enumerate(letter_indices):
                new_x = start_x + i * letter_spacing
                animations.append(letters[original_index].animate.move_to([new_x, 0, 0]))
            
            self.play(*animations, run_time=0.8)
            
            # Update tracking
            last_position = new_position
            letters = [letters[i] for i in letter_indices]  # Reorder letters list
            
            self.wait(0.5)
        
        reasoning_trace.append(f"Final answer: {answer}")
        
        self.wait(2)
        
        # Save solution, question text, and reasoning trace
        with open(f"solutions/anagram_all_diff_n{self.num_shuffles}_seed{self.seed}.txt", "w") as f:
            f.write(str(answer))
        
        question_text_content = (
            f"What is the sum of the distance of all shifts that happened to the letter '{self.target_letter}' across all word moves?\n"
            "Return the answer as a number."
        )
        with open(f"question_text/anagram_all_diff_n{self.num_shuffles}_seed{self.seed}.txt", "w") as f:
            f.write(question_text_content)
            
        # Save detailed reasoning trace
        with open(f"reasoning_traces/anagram_all_diff_n{self.num_shuffles}_seed{self.seed}.txt", "w") as f:
            f.write("\n".join(reasoning_trace))


if __name__ == "__main__":
    # Generate the anagram video
    scene = anagram_distance()
    scene.render()

    # Move the output file with descriptive name
    output = Path("manim_output/videos/1080p30/anagram_distance.mp4")
    if output.exists():
        filename = f"anagram_all_diff_n{scene.num_shuffles}_seed{scene.seed}.mp4"
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