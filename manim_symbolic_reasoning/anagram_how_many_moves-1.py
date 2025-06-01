from manim import *
import numpy as np

class AnagramAnimation(Scene):
    def construct(self):
        # The original word
        word = "UNDERSTATED"
        
        # Create and add a title
        question = "How many times was the word rearranged?\nPlease answer with just a number and no other text"
        title = Text(question, font_size=36)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(0.5)

        # Create a list of Text objects for each letter in the word
        letters = [Text(letter) for letter in word]
        
        # Position the letters in a line
        for i, letter in enumerate(letters):
            letter.move_to(LEFT * (len(word) / 2 - i))

        # Display the word
        self.play(*[Write(letter) for letter in letters])
        self.wait(1)
        
        answer = 7
        with open('combined_questions.csv', 'a+') as file:
            file.write(f"media/videos/anagram-2/480p15/AnagramAnimation.mp4, {answer}, {question}\n")

        # Shuffle the letters to create an anagram
        for _ in range(7):
            shuffled_letters = letters.copy()
            np.random.shuffle(shuffled_letters)

            # Animate the letters rearranging
            self.play(
                *[letter.animate.move_to(LEFT * (len(word) / 2 - i)) for i, letter in enumerate(shuffled_letters)]
            )
        self.wait(1)
