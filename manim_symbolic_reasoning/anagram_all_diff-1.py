from manim import *
import numpy as np

class AnagramAnimation(Scene):
    def construct(self):
        # The original word
        word = "GOVERNMENT"
        
        # Create and add a title
        question = "What is the sum of the distance of all shifts\nthat happened to the letter 'V' across all word moves?\nPlease answer with just a number and no other text"
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
        
        answer = 0
        last_word = word

        # Shuffle the letters to create an anagram
        for _ in range(10):
            shuffled_letters = letters.copy()
            np.random.shuffle(shuffled_letters)
            
            shuffled_word = ''.join([l.text for l in shuffled_letters])
            answer += np.abs(shuffled_word.find('V') - last_word.find('V'))
            last_word = shuffled_word

            # Animate the letters rearranging
            self.play(
                *[letter.animate.move_to(LEFT * (len(word) / 2 - i)) for i, letter in enumerate(shuffled_letters)]
            )
            
        with open('combined_questions.csv', 'a+') as file:
                file.write(f"media/videos/anagram-5/480p15/AnagramAnimation.mp4, {answer}, {question}\n")
        self.wait(1)
