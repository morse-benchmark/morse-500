from manim import *
import numpy as np

class AnagramAnimation(Scene):
    def construct(self):
        # The original word
        word = "DAFFODILLY"
        
        # Create and add a title
        question = "How many spaces did the letter 'O' move?\nPlease answer with just a number and no other text"
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

        # Shuffle the letters to create an anagram
        shuffled_indices = list(range(len(letters)))
        np.random.shuffle(shuffled_indices)
        shuffled_letters = [letters[i] for i in shuffled_indices]
        shuffled_word = ''.join([l.text for l in shuffled_letters])
        answer = np.abs(shuffled_word.find('O') - word.find('O'))
        with open('combined_questions.csv', 'a+') as file:
            file.write(f"media/videos/anagram-1/480p15/AnagramAnimation.mp4, {answer}, {question}\n")

        # Animate the letters rearranging
        self.play(
            *[letter.animate.move_to(LEFT * (len(word) / 2 - i)) for i, letter in enumerate(shuffled_letters)]
        )
        self.wait(1)
