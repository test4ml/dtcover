
import spacy
from spacy.tokens import Doc

from textattack.shared.utils.strings import words_from_text

class FlairTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = words_from_text(text)
        spaces = [True] * len(words)
        spaces[-1] = False

        return Doc(self.vocab, words=words, spaces=spaces)

from textattack.attack import AttackedText

class AttackedTextStack(object):
    def __init__(self, text: AttackedText) -> None:
        self.text = text
        self.replace_indices = []
        self.replace_words = []
    
    def compute_stack(self) -> AttackedText:
        return self.text.replace_words_at_indices(self.replace_indices, self.replace_words)
    
    def delete_word_at_index(self, idx: int):
        self.replace_indices.append(idx)
        self.replace_words.append("")
    
    def replace_word_at_index(self, idx: int, word: str):
        self.replace_indices.append(idx)
        self.replace_words.append(word)
    
    def insert_text_after_word_index(self, index: int, text: str):
        word_at_index = self.text.words[index]
        new_text = " ".join((word_at_index, text))
        
        self.replace_indices.append(index)
        self.replace_words.append(new_text)

    def insert_text_before_word_index(self, index: int, text: str):
        word_at_index = self.text.words[index]
        # TODO if ``word_at_index`` is at the beginning of a sentence, we should
        # optionally capitalize ``text``.
        new_text = " ".join((text, word_at_index))
        
        self.replace_indices.append(index)
        self.replace_words.append(new_text)