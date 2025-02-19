from typing import Iterable

from .mutation import MutationResult
from .mutation.gender import has_person_name, replace_gender_token
from .mutation.gender.vocab import is_female_word, is_male_word
from .utils import FlairTokenizer

from textattack.transformations import WordSwap
from textattack.attack import AttackedText

import spacy

class GenderWordMutation(WordSwap):
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        self.nlp.tokenizer = FlairTokenizer(self.nlp.vocab)
        
    def _get_transformations(self, attacked_text: AttackedText, indices_to_modify):
        words = attacked_text.words
        
        src_line_nlps = self.nlp.__call__(attacked_text.text)
        src_line_nlp = src_line_nlps

        original_tokens = [token.text_with_ws for token in src_line_nlp]
        original_pos = [token.pos_ for token in src_line_nlp]
        original_entities = [entity for entity in src_line_nlp.ents]

        transformations = []

        gender = ""
        gender_words = list()
        mutant_tokens = list()
        if has_person_name(original_entities):
            # return gender, gender_words, "".join(mutant_tokens), " ".join(pure_tokens), MutationResult.NER, identical_align
            return transformations

        for (idx, (token, pos)) in enumerate(zip(original_tokens, original_pos)):
            current_gender = ""
            if pos in ["DET", "NOUN", "PRON", "ADJ"]:
                t = token.strip().lower()
                if is_male_word(t):
                    current_gender = "male"
                    gender_words.append((t, idx))
                elif is_female_word(t):
                    current_gender = "female"
                    gender_words.append((t, idx))
                else:
                    mutant_tokens.append(token)
                    continue

                # First occupation or consistent with previous gender words
                if gender == "" or (gender != "" and current_gender == gender):
                    gender = current_gender
                    mutant_tokens.append(replace_gender_token(token, pos))
                # Kick out sentences which contain conflict gender
                # For example: He has a daughter
                else:
                    gender = ""
                    return transformations
            else:
                mutant_tokens.append(token)

        # generate transformations
        indices = []
        new_words = []
        for idx, word in enumerate(words):
            if idx not in indices_to_modify:
                continue
            if mutant_tokens[idx].rstrip() != words[idx]:
                indices.append(idx)
                new_words.append(mutant_tokens[idx].rstrip())
        
        transformations.append(attacked_text.replace_words_at_indices(indices, new_words))
        return transformations
