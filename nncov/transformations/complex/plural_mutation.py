from typing import Iterable

from nncov.transformations.complex.mutation.plural.det import starts_with_vowel_sound

from .mutation import MutationResult
from .utils import AttackedTextStack, FlairTokenizer

from .mutation.syntax import find_child_index, recover_word
from .mutation.negative.detection import is_negative
from .mutation.tense.detection import is_simple_past
from .mutation.tense.mut import past_to_present
from .mutation.plural import is_plural, is_singular, is_verb_plural, is_verb_singular, is_word_plural, pluralize, pluralize_verb, singularize, singularize_verb

from textattack.transformations import WordSwap
from textattack.attack import AttackedText

import spacy


def mutate_singular_to_plural(sentence: AttackedText, noun_index: int, original_pos, original_head, original_deps):
    new_sentence = AttackedTextStack(sentence)

    token = sentence.words[noun_index]
    det_index = find_child_index(original_deps, original_head, noun_index, "det")
    nount_token = sentence.words[det_index].strip().lower()

    if det_index != -1 and nount_token in ["a", "an"]: # "this", "that"可以修饰不可数
        new_sentence.replace_word_at_index(noun_index, pluralize(token))
        new_sentence.delete_word_at_index(det_index)

        if original_deps[noun_index] == "nsubj":
            parent_index = original_head[noun_index].i
            parent_token = sentence.words[parent_index]
            cleaned_parent_token = parent_token.lower().strip()
            # search have had has
            verb_aux_token_index = find_child_index(original_deps, original_head, parent_index, "aux")
            if verb_aux_token_index != -1:
                verb_aux_token = sentence.words[verb_aux_token_index].lower().strip()
                if verb_aux_token == "has":
                    new_sentence.replace_word_at_index(verb_aux_token_index, recover_word("have", sentence.words[verb_aux_token_index]))
                else:
                    new_sentence.replace_word_at_index(verb_aux_token_index, recover_word(pluralize_verb(verb_aux_token), sentence.words[verb_aux_token_index]))
            else:
                if is_verb_singular(cleaned_parent_token):
                    new_sentence.replace_word_at_index(parent_index, recover_word(pluralize_verb(cleaned_parent_token), parent_token))
        elif original_deps[noun_index] == "nsubjpass":
            parent_index = original_head[noun_index].i
            auxpass_index = find_child_index(original_deps, original_head, parent_index, "auxpass")
            auxpass_token = sentence.words[auxpass_index]
            cleaned_auxpass_token = auxpass_token.lower().strip()
            if is_verb_singular(cleaned_auxpass_token):
                new_sentence.replace_word_at_index(auxpass_index, recover_word(pluralize_verb(cleaned_auxpass_token), auxpass_token))
    else:
        return None
    return new_sentence.compute_stack()

def mutate_plural_to_singular(sentence: AttackedText, noun_index: int, original_pos, original_head, original_deps):
    new_sentence = AttackedTextStack(sentence)

    token = sentence.words[noun_index]
    det_index = find_child_index(original_deps, original_head, noun_index, "det")
    amod_index = find_child_index(original_deps, original_head, noun_index, "amod")

    prefix_token = None
    prefix_index = None
    if det_index != -1:
        det_token = sentence.words[det_index].strip().lower()
        if det_token in ["some", "these", "those"]:
            prefix_token = det_token
            prefix_index = det_index
        else:
            return None
    elif amod_index != -1:
        amod_token = sentence.words[amod_index].strip().lower()
        if amod_token in ["many"]:
            prefix_token = amod_token
            prefix_index = amod_index
        else:
            return None
    else:
        return None

    new_sentence.replace_word_at_index(noun_index, singularize(token))
    new_sentence.delete_word_at_index(prefix_index)

    if prefix_token in ["many", "some"]:
        if starts_with_vowel_sound(token.lower().strip()):
            new_sentence.insert_text_before_word_index(prefix_index, "an")
        else:
            new_sentence.insert_text_before_word_index(prefix_index, "a")
    elif prefix_token in ["these", "those"]:
        new_sentence.insert_text_before_word_index(prefix_index, "the")
    
    if original_deps[noun_index] == "nsubj":
        parent_index = original_head[noun_index].i
        parent_token = sentence.words[parent_index]
        cleaned_parent_token = parent_token.lower().strip()
        # search have had has
        verb_aux_token_index = find_child_index(original_deps, original_head, parent_index, "aux")
        if verb_aux_token_index != -1:
            verb_aux_token = sentence.words[verb_aux_token_index].lower().strip()
            if verb_aux_token == "have":
                new_sentence.replace_word_at_index(verb_aux_token_index, recover_word("has", sentence.words[verb_aux_token_index]))
            else:
                new_sentence.replace_word_at_index(verb_aux_token_index, recover_word(singularize_verb(verb_aux_token), sentence.words[verb_aux_token_index]))
        else:
            if is_verb_plural(cleaned_parent_token):
                new_sentence.replace_word_at_index(parent_index, recover_word(singularize_verb(cleaned_parent_token), parent_token))
    elif original_deps[noun_index] == "nsubjpass":
        parent_index = original_head[noun_index].i
        auxpass_index = find_child_index(original_deps, original_head, parent_index, "auxpass")
        auxpass_token = sentence.words[auxpass_index]
        cleaned_auxpass_token = auxpass_token.lower().strip()
        if is_verb_plural(cleaned_auxpass_token):
            new_sentence.replace_word_at_index(auxpass_index, recover_word(singularize_verb(cleaned_auxpass_token), auxpass_token))

    return new_sentence.compute_stack()

class PluralWordMutation(WordSwap):
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
        original_deps = [token.dep_ for token in src_line_nlp]
        original_head = [token.head for token in src_line_nlp]

        transformations = []

        mutkind = []
        mutkind_word = []

        # get all nouns
        noun_indices = []

        for i in range(len(original_pos)):
            head = original_head[i].i
            if original_pos[i] == "NOUN" and original_deps[i] in ["nsubj", "dobj"] and head != i:
                if is_singular(original_tokens, original_pos, original_entities, i) or \
                    is_plural(original_tokens, original_pos, original_entities, i):
                    noun_indices.append(i)
        
        if len(noun_indices) == 0:
            return transformations
        
        # mutation
        for noun_index in noun_indices:
            if noun_index not in indices_to_modify:
                continue
            # Found target NOUN
            if is_singular(original_tokens, original_pos, original_entities, noun_index):
                mut_retcode = mutate_singular_to_plural(attacked_text, noun_index, original_pos, original_head, original_deps)
                if mut_retcode is not None:
                    mutkind.append("singular")
                    mutkind_word.append((original_tokens[noun_index], noun_index))
                    transformations.append(mut_retcode)

            elif is_plural(original_tokens, original_pos, original_entities, noun_index):
                mut_retcode = mutate_plural_to_singular(attacked_text, noun_index, original_pos, original_head, original_deps)
                if mut_retcode is not None:
                    mutkind.append("plural")
                    mutkind_word.append((original_tokens[noun_index], noun_index))
                    transformations.append(mut_retcode)

        return transformations
