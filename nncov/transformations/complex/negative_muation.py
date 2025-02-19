from typing import Iterable, List

import spacy


from .utils import AttackedTextStack, FlairTokenizer
from .mutation import MutationResult
from .mutation.syntax import find_child_index, recover_word
from .mutation.negative.detection import is_negative
from .mutation.tense.detection import is_simple_past
from .mutation.tense.mut import past_to_present
from .mutation.plural import is_word_plural

from textattack.transformations import WordSwap
from textattack.attack import AttackedText


def mutate_negative_to_positive(sentence: AttackedText, verb_index: int, original_pos: List[str], original_head, original_deps):
    new_sentence = AttackedTextStack(sentence)
    verb_token = sentence.words[verb_index]
    cleaned_verb_token = sentence.words[verb_index].strip().lower()

    neg_index = find_child_index(original_deps=original_deps, original_head=original_head, index=verb_index, type="neg")
    is_neg = neg_index != -1

    if is_neg:
        new_sentence.delete_word_at_index(neg_index)

        aux_index = find_child_index(original_deps=original_deps, original_head=original_head, index=verb_index, type=["aux", "auxpass"])
        if aux_index != -1:
            aux_token = sentence.words[aux_index].strip().lower()
            if aux_token in ["do", "does", "did"]:
                new_sentence.delete_word_at_index(aux_index)
                
            if aux_token == "wo":
                new_sentence.replace_word_at_index(aux_index, recover_word("will", sentence.words[aux_index]))
    else:
        return None
    return new_sentence.compute_stack()

def mutate_positive_to_negative(sentence: AttackedText, verb_index: int, original_pos: List[str], original_head, original_deps):
    new_sentence = AttackedTextStack(sentence)
    verb_token = sentence.words[verb_index]
    cleaned_verb_token = sentence.words[verb_index].strip().lower()

    neg_index = find_child_index(original_deps=original_deps, original_head=original_head, index=verb_index, type="neg")
    is_neg = neg_index != -1

    if not is_neg:
        cleaned_tokens = [tok.lower().strip() for tok in sentence.words]
        if "?" in cleaned_tokens or "what" in cleaned_tokens or \
                "why" in cleaned_tokens or "how" in cleaned_tokens:
            return False
        
        aux_index = find_child_index(original_deps=original_deps, original_head=original_head, index=verb_index, type=["aux", "auxpass"])
        if aux_index != -1:
            new_sentence.insert_text_before_word_index(aux_index + 1, "not")
            return new_sentence.compute_stack()
        if original_pos[verb_index] == "AUX":
            new_sentence.insert_text_before_word_index(verb_index + 1, "not")
            return new_sentence.compute_stack()

        nsubj_index = find_child_index(original_deps=original_deps, original_head=original_head, index=verb_index, type="nsubj")
        if nsubj_index != -1:
            nsubj = sentence.words[nsubj_index].strip().lower()
            verb = sentence.words[verb_index].strip().lower()
            if is_simple_past(verb):
                new_sentence.insert_text_before_word_index(verb_index, "did not")
            else:
                if is_word_plural(nsubj) or nsubj in ["you", "we", "they"]:
                    new_sentence.insert_text_before_word_index(verb_index, "do not")
                else:
                    new_sentence.insert_text_before_word_index(verb_index, "does not")
            # 动词时态还原
            new_sentence.replace_word_at_index(verb_index, recover_word(past_to_present(cleaned_verb_token), verb_token))
            return new_sentence.compute_stack()

        expl_index = find_child_index(original_deps=original_deps, original_head=original_head, index=verb_index, type="expl")
        if expl_index == -1:
            # 祈使句
            if verb_index == 0:
                new_sentence.insert_text_before_word_index(verb_index, "Do not")
                new_sentence.replace_word_at_index(verb_index, sentence.words[verb_index].lower())
            else:
                new_sentence.insert_text_before_word_index(verb_index, "do not")
            return new_sentence.compute_stack()
        else:
            return None
    else:
        return None

class NegativeWordMutation(WordSwap):
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

        verb_indices = []
        # replace
        for i, token in enumerate(original_tokens):
            if original_deps[i] in ["ROOT", "conj"] and original_pos[i] in ["VERB", "AUX"]:
                verb_indices.append(i)

        for verb_index in verb_indices:
            if verb_index not in indices_to_modify:
                continue
            token = original_tokens[verb_index]
            cleaned_token = original_tokens[verb_index].strip().lower()

            neg = is_negative(original_tokens, original_pos, original_deps, original_head, verb_index)

            if not neg:
                cleaned_tokens = [tok.lower().strip() for tok in attacked_text.words]
                if "?" in cleaned_tokens or "what" in cleaned_tokens or \
                        "why" in cleaned_tokens or "how" in cleaned_tokens:
                    continue
                mut_retcode = mutate_positive_to_negative(attacked_text, verb_index, original_pos, original_head, original_deps)
                if mut_retcode is not None:
                    mutkind.append("positive")
                    mutkind_word.append((cleaned_token, verb_index))
                    transformations.append(mut_retcode)
            else:
                mut_retcode = mutate_negative_to_positive(attacked_text, verb_index, original_pos, original_head, original_deps)
                if mut_retcode:
                    mutkind.append("negative")
                    mutkind_word.append((cleaned_token, verb_index))
                    transformations.append(mut_retcode)

        if len(mutkind) != 1:
            transformations.clear()
        return transformations
