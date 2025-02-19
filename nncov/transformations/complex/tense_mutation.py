
from typing import Iterable, List



from .mutation import MutationResult
from .utils import AttackedTextStack, FlairTokenizer

from .mutation.tense.irregular import AUX_PAST_PRESENT, VERB_PAST, AUX_PRESENT_PAST, PAST_VERB
from .mutation.syntax import find_child_index, recover_word
from .mutation.tense.detection import TenseDetector
from .mutation.tense import TenseType

from textattack.transformations import WordSwap
from textattack.attack import AttackedText

import spacy
import nltk




detector = TenseDetector()

LEMMA = nltk.wordnet.WordNetLemmatizer()

def past_to_present(word: str) -> str:
    cleaned_word = word.strip().lower()
    if cleaned_word in AUX_PAST_PRESENT:
        return AUX_PAST_PRESENT[cleaned_word]
    if cleaned_word in PAST_VERB:
        return PAST_VERB[cleaned_word]
    
    ret = LEMMA.lemmatize(word.lower().strip(), "v")
    return ret

def present_to_past(word: str) -> str:
    cleaned_word = word.lower().strip()
    if cleaned_word == "are":
        return "were"
    elif cleaned_word == "is":
        return "was"
    elif cleaned_word == "am":
        return "was"
    token = LEMMA.lemmatize(cleaned_word, "v")
    if token in VERB_PAST:
        return VERB_PAST[token][0]
    if token.endswith("e"):
        return token + "d"
    else:
        return token + "ed"

def aux_past_to_present(word: str, subj: str) -> str:
    if subj.lower().strip() == "i":
        return "am"
    if word in AUX_PAST_PRESENT:
        return AUX_PAST_PRESENT[word]
    return word

def aux_present_to_past(word: str, subj: str) -> str:
    if subj.lower().strip() == "i":
        return "was"
    if word in AUX_PRESENT_PAST:
        return AUX_PRESENT_PAST[word]
    return word

def mutate_simple_past_to_simple_present(sentence: AttackedText, verb_index: int, original_pos: List[str], original_head, original_deps):
    new_sentence = AttackedTextStack(sentence)
    verb_token = sentence.words[verb_index]
    cleaned_verb_token = sentence.words[verb_index].strip().lower()

    auxpass_index = find_child_index(original_deps=original_deps, original_head=original_head, index=verb_index, type="auxpass")
    is_pass = auxpass_index != -1

    if not is_pass:
        aux_index = find_child_index(original_deps=original_deps, original_head=original_head, index=verb_index, type="aux")
        nsubj_index = find_child_index(original_deps=original_deps, original_head=original_head, index=verb_index, type="nsubj")
        
        if aux_index != -1 and nsubj_index != -1:
            aux_token = sentence.words[aux_index].strip().lower()
            cleand_nsubj = sentence.words[nsubj_index].strip().lower()
            # 一般疑问句
            new_sentence.replace_word_at_index(aux_index, recover_word(aux_past_to_present(aux_token, cleand_nsubj), sentence.words[aux_index]))
        else:
            if original_pos[verb_index] == "AUX" and nsubj_index != -1:
                cleand_nsubj = sentence.words[nsubj_index].strip().lower()
                # 特殊疑问句
                new_sentence.replace_word_at_index(verb_index, recover_word(aux_past_to_present(cleaned_verb_token, cleand_nsubj), verb_token))
            else:
                new_sentence.replace_word_at_index(verb_index, recover_word(past_to_present(cleaned_verb_token), verb_token))
    else:
        auxpass_token = sentence.words[auxpass_index].strip().lower()
        new_sentence.replace_word_at_index(auxpass_index, recover_word(past_to_present(auxpass_token), verb_token))

    return new_sentence.compute_stack()

def mutate_simple_present_to_simple_past(sentence: AttackedText, verb_index: int, original_pos: List[str], original_head, original_deps):
    new_sentence = AttackedTextStack(sentence)
    verb_token = sentence.words[verb_index]
    cleaned_verb_token = sentence.words[verb_index].strip().lower()

    auxpass_index = find_child_index(original_deps=original_deps, original_head=original_head, index=verb_index, type="auxpass")
    is_pass = auxpass_index != -1

    if not is_pass:
        aux_index = find_child_index(original_deps=original_deps, original_head=original_head, index=verb_index, type="aux")
        nsubj_index = find_child_index(original_deps=original_deps, original_head=original_head, index=verb_index, type="nsubj")
        
        if aux_index != -1 and nsubj_index != -1:
            aux_token = sentence.words[aux_index].strip().lower()
            cleand_nsubj = sentence.words[nsubj_index].strip().lower()
            if aux_token in ["should", "shall", "must"]:
                return False
            # 一般疑问句
            new_sentence.replace_word_at_index(aux_index, recover_word(aux_present_to_past(aux_token, cleand_nsubj), sentence.words[aux_index]))
        else:
            if original_pos[verb_index] == "AUX" and nsubj_index != -1:
                cleand_nsubj = sentence.words[nsubj_index].strip().lower()
                # 特殊疑问句
                new_sentence.replace_word_at_index(verb_index, recover_word(aux_present_to_past(cleaned_verb_token, cleand_nsubj), verb_token))
            else:
                if cleaned_verb_token == "'s":
                    new_sentence.replace_word_at_index(verb_index, recover_word(' was', verb_token))
                elif cleaned_verb_token == "'re":
                    new_sentence.replace_word_at_index(verb_index, recover_word(' were', verb_token))
                else:
                    new_sentence.replace_word_at_index(verb_index, recover_word(present_to_past(cleaned_verb_token), verb_token))
    else:
        auxpass_token = sentence.words[auxpass_index].strip().lower()
        new_sentence.replace_word_at_index(auxpass_index, recover_word(present_to_past(auxpass_token), verb_token))

    return new_sentence.compute_stack()


class TenseWordMutation(WordSwap):
    def __init__(self, tense = TenseType.SimplePresent):
        self.nlp = spacy.load('en_core_web_sm')
        self.nlp.tokenizer = FlairTokenizer(self.nlp.vocab)
        self.tense = tense
        
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
            if original_deps[i] in ["ROOT", "conj"] and original_pos[i] == "VERB":
                verb_indices.append(i)

        # mutation
        for verb_index in verb_indices:
            if verb_index not in indices_to_modify:
                continue
            token = original_tokens[verb_index]

            try:
                t = detector.detect_tense_verb(original_tokens, original_pos, original_deps, original_head, verb_index)
            except Exception as e:
                if isinstance(e, KeyboardInterrupt):
                    raise e
                else:
                    continue

            if self.tense == t and t == TenseType.SimplePast:
                mut_retcode = mutate_simple_past_to_simple_present(attacked_text, verb_index, original_pos, original_head, original_deps)
                if mut_retcode is not None:
                    mutkind.append(t.name)
                    mutkind_word.append((token, verb_index))
                    transformations.append(mut_retcode)
                continue
            elif self.tense == t and t == TenseType.SimplePresent:
                mut_retcode = mutate_simple_present_to_simple_past(attacked_text, verb_index, original_pos, original_head, original_deps)
                if mut_retcode is not None:
                    mutkind.append(t.name)
                    mutkind_word.append((token, verb_index))
                    transformations.append(mut_retcode)
                continue

        return transformations
