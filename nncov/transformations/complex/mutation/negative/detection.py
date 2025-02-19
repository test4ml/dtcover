


from typing import List

from ..tense import TenseType
from ..syntax import find_child_index

def is_negative(original_tokens, original_tags, original_deps, original_head, verb_index) -> TenseType:
    neg_index = find_child_index(original_deps, original_head, verb_index, "neg")

    return neg_index != -1