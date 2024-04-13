"""

Multi Choice Options Modification
--------------------------

"""

from typing import Set

from textattack.attack import AttackedText
from textattack.constraints import PreTransformationConstraint
from textattack.shared.validators import transformation_consists_of_word_swaps


class MultiChoiceOptionsModification(PreTransformationConstraint):
    """A constraint disallowing the modification of options."""

    def __init__(self):
        pass

    def _get_modifiable_indices(self, current_text: AttackedText) -> Set[int]:
        """Returns the word indices in ``current_text`` which are able to be
        modified."""
        options = ["A", "B", "C", "D"]
        non_option_indices = list()
        is_option = False
        for i, word in enumerate(current_text.words):
            if word in options:
                is_option = True
            if not is_option:
                non_option_indices.append(i)
            if word == "Answer":
                is_option = False
        return set(non_option_indices)
