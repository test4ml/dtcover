"""

Multi Choice Options Modification
--------------------------

"""

from typing import Set

from textattack.attack import AttackedText
from textattack.constraints import PreTransformationConstraint
from textattack.shared.validators import transformation_consists_of_word_swaps


class MultiChoiceInstructionModification(PreTransformationConstraint):
    """A constraint disallowing the modification of Instruction."""

    def __init__(self):
        pass

    def _get_modifiable_indices(self, current_text: AttackedText) -> Set[int]:
        """Returns the word indices in ``current_text`` which are able to be
        modified."""
        non_instruction_indices = list()
        is_instruction = True

        for i, word in enumerate(current_text.words):
            if "\n" in current_text.text_until_word_index(i):
                is_instruction = False
            if not is_instruction:
                non_instruction_indices.append(i)

        return set(non_instruction_indices)
