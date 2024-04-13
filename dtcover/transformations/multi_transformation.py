"""
Composite Transformation
============================================
Multiple transformations can be used by providing a list of ``Transformation`` to ``CompositeTransformation``

"""

import random
from textattack.shared import utils
from textattack.transformations import Transformation


class MultiTransformation(Transformation):
    """A transformation which applies each of a list of transformations,
    returning a set of all optoins.

    Args:
        transformations: The list of ``Transformation`` to apply.
    """

    def __init__(self, transformations, select_n: int = 3, first_k: int=1):
        if not (
            isinstance(transformations, list) or isinstance(transformations, tuple)
        ):
            raise TypeError("transformations must be list or tuple")
        elif not len(transformations):
            raise ValueError("transformations cannot be empty")
        self.transformations = transformations
        self.select_n = select_n
        self.first_k = first_k
        self.random_state = None

    def _get_transformations(self, *_):
        """Placeholder method that would throw an error if a user tried to
        treat the MultiTransformation as a 'normal' transformation."""
        raise RuntimeError(
            "MultiTransformation does not support _get_transformations()."
        )


    def __call__(self, *args, **kwargs):
        recover_state = random.getstate()
        if self.random_state is None:
            self.random_state = random.getstate()
        
        random.setstate(self.random_state)

        new_attacked_texts = set()
        pre_attacked_texts = set()
        for transformation in self.transformations:
            random.setstate(self.random_state)
            next_attacked_texts = set()
            if len(new_attacked_texts) == 0:
                next_attacked_texts.update(transformation(*args, **kwargs))
            else:
                if self.select_n > 0:
                    pre_attacked_texts = random.sample(list(pre_attacked_texts), k=min(self.select_n, len(pre_attacked_texts)))
                else:
                    pre_attacked_texts = list(pre_attacked_texts)
                for pre_attacked_text in pre_attacked_texts:
                    out = transformation(pre_attacked_text, **kwargs)
                    random.shuffle(out)
                    next_attacked_texts.update(out[:min(self.first_k, len(out))])
            pre_attacked_texts = next_attacked_texts
            new_attacked_texts |= next_attacked_texts
        attacked_texts = list(new_attacked_texts)
        random.shuffle(attacked_texts)

        self.random_state = random.getstate()
        random.setstate(recover_state)
        return attacked_texts

    def __repr__(self):
        main_str = "MultiTransformation" + "("
        transformation_lines = []
        for i, transformation in enumerate(self.transformations):
            transformation_lines.append(utils.add_indent(f"({i}): {transformation}", 2))
        transformation_lines.append(")")
        main_str += utils.add_indent("\n" + "\n".join(transformation_lines), 2)
        return main_str

    __str__ = __repr__
