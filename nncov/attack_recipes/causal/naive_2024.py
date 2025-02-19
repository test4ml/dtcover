"""

HotFlip
===========
(HotFlip: White-Box Adversarial Examples for Text Classification)

"""
from textattack import Attack
from textattack.constraints.grammaticality import PartOfSpeech
from textattack.constraints.overlap import MaxWordsPerturbed
from textattack.constraints.pre_transformation import (
    RepeatModification,
    StopwordModification,
    MaxNumWordsModified,
)
from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.goal_functions import UntargetedClassification
from textattack.search_methods import BeamSearch
from textattack.transformations import WordSwapGradientBased

from textattack.attack_recipes.attack_recipe import AttackRecipe

from nncov.constraints.pre_transformation.multi_choice_instruction import MultiChoiceInstructionModification
from nncov.constraints.pre_transformation.multi_choice_options import MultiChoiceOptionsModification
from nncov.goal_functions.classification.mcq import UntargetedMultipleChoiceQuestionClassification
from nncov.goal_functions.classification.next_token_classification import UntargetedNextTokenClassification
from nncov.transformations.word_swaps.fgsm_based import WordSwapFGSMBased
from textattack.transformations.word_swaps import WordSwapRandomCharacterDeletion
from textattack.transformations.word_swaps import WordSwapRandomCharacterInsertion
from textattack.transformations.word_swaps import WordSwapNeighboringCharacterSwap
from nncov.transformations.word_swaps.word_swap_qwerty import WordSwapQWERTY

QUERY_BUDGET = 100

class Naive2024(AttackRecipe):
    @staticmethod
    def build(model_wrapper, naive_method: str):
        # transformation = WordSwapFGSMBased(model_wrapper, scale_max=scale_max, scale_step=scale_step)
        if naive_method == "WordSwapNeighboringCharacterSwap":
            transformation = WordSwapNeighboringCharacterSwap()
        elif naive_method == "WordSwapRandomCharacterDeletion":
            transformation = WordSwapRandomCharacterDeletion()
        elif naive_method == "WordSwapRandomCharacterInsertion":
            transformation = WordSwapRandomCharacterInsertion()
        elif naive_method == "WordSwapQWERTY":
            transformation = WordSwapQWERTY()
        else:
            raise Exception("Unsupported naive transformation method")
        #
        # Don't modify the same word twice or stopwords
        #
        constraints = [
            RepeatModification(),
            StopwordModification(),
            MultiChoiceOptionsModification(),
            MultiChoiceInstructionModification(),
            MaxNumWordsModified(max_num_words=5),
        ]

        #
        # Goal Function
        #
        ignore_special_tokens = 100
        model_config = model_wrapper.model.config
        if model_config.model_type == "falcon":
            ignore_special_tokens = 10
        """
        goal_function = UntargetedMultipleChoiceQuestionClassification(
            model_wrapper=model_wrapper,
            use_cache=False,
            model_cache_size=4,
            # query_budget=3,
            ignore_special_tokens=ignore_special_tokens,
        )
        """
        goal_function = UntargetedMultipleChoiceQuestionClassification(
            model_wrapper=model_wrapper,
            num_options=4,
            use_cache=False,
            model_cache_size=4,
            query_budget=QUERY_BUDGET,
        )

        search_method = BeamSearch(beam_width=10)

        return Attack(goal_function, constraints, transformation, search_method)
