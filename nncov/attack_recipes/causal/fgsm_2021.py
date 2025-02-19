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

QUERY_BUDGET = 100

class FGSM2021(AttackRecipe):
    @staticmethod
    def build(model_wrapper, scale_max, scale_step):
        #
        # "HotFlip ... uses the gradient with respect to a one-hot input
        # representation to efficiently estimate which individual change has the
        # highest estimated loss."
        transformation = WordSwapFGSMBased(model_wrapper, scale_max=scale_max, scale_step=scale_step)
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
        #
        # "HotFlip ... uses a beam search to find a set of manipulations that work
        # well together to confuse a classifier ... The adversary uses a beam size
        # of 10."
        #
        search_method = BeamSearch(beam_width=10)

        return Attack(goal_function, constraints, transformation, search_method)
