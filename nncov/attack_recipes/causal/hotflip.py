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
# from textattack.transformations import WordSwapGradientBased

from textattack.attack_recipes.attack_recipe import AttackRecipe

from nncov.constraints.pre_transformation.multi_choice_instruction import MultiChoiceInstructionModification
from nncov.constraints.pre_transformation.multi_choice_options import MultiChoiceOptionsModification
from nncov.goal_functions.classification.mcq import UntargetedMultipleChoiceQuestionClassification
from nncov.goal_functions.classification.next_token_classification import UntargetedNextTokenClassification
from nncov.transformations.word_swaps.hotflip_based import WordSwapHotFlipBased

QUERY_BUDGET = 100

class HotFlipEbrahimi2017(AttackRecipe):
    @staticmethod
    def build(model_wrapper):
        #
        # "HotFlip ... uses the gradient with respect to a one-hot input
        # representation to efficiently estimate which individual change has the
        # highest estimated loss."
        transformation = WordSwapHotFlipBased(model_wrapper, top_n=1)
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
        # 0. "We were able to create only 41 examples (2% of the correctly-
        # classified instances of the SST test set) with one or two flips."
        #
        # constraints.append(MaxWordsPerturbed(max_num_words=2))
        #
        # 1. "The cosine similarity between the embedding of words is bigger than a
        #   threshold (0.8)."
        #
        # constraints.append(WordEmbeddingDistance(min_cos_sim=0.8))
        #
        # 2. "The two words have the same part-of-speech."
        #
        # constraints.append(PartOfSpeech())

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
