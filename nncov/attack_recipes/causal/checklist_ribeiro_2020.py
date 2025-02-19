"""
CheckList:
=========================

(Beyond Accuracy: Behavioral Testing of NLP models with CheckList)

"""
from textattack import Attack
from textattack.constraints.pre_transformation import RepeatModification
from textattack.goal_functions import UntargetedClassification
from textattack.search_methods import GreedySearch
from textattack.transformations import (
    CompositeTransformation,
    WordSwapChangeLocation,
    WordSwapChangeName,
    WordSwapChangeNumber,
    WordSwapContract,
    WordSwapExtend,
)

from textattack.attack_recipes.attack_recipe import AttackRecipe
from textattack.constraints.pre_transformation import (
    RepeatModification,
    StopwordModification,
    MaxNumWordsModified,
)
from nncov.constraints.pre_transformation.multi_choice_instruction import MultiChoiceInstructionModification
from nncov.constraints.pre_transformation.multi_choice_options import MultiChoiceOptionsModification
from nncov.goal_functions.classification.mcq import UntargetedMultipleChoiceQuestionClassification

QUERY_BUDGET = 100

class CheckList2020(AttackRecipe):
    """An implementation of the attack used in "Beyond Accuracy: Behavioral
    Testing of NLP models with CheckList", Ribeiro et al., 2020.

    This attack focuses on a number of attacks used in the Invariance Testing
    Method: Contraction, Extension, Changing Names, Number, Location

    https://arxiv.org/abs/2005.04118
    """

    @staticmethod
    def build(model_wrapper):
        transformation = CompositeTransformation(
            [
                WordSwapExtend(),
                WordSwapContract(),
                WordSwapChangeName(),
                WordSwapChangeNumber(),
                WordSwapChangeLocation(),
            ]
        )

        # Need this constraint to prevent extend and contract modifying each others' changes and forming infinite loop
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
        search_method = GreedySearch()

        return Attack(goal_function, constraints, transformation, search_method)
