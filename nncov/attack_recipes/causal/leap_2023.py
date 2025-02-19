"""

LEAP
==================================

(LEAP: Efficient and Automated Test Method for NLP Software)

"""
from textattack import Attack
from textattack.constraints.pre_transformation import (
    RepeatModification,
    StopwordModification,
    MaxNumWordsModified,
)
from textattack.goal_functions import UntargetedClassification
from nncov.constraints.pre_transformation.multi_choice_instruction import MultiChoiceInstructionModification
from nncov.constraints.pre_transformation.multi_choice_options import MultiChoiceOptionsModification
from nncov.goal_functions.classification.mcq import UntargetedMultipleChoiceQuestionClassification
from nncov.goal_functions.classification.next_token_classification import UntargetedNextTokenClassification
from ...search_methods.particle_swarm_optimization_leap import ParticleSwarmOptimizationLEAP
from textattack.transformations import WordSwapWordNet

from textattack.attack_recipes import AttackRecipe

QUERY_BUDGET = 100

class LEAP2023(AttackRecipe):
    @staticmethod
    def build(model_wrapper):
        #
        # Swap words with their synonyms extracted based on the WordNet.
        #
        transformation = WordSwapWordNet()
        #
        # MaxModificationRate = 0.16 in AG's News
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
        # Perform word substitution with LEAP algorithm.
        #
        search_method = ParticleSwarmOptimizationLEAP(
            pop_size=60, max_iters=20, post_turn_check=True, max_turn_retries=20
        )

        return Attack(goal_function, constraints, transformation, search_method)
