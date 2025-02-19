"""

Improved Genetic Algorithm
=============================

(Natural Language Adversarial Attacks and Defenses in Word Level)

"""
from textattack import Attack
from textattack.constraints.overlap import MaxWordsPerturbed
from textattack.constraints.pre_transformation import (
    RepeatModification,
    StopwordModification,
    MaxNumWordsModified,
)
from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.goal_functions import UntargetedClassification
from textattack.search_methods import ImprovedGeneticAlgorithm
from textattack.transformations import WordSwapEmbedding

from textattack.attack_recipes.attack_recipe import AttackRecipe

from nncov.constraints.pre_transformation.multi_choice_instruction import MultiChoiceInstructionModification
from nncov.constraints.pre_transformation.multi_choice_options import MultiChoiceOptionsModification
from nncov.goal_functions.classification.mcq import UntargetedMultipleChoiceQuestionClassification

QUERY_BUDGET = 100

class IGAWang2019(AttackRecipe):
    """Xiaosen Wang, Hao Jin, Kun He (2019).

    Natural Language Adversarial Attack and Defense in Word Level.

    http://arxiv.org/abs/1909.06723
    """

    @staticmethod
    def build(model_wrapper):
        #
        # Swap words with their embedding nearest-neighbors.
        # Embedding: Counter-fitted Paragram Embeddings.
        # Fix the hyperparameter value to N = Unrestricted (50)."
        #
        transformation = WordSwapEmbedding(max_candidates=50)
        #
        # Don't modify the stopwords
        #
        constraints = [
            RepeatModification(),
            StopwordModification(),
            MultiChoiceOptionsModification(),
            MultiChoiceInstructionModification(),
            MaxNumWordsModified(max_num_words=5),
        ]
        #
        # Maximum words perturbed percentage of 20%
        #
        constraints.append(MaxWordsPerturbed(max_percent=0.2))
        #
        # Maximum word embedding euclidean distance δ of 0.5.
        #
        constraints.append(
            WordEmbeddingDistance(max_mse_dist=0.5, compare_against_original=False)
        )
        
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
        # Perform word substitution with an improved genetic algorithm.
        # Fix the hyperparameter values to S = 60, M = 20, λ = 5."
        #
        search_method = ImprovedGeneticAlgorithm(
            pop_size=60,
            max_iters=20,
            max_replace_times_per_index=5,
            post_crossover_check=False,
        )

        return Attack(goal_function, constraints, transformation, search_method)
