"""

LEAP
==================================

(LEAP: Efficient and Automated Test Method for NLP Software)

"""

from typing import Any, Dict, List, Tuple
from textattack import Attack
from textattack.constraints.pre_transformation import (
    MaxModificationRate,
    StopwordModification,
)
from textattack.goal_functions import UntargetedClassification
from nncov.coverage.coverage_data import NeuronCoverage
from nncov.goal_functions.classification.mcq import (
    UntargetedMultipleChoiceQuestionClassification,
)
from nncov.goal_functions.classification.next_token_classification import (
    UntargetedNextTokenClassification,
)
from nncov.injection import get_injection_module
from nncov.metrics import CovKind
from nncov.metrics.utils import create_metric
from nncov.search_methods.coverage_beam_search import CoverageBeamSearch
from nncov.transformations.word_swaps.fgsm_based import WordSwapFGSMBased
from nncov.transformations.word_swaps.hotflip_based import WordSwapHotFlipBased
from ...search_methods.particle_swarm_optimization_leap import (
    ParticleSwarmOptimizationLEAP,
)
from textattack.transformations import WordSwapWordNet

from textattack.attack_recipes import AttackRecipe

from textattack.search_methods import GreedySearch, BeamSearch
from textattack.constraints.pre_transformation import (
    RepeatModification,
    StopwordModification,
    MaxNumWordsModified,
)

from nncov.constraints.pre_transformation.multi_choice_options import (
    MultiChoiceOptionsModification,
)
from nncov.constraints.pre_transformation.multi_choice_instruction import (
    MultiChoiceInstructionModification,
)

QUERY_BUDGET = 100


def get_module_name(coverage_name: str, model_type: str) -> str:
    if "mlp" in coverage_name.lower():
        mlp_re = get_injection_module(model_type, CovKind.Mlp)
        module_name = mlp_re
    elif "atten" in coverage_name.lower():
        atten_re = get_injection_module(model_type, CovKind.AttenWeights)
        module_name = atten_re
    else:
        raise Exception(f"Unsupported coverage: {coverage_name}")
    return module_name


class DTCover2024(AttackRecipe):
    @staticmethod
    def build(model_wrapper, metrics: List[Tuple[str, NeuronCoverage, Dict[str, Any]]]):
        """coverage_name: str, cover_neurons: NeuronCoverage, coverage_params: Dict[str, Any]"""
        model_device = next(model_wrapper.model.parameters()).device
        model_config = model_wrapper.model.config

        metric_objects = []  # metric, cover_neurons, module_name
        for coverage_name, cover_neurons, coverage_params in metrics:
            metric = create_metric(
                metric_name=coverage_name,
                params=coverage_params,
                model_device=model_device,
            )
            metric_objects.append(
                (
                    metric,
                    cover_neurons,
                    get_module_name(coverage_name, model_config.model_type),
                )
            )
        # build transformations
        transformation = WordSwapHotFlipBased(model_wrapper, top_n=-1)

        # We'll constrain modification of already modified indices and stopwords
        constraints = [
            RepeatModification(),
            StopwordModification(),
            MultiChoiceOptionsModification(),
            MultiChoiceInstructionModification(),
            MaxNumWordsModified(max_num_words=5),
        ]

        # Search Method
        search_method = CoverageBeamSearch(metric_objects)

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

        # Now, let's make the attack from the 4 components:
        attack = Attack(
            goal_function,
            constraints,
            transformation,
            search_method,
            transformation_cache_size=4,
            constraint_cache_size=4,
        )

        return attack
