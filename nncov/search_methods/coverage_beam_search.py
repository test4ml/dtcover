"""
Coverage Guided Beam Search
===============

"""

from typing import List, Tuple
import numpy as np

from textattack.goal_function_results import GoalFunctionResultStatus
from textattack.search_methods import SearchMethod
from textattack.attack_results.attack_result import AttackResult
from textattack.attack import AttackedText

from nncov.coverage.coverage_data import NeuronCoverage
from nncov.coverage.coverage_infer import get_covered_neurons
from nncov.metrics import NeuralMetric

# TODO: here
class CoverageBeamSearch(SearchMethod):
    def __init__(self, metrics: List[Tuple[NeuralMetric, NeuronCoverage, str]], beam_width=8):
        """metric: NeuralMetric, cover_neurons: NeuronCoverage, module_name: str"""
        self.metrics = metrics
        self.beam_width = beam_width
    
    def get_coverage(self, each_transformation: AttackedText) -> float:
        coverage_value_sum = 0
        for metric, accumulate_cover_neurons, module_name in self.metrics:
            cover_neurons = get_covered_neurons(
                self.goal_function.model,
                text=each_transformation.text,
                metric=metric,
                metric_name=type(metric).__name__,
                module_name=module_name
            )
            metric.reset()
            metric.state = accumulate_cover_neurons.get_logical_not().get_logical_and(cover_neurons).get_logical_or(accumulate_cover_neurons).states
            coverage_value = metric.compute()
            metric.reset()
            coverage_value_sum += coverage_value
        return coverage_value_sum

    def perform_search(self, initial_result):
        beam = [initial_result.attacked_text]
        best_result = initial_result
        while not best_result.goal_status == GoalFunctionResultStatus.SUCCEEDED:
            potential_next_beam = []
            potential_next_beam_coverage = []
            for text in beam:
                transformations: List[AttackedText] = self.get_transformations(
                    text, original_text=initial_result.attacked_text
                )
                potential_next_beam += transformations
                for each_transformation in transformations:
                    coverage_value = self.get_coverage(each_transformation)
                    potential_next_beam_coverage.append(coverage_value)

            if len(potential_next_beam) == 0:
                # If we did not find any possible perturbations, give up.
                return best_result
            results, search_over = self.get_goal_results(potential_next_beam)
            # print([(r.score, potential_next_beam_coverage[i]) for i, r in enumerate(results)])

            target_scores = np.array([r.score for i, r in enumerate(results)])
            cover_scores = np.array([potential_next_beam_coverage[i] for i, r in enumerate(results)])

            min_val = np.min(target_scores)
            max_val = np.max(target_scores)
            if max_val - min_val == 0:
                target_scores = target_scores - min_val
            else:
                target_scores = (target_scores - min_val) / (max_val - min_val)

            min_val = np.min(cover_scores)
            max_val = np.max(cover_scores)
            if max_val - min_val == 0:
                cover_scores = cover_scores - min_val
            else:
                cover_scores = (cover_scores - min_val) / (max_val - min_val)

            scores = target_scores + cover_scores
            best_result = results[scores.argmax()]
            if search_over:
                return best_result

            # Refill the beam. This works by sorting the scores
            # in descending order and filling the beam from there.
            best_indices = (-scores).argsort()[: self.beam_width]
            beam = [potential_next_beam[i] for i in best_indices]

        return best_result

    @property
    def is_black_box(self):
        return False

    def extra_repr_keys(self):
        return ["beam_width"]
