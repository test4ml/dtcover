"""
Determine for if an attack has been successful in Next Token Prediction
---------------------------------------------------------------------
"""

import numpy as np
import torch

from textattack.goal_functions import GoalFunction

from nncov.goal_function_results.next_token_goal_function_result import NextTokenGoalFunctionResult


class NextTokenGoalFunction(GoalFunction):
    """A goal function defined on a model that outputs a probability for next token."""
    def __init__(
        self,
        model_wrapper,
        maximizable=False,
        use_cache=True,
        query_budget=float("inf"),
        model_batch_size=32,
        model_cache_size=2**20,
    ):
        super().__init__(model_wrapper,
            maximizable=maximizable,
            use_cache=use_cache,
            query_budget=query_budget,
            model_batch_size=model_batch_size,
            model_cache_size=model_cache_size
            )
    def _process_model_outputs(self, inputs, logits):
        """Processes and validates a list of model outputs.

        This is a task-dependent operation. For example, classification
        outputs need to have a softmax applied.
        """
        if len(logits.shape) == 3:
            scores = logits[:, -1, :].squeeze(0)
        elif len(logits.shape) == 2:
            scores = logits.squeeze(0)
        else:
            raise Exception("Unsupported logits shape")
        # Automatically cast a list or ndarray of predictions to a tensor.
        if isinstance(scores, list) or isinstance(scores, np.ndarray):
            scores = torch.tensor(scores)

        # Ensure the returned value is now a tensor.
        if not isinstance(scores, torch.Tensor):
            raise TypeError(
                "Must have list, np.ndarray, or torch.Tensor of "
                f"scores. Got type {type(scores)}"
            )

        # Validation check on model score dimensions
        if scores.ndim == 1:
            # Unsqueeze prediction, if it's been squeezed by the model.
            if len(inputs) == 1:
                scores = scores.unsqueeze(dim=0)
            else:
                raise ValueError(
                    f"Model return score of shape {scores.shape} for {len(inputs)} inputs."
                )
        elif scores.ndim != 2:
            # If model somehow returns too may dimensions, throw an error.
            raise ValueError(
                f"Model return score of shape {scores.shape} for {len(inputs)} inputs."
            )
        elif scores.shape[0] != len(inputs):
            # If model returns an incorrect number of scores, throw an error.
            raise ValueError(
                f"Model return score of shape {scores.shape} for {len(inputs)} inputs."
            )
 
        return scores.cpu()

    def _goal_function_result_type(self):
        """Returns the class of this goal function's results."""
        return NextTokenGoalFunctionResult

    def extra_repr_keys(self):
        return []
