"""

Determine successful in untargeted Multiple Choice Question
----------------------------------------------------
"""

import string
import torch
from nncov.goal_function_results.mcq_goal_function_result import MutipleChoiceQuestionGoalFunctionResult
from nncov.goal_functions.classification.next_token import NextTokenGoalFunction


class UntargetedMultipleChoiceQuestionClassification(NextTokenGoalFunction):
    def __init__(self, num_options: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        alphabet = list(string.ascii_uppercase)[:num_options]
        self.alphabet_ids = []
        for c in alphabet:
            labels_ids = self.model.tokenizer(c,
                                    return_tensors="pt",
                                    padding="longest",
                                    max_length=4,
                                    truncation=True,
                                    add_special_tokens=False
                                )["input_ids"]
            self.alphabet_ids.append(int(labels_ids[:, -1].item()))


    def _is_goal_complete(self, model_output, _):
        sorted_score_indices = (-model_output).argsort().tolist()
        token_id = int(sorted_score_indices[0])
        for each_token_id in sorted_score_indices:
            each_token_id = int(each_token_id)
            if each_token_id in self.alphabet_ids:
                token_id = each_token_id
                break

        encoded_labels = self.model.tokenizer(self.ground_truth_output,
                                              return_tensors="pt",
                                              padding="longest",
                                              max_length=4,
                                              truncation=True,
                                              add_special_tokens=False
                                            )
        labels_ids = encoded_labels["input_ids"]
        single_labels = labels_ids[:, -1]
        return token_id != single_labels.item()

    def _get_score(self, model_output, _):
        # If the model outputs a single number and the ground truth output is
        # a float, we assume that this is a regression task.
        encoded_labels = self.model.tokenizer(self.ground_truth_output,
                                              return_tensors="pt",
                                              padding="longest",
                                              max_length=4,
                                              truncation=True,
                                              add_special_tokens=False
                                            )
        labels_ids = encoded_labels["input_ids"]
        single_labels = labels_ids[:, -1]

        label = single_labels.item()
        
        total_sum = torch.sum(model_output)
        normalized_model_output = model_output / total_sum
        return 1 - normalized_model_output[label].item()

    def _get_displayed_output(self, raw_output):
        sorted_score_indices = (-raw_output).argsort().tolist()
        token_id = int(sorted_score_indices[0])
        for each_token_id in sorted_score_indices:
            each_token_id = int(each_token_id)
            if each_token_id in self.alphabet_ids:
                token_id = each_token_id
                break
        
        if hasattr(self.model.tokenizer , "convert_id_to_word"):
            output = self.model.tokenizer.convert_id_to_word(token_id)
        else:
            output = self.model.tokenizer.convert_ids_to_tokens(token_id)
        return (token_id, output)

    def _goal_function_result_type(self):
        """Returns the class of this goal function's results."""
        return MutipleChoiceQuestionGoalFunctionResult