"""

Determine successful in untargeted Classification
----------------------------------------------------
"""


import torch
from nncov.goal_functions.classification.next_token import NextTokenGoalFunction


class UntargetedNextTokenClassification(NextTokenGoalFunction):
    def __init__(self, *args, **kwargs):
        
        if "ignore_special_tokens" in kwargs:
            self.ignore_special_tokens: int = kwargs["ignore_special_tokens"]
            del kwargs["ignore_special_tokens"]
        super().__init__(*args, **kwargs)

    def _is_goal_complete(self, model_output, _):
        encoded_labels = self.model.tokenizer(self.ground_truth_output,
                                              return_tensors="pt",
                                              padding="longest",
                                              max_length=4,
                                              truncation=True,
                                              add_special_tokens=False
                                            )
        labels_ids = encoded_labels["input_ids"]
        single_labels = labels_ids[:, -1]
        clone_model_output = model_output.clone()
        clone_model_output[:self.ignore_special_tokens] = -4096.0
        max_token_id = clone_model_output.argmax().item()
        # Refuse special tokens as final attack results
        if max_token_id < self.ignore_special_tokens:
            return False
        
        max_token: str = self.model.tokenizer.convert_ids_to_tokens([max_token_id])[0]
        if "<" in max_token and ">" in max_token:
            return False
        
        if max_token.startswith("▁"):
            no_prefix_max_token = max_token[len("▁"):]
        elif max_token.startswith("Ġ"):
            no_prefix_max_token = max_token[len("Ġ"):]
        else:
            no_prefix_max_token = max_token

        gt_token: str = self.model.tokenizer.convert_ids_to_tokens([single_labels.item()])[0]
        
        if gt_token.startswith("▁"):
            no_prefix_gt_token = gt_token[len("▁"):]
        elif gt_token.startswith("Ġ"):
            no_prefix_gt_token = gt_token[len("Ġ"):]
        else:
            no_prefix_gt_token = gt_token

        # print(no_prefix_max_token, no_prefix_gt_token)
        return max_token_id != single_labels.item() and no_prefix_max_token != no_prefix_gt_token

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
        
        total_sum = torch.sum(model_output.to(torch.float32))
        normalized_model_output = model_output / total_sum
        return 1 - normalized_model_output[label].item()

    def _get_displayed_output(self, raw_output):
        clone_raw_output = raw_output.clone()
        clone_raw_output[:self.ignore_special_tokens] = -4096.0
        token_id = int(clone_raw_output.argmax().item())
            
        # if hasattr(self.model.tokenizer , "convert_id_to_word"):
        #     output = self.model.tokenizer.convert_id_to_word(token_id)
        # else:
        #     output = self.model.tokenizer.convert_ids_to_tokens(token_id)
        return token_id
