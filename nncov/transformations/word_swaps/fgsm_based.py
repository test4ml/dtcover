"""
Word Swap by Gradient
-------------------------------

"""
from typing import List
import torch

import textattack
from textattack.attack import AttackedText
from textattack.shared import utils
from textattack.shared.validators import validate_model_gradient_word_swap_compatibility

from textattack.transformations.word_swaps.word_swap import WordSwap

import tqdm

class WordSwapFGSMBased(WordSwap):
    def __init__(self, model_wrapper, scale_max=3, scale_step=0.3):
        # Unwrap model wrappers. Need raw model for gradient.
        if not isinstance(model_wrapper, textattack.models.wrappers.ModelWrapper):
            raise TypeError(f"Got invalid model wrapper type {type(model_wrapper)}")
        self.model = model_wrapper.model
        self.model_wrapper = model_wrapper
        self.tokenizer = self.model_wrapper.tokenizer
        # Make sure we know how to compute the gradient for this model.
        # validate_model_gradient_word_swap_compatibility(self.model)
        # Make sure this model has all of the required properties.
        if not hasattr(self.model, "get_input_embeddings"):
            raise ValueError(
                "Model needs word embedding matrix for gradient-based word swap"
            )
        if not hasattr(self.tokenizer, "pad_token_id") and self.tokenizer.pad_token_id:
            raise ValueError(
                "Tokenizer needs to have `pad_token_id` for gradient-based word swap"
            )

        self.scale_max = scale_max
        self.scale_step = scale_step
        self.is_black_box = False
    
    def _gen_adv(self, text_ids: List[int], word_idx: int, matched_tokens: List[int], emb_grad: torch.Tensor, lookup_table: torch.Tensor):
        candidates = []
        new_token_ids = []
        for token_pos in matched_tokens:
            # Make sure the word is in bounds.
            if token_pos >= len(emb_grad):
                continue

            token_idx = text_ids[token_pos]
            
            new_token_id = text_ids[token_pos]
            x_t = lookup_table[token_idx]
            # Scale grad
            scale = 1.0
            while scale < self.scale_max:
                pert = (emb_grad[token_pos] * scale).to(x_t.device)
                t_emb = (x_t + pert).to(lookup_table.device)
                similarity_grads = lookup_table.mv(t_emb).squeeze()
                # word_idxs_sorted_by_grad = similarity_grads.flatten().argsort()
                # nearest_token_id = int(word_idxs_sorted_by_grad[0].item())

                nearest_token_id = torch.argmin(similarity_grads.flatten()).item()
                if hasattr(self.tokenizer , "convert_id_to_word"):
                    word = self.tokenizer.convert_id_to_word(nearest_token_id)
                else:
                    word = self.tokenizer.convert_ids_to_tokens(nearest_token_id, skip_special_tokens=True)
                if (not utils.has_letter(word)) or (len(utils.words_from_text(word)) != 1):
                    # Do not consider words that are solely letters or punctuation.
                    scale += self.scale_step
                    continue
                if nearest_token_id == text_ids[token_pos]:
                    scale += self.scale_step
                    continue
                new_token_id = nearest_token_id
                break
            new_token_ids.append(new_token_id)
        
        if hasattr(self.tokenizer , "convert_id_to_word"):
            raw_tokens = [self.tokenizer.convert_id_to_word(token_id) for token_id in new_token_ids]
        else:
            raw_tokens = self.tokenizer.convert_ids_to_tokens(new_token_ids, skip_special_tokens=True)
        
        tokens = []
        for each_token in raw_tokens:
            if each_token.startswith("▁"):
                tokens.append(" " + each_token[len("▁"):])
            elif each_token.startswith("Ġ"):
                tokens.append(" " + each_token[len("Ġ"):])
            else:
                tokens.append(each_token)
            
        word = "".join(tokens)

        candidates.append((word, word_idx))
        return candidates

    def _get_replacement_words_by_grad(self, attacked_text: AttackedText, indices_to_replace: List[int]):
        """Returns returns a list containing all possible words to replace
        `word` with, based off of the model's gradient.

        Arguments:
            attacked_text (AttackedText): The full text input to perturb
            word_index (int): index of the word to replace
        """

        word2token_mapping = attacked_text.align_with_model_tokens(self.model_wrapper)

        lookup_table = self.model.get_input_embeddings().weight.data.float()# .cpu()

        grad_output = self.model_wrapper.get_grad(attacked_text.tokenizer_input)
        emb_grad = torch.tensor(grad_output["gradient"]).float()
        # emb_grad = grad_output["gradient"].float().clone().detach()
        text_ids = grad_output["ids"]
        
        vocab_size = lookup_table.size(0)

        all_candidates = []

        for word_pos in indices_to_replace:
            matched_tokens = word2token_mapping[word_pos]
            if matched_tokens is None:
                continue
            candidates = self._gen_adv(text_ids, word_pos, matched_tokens, emb_grad, lookup_table)
            all_candidates.extend(candidates)
            
        return all_candidates

    def _get_transformations(self, attacked_text, indices_to_replace):
        """Returns a list of all possible transformations for `text`.

        If indices_to_replace is set, only replaces words at those
        indices.
        """
        transformations = []
        for word, idx in self._get_replacement_words_by_grad(
            attacked_text, indices_to_replace
        ):
            transformations.append(attacked_text.replace_word_at_index(idx, word))
        return transformations

    def extra_repr_keys(self):
        return ["scale_max", "scale_step"]
