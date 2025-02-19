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


class WordSwapHotFlipBased(WordSwap):
    """Uses the model's gradient to suggest replacements for a given word.

    Based off of HotFlip: White-Box Adversarial Examples for Text
    Classification (Ebrahimi et al., 2018).
    https://arxiv.org/pdf/1712.06751.pdf

    Arguments:
        model (nn.Module): The model to attack. Model must have a
            `word_embeddings` matrix and `convert_id_to_word` function.
        top_n (int): the number of top words to return at each index
    >>> from textattack.transformations import WordSwapGradientBased
    >>> from textattack.augmentation import Augmenter

    >>> transformation = WordSwapGradientBased()
    >>> augmenter = Augmenter(transformation=transformation)
    >>> s = 'I am fabulous.'
    >>> augmenter.augment(s)
    """

    def __init__(self, model_wrapper, top_n=1):
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

        self.top_n = top_n
        self.is_black_box = False

    def _get_replacement_words_by_grad(self, attacked_text: AttackedText, indices_to_replace):
        """Returns returns a list containing all possible words to replace
        `word` with, based off of the model's gradient.

        Arguments:
            attacked_text (AttackedText): The full text input to perturb
            word_index (int): index of the word to replace
        """

        word2token_mapping = attacked_text.align_with_model_tokens(self.model_wrapper)

        lookup_table = self.model.get_input_embeddings().weight.data.float()

        grad_output = self.model_wrapper.get_grad(attacked_text.tokenizer_input)
        emb_grad = torch.tensor(grad_output["gradient"]).float()
        # emb_grad = grad_output["gradient"].float().clone().detach()
        text_ids = grad_output["ids"]
        # grad differences between all flips and original word (eq. 1 from paper)
        vocab_size = lookup_table.size(0)

        token_pos_2_word_pos = {}
        token_indices_to_replace = []
        num_token_to_replace = 0
        for word_pos in indices_to_replace:
            matched_tokens = word2token_mapping[word_pos]
            if matched_tokens is None:
                continue
            for token_pos in matched_tokens:
                # Make sure the word is in bounds.
                if token_pos >= len(emb_grad):
                    continue
                token_pos_2_word_pos[token_pos] = word_pos
                token_indices_to_replace.append(token_pos)
                num_token_to_replace += 1

        diffs = torch.zeros(num_token_to_replace, vocab_size, device=lookup_table.device)
        indices_to_replace = list(indices_to_replace)

        token_num=0
        for word_pos in indices_to_replace:
            matched_tokens = word2token_mapping[word_pos]
            if matched_tokens is None:
                continue
            for token_pos in matched_tokens:
                # Make sure the word is in bounds.
                if token_pos >= len(emb_grad):
                    continue
                token_idx = text_ids[token_pos]
                # Get the grad w.r.t the one-hot index of the word.
                b_grads = lookup_table.mv(emb_grad[token_pos].to(lookup_table.device)).squeeze()
                a_grad = b_grads[token_idx]
                diffs[token_num] += b_grads - a_grad
                token_num += 1

        # Don't change to the pad token.
        diffs[:, self.tokenizer.pad_token_id] = float("-inf")

        # Find best indices within 2-d tensor by flattening.
        token_idxs_sorted_by_grad = (-diffs).flatten().argsort()

        candidates = []
        num_tokens_in_text, num_tokens_in_vocab = diffs.shape
        for idx in token_idxs_sorted_by_grad.tolist():
            idx_in_diffs = idx // num_tokens_in_vocab
            idx_in_vocab = idx % (num_tokens_in_vocab)
            
            word_pos = token_pos_2_word_pos[token_indices_to_replace[idx_in_diffs]]
            matched_tokens = word2token_mapping[word_pos]
            if matched_tokens is None:
                continue

            token_ids = []
            for token_pos in matched_tokens:
                if token_indices_to_replace[idx_in_diffs] == token_pos:
                    token_ids.append(idx_in_vocab)
                else:
                    token_ids.append(text_ids[token_pos])

            if hasattr(self.tokenizer , "convert_id_to_word"):
                raw_tokens = [self.tokenizer.convert_id_to_word(token_id) for token_id in token_ids]
                
            else:
                raw_tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
            
            tokens = []
            for each_token in raw_tokens:
                if each_token.startswith("▁"):
                    tokens.append(" " + each_token[len("▁"):])
                elif each_token.startswith("Ġ"):
                    tokens.append(" " + each_token[len("Ġ"):])
                else:
                    tokens.append(each_token)
                
            word = "".join(tokens)

            if (not utils.has_letter(word)) or (len(utils.words_from_text(word)) != 1):
                # Do not consider words that are solely letters or punctuation.
                continue
            candidates.append((word, word_pos))
            if self.top_n == -1 and len(candidates) == len(indices_to_replace):
                break
            if self.top_n > 0 and len(candidates) == self.top_n:
                break

        return candidates

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
        return ["top_n"]
