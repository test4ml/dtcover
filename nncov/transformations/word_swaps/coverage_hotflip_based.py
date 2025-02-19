"""
Word Swap by Gradient
-------------------------------

"""
import torch
import numpy as np
import transformers

import textattack
from textattack.attack import AttackedText
from textattack.shared import utils
from textattack.shared.validators import validate_model_gradient_word_swap_compatibility

from textattack.transformations.word_swaps.word_swap import WordSwap

from nncov.core import inject_module, recover_module
from nncov.cover_target import get_cover_grad
from nncov.coverage.coverage_infer import get_covered_neurons
from nncov.inference import lm_forward


# from nncov.monitors.reduce_ops import max_fn, maxmin_fn, min_fn
# from nncov.monitor import ReduceMonitor

from nncov.injection import get_injection_module
from nncov.metrics import CovKind, CovType, NeuralMetric
from nncov.metrics.ncov_pos_atten import NCovPosAtten
from nncov.metrics.ncov_mlp import NCovMlp
from nncov.metrics.kmncov_mlp import KMNCovMlp
from nncov.metrics.kmncov_pos_atten import KMNCovPosAtten
from nncov.metrics.nbcov_mlp import NBCovMlp
from nncov.metrics.nbcov_pos_atten import NBCovPosAtten
from nncov.metrics.snacov_mlp import SNACovMlp
from nncov.metrics.snacov_pos_atten import SNACovPosAtten

class WordSwapCoverageHotFlipBased(WordSwap):
    def __init__(self, model_wrapper, samplek=1, alpha=0.5, top_n=1, cov_kind=CovKind.Mlp, logger=None):
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
        
        self.samplek = samplek
        self.alpha = alpha
        self.top_n = top_n
        self.cov_kind = cov_kind

        self.is_black_box = False

        self.logger = logger

        model_config = self.model_wrapper.model.config
        self.atten_re = get_injection_module(model_config.model_type, CovKind.AttenWeights)
        self.mlp_re = get_injection_module(model_config.model_type, CovKind.Mlp)

        if self.cov_kind == CovKind.Mlp:
            self.module_re = self.mlp_re
        elif self.cov_kind == CovKind.AttenOutput:
            self.module_re = self.atten_re
        elif self.cov_kind == CovKind.AttenWeights:
            self.module_re = self.atten_re
        elif self.cov_kind == CovKind.PairAttenOutput:
            self.module_re = self.atten_re
        elif self.cov_kind == CovKind.PairAttenWeights:
            self.module_re = self.atten_re
        else:
            raise Exception("Invalid cov type")

        self.cover_neurons_states = None
    

    def _idx_to_word(self, idx_in_vocab: int) -> str:
        if hasattr(self.tokenizer , "convert_id_to_word"):
            word = self.tokenizer.convert_id_to_word(idx_in_vocab)
        else:
            word = self.tokenizer.convert_ids_to_tokens(idx_in_vocab)
        return word


    def _get_replacement_words_by_grad(self, attacked_text: AttackedText, indices_to_replace):
        """Returns returns a list containing all possible words to replace
        `word` with, based off of the model's gradient.

        Arguments:
            attacked_text (AttackedText): The full text input to perturb
            word_index (int): index of the word to replace
        """

        if self.cover_neurons_states is None:
            grad_output = self.model_wrapper.get_grad(attacked_text.tokenizer_input)
        else:
            grad_output = get_cover_grad(
                self.model_wrapper,
                attacked_text.tokenizer_input,
                self.cover_neurons_states,
                self.module_re,
                use_re=True,
                cov_kind=self.cov_kind,
                alpha=self.alpha,
                samplek=self.samplek,
            )
        
        word2token_mapping = attacked_text.align_with_model_tokens(self.model_wrapper)

        lookup_table = self.model.get_input_embeddings().weight.data.float()

        # grad_output = self.model_wrapper.get_grad(attacked_text.tokenizer_input)
        # emb_grad = torch.tensor(grad_output["gradient"]).float()
        emb_grad = grad_output["gradient"].float().clone().detach()
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
        # Setup model monitor and metrics
        transformations = []
        for word, idx in self._get_replacement_words_by_grad(
            attacked_text, indices_to_replace
        ):
            transformations.append(attacked_text.replace_word_at_index(idx, word))
        return transformations

    def extra_repr_keys(self):
        return ["top_n"]
