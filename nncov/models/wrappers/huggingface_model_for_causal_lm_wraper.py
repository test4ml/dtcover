"""
HuggingFace Model For CausalLM Wrapper
--------------------------
"""

import torch
from torch import nn
import transformers

import textattack
from textattack.models.helpers import T5ForTextToText
from textattack.models.tokenizers import T5Tokenizer

from textattack.models.wrappers.pytorch_model_wrapper import PyTorchModelWrapper

torch.cuda.empty_cache()


class HuggingFaceModelForCausalLMWrapper(PyTorchModelWrapper):
    """Loads a HuggingFace ``transformers`` model and tokenizer."""

    def __init__(self, model, tokenizer, max_length=None, padding="max_length"):
        assert isinstance(
            model, (transformers.PreTrainedModel, T5ForTextToText)
        ), f"`model` must be of type `transformers.PreTrainedModel`, but got type {type(model)}."
        assert isinstance(
            tokenizer,
            (
                transformers.PreTrainedTokenizer,
                transformers.PreTrainedTokenizerFast,
                T5Tokenizer,
            ),
        ), f"`tokenizer` must of type `transformers.PreTrainedTokenizer` or `transformers.PreTrainedTokenizerFast`, but got type {type(tokenizer)}."

        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding
        self.loss_fn = nn.CrossEntropyLoss()
    
    def __call__(self, text_input_list):
        """Passes inputs to HuggingFace models as keyword arguments.

        (Regular PyTorch ``nn.Module`` models typically take inputs as
        positional arguments.)
        """
        # Default max length is set to be int(1e30), so we force 512 to enable batching.
        max_length = (
            512
            if self.tokenizer.model_max_length == int(1e30)
            else self.tokenizer.model_max_length
        )
        if self.max_length is not None:
            max_length = self.max_length
        inputs_dict = self.tokenizer(
            text_input_list,
            add_special_tokens=True,
            padding=self.padding,
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        model_device = next(self.model.parameters()).device
        inputs_dict.to(model_device)

        with torch.no_grad():
            outputs = self.model(**inputs_dict)

        if isinstance(outputs[0], str):
            # HuggingFace sequence-to-sequence models return a list of
            # string predictions as output. In this case, return the full
            # list of outputs.
            return outputs
        else:
            # HuggingFace classification models return a tuple as output
            # where the first item in the tuple corresponds to the list of
            # scores for each input.
            return outputs.logits[:, -1, :]

    def get_grad(self, text_input: str):
        """Get gradient of loss with respect to input tokens.

        Args:
            text_input (str): input string
        Returns:
            Dict of ids, tokens, and gradient as numpy array.
        """
        if isinstance(self.model, textattack.models.helpers.T5ForTextToText):
            raise NotImplementedError(
                "`get_grads` for T5FotTextToText has not been implemented yet."
            )

        self.model.train()
        embedding_layer = self.model.get_input_embeddings()
        original_state = embedding_layer.weight.requires_grad
        embedding_layer.weight.requires_grad = True

        emb_grads = []

        def grad_hook(module, grad_in, grad_out):
            emb_grads.append(grad_out[0])

        emb_hook = embedding_layer.register_backward_hook(grad_hook)

        self.model.zero_grad()
        model_device = next(self.model.parameters()).device
        input_dict = self.tokenizer(
            [text_input],
            add_special_tokens=True,
            return_tensors="pt",
            padding=self.padding,
            truncation=True,
        )
        input_dict.to(model_device)
        predictions = self.model(**input_dict).logits

        try:
            # predictions: [batch, seq, vocab]
            labels = predictions.argmax(dim=2)
            # lm_output = self.model(**input_dict)# [0]
            last_logits = predictions[:, -1, :]
            single_labels = labels[:, -1]
            loss = self.loss_fn(last_logits, single_labels)

        except TypeError:
            raise TypeError(
                f"{type(self.model)} class does not take in `labels` to calculate loss. "
                "One cause for this might be if you instantiatedyour model using `transformer.AutoModel` "
                "(instead of `transformers.AutoModelForSequenceClassification`)."
            )

        loss.backward()

        # grad w.r.t to word embeddings
        grad = emb_grads[0][0].detach().cpu().numpy()

        embedding_layer.weight.requires_grad = original_state
        emb_hook.remove()
        self.model.eval()

        output = {"ids": input_dict["input_ids"].squeeze().detach(), "gradient": grad}

        return output

    def _tokenize(self, inputs):
        """Helper method that for `tokenize`
        Args:
            inputs (list[str]): list of input strings
        Returns:
            tokens (list[list[str]]): List of list of tokens as strings
        """
        tokens = [
            self.tokenizer.convert_ids_to_tokens(
                self.tokenizer([x], truncation=True)["input_ids"][0]
            )
            for x in inputs
        ]

        if isinstance(self.tokenizer, transformers.PreTrainedTokenizerBase):
            out = []
            for each_input in tokens:
                each_out = []
                for token in each_input:
                    if token.startswith("▁"):
                        each_out.append(token[1:])
                    elif token.startswith("Ġ"):
                        each_out.append(token[1:])
                    else:
                        each_out.append(token)
                out.append(each_out)
            return out
        else:
            return tokens