
import random
from typing import Dict, List, Optional, Set, Tuple
import textattack
from textattack.models.wrappers import PyTorchModelWrapper
import torch

from nncov.core import match_module
from nncov.coverage.coverage_data import NeuronCoverage, NeuronPosition
from nncov.helper import recursive_getattr
from nncov.metrics import CovKind

class UncoveredMlpHook(object):
    def __init__(self, target_neurons: List[NeuronPosition]) -> None:
        self.loss = []
        self.target_neurons: Dict[str, Set[int]] = {}
        for each_pos in target_neurons:
            if each_pos.layer not in self.target_neurons:
                self.target_neurons[each_pos.layer] = set()
            self.target_neurons[each_pos.layer].add(each_pos.i)
    
    def get_activation(self, module_name: str):
        def hook(module, fea_in, fea_out):
            # fea_out: [batch, seq, embed]
            if module_name in self.target_neurons:
                for neuron_id in self.target_neurons[module_name]:
                    uncovered_loss = torch.mean(fea_out[..., neuron_id])
                    self.loss.append(uncovered_loss)
        return hook

class UncoveredPosAttenWeightHook(object):
    def __init__(self, target_neurons: List[NeuronPosition], atten_idx: int = 1) -> None:
        self.atten_idx = atten_idx
        self.loss = []
        self.target_neurons: Dict[str, Set[Tuple[int, int]]] = {}
        for each_pos in target_neurons:
            if each_pos.layer not in self.target_neurons:
                self.target_neurons[each_pos.layer] = set()
            self.target_neurons[each_pos.layer].add(tuple(each_pos.i))
    
    def get_activation(self, module_name: str):
        def hook(module, fea_in, fea_out):
            # attn_output, attn_weights, past_key_value = fea_out
            attn_weights = fea_out[self.atten_idx]
            batch, num_heads, seq_len, seq_len = attn_weights.shape
            # attn_weights: [batch, num_head, seq, seq]
            if module_name in self.target_neurons:
                for neuron_id_tuple in self.target_neurons[module_name]:
                    uncovered_loss = torch.mean(attn_weights[..., neuron_id_tuple[1], neuron_id_tuple[0], :])
                    self.loss.append(uncovered_loss)
        return hook

class UncoveredPairAttenWeightHook(object):
    def __init__(self, target_neurons: List[NeuronPosition], atten_idx: int = 1) -> None:
        self.atten_idx = atten_idx
        self.loss = []
        self.target_neurons: Dict[str, Set[Tuple[int, int]]] = {}
        for each_pos in target_neurons:
            if each_pos.layer not in self.target_neurons:
                self.target_neurons[each_pos.layer] = set()
            self.target_neurons[each_pos.layer].add(tuple(each_pos.i))
    
    def get_activation(self, module_name: str):
        def hook(module, fea_in, fea_out):
            # attn_output, attn_weights, past_key_value = fea_out
            attn_weights = fea_out[self.atten_idx]
            batch, num_heads, seq_len, seq_len = attn_weights.shape
            # attn_weights: [batch, num_head, seq, seq]
            if module_name in self.target_neurons:
                for neuron_id_tuple in self.target_neurons[module_name]:
                    uncovered_loss = torch.mean(attn_weights[..., neuron_id_tuple[0], neuron_id_tuple[1]])
                    self.loss.append(uncovered_loss)
        return hook

def random_sample_neurons(uncover_neurons: NeuronCoverage, k=1) -> List[NeuronPosition]:
    target_neurons: List[NeuronPosition] = []
    # Random get k neurons
    total_num = 0
    layers_nonzero = {}
    for layer_name, layer_value in uncover_neurons.items():
        # total_num += torch.count_nonzero(layer_value).item()
        true_indices = torch.nonzero(layer_value, as_tuple=True)
        first_item = true_indices[0]

        total_num += len(first_item)
        layers_nonzero[layer_name] = true_indices
    
    random_neurons = random.sample(list(range(total_num)), k=min(k, total_num))
    random_neurons = sorted(random_neurons)

    random_neuron_i = 0
    neuron_i = 0
    for layer_name, layer_value in uncover_neurons.items():
        # Get indices where the tensor is True
        # true_indices = torch.nonzero(layer_value, as_tuple=True)
        true_indices = layers_nonzero[layer_name]
        first_item = true_indices[0]
        if random_neuron_i >= len(random_neurons):
            break
        if neuron_i + len(first_item) <= random_neurons[random_neuron_i]:
            neuron_i += len(first_item)
        else:
            for index in range(len(first_item)):
                if neuron_i == random_neurons[random_neuron_i]:
                    if len(true_indices) == 1:
                        target_index = true_indices[0][index].item()
                    else:
                        target_index = [tuple_item[index].item() for tuple_item in true_indices]
                    target_neurons.append(NeuronPosition(layer_name, target_index))
                    random_neuron_i += 1
                
                if random_neuron_i >= len(random_neurons):
                    break
                neuron_i += 1
        if random_neuron_i >= len(random_neurons):
            break
    return target_neurons


def sample_all_neurons(uncover_neurons: NeuronCoverage, k=1) -> List[NeuronPosition]:
    target_neurons: List[NeuronPosition] = []
    # Get all neurons

    neuron_i = 0
    for layer_name, layer_value in uncover_neurons.items():
        # Get indices where the tensor is True
        true_indices = torch.nonzero(torch.ones_like(layer_value), as_tuple=True)
        first_item = true_indices[0]

        for index in range(len(first_item)):
            if len(true_indices) == 1:
                target_index = true_indices[0][index].item()
            else:
                target_index = [tuple_item[index].item() for tuple_item in true_indices]
            target_neurons.append(NeuronPosition(layer_name, target_index))
            neuron_i += 1
    return target_neurons

def get_cover_grad(
        model_wrapper: PyTorchModelWrapper,
        text_input: str,
        cover_neurons: NeuronCoverage,
        module_name: str,
        use_re: bool,
        cov_kind: CovKind=CovKind.Mlp,
        alpha: float=0.5,
        samplek: int=1):
    """Get gradient of loss with respect to input tokens.

    Args:
        text_input (str): input string
    Returns:
        Dict of ids, tokens, and gradient as numpy array.
    """
    if isinstance(model_wrapper.model, textattack.models.helpers.T5ForTextToText):
        raise NotImplementedError(
            "`get_grads` for T5FotTextToText has not been implemented yet."
        )

    model_wrapper.model.train()
    embedding_layer = model_wrapper.model.get_input_embeddings()
    original_state = embedding_layer.weight.requires_grad
    embedding_layer.weight.requires_grad = True

    # token num
    input_dict = model_wrapper.tokenizer(
        [text_input],
        add_special_tokens=True,
        return_tensors="pt",
        padding=model_wrapper.padding,
        truncation=True,
    )
    token_num = input_dict["input_ids"].numel()

    # embed_grads hook
    emb_grads = []

    def grad_hook(module, grad_in, grad_out):
        emb_grads.append(grad_out[0])

    emb_hook = embedding_layer.register_backward_hook(grad_hook)
    
    uncover_neurons = cover_neurons.get_logical_not()
    if cov_kind == CovKind.AttenWeights or cov_kind == CovKind.AttenOutput:
        # clip by seq len
        uncover_neurons = NeuronCoverage.from_dict({k:v[:token_num,:] for k, v in uncover_neurons.items()})
    elif cov_kind == CovKind.PairAttenWeights or cov_kind == CovKind.PairAttenOutput:
        # clip by seq len
        uncover_neurons = NeuronCoverage.from_dict({k:v[:token_num,:token_num] for k, v in uncover_neurons.items()})
    if samplek <= 0:
        target_uncover_neurons = sample_all_neurons(uncover_neurons)
    else:
        target_uncover_neurons = random_sample_neurons(uncover_neurons, k=samplek)

    # uncover hook
    atten_idx = 1
    atten_out_idx = 0
    if model_wrapper.model.config.model_type == "falcon":
        atten_idx = 2
        atten_out_idx = 0
    
    if cov_kind == CovKind.Mlp:
        uncover_hook = UncoveredMlpHook(target_uncover_neurons)
    elif cov_kind == CovKind.AttenWeights:
        uncover_hook = UncoveredPosAttenWeightHook(target_uncover_neurons, atten_idx=atten_idx)
    elif cov_kind == CovKind.AttenOutput:
        uncover_hook = UncoveredPosAttenWeightHook(target_uncover_neurons, atten_idx=atten_out_idx)
    elif cov_kind == CovKind.PairAttenWeights:
        uncover_hook = UncoveredPairAttenWeightHook(target_uncover_neurons, atten_idx=atten_idx)
    elif cov_kind == CovKind.PairAttenOutput:
        uncover_hook = UncoveredPairAttenWeightHook(target_uncover_neurons, atten_idx=atten_out_idx)
    uncover_hooks = []
    replace_name = []
    for name, module in model_wrapper.model.named_modules():
        if match_module(name, [module_name], use_re):
            replace_name.append(name)
            # break
    for name in replace_name:
        module = recursive_getattr(model_wrapper.model, name)
        hook_handle = module.register_forward_hook(hook=uncover_hook.get_activation(name))
        uncover_hooks.append(hook_handle)

    # Forward
    model_wrapper.model.zero_grad()
    model_device = next(model_wrapper.model.parameters()).device
    input_dict = model_wrapper.tokenizer(
        [text_input],
        add_special_tokens=True,
        return_tensors="pt",
        padding=model_wrapper.padding,
        truncation=True,
    )
    input_dict.to(model_device)
    predictions = model_wrapper.model(**input_dict, output_attentions=True).logits

    # sum coverage loss
    uncov_loss = None
    for item in uncover_hook.loss:
        if uncov_loss is None:
            uncov_loss = item
        else:
            uncov_loss += item.to(uncov_loss.device)
    
    # remove coverage hook
    for hook in uncover_hooks:
        hook.remove()

    # Get Next Token Loss
    try:
        # predictions: [batch, seq, vocab]
        labels = predictions.argmax(dim=2)
        # lm_output = model_wrapper.model(**input_dict, output_attentions=True)# [0]
        last_logits = predictions[:, -1, :]
        single_labels = labels[:, -1]

        next_token_loss = model_wrapper.loss_fn(last_logits, single_labels)

        if uncov_loss is None:
            loss = next_token_loss
        else:
            loss = alpha * next_token_loss + (1.0 - alpha) * uncov_loss.to(next_token_loss.device)

    except TypeError:
        raise TypeError(
            f"{type(model_wrapper.model)} class does not take in `labels` to calculate loss. "
            "One cause for this might be if you instantiatedyour model using `transformer.AutoModel` "
            "(instead of `transformers.AutoModelForSequenceClassification`)."
        )

    loss.backward()

    # grad w.r.t to word embeddings
    grad = emb_grads[0][0].detach()# .cpu().numpy()

    embedding_layer.weight.requires_grad = original_state
    emb_hook.remove()

    model_wrapper.model.eval()

    output = {"ids": input_dict["input_ids"].squeeze().detach(), "gradient": grad}

    return output

def get_grad(model_wrapper: PyTorchModelWrapper, text_input: str):
    """Get gradient of loss with respect to input tokens.

    Args:
        text_input (str): input string
    Returns:
        Dict of ids, tokens, and gradient as numpy array.
    """
    if isinstance(model_wrapper.model, textattack.models.helpers.T5ForTextToText):
        raise NotImplementedError(
            "`get_grads` for T5FotTextToText has not been implemented yet."
        )

    model_wrapper.model.train()
    embedding_layer = model_wrapper.model.get_input_embeddings()
    original_state = embedding_layer.weight.requires_grad
    embedding_layer.weight.requires_grad = True

    emb_grads = []

    def grad_hook(module, grad_in, grad_out):
        emb_grads.append(grad_out[0])

    emb_hook = embedding_layer.register_backward_hook(grad_hook)

    model_wrapper.model.zero_grad()
    model_device = next(model_wrapper.model.parameters()).device
    input_dict = model_wrapper.tokenizer(
        [text_input],
        add_special_tokens=True,
        return_tensors="pt",
        padding=model_wrapper.padding,
        truncation=True,
    )
    input_dict.to(model_device)
    predictions = model_wrapper.model(**input_dict).logits

    try:
        # predictions: [batch, seq, vocab]
        labels = predictions.argmax(dim=2)
        # lm_output = self.model(**input_dict)# [0]
        last_logits = predictions[:, -1, :]
        single_labels = labels[:, -1]
        loss = model_wrapper.loss_fn(last_logits, single_labels)

    except TypeError:
        raise TypeError(
            f"{type(model_wrapper.model)} class does not take in `labels` to calculate loss. "
            "One cause for this might be if you instantiatedyour model using `transformer.AutoModel` "
            "(instead of `transformers.AutoModelForSequenceClassification`)."
        )

    loss.backward()

    # grad w.r.t to word embeddings
    grad = emb_grads[0][0].detach()

    embedding_layer.weight.requires_grad = original_state
    emb_hook.remove()
    model_wrapper.model.eval()

    output = {"ids": input_dict["input_ids"].squeeze().detach(), "gradient": grad}

    return output
