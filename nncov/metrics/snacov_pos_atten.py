

from typing import Dict, Tuple, Union
import torch
from nncov.coverage.coverage_data import NeuronCoverage
from nncov.metrics import CovKind, NeuralMetric
from nncov.monitors.base_monitor import LayerMonitor
from nncov.monitors.reduce_ops import compare_size, pad_to_same_size, shrink_to_same_size

class SNACovPosAtten(NeuralMetric):
    def __init__(self, layers: Dict[str, LayerMonitor], max_seq_len: int = 2048, device="cpu") -> None:
        super().__init__(device=device)
        self.kind = CovKind.AttenWeights
        
        self.layers = layers
        self.max_seq_len = max_seq_len

        self.state = {}
        self.samples = 0

    def update(self, layer_name: str, layer: Union[torch.Tensor, Tuple[torch.Tensor]]):
        # print(name, layer.data)
        attn_output, attn_weights, past_key_value = layer
        batch_size = attn_weights.shape[0]
        num_heads = attn_weights.shape[1]
        sequence_length = attn_weights.shape[2]

        layer_device = attn_weights.device

        if layer_name not in self.state:
            self.state[layer_name] = torch.zeros(self.max_seq_len, num_heads, dtype=torch.bool, device=self.device)

        if layer_name not in self.layers:
            print(f"Warning: {layer_name} is not a valid layer name. Please make sure you are providing the correct layer name.")
            return
        
        # Get layer max and min
        max_tensor_data, min_tensor_data = self.layers[layer_name].data
        max_attn_output, max_tensor, max_past_key_value = max_tensor_data
        min_attn_output, min_tensor, min_past_key_value = min_tensor_data

        # Move to layer device
        if max_tensor.device != layer_device or min_tensor.device != layer_device:
            if max_tensor.device != layer_device:
                max_tensor = max_tensor.to(layer_device)
            if min_tensor.device != layer_device:
                min_tensor = min_tensor.to(layer_device)
            self.layers[layer_name].data = (
                (max_attn_output, max_tensor, max_past_key_value),
                (min_attn_output, min_tensor, min_past_key_value)
            )

        # Padding
        c_status = compare_size(attn_weights, min_tensor)
        # layer is shorter seq
        if c_status == "lt":
            min_tensor_padded = shrink_to_same_size(min_tensor, attn_weights)
            max_tensor_padded = shrink_to_same_size(max_tensor, attn_weights)
        elif c_status == "gt":
            min_tensor_padded = min_tensor
            max_tensor_padded = max_tensor
            attn_weights = pad_to_same_size(attn_weights, min_tensor, 0)
            print("Warning: the sequence length of layer is longer than any training text.")
        else:
            min_tensor_padded = min_tensor
            max_tensor_padded = max_tensor
        
        # Need higher precision (Maybe)
        min_tensor_padded = min_tensor_padded.to(layer_device) # .to(dtype=torch.float32)
        max_tensor_padded = max_tensor_padded.to(layer_device) # .to(dtype=torch.float32)

        activation_tensor = attn_weights < max_tensor_padded
        activation_tensor = torch.any(activation_tensor, dim=3)
        activation_tensor = torch.any(activation_tensor, dim=0)

        self.state[layer_name][:sequence_length, ...] |= torch.permute(activation_tensor, (1, 0)).to(self.device)
        self.samples += batch_size

    def compute(self):
        cover = 0
        total = 0
        for l, v in self.state.items():
            cover += torch.count_nonzero(v)
            total += v.numel()
        return cover / total

    def covered(self):
        return NeuronCoverage.from_dict(self.state)

    def reset(self):
        self.state = {}
        self.samples = 0

    def clear(self):
        for layer_name in self.state.keys():
            self.state[layer_name].zero_()
        self.samples = 0


