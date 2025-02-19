

from typing import Dict, Tuple, Union
import torch
from nncov.coverage.coverage_data import NeuronCoverage
from nncov.metrics import CovKind, NeuralMetric
from nncov.monitors.base_monitor import LayerMonitor
from nncov.monitors.reduce_ops import compare_size, pad_to_same_size, shrink_to_same_size

class SNACovMlp(NeuralMetric):
    def __init__(self, layers: Dict[str, LayerMonitor], max_seq_len: int = 2048, device="cpu") -> None:
        super().__init__(device=device)
        self.kind = CovKind.Mlp
        self.layers = layers
        self.max_seq_len = max_seq_len

        self.state = {}
        self.samples = 0

    def update(self, layer_name: str, layer: Union[torch.Tensor, Tuple[torch.Tensor]]):
        # print(name, layer.data)
        batch_size = layer.shape[0]
        sequence_length = layer.shape[1]
        embedding_size = layer.shape[2]

        layer_device = layer.device

        if layer_name not in self.state:
            self.state[layer_name] = torch.zeros(embedding_size, dtype=torch.bool, device=self.device)

        if layer_name not in self.layers:
            print(f"Warning: {layer_name} is not a valid layer name. Please make sure you are providing the correct layer name.")
            return
        
        # Get layer max and min
        max_tensor, min_tensor = self.layers[layer_name].data

        # Move to layer device
        if max_tensor.device != layer_device or min_tensor.device != layer_device:
            if max_tensor.device != layer_device:
                max_tensor = max_tensor.to(layer_device)
            if min_tensor.device != layer_device:
                min_tensor = min_tensor.to(layer_device)
            self.layers[layer_name].data = (max_tensor, min_tensor)

        # Padding
        c_status = compare_size(layer, min_tensor)
        # layer is shorter seq
        if c_status == "lt":
            min_tensor_padded = shrink_to_same_size(min_tensor, layer)
            max_tensor_padded = shrink_to_same_size(max_tensor, layer)
        elif c_status == "gt":
            min_tensor_padded = min_tensor
            max_tensor_padded = max_tensor
            layer = pad_to_same_size(layer, min_tensor, 0)
            print("Warning: the sequence length of layer is longer than any training text.")
        else:
            min_tensor_padded = min_tensor
            max_tensor_padded = max_tensor
        
        # Need higher precision (Maybe)
        min_tensor_padded = min_tensor_padded.to(layer_device) # .to(dtype=torch.float32)
        max_tensor_padded = max_tensor_padded.to(layer_device) # .to(dtype=torch.float32)

        activation_tensor = layer < max_tensor_padded

        self.state[layer_name] |= torch.any(torch.any(activation_tensor, dim=1), dim=0).to(self.device)
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

