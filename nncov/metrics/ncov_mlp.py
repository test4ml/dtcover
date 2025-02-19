

from typing import Tuple, Union
import torch
from nncov.coverage.coverage_utils import scale_batch, scale_batch_clean, scale_batch_dim
from nncov.coverage.coverage_data import NeuronCoverage
from nncov.metrics import CovKind, NeuralMetric

class NCovMlp(NeuralMetric):
    def __init__(self, t: float, max_seq_len: int = 2048, device="cpu") -> None:
        super().__init__(device=device)
        self.kind = CovKind.Mlp
        
        self.t = t
        self.max_seq_len = max_seq_len

        self.state = {}
        self.samples = 0

    def update(self, layer_name: str, layer: Union[torch.Tensor, Tuple[torch.Tensor]]):
        # print(layer_name, layer.data.shape)
        # ([1, 25, 4096])
        batch_size = layer.shape[0]
        sequence_length = layer.shape[1]
        embedding_size = layer.shape[2]

        # Update Coverage States
        if layer_name not in self.state:
            self.state[layer_name] = torch.zeros(embedding_size, dtype=torch.bool, device=self.device)

        # MinMax Scale along the Batch Dimension
        scaled_layer = scale_batch(layer, rmax=1.0, rmin=0.0)
        # scaled_layer = layer

        # shape: [batch, seq, embed]
        diff = (scaled_layer > self.t)
        self.state[layer_name] |= torch.any(torch.any(diff, 1), 0).to(self.device)

        self.samples += batch_size

    def compute(self):
        cover = 0
        total = 0
        for l, v in self.state.items():
            cover += torch.count_nonzero(v)
            total += v.numel()
        return (cover / total).item()

    def covered(self):
        return NeuronCoverage.from_dict(self.state)
    
    def reset(self):
        self.state = {}
        self.samples = 0

    def clear(self):
        for layer_name in self.state.keys():
            self.state[layer_name].zero_()
        self.samples = 0

