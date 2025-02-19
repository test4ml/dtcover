

from typing import Tuple, Union
import torch
from nncov.coverage.coverage_data import NeuronCoverage
from nncov.metrics import CovKind, NeuralMetric

class TKNCovMlp(NeuralMetric):
    def __init__(self, t: float, topk: int, max_seq_len: int = 2048, device="cpu") -> None:
        super().__init__(device=device)
        self.kind = CovKind.Mlp
        self.t = t
        self.topk = topk
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
        
        # shape: [batch, seq, embed]
        # Find the indices of topk elements along the last dimension
        topk_values, topk_indices = torch.topk(layer, k=self.topk, dim=-1)

        # Create a mask tensor with the same shape as 'x', initialized with zeros.
        mask = torch.zeros_like(layer, dtype=torch.bool)
        true_mask = torch.ones_like(topk_values, dtype=torch.bool)

        # Set the top-k values to their original values in 'x'.
        mask.scatter_(dim=-1, index=topk_indices, src=true_mask)

        self.state[layer_name] |= torch.any(torch.any(mask, 1), 0).to(self.device)

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

