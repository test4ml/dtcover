

from typing import Tuple, Union
import torch
from nncov.coverage.coverage_data import NeuronCoverage
from nncov.metrics import CovKind, NeuralMetric

class TKNCovPosAtten(NeuralMetric):
    def __init__(self, t: float, topk: int, max_seq_len: int = 2048, device="cpu") -> None:
        super().__init__(device=device)
        self.kind = CovKind.AttenWeights
        
        self.t = t
        self.topk = topk
        self.max_seq_len = max_seq_len

        self.state = {}
        self.samples = 0

    def update(self, layer_name: str, layer: Union[torch.Tensor, Tuple[torch.Tensor]]):
        # print(name, layer.data)
        attn_output, attn_weights, past_key_value = layer
        batch_size = attn_weights.shape[0]
        num_heads = attn_weights.shape[1]
        sequence_length = attn_weights.shape[2]
        # [batch, num_heads, seq, seq]

        # Update Coverage States
        if layer_name not in self.state:
            self.state[layer_name] = torch.zeros(self.max_seq_len, num_heads, dtype=torch.bool, device=self.device)
        
        # Find the indices of topk elements along the last dimension
        topk_values, topk_indices = torch.topk(layer, k=self.topk, dim=-1)

        # Create a mask tensor with the same shape as 'x', initialized with zeros.
        mask = torch.zeros_like(layer, dtype=torch.bool)
        true_mask = torch.ones_like(topk_values, dtype=torch.bool)

        # Set the top-k values to their original values in 'x'.
        mask.scatter_(dim=-1, index=topk_indices, src=true_mask)

        diff = mask
        diff_sum = torch.any(torch.any(diff, 2), 0).to(self.device)
        # [batch, num_heads, seq, seq] -> [num_heads, seq]
        self.state[layer_name][:sequence_length, ...] |= torch.permute(diff_sum, (1, 0))

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
