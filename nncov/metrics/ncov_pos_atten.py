

from typing import Tuple, Union
import torch
from nncov.coverage.coverage_utils import scale_batch, scale_batch_clean, scale_batch_dim
from nncov.coverage.coverage_data import NeuronCoverage
from nncov.metrics import CovKind, NeuralMetric

class NCovPosAtten(NeuralMetric):
    def __init__(self, t: float, atten_idx: int=1, max_seq_len: int = 2048, device="cpu") -> None:
        super().__init__(device=device)
        self.kind = CovKind.AttenWeights
        
        self.t = t
        self.atten_idx = atten_idx
        self.max_seq_len = max_seq_len
        self.state = {}
        self.samples = 0

    def update(self, layer_name: str, layer: Union[torch.Tensor, Tuple[torch.Tensor]]):
        # print(name, layer.data)
        # attn_output, attn_weights, past_key_value = layer
        assert isinstance(layer, list) or isinstance(layer, tuple)
        attn_weights = layer[self.atten_idx]

        batch_size = attn_weights.shape[0]
        num_heads = attn_weights.shape[1]
        sequence_length = attn_weights.shape[2]
        # [batch, num_heads, seq, seq]
        
        # Update Coverage States
        if layer_name not in self.state:
            self.state[layer_name] = torch.zeros(self.max_seq_len, num_heads, dtype=torch.bool, device=self.device)
        
        # MinMax Scale along the Batch Dimension
        scaled_attn_weights = scale_batch(attn_weights, rmax=1.0, rmin=0.0)
        # scaled_attn_weights = attn_weights

        # shape: [batch, num_heads, seq, seq]
        diff = (scaled_attn_weights > self.t)
        diff_sum = torch.any(torch.any(diff, 3), 0).to(self.device)
        # [batch, num_heads, seq, seq] -> [num_heads, seq]
        self.state[layer_name][:sequence_length, ...] |= torch.permute(diff_sum, (1, 0))

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
