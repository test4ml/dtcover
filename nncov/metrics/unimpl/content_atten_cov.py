

from typing import Tuple, Union
import torch
from nncov.metrics import NeuralMetric

class ContentAttenCov(NeuralMetric):
    def __init__(self, t: float) -> None:
        super().__init__()
        self.t = t
        self.state = {}
        self.samples = 0

    def update(self, layer_name: str, layer: Union[torch.Tensor, Tuple[torch.Tensor]]):
        # print(name, layer.data)
        attn_output, attn_weights, past_key_value = layer
        batch_size = attn_weights.shape[0]

        if layer_name not in self.state:
            self.state[layer_name] = torch.zeros(4096, device=self.device, dtype=torch.long)
        
        acti_indices = (attn_weights > self.t).nonzero(as_tuple=True)
        indices_batch, indices_head, indices_x, indices_y = acti_indices[0], acti_indices[1], acti_indices[2], acti_indices[3]

        indices_x_y = torch.cat([indices_x, indices_y], dim=0)
        bincount = torch.bincount(indices_x_y)
        self.state[layer_name][:len(bincount)] += bincount.to(self.device)
        
        self.samples += batch_size

        # print ((t == 2).nonzero(as_tuple=True)[0])

    def compute(self):
        out = {}
        for name, value in self.state.items():
            out[name] = 0
            for item in value:
                if item != 0:
                    out[name] += 1
        return out
            
    
    def clear(self):
        self.state = {}
        self.samples = 0