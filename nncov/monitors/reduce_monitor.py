from typing import Dict, List, Optional, Union

import torch
import pickle

from nncov.utils import transfer_device
from nncov.monitors.base_monitor import LayerMonitor, NeuralMonitor


class ReduceMonitor(NeuralMonitor):
    def __init__(self, reduce_fn, dtype: Union[str, torch.dtype] = torch.float16, device:Optional[str]="cpu") -> None:
        super(ReduceMonitor, self).__init__(dtype, device)
        self.reduce_fn = reduce_fn
    
    #@profile
    def update(self, layer_name: str, data: Union[torch.Tensor, List, Dict]):
        if layer_name not in self.layers:
            self.layers[layer_name] = LayerMonitor()
        if isinstance(data, torch.Tensor):
            if len(data.shape) <= 1:
                raise Exception("The shape of output tensor should be (batch, features, ...)")
            # d = data.type(self.dtype).to(self.device)
            if self.device is not None:
                d = data.to(self.device)
            else:
                d = data
            self.layers[layer_name].reduce(d, self.reduce_fn)
        else:
            if self.device is not None:
                d = transfer_device(data, self.device)
            else:
                d = data
            self.layers[layer_name].reduce(d, self.reduce_fn)
    
    #@profile
    def reduce(self, other: "ReduceMonitor"):
        for layer_name, layer_data in other.layers.items():
            if layer_name not in self.layers:
                continue
            self.layers[layer_name].reduce(layer_data, self.reduce_fn)