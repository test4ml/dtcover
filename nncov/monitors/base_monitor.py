from typing import Dict, List, Optional, Union

import torch
import pickle

from nncov.utils import transfer_device

class LayerMonitor(object):
    def __init__(self, device: str = "cpu") -> None:
        self.name: Optional[str] = None
        self.data: Optional[Union[torch.Tensor, List, Dict]] = None
        self.device = device
    
    def to_(self, device: str):
        self.data = transfer_device(self.data, device)
        self.device = device
    
    def update(self, data: torch.FloatTensor):
        self.data = data
    
    def reduce(self, other_data, reduce_fn):
        self.data = reduce_fn(self.data, other_data)

class NeuralMonitor(object):
    def __init__(self, dtype: Union[str, torch.dtype] = torch.float16, device:str="cpu") -> None:
        self.dtype = dtype
        self.layers: Dict[str, LayerMonitor] = {}
        self.device = device
    
    def to_(self, device: str):
        for layer_name, layer in self.layers.items():
            layer.to_(device)
    
    def update(self, layer_name: str, data: Union[torch.Tensor, List, Dict]):
        if layer_name not in self.layers:
            self.layers[layer_name] = LayerMonitor()
        if isinstance(data, torch.Tensor):
            if len(data.shape) <= 1:
                raise Exception("The shape of output tensor should be (batch, features, ...)")
            self.layers[layer_name].update(data.type(self.dtype))
        else:
            self.layers[layer_name].update(data)
        
    def save(self, path: str):
        with open(path, "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(self, path: str) -> "NeuralMonitor":
        with open(path, "rb") as file:
            out = pickle.load(file)
        return out

