from typing import Dict, List, Optional, Union

import torch
import pickle


class NeuralLayerSnapshot(object):
    def __init__(self) -> None:
        self.name: Optional[str] = None
        self.data: Optional[Union[torch.Tensor, List, Dict]] = None
    
    def _transfer_device(self, obj, device: str):
        if isinstance(obj, torch.Tensor):
            return obj.to(device)
        elif isinstance(obj, tuple):
            out = []
            for item in obj:
                if item is None:
                    out.append(None)
                else:
                    out.append(item.to(device))
            return tuple(out)
        elif isinstance(obj, list):
            out = []
            for item in obj:
                if item is None:
                    out.append(None)
                else:
                    out.append(item.to(device))
            return out
        elif isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                if item is None:
                    out[k] = None
                else:
                    out[k] = v.to(device)
            return out
        else:
            raise Exception(f"Unknown data type {type(obj)} and cannot to the device")

    def to_(self, device: str):
        self.data = self._transfer_device(self.data, device)
    
    def update(self, data: torch.FloatTensor):
        self.data = data

class NeuralSnapshot(object):
    def __init__(self, dtype: Union[str, torch.dtype] = torch.float16) -> None:
        self.dtype = dtype
        self.layers: Dict[str, NeuralLayerSnapshot] = {}
    
    def to_(self, device: str):
        for layer_name, layer in self.layers.items():
            layer.to_(device)
    
    def update(self, layer_name: str, data: Union[torch.Tensor, List, Dict]):
        if layer_name not in self.layers:
            self.layers[layer_name] = NeuralLayerSnapshot()
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
    def load(self, path: str) -> "NeuralSnapshot":
        with open(path, "rb") as file:
            out = pickle.load(file)
        return out
