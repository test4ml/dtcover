

import torch
from nncov.metrics import NeuralMetric

class NCov(NeuralMetric):
    def __init__(self, t: float) -> None:
        super().__init__()
        self.t = t
        self.state = NeuralMask()

    def update(self, snapshot: NeuralSnapshot):
        for name, layer in snapshot.layers.items():
            activation = (layer.data > self.t)
            self.state.logical_or(name, activation)

    def compute(self):
        return self.state.positive() / self.state.numel()
    
    def clear(self):
        self.state = NeuralMask()