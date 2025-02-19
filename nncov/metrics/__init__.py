
from torch import Tensor
from typing import Callable, Optional, Tuple, Union

import torch
from torch import nn

from nncov.coverage.coverage_data import NeuronCoverage

class NeuralMetric(object):
    def __init__(self, device="cpu") -> None:
        super().__init__()
        self.device = device
    
    def to(self, device: str):
        self.device = device
        return self

    def update(self, layer_name: str, layer: Union[torch.Tensor, Tuple[torch.Tensor]]):
        raise NotImplementedError()
    
    def compute(self) -> float:
        raise NotImplementedError()

    def covered(self) -> NeuronCoverage:
        raise NotImplementedError()

    def clear(self):
        raise NotImplementedError()
    
    def reset(self):
        raise NotImplementedError()
    

from enum import Enum, auto
class CovKind(Enum):
    Mlp = auto()
    AttenWeights = auto()
    AttenOutput = auto()
    PairAttenWeights = auto()
    PairAttenOutput = auto()

class CovType(Enum):
    NCovPosAtten = auto()
    NCovAbsAtten = auto()
    NCovRelAtten = auto()
    NCovMlp = auto()
    KMNCovMlp = auto()
    KMNCovPosAtten = auto()
    NBCovMlp = auto()
    NBCovPosAtten = auto()
    SNACovMlp = auto()
    SNACovPosAtten = auto()