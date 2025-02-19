

from typing import Dict, Union, List
import torch

class NeuronCoverage(object):
    def __init__(self) -> None:
        self.states: Dict[str, torch.BoolTensor] = {}
    
    @classmethod
    def from_dict(cls, obj: Dict[str, torch.BoolTensor], clone: bool=True) -> "NeuronCoverage":
        # check typing
        if not isinstance(obj, dict):
            raise TypeError("Input object must be a dictionary.")
        for key, value in obj.items():
            if not isinstance(key, str):
                raise TypeError("Keys in the dictionary must be strings.")
            
            if not isinstance(value, torch.Tensor):
                raise TypeError("Values in the dictionary must be torch.BoolTensor objects.")

        # Create a new instance of NeuronCoverage
        coverage = NeuronCoverage()
        if clone:
            coverage.states = {k: v.detach().clone() for k, v in obj.items()}
        else:
            coverage.states = {k: v.detach() for k, v in obj.items()}
        return coverage
                    
    def update_coverage(self, other: "NeuronCoverage"):
        for layer_name, layer_value in other.states.items():
            if layer_name in self.states:
                # Update the coverage for each neuron in the layer
                self.states[layer_name].logical_or_(layer_value)
            else:
                # If the layer is not present in the self NeuronCoverage object,
                # init the coverage
                self.states[layer_name] = layer_value.detach().clone()
    
    def update_uncoverage(self, other: "NeuronCoverage"):
        for layer_name, layer_value in other.states.items():
            if layer_name in self.states:
                # Update the coverage for each neuron in the layer
                self.states[layer_name].logical_and_(layer_value)
            else:
                # If the layer is not present in the self NeuronCoverage object,
                # init the coverage
                self.states[layer_name] = layer_value.detach().clone()
    def get_logical_and(self, other: "NeuronCoverage"):
        # Create a new instance of NeuronCoverage
        coverage = NeuronCoverage()
        coverage.states = {k: torch.logical_and(self.states[k], v) for k,v in other.states.items()}

        return coverage
    
    def get_logical_or(self, other: "NeuronCoverage"):
        # Create a new instance of NeuronCoverage
        coverage = NeuronCoverage()
        coverage.states = {k: torch.logical_or(self.states[k], v) for k,v in other.states.items()}

        return coverage
    
    def get_logical_xor(self, other: "NeuronCoverage"):
        # Create a new instance of NeuronCoverage
        coverage = NeuronCoverage()
        coverage.states = {k: torch.logical_xor(self.states[k], v) for k,v in other.states.items()}

        return coverage

    def get_logical_not(self):
        # Create a new instance of NeuronCoverage
        coverage = NeuronCoverage()
        coverage.states = {k: torch.logical_not(v) for k,v in self.states.items()}

        return coverage
    
    def clone(self) -> "NeuronCoverage":
        # Create a new instance of NeuronCoverage
        coverage = NeuronCoverage()
        coverage.states = {k: v.clone() for k, v in self.states.items()}

        return coverage

    def to_dict(self) -> Dict[str, torch.BoolTensor]:
        return self.states

    def __repr__(self) -> str:
        return f"NeuronCoverage(states={len(self.states)})"

    def __getitem__(self, key: str) -> torch.BoolTensor:
        if key not in self.states:
            raise KeyError(f"Layer '{key}' is not present in the NeuronCoverage object.")
        return self.states[key]
    
    def items(self) -> Dict[str, torch.BoolTensor]:
        return self.states.items()

class NeuronPosition(object):
    def __init__(self, layer: int, i: Union[int, List[int]]) -> None:
        self.layer = layer
        self.i = i
