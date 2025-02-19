

from typing import Dict, Tuple, Union
import torch
from nncov.coverage.coverage_data import NeuronCoverage
from nncov.metrics import CovKind, NeuralMetric
from nncov.monitors.base_monitor import LayerMonitor
from nncov.monitors.reduce_ops import compare_size, pad_to_same_size, shrink_to_same_size

class KMNCovMlp(NeuralMetric):
    def __init__(self, layers: Dict[str, LayerMonitor], k: int, max_seq_len: int = 2048, device="cpu") -> None:
        super().__init__(device=device)
        self.kind = CovKind.Mlp

        self.layers = layers
        self.k = k
        self.max_seq_len = max_seq_len

        self.state = {}
        self.samples = 0
    
    def update(self, layer_name: str, layer: Union[torch.Tensor, Tuple[torch.Tensor]]):
        # print(name, layer.data)
        batch_size = layer.shape[0]
        sequence_length = layer.shape[1]
        embedding_size = layer.shape[2]

        if layer_name not in self.state:
            self.state[layer_name] = torch.zeros(embedding_size, self.k, dtype=torch.bool, device=layer.device)

        if layer_name not in self.layers:
            print(f"Warning: {layer_name} is not a valid layer name. Please make sure you are providing the correct layer name.")
            return
        
        # Get layer max and min
        max_tensor, min_tensor = self.layers[layer_name].data

        # Move to layer device
        if max_tensor.device != layer.device:
            max_tensor = max_tensor.to(layer.device)
        if min_tensor.device != layer.device:
            min_tensor = min_tensor.to(layer.device)
        self.layers[layer_name].data = (max_tensor, min_tensor)

        # Padding
        c_status = compare_size(layer, min_tensor)
        # layer is shorter seq
        if c_status == "lt":
            min_tensor_padded = shrink_to_same_size(min_tensor, layer)
            max_tensor_padded = shrink_to_same_size(max_tensor, layer)
        elif c_status == "gt":
            min_tensor_padded = min_tensor
            max_tensor_padded = max_tensor
            layer = pad_to_same_size(layer, min_tensor, 0)
            print("Warning: the sequence length of layer is longer than any training text.")
        else:
            min_tensor_padded = min_tensor
            max_tensor_padded = max_tensor
        
        # Need higher precision (Maybe)
        min_tensor_padded = min_tensor_padded.to(layer.device) # .to(dtype=torch.float32)
        max_tensor_padded = max_tensor_padded.to(layer.device) # .to(dtype=torch.float32)
        layer_fp = layer # .to(dtype=torch.float32)

        # Compute Unit Tensor
        inv_unit_tensor = (max_tensor_padded - min_tensor_padded) * (1.0 / self.k)
        # Compite section ID
        section_tensor = torch.floor((layer_fp - min_tensor_padded) * inv_unit_tensor).to(dtype=torch.int)
        
        # shape: [batch, seq, embed]
        
        # 将输入矩阵扩展为 [batch, seq_len, embed, 1] 形状的张量
        expanded_section_tensor = section_tensor.unsqueeze(3)
        # Create mask for invalid values
        mask = (expanded_section_tensor < 0) | (expanded_section_tensor >= self.k)
        # Set invalid values to 0
        mask_full_value = torch.full_like(expanded_section_tensor, fill_value=self.k, device=layer.device, dtype=torch.int)
        mask_section_tensor = torch.where(mask, mask_full_value, expanded_section_tensor).to(dtype=torch.long)

        # 使用 `torch.nn.functional.one_hot()` 将扩展的输入矩阵转换为 one-hot 编码
        one_section_tensor = torch.nn.functional.one_hot(mask_section_tensor, num_classes=self.k+1)

        # 调整张量的形状为 [batch, seq_len, embed, k + 1]
        one_hot_matrix = one_section_tensor.view(batch_size, sequence_length, embedding_size, self.k + 1)[..., :self.k]

        k_counter = torch.any(torch.any(one_hot_matrix, dim=1), dim=0)
        k_counter = k_counter.to(self.state[layer_name].device)
        self.state[layer_name][:, :].logical_or_(k_counter)
        self.samples += batch_size

    def compute(self):
        cover = 0
        total = 0
        for l, v in self.state.items():
            cover += torch.count_nonzero(v).item()
            # cover += torch.sum(torch.sum(v, dim=-1), dtype=torch.int).item()
            total += v.numel()
        return cover / total

    def covered(self):
        # [embed, k] -> [embed]
        covered_state = {}
        for l, v in self.state.items():
            covered_state[l] = torch.all(v, dim=-1)
        return NeuronCoverage.from_dict(covered_state)
    
    def reset(self):
        self.state = {}
        self.samples = 0

    def clear(self):
        for layer_name in self.state.keys():
            self.state[layer_name].zero_()
        self.samples = 0
