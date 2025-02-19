

from typing import Dict, Tuple, Union
import torch
from nncov.coverage.coverage_data import NeuronCoverage
from nncov.metrics import CovKind, NeuralMetric
from nncov.monitors.base_monitor import LayerMonitor
from nncov.monitors.reduce_ops import compare_size, pad_to_same_size, shrink_to_same_size

class KMNCovPosAtten(NeuralMetric):
    def __init__(self, layers: Dict[str, LayerMonitor], k: int, max_seq_len: int = 2048, device="cpu") -> None:
        super().__init__(device=device)
        self.kind = CovKind.AttenWeights
        
        self.layers = layers
        self.k = k
        self.max_seq_len = max_seq_len

        self.state = {}
        self.samples = 0
    
    #@profile
    def update(self, layer_name: str, layer: Union[torch.Tensor, Tuple[torch.Tensor]]):
        attn_output, attn_weights, past_key_value = layer
        batch_size = attn_weights.shape[0]
        num_heads = attn_weights.shape[1]
        sequence_length = attn_weights.shape[2]

        layer_device = attn_weights.device

        if layer_name not in self.state:
            self.state[layer_name] = torch.zeros(self.max_seq_len, num_heads, self.k, dtype=torch.int8, device=layer_device)

        if layer_name not in self.layers:
            print(f"Warning: {layer_name} is not a valid layer name. Please make sure you are providing the correct layer name.")
            return
        
        # Get layer max and min
        max_tensor_data, min_tensor_data = self.layers[layer_name].data
        max_attn_output, max_tensor, max_past_key_value = max_tensor_data
        min_attn_output, min_tensor, min_past_key_value = min_tensor_data
        # max_tensor: [batch, head, seq, seq], min_tensor: [batch, head, seq, seq]

        # Move to layer device
        if max_tensor.device != layer_device or min_tensor.device != layer_device:
            if max_tensor.device != layer_device:
                max_tensor = max_tensor.to(layer_device)
            if min_tensor.device != layer_device:
                min_tensor = min_tensor.to(layer_device)
            self.layers[layer_name].data = (
                (max_attn_output, max_tensor, max_past_key_value),
                (min_attn_output, min_tensor, min_past_key_value)
            )

        # Padding
        c_status = compare_size(attn_weights, min_tensor)
        # layer is shorter seq
        if c_status == "lt":
            min_tensor_padded = shrink_to_same_size(min_tensor, attn_weights)
            max_tensor_padded = shrink_to_same_size(max_tensor, attn_weights)
        elif c_status == "gt":
            min_tensor_padded = min_tensor
            max_tensor_padded = max_tensor
            attn_weights = pad_to_same_size(attn_weights, min_tensor, 0)
            print("Warning: the sequence length of layer is longer than any training text.")
        else:
            min_tensor_padded = min_tensor
            max_tensor_padded = max_tensor
        
        # Need higher precision (Maybe)
        min_tensor_padded = min_tensor_padded.to(layer_device) # .to(dtype=torch.float32)
        max_tensor_padded = max_tensor_padded.to(layer_device) # .to(dtype=torch.float32)
        layer_fp = attn_weights # .to(dtype=torch.float32)

        # Compute Unit Tensor
        inv_unit_tensor = (max_tensor_padded - min_tensor_padded) * (1.0 / self.k)
        # Compite section ID
        section_tensor = torch.floor((layer_fp - min_tensor_padded) * inv_unit_tensor).to(dtype=torch.int)
        
        # shape: [batch, seq, embed]
        
        # 将输入矩阵扩展为 [batch, head, seq_len, seq_len, 1] 形状的张量
        expanded_section_tensor = section_tensor.unsqueeze(4)
        # Create mask for invalid values
        mask = (expanded_section_tensor < 0) | (expanded_section_tensor >= self.k)
        # Set invalid values to 0
        mask_full_value = torch.full_like(expanded_section_tensor, fill_value=self.k, device=layer_device, dtype=torch.int)
        mask_section_tensor = torch.where(mask, mask_full_value, expanded_section_tensor).to(dtype=torch.long)

        # 使用 `torch.nn.functional.one_hot()` 将扩展的输入矩阵转换为 one-hot 编码
        one_section_tensor = torch.nn.functional.one_hot(mask_section_tensor, num_classes=self.k+1)

        # 调整张量的形状为 [batch, seq_len, k + 1]
        one_hot_matrix = one_section_tensor.view(batch_size, num_heads, sequence_length, sequence_length, self.k + 1)[..., :self.k]
        one_hot_matrix = torch.sum(one_hot_matrix, dim=3)

        k_counter = torch.permute(torch.any(one_hot_matrix, dim=0), (1, 0, 2))
        k_counter = k_counter.to(self.state[layer_name].device)
        self.state[layer_name][:sequence_length, :, :].logical_or_(k_counter)
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
        # [seq, head, k] -> [seq, head]
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
