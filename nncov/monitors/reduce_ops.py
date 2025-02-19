
from typing import Dict, List, Union
import torch
import torch.nn.functional as F
from nncov.utils import clone_tensor, transfer_device
#@profile
def pad_to_same_size(lhs, rhs, value):
    """
    Pad lhs tensor to the size of rhs
    """
    lhs_size = lhs.size()
    rhs_size = rhs.size()
    if lhs_size == rhs_size:
        return lhs  # 如果lhs已经与rhs具有相同的大小，则无需填充
    
    final_size = []
    for i in range(len(lhs_size)):
        final_size.append(max(lhs_size[i], rhs_size[i]))
    
    final_tensor = torch.full(size=final_size, fill_value=value, dtype=lhs.dtype, device=lhs.device)
    select_indices = []
    for i in range(len(lhs_size)):
        select_indices.append(slice(0, lhs_size[i]))
    final_tensor.__setitem__(select_indices, lhs)
    return final_tensor

#@profile
def shrink_to_same_size(lhs, rhs):
    lhs_size = list(lhs.shape)
    rhs_size = list(rhs.shape)
    if lhs_size == rhs_size:
        return lhs  # 如果lhs已经与rhs具有相同的大小，则无需填充
    
    final_size = []
    for i, s in enumerate(lhs_size):
        final_size.append(min(lhs_size[i], rhs_size[i]))
    
    select_indices = []
    for i in range(len(lhs_size)):
        select_indices.append(slice(0, final_size[i]))
    return lhs.__getitem__(select_indices)

#@profile
def compare_size(lhs, rhs) -> str:
    lhs_shape = list(lhs.shape)
    rhs_shape = list(rhs.shape)
    if lhs_shape == rhs_shape:
        return "eq"
    else:
        if len(lhs_shape) == len(rhs_shape):
            for i, s in enumerate(lhs_shape):
                if s < rhs_shape[i]:
                    return "lt"
                elif s > rhs_shape[i]:
                    return "gt"
            return "eq"
        else:
            return "unk"

#@profile
def check_obj_structure(lhs, rhs) -> bool:
    if lhs is None and rhs is None:
        return True
    elif isinstance(lhs, torch.Tensor) and isinstance(rhs, torch.Tensor):
        return True
    elif isinstance(lhs, tuple) and isinstance(rhs, tuple):
        out = []
        for pair_lhs, pair_rhs in zip(lhs, rhs):
            out.append(check_obj_structure(pair_lhs, pair_rhs))
        return all(out)
    elif isinstance(lhs, list) and isinstance(rhs, list):
        out = []
        for pair_lhs, pair_rhs in zip(lhs, rhs):
            out.append(check_obj_structure(pair_lhs, pair_rhs))
        return all(out)
    elif isinstance(lhs, dict) and isinstance(rhs, dict):
        if lhs.keys() != rhs.keys():
            return False
        out = []
        for key in lhs.keys():
            out.append(check_obj_structure(lhs[key], rhs[key]))
        return all(out)
    else:
        return False

def max_fn(lhs, rhs):
    return torch.max(lhs, rhs)

def min_fn(lhs, rhs):
    return torch.min(lhs, rhs)

#@profile
def _unry_op_tensor(lhs, rhs, op=torch.max, pad_value=0):
    if not check_obj_structure(lhs, rhs):
        raise Exception("Different shape or struct")
    lhs_ori = lhs
    rhs_ori = rhs
    if lhs is None and rhs is None:
        return None
    elif isinstance(lhs, torch.Tensor) and isinstance(rhs, torch.Tensor):
        if lhs.device != rhs.device:
            rhs = rhs.to(lhs.device)
        c_status = compare_size(lhs, rhs)
        if c_status == "lt":
            lhs = pad_to_same_size(lhs, rhs, pad_value)
            return op(lhs, rhs)
        elif c_status == "gt":
            shrinked_lhs = shrink_to_same_size(lhs, rhs)
            op(shrinked_lhs, rhs, out=shrinked_lhs)
            return lhs
        elif c_status == "eq":
            return op(lhs, rhs)
        elif c_status == "unk":
            raise Exception(f"Failed to pad the lhs tensor as the size of rhs. (c_status: {c_status})")
        else:
            raise Exception(f"Failed to pad the lhs tensor as the size of rhs. (c_status: {c_status})")
    elif isinstance(lhs, tuple) and isinstance(rhs, tuple):
        out = []
        for pair_lhs, pair_rhs in zip(lhs, rhs):
            out.append(_unry_op_tensor(pair_lhs, pair_rhs, op, pad_value))
        return tuple(out)
    elif isinstance(lhs, list) and isinstance(rhs, list):
        out = []
        for pair_lhs, pair_rhs in zip(lhs, rhs):
            out.append(_unry_op_tensor(pair_lhs, pair_rhs, op, pad_value))
        return list(out)
    elif isinstance(lhs, dict) and isinstance(rhs, dict):
        if lhs.keys() != rhs.keys():
            return False
        out = {}
        for key in lhs.keys():
            out[key] = _unry_op_tensor(lhs[key], rhs[key], op, pad_value)
        return out
    else:
        raise Exception(f"Mismatch lhs and rhs data type or shape")

#@profile
def maxmin_fn(lhs, rhs):
    if lhs is None:
        old_max_tensor = None
        old_min_tensor = None
    else:
        old_max_tensor, old_min_tensor = lhs
    if old_max_tensor is None:
        max_tensor = clone_tensor(rhs)
    else:
        max_tensor = _unry_op_tensor(old_max_tensor, rhs, torch.max, -1000)
    if old_min_tensor is None:
        min_tensor = clone_tensor(rhs)
    else:
        min_tensor = _unry_op_tensor(old_min_tensor, rhs, torch.min, 1000)
    return (max_tensor, min_tensor)