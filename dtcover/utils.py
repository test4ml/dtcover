
from typing import Dict, List, Tuple, Union
import torch
from torch import nn

def get_token_num(tokenizer, text):
    return len(tokenizer(text)["input_ids"])

def dump_modules(model: nn.Module) -> Dict[str, str]:
    ret = []
    for name, module in model.named_modules():
        item = {
            "name": str(name),
            "type": str(type(module)),
        }
        if hasattr(module, "dtype"):
            item["dtype"] = str(getattr(module, "dtype"))
        if hasattr(module, "device"):
            item["device"] = str(getattr(module, "device"))
        ret.append(item)
    return ret


def batched(lst, batch_size: int):
    batches = []
    for i in range(0, len(lst), batch_size):
        batch = lst[i:i+batch_size]
        batches.append(batch)
    return batches


def transfer_device(obj, device: str):
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


def clone_tensor(t: Union[torch.Tensor, List, Dict, Tuple]):
    if t is None:
        return None
    elif isinstance(t, torch.Tensor):
        return t.clone()
    elif isinstance(t, list):
        return [clone_tensor(item) for item in t]
    elif isinstance(t, tuple):
        return tuple([clone_tensor(item) for item in t])
    elif isinstance(t, dict):
        return {key: clone_tensor(value) for key, value in t.items()}
    else:
        raise TypeError(f"Invalid input type. Must be a torch.Tensor, list, tuple, or dict. (type: {type(t)})")
