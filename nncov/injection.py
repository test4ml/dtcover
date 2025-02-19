
import re
from typing import Dict, List, Union
import warnings
import torch
from torch import nn


class InjectionConfig(object):
    def __init__(self, module_name: List[str], use_re: bool=False) -> None:
        self.module_name = module_name
        self.use_re = use_re

from nncov.metrics import CovKind, CovType


def get_injection_module(model_type: str, target_module: CovKind) -> str:
    if model_type == "llama":
        if target_module in [CovKind.AttenWeights, CovKind.AttenOutput, CovKind.PairAttenWeights, CovKind.PairAttenOutput]:
            atten_re = r"^model\.layers\.[0-9]+\.self_attn$"
            return atten_re
        elif target_module in [CovKind.Mlp]:
            mlp_re = r"^model\.layers\.[0-9]+\.mlp$"
            return mlp_re
        else:
            raise Exception("Unknown Coverage Type.")
    elif model_type == "mistral":
        if target_module in [CovKind.AttenWeights, CovKind.AttenOutput, CovKind.PairAttenWeights, CovKind.PairAttenOutput]:
            atten_re = r"^model\.layers\.[0-9]+\.self_attn$"
            return atten_re
        elif target_module in [CovKind.Mlp]:
            mlp_re = r"^model\.layers\.[0-9]+\.mlp$"
            return mlp_re
        else:
            raise Exception("Unknown Coverage Type.")
    elif model_type == "internlm2":
        if target_module in [CovKind.AttenWeights, CovKind.AttenOutput, CovKind.PairAttenWeights, CovKind.PairAttenOutput]:
            atten_re = r"^model\.layers\.[0-9]+\.attention$"
            return atten_re
        elif target_module in [CovKind.Mlp]:
            mlp_re = r"^model\.layers\.[0-9]+\.feed_forward$"
            return mlp_re
        else:
            raise Exception("Unknown Coverage Type.")
    elif model_type == "falcon":
        if target_module in [CovKind.AttenWeights, CovKind.AttenOutput, CovKind.PairAttenWeights, CovKind.PairAttenOutput]:
            atten_re = r"^transformer\.h\.[0-9]+\.self_attention$"
            return atten_re
        elif target_module in [CovKind.Mlp]:
            mlp_re = r"^transformer\.h\.[0-9]+\.mlp$"
            return mlp_re
        else:
            raise Exception("Unknown Coverage Type.")
    else:
        raise Exception("Unknown Model Type.")