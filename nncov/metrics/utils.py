


from typing import Any, Dict
from nncov.metrics.ncov_mlp import NCovMlp
from nncov.metrics.ncov_pair_atten import NCovPairAtten
from nncov.metrics.ncov_pos_atten import NCovPosAtten


def create_metric(metric_name: str, params: Dict[str, Any], model_device: str="cpu"):
    if metric_name == "NCovMlp":
        metric = NCovMlp(t=params["threshold"] / 100.0, device=model_device)
    elif metric_name == "NCovPosAtten":
        metric = NCovPosAtten(t=params["attn_threshold"] / 100.0, device=model_device)
    elif metric_name == "NCovPairAtten":
        metric = NCovPairAtten(t=params["attn_threshold"] / 100.0, device=model_device)
    else:
        raise Exception(f"Unsupported coverage: {metric_name}")
    return metric
