
import re
from types import MethodType
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import warnings
import torch
from torch import nn

from pydantic import BaseModel, Field

from nncov.hooks import NNCovProberHook
from nncov.injection import InjectionConfig
from nncov.metrics import NeuralMetric

from nncov.helper import recursive_getattr, recursive_setattr
from nncov.monitors.base_monitor import NeuralMonitor

META_PROPERTY_NAME = "_nncov_meta"

class ProberMeta(object):
    def __init__(self, module_name: List[str], use_re: bool=False) -> None:
        self.injection: Optional[InjectionConfig] = InjectionConfig(module_name, use_re)
        # self.monitor: Optional[NeuralMonitor] = NeuralMonitor(self.metrics)
        self.metrics: Dict[str, NeuralMetric] = {}
        self.monitors: Dict[str, NeuralMonitor] = {}
        self.hooks: Dict[str, Tuple[NNCovProberHook, Any]] = {}

    def add_hook(self, name: str, hook: NNCovProberHook, handle):
        self.hooks[name] = (hook, handle)
    
    def remove_hook(self, name: str):
        if name in self.hooks:
            self.hooks.pop(name)


def match_module(target_module_name, module_name_list: List[str], use_re: bool=True) -> bool:
    for module_name in module_name_list:
        if use_re:
            match = re.match(module_name, target_module_name)
            if match:
                return True
        else:
            if module_name in target_module_name:
                return True
    return False

def add_prober(self):
    meta: ProberMeta = getattr(self, META_PROPERTY_NAME)
    use_re = meta.injection.use_re
    module_name = meta.injection.module_name

    replace_name = []
    for name, module in self.named_modules():
        if match_module(name, module_name, use_re):
            replace_name.append(name)
            # break
    for name in replace_name:
        module = recursive_getattr(self, name)
        hook = NNCovProberHook(name, meta.metrics, meta.monitors)
        hook_handle = module.register_forward_hook(hook=hook)
        meta.add_hook(name, hook, hook_handle)

def remove_prober(self):
    meta: ProberMeta = getattr(self, META_PROPERTY_NAME)
    use_re = meta.injection.use_re
    module_name = meta.injection.module_name

    for hook_name, (hook, hook_handle) in meta.hooks.items():
        hook_handle.remove()

class MonitorInfo(object):
    def __init__(self, monitor: NeuralMonitor, module_name: str, use_re: bool) -> None:
        self.monitor = monitor
        self.module_name = module_name
        self.use_re = use_re

def add_monitor(self, monitor_name: str, module_name: str, monitor: NeuralMonitor, use_re: bool=True) -> NeuralMonitor:
    meta: ProberMeta = getattr(self, META_PROPERTY_NAME)
    if monitor_name in meta.monitors:
        raise Exception("the monitor has been added.")
    meta.monitors[monitor_name] = MonitorInfo(monitor=monitor, module_name=module_name, use_re=use_re)
    return monitor

def get_monitor(self, monitor_name: str) -> MonitorInfo:
    meta: ProberMeta = getattr(self, META_PROPERTY_NAME)
    return meta.monitors[monitor_name]

def remove_monitor(self, monitor_name: str):
    meta: ProberMeta = getattr(self, META_PROPERTY_NAME)
    del meta.monitors[monitor_name]

def clear_monitors(self):
    meta: ProberMeta = getattr(self, META_PROPERTY_NAME)
    meta.monitors.clear()

class MetricInfo(object):
    def __init__(self, metric: NeuralMetric, module_name: str, use_re: bool) -> None:
        self.metric = metric
        self.module_name = module_name
        self.use_re = use_re

def add_metric(self, metric_name: str, module_name: str, metric: NeuralMetric, use_re: bool=True) -> NeuralMetric:
    meta: ProberMeta = getattr(self, META_PROPERTY_NAME)
    if metric_name in meta.metrics:
        raise Exception("the metric has been added.")
    meta.metrics[metric_name] = MetricInfo(metric=metric, module_name=module_name, use_re=use_re)
    return metric

def get_metric(self, metric_name: str) -> Optional[MetricInfo]:
    meta: ProberMeta = getattr(self, META_PROPERTY_NAME)
    if metric_name not in meta.metrics:
        return None
    else:
        return meta.metrics[metric_name]

def remove_metric(self, metric_name: str):
    meta: ProberMeta = getattr(self, META_PROPERTY_NAME)
    del meta.metrics[metric_name]

def clear_metrics(self):
    meta: ProberMeta = getattr(self, META_PROPERTY_NAME)
    meta.metrics.clear()

def inject_module(model: nn.Module, module_name: Union[str, List[str]], use_re: bool=False):
    if hasattr(model, META_PROPERTY_NAME):
        warnings.warn("This model has been injected prober.")
        return model
    if isinstance(module_name, str):
        module_name = [module_name]
    meta = ProberMeta(module_name, use_re)
    setattr(model, META_PROPERTY_NAME, meta)
    model.add_prober = MethodType(add_prober, model)
    model.remove_prober = MethodType(remove_prober, model)

    model.add_metric = MethodType(add_metric, model)
    model.get_metric = MethodType(get_metric, model)
    model.remove_metric = MethodType(remove_metric, model)
    model.clear_metrics = MethodType(clear_metrics, model)

    model.add_monitor = MethodType(add_monitor, model)
    model.get_monitor = MethodType(get_monitor, model)
    model.remove_monitor = MethodType(remove_monitor, model)
    model.clear_monitors = MethodType(clear_monitors, model)

    model.add_prober()
    return model

def recover_module(model: nn.Module):
    if not hasattr(model, META_PROPERTY_NAME):
        warnings.warn("This model is not injected module.")
        return model
    model.remove_prober()
    delattr(model, META_PROPERTY_NAME)
    delattr(model, "add_prober")
    delattr(model, "remove_prober")

    delattr(model, "add_metric")
    delattr(model, "get_metric")
    delattr(model, "remove_metric")
    delattr(model, "clear_metrics")

    delattr(model, "add_monitor")
    delattr(model, "get_monitor")
    delattr(model, "remove_monitor")
    delattr(model, "clear_monitors")
    return model

