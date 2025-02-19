

from typing import Dict

class NNCovProberHook(object):
    def __init__(self, name: str, metrics: Dict[str, "MetricInfo"], monitors: Dict[str, "MonitorInfo"]) -> None:
        super().__init__()
        self.name = name
        self.metrics: Dict[str, "MetricInfo"] = metrics
        self.monitors: Dict[str, "MonitorInfo"] = monitors

    def __call__(self, module, fea_in, fea_out):
        for metric_name, metric in self.metrics.items():
            metric.metric.update(self.name, fea_out)
        for monitor_name, monitor in self.monitors.items():
            monitor.monitor.update(self.name, fea_out)
        return None

    def pre_hook(module, args) -> None:
        pass

    def post_hook(module, args) -> None:
        pass