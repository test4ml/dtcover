
from nncov.core import inject_module, recover_module
from nncov.inference import lm_forward
from nncov.metrics import CovKind, NeuralMetric
from textattack.models.wrappers import PyTorchModelWrapper

def get_covered_neurons(model_wrapper: PyTorchModelWrapper, text: str, metric: NeuralMetric, metric_name: str, module_name: str):
    model = model_wrapper.model
    tokenizer = model_wrapper.tokenizer
    model_device = next(model.parameters()).device

    model = inject_module(model, module_name, use_re=True)
    model.add_metric(metric_name, module_name, metric)

    output_attentions = False
    if hasattr(metric, "kind") and metric.kind in [CovKind.AttenWeights, CovKind.PairAttenWeights]:
        output_attentions = True
    lm_forward(text, model_wrapper=model_wrapper, output_attentions=output_attentions)

    # metric_value = metric.compute()
    covered_neurons = metric.covered()
    metric.clear()
    model.clear_metrics()

    recover_module(model)
    return covered_neurons
