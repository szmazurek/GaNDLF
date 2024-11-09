import torch
from copy import deepcopy
from GANDLF.metrics import get_metrics
from abc import ABC, abstractmethod


class AbstractMetricCalculator(ABC):
    def __init__(self, params: dict):
        super().__init__()
        self.params = deepcopy(params)
        self._initialize_metrics_dict()

    def _initialize_metrics_dict(self):
        self.metrics_calculators = get_metrics(self.params)

    def _process_metric_value(self, metric_value: torch.Tensor):
        if metric_value.dim() == 0:
            return metric_value.item()
        else:
            return metric_value.tolist()

    @abstractmethod
    def __call__(
        self, prediction: torch.Tensor, target: torch.Tensor, *args
    ) -> torch.Tensor:
        pass


class MetricCalculatorStdnet(AbstractMetricCalculator):
    def __init__(self, params):
        super().__init__(params)

    def __call__(self, prediction: torch.Tensor, target: torch.Tensor, *args):
        metric_results = {}
        # TODO what do we do with edge case in GaNDLF/GANDLF/compute/loss_and_metric.py?
        for metric_name, metric_calculator in self.metrics_calculators.items():
            metric_value = (
                metric_calculator(prediction, target, self.params).detach().cpu()
            )
            metric_results[metric_name] = self._process_metric_value(metric_value)
        return metric_results


class MetricCalculatorDeepSupervision(AbstractMetricCalculator):
    def __init__(self, params):
        super().__init__(params)

    def __call__(self, prediction: torch.Tensor, target: torch.Tensor, *args):
        metric_results = {}

        for metric_name, metric_calculator in self.metrics_calculators.items():
            metric_results[metric_name] = 0.0
            for i, _ in enumerate(prediction):
                metric_value = (
                    metric_calculator(prediction[i], target[i], self.params)
                    .detach()
                    .cpu()
                )
                metric_results[metric_name] += self._process_metric_value(metric_value)
        return metric_results


class MetricCalculatorSimple(AbstractMetricCalculator):
    def __init__(self, params):
        super().__init__(params)

    def __call__(self, prediction: torch.Tensor, target: torch.Tensor, *args):
        metric_results = {}

        for metric_name, metric_calculator in self.metrics_calculators.items():
            metric_value = (
                metric_calculator(prediction, target, self.params).detach().cpu()
            )
            metric_results[metric_name] = self._process_metric_value(metric_value)
        return metric_results


class MetricCalculatorFactory:
    def __init__(self, params: dict):
        self.params = params

    def get_metric_calculator(self) -> AbstractMetricCalculator:
        if self.params["model"]["architecture"] == "sdnet":
            return MetricCalculatorStdnet(self.params)
        elif "deep" in self.params["model"]["architecture"].lower():
            return MetricCalculatorDeepSupervision(self.params)
        else:
            return MetricCalculatorSimple(self.params)
