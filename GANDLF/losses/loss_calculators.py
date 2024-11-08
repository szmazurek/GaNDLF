import torch
import torch.nn.functional as F
from GANDLF.losses import get_loss
from GANDLF.utils.tensor import reverse_one_hot, get_linear_interpolation_mode
from abc import ABC, abstractmethod
from typing import List


class AbstractLossCalculator(ABC):
    def __init__(self, params: dict):
        super().__init__()
        self.params = params
        self._initialize_loss()

    def _initialize_loss(self):
        self.loss = get_loss(self.params)

    @abstractmethod
    def __call__(
        self, prediction: torch.Tensor, target: torch.Tensor, *args
    ) -> torch.Tensor:
        pass


class LossCalculatorStdnet(AbstractLossCalculator):
    def __init__(self, params):
        super().__init__(params)
        self.l1_loss = get_loss(params)
        self.kld_loss = get_loss(params)
        self.mse_loss = get_loss(params)

    def __call__(self, prediction: torch.Tensor, target: torch.Tensor, *args):
        if len(prediction) < 2:
            image: torch.Tensor = args[0]
            loss_seg = self.loss(prediction[0], target.squeeze(-1), self.params)
            loss_reco = self.l1_loss(prediction[1], image[:, :1, ...], None)
            loss_kld = self.kld_loss(prediction[2], prediction[3])
            loss_cycle = self.mse_loss(prediction[2], prediction[4], None)
            return 0.01 * loss_kld + loss_reco + 10 * loss_seg + loss_cycle
        else:
            return self.loss(prediction, target, self.params)


class LossCalculatorDeepSupervision(AbstractLossCalculator):
    def __init__(self, params):
        super().__init__(params)
        # This was taken from current Gandlf code, but I am not sure if
        # we should have this set rigidly here, as it enforces the number of
        # classes to be 4.
        self.loss_weights = [0.5, 0.25, 0.175, 0.075]

    def __call__(
        self, prediction: torch.Tensor, target: torch.Tensor, *args
    ) -> torch.Tensor:
        if len(prediction) > 1:
            loss = torch.tensor(0.0, requires_grad=True)
            ground_truth_resampled = self._resample_ground_truth(target, prediction)
            for i in range(len(prediction)):
                loss += (
                    self.loss(prediction[i], ground_truth_resampled[i], self.params)
                    * self.loss_weights[i]
                )
        else:
            loss = self.loss(prediction, target, self.params)

        return loss

    def _resample_ground_truth(
        self, target: torch.Tensor, prediction: torch.Tensor
    ) -> List[torch.Tensor]:
        ground_truth_resampled = []
        ground_truth_prev = target.detach()
        for i, _ in enumerate(prediction):
            if ground_truth_prev[0].shape != prediction[i][0].shape:
                expected_shape = reverse_one_hot(
                    prediction[i][0].detach(), self.params["model"]["class_list"]
                ).shape
                ground_truth_prev = F.interpolate(
                    ground_truth_prev,
                    size=expected_shape,
                    mode=get_linear_interpolation_mode(len(expected_shape)),
                    align_corners=False,
                )
            else:
                ground_truth_resampled.append(ground_truth_prev)
        return ground_truth_resampled


class LossCalculatorSimple(AbstractLossCalculator):
    def __call__(
        self, prediction: torch.Tensor, target: torch.Tensor, *args
    ) -> torch.Tensor:
        return self.loss(prediction, target, self.params)


class LossCalculatorFactory:
    def __init__(self, params: dict):
        self.params = params

    def get_loss_calculator(self) -> AbstractLossCalculator:
        if self.params["model"]["architecture"] == "sdnet":
            return LossCalculatorStdnet(self.params)
        elif "deep" in self.params["model"]["architecture"]:
            return LossCalculatorDeepSupervision(self.params)
        else:
            return LossCalculatorSimple(self.params)
