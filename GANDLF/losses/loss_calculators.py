import torch
from GANDLF.losses import get_loss_new
from abc import ABC, abstractmethod
from copy import deepcopy


class AbstractLossCalculator(ABC):
    def __init__(self, params: dict):
        super().__init__()
        self.params = params
        self._initialize_loss()

    def _initialize_loss(self):
        self.loss = self._get_loss(self.params)

    @staticmethod
    def _get_loss(params: dict):
        return get_loss_new(params)

    @abstractmethod
    def __call__(
        self, prediction: torch.Tensor, target: torch.Tensor, *args
    ) -> torch.Tensor:
        pass


class LossCalculatorSDNet(AbstractLossCalculator):
    def __init__(self, params):
        super().__init__(params)
        self._get_proxy_losses()

    def _get_proxy_losses(self):
        copied_params = deepcopy(self.params)
        copied_params["loss_function"] = "l1"
        self.l1_loss = self._get_loss(copied_params)
        copied_params["loss_function"] = "kld"
        self.kld_loss = self._get_loss(copied_params)
        copied_params["loss_function"] = "mse"
        self.mse_loss = self._get_loss(copied_params)

    def __call__(self, prediction: torch.Tensor, target: torch.Tensor, *args):
        if len(prediction) < 2:
            image: torch.Tensor = args[0]
            loss_seg = self.loss(prediction[0], target.squeeze(-1))
            loss_reco = self.l1_loss(prediction[1], image[:, :1, ...], None)
            loss_kld = self.kld_loss(prediction[2], prediction[3])
            loss_cycle = self.mse_loss(prediction[2], prediction[4], None)
            return 0.01 * loss_kld + loss_reco + 10 * loss_seg + loss_cycle
        else:
            return self.loss(prediction, target)


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
            for i in range(len(prediction)):
                loss += self.loss(prediction[i], target[i]) * self.loss_weights[i]
        else:
            loss = self.loss(prediction, target)

        return loss


class LossCalculatorSimple(AbstractLossCalculator):
    def __call__(
        self, prediction: torch.Tensor, target: torch.Tensor, *args
    ) -> torch.Tensor:
        return self.loss(prediction, target)


class LossCalculatorFactory:
    def __init__(self, params: dict):
        self.params = params

    def get_loss_calculator(self) -> AbstractLossCalculator:
        if self.params["model"]["architecture"] == "sdnet":
            return LossCalculatorSDNet(self.params)
        elif "deep" in self.params["model"]["architecture"].lower():
            return LossCalculatorDeepSupervision(self.params)
        else:
            return LossCalculatorSimple(self.params)
