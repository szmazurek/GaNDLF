import lightning.pytorch as pl
from GANDLF.models import get_model
from GANDLF.optimizers import get_optimizer
from GANDLF.schedulers import get_scheduler
from GANDLF.losses.loss_calculators import LossCalculatorFactory
from GANDLF.models.modelBase import ModelBase

from copy import deepcopy


class GandlfLightningModule(pl.LightningModule):
    def __init__(self, params: dict):
        super().__init__()
        self.params = deepcopy(params)
        self.model: ModelBase = get_model(params)
        self.loss = LossCalculatorFactory(params).get_loss_calculator()

    def configure_optimizers(self):
        params = deepcopy(self.params)
        params["model_parameters"] = self.model.parameters()
        optimizer = get_optimizer(params)
        if "scheduler" in self.params:
            params["optimizer_object"] = optimizer
            scheduler = get_scheduler(params)
            return [optimizer], [scheduler]
        return optimizer
