import yaml
import torch
from pathlib import Path
from GANDLF.models.lightning_module import GandlfLightningModule
from GANDLF.losses.loss_calculators import (
    LossCalculatorFactory,
    LossCalculatorSimple,
    LossCalculatorStdnet,
    AbstractLossCalculator,
    LossCalculatorDeepSupervision,
)


def add_mock_config_params(config):
    config["penalty_weights"] = [0.5, 0.25, 0.175, 0.075]
    config["model"]["num_channels"] = 4


def read_config_file():
    path = Path(
        "/net/tscratch/people/plgmazurekagh/lightning_port_gandlf/GaNDLF/testing/config_segmentation.yaml"
    )
    with open(path, "r") as file:
        config = yaml.safe_load(file)
    add_mock_config_params(config)
    return config


def loss_call(loss_calculator):
    dummy_preds = torch.rand(4, 4, 4, 4)
    dummy_target = torch.rand(4, 4, 4, 4)
    image_dummy = torch.rand(4, 4, 4, 4)
    return loss_calculator(dummy_preds, dummy_target, image_dummy)


def test_port_loss_calculator_simple():
    config = read_config_file()
    loss_calculator = LossCalculatorFactory(config).get_loss_calculator()
    assert isinstance(loss_calculator, LossCalculatorSimple)
    loss = loss_call(loss_calculator)
    # assert loss not nan
    assert not torch.isnan(loss).any()


def test_port_loss_calculator_sdnet():
    config = read_config_file()
    config["model"]["architecture"] = "sdnet"
    loss_calculator = LossCalculatorFactory(config).get_loss_calculator()
    assert isinstance(loss_calculator, LossCalculatorStdnet)
    loss = loss_call(loss_calculator)
    assert not torch.isnan(loss).any()


# # TODO this is failing due to interpolation size mismatch - check it out
# def test_port_loss_calculator_deep_supervision():
#     config = read_config_file()
#     config["model"]["architecture"] = "deep_supervision"
#     loss_calculator = LossCalculatorFactory(config).get_loss_calculator()
#     assert isinstance(loss_calculator, LossCalculatorDeepSupervision)
#     loss = loss_call(loss_calculator)
#     assert not torch.isnan(loss).any()


def test_port_model_initalization():
    config = read_config_file()
    model = GandlfLightningModule(config)
    assert model is not None
    assert model.model is not None
    assert isinstance(model.loss, AbstractLossCalculator)
