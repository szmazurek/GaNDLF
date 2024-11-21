"""
All the losses are to be called from here
"""
from .segmentation import (
    MCD_loss,
    MCD_log_loss,
    MCT_loss,
    KullbackLeiblerDivergence,
    FocalLoss,
    MCC_loss,
    MCC_log_loss,
)
from .regression import CE, CEL, MSE_loss, L1_loss
from .hybrid import DCCE, DCCE_Logits, DC_Focal

### New interfaces
from .segmentation_new import (
    MulticlassDiceLoss,
    MulticlassDiceLogLoss,
    MulticlassMCCLoss,
    MulticlassFocalLoss,
    MulticlassTverskyLoss,
    MulticlassMCLLogLoss,
    KullbackLeiblerDivergenceLoss,
)

from .regression_new import CrossEntropyLoss, BinaryCrossEntropyLoss, L1Loss, MSELoss

from .hybrid_new import DiceCrossEntropyLoss, DiceCrossEntropyLossLogits, DiceFocalLoss

from .loss_interface import AbstractLossFunction
from typing import Type

# global defines for the losses
global_losses_dict = {
    "dc": MCD_loss,
    "dice": MCD_loss,
    "dc_log": MCD_log_loss,
    "dclog": MCD_log_loss,
    "dice_log": MCD_log_loss,
    "dicelog": MCD_log_loss,
    "mcc": MCC_loss,
    "mcc_log": MCC_log_loss,
    "mcclog": MCC_log_loss,
    "mathews": MCC_loss,
    "mathews_log": MCC_log_loss,
    "dcce": DCCE,
    "dcce_logits": DCCE_Logits,
    "ce": CE,
    "mse": MSE_loss,
    "cel": CEL,
    "tversky": MCT_loss,
    "kld": KullbackLeiblerDivergence,
    "l1": L1_loss,
    "focal": FocalLoss,
    "dc_focal": DC_Focal,
}

global_losses_dict_new = {
    "dc": MulticlassDiceLoss,
    "dice": MulticlassDiceLoss,
    "dc_log": MulticlassDiceLogLoss,
    "dclog": MulticlassDiceLogLoss,
    "dice_log": MulticlassDiceLogLoss,
    "dicelog": MulticlassDiceLogLoss,
    "mcc": MulticlassMCCLoss,
    "mcc_log": MulticlassMCLLogLoss,
    "mcclog": MulticlassMCLLogLoss,
    "mathews": MulticlassMCCLoss,
    "mathews_log": MulticlassMCLLogLoss,
    "dcce": DiceCrossEntropyLoss,
    "dcce_logits": DiceCrossEntropyLossLogits,
    "ce": BinaryCrossEntropyLoss,
    "mse": MSELoss,
    "cel": CrossEntropyLoss,
    "tversky": MulticlassTverskyLoss,
    "kld": KullbackLeiblerDivergenceLoss,
    "l1": L1Loss,
    "focal": MulticlassFocalLoss,
    "dc_focal": DiceFocalLoss,
}


def get_loss_new(params: dict) -> AbstractLossFunction:
    """
    Function to get the loss definition.

    Args:
        params (dict): The parameters' dictionary.

    Returns:
        loss (object): The loss definition.
    """
    # TODO This check looks like legacy code, should we have it?

    if isinstance(params["loss_function"], dict):
        chosen_loss = list(params["loss_function"].keys())[0].lower()
    else:
        chosen_loss = params["loss_function"].lower()
    assert (
        chosen_loss in global_losses_dict_new
    ), f"Could not find the requested loss function '{params['loss_function']}'"

    return global_losses_dict_new[chosen_loss](params) # type: ignore


def get_loss(params: dict) -> object:
    """
    Function to get the loss definition.

    Args:
        params (dict): The parameters' dictionary.

    Returns:
        loss (object): The loss definition.
    """
    # TODO This check looks like legacy code, should we have it?

    if isinstance(params["loss_function"], dict):
        chosen_loss = list(params["loss_function"].keys())[0].lower()
    else:
        chosen_loss = params["loss_function"].lower()
    assert (
        chosen_loss in global_losses_dict
    ), f"Could not find the requested loss function '{params['loss_function']}'"

    return global_losses_dict[chosen_loss]
