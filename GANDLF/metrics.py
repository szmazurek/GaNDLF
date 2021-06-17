"""
All the metrics are to be called from here
"""
import torch
from .losses import MSE, MSE_loss
from .utils import one_hot


# Dice scores and dice losses
def dice(output, label):
    """
    This function computes a dice score between two tensors

    Parameters
    ----------
    output : Tensor
        Output predicted generally by the network
    label : Tensor
        Required target label to match the output with

    Returns
    -------
    Tensor
        Computed Dice Score

    """
    smooth = 1e-7
    iflat = output.contiguous().view(-1)
    tflat = label.contiguous().view(-1)
    intersection = (iflat * tflat).sum()
    return (2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)


def multi_class_dice(output, label, params):
    """
    This function computes a multi-class dice

    Parameters
    ----------
    output : TYPE
        DESCRIPTION.
    label : TYPE
        DESCRIPTION.
    num_class : TYPE
        DESCRIPTION.
    weights : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    total_dice : TYPE
        DESCRIPTION.

    """
    label = one_hot(label, params["model"]["class_list"])
    total_dice = 0
    num_class = params["model"]["num_classes"]
    # print("Number of classes : ", params["model"]["num_classes"])
    for i in range(0, num_class):  # 0 is background
        if (
            num_class != params["model"]["ignore_label_validation"]
        ):  # this check should only happen during validation
            total_dice += dice(output[:, i, ...], label[:, i, ...])
        # currentDiceLoss = 1 - currentDice # subtract from 1 because this is a loss
    total_dice /= num_class
    return total_dice


def accuracy(output, label, params):
    """
    Calculates the accuracy between output and a label

    Parameters
    ----------
    output : TYPE
        DESCRIPTION.
    label : TYPE
        DESCRIPTION.
    thresh : TYPE, optional
        DESCRIPTION. The default is 0.5.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    if params["metrics"]["accuracy"]["threshold"] is not None:
        output = (output >= params["metrics"]["accuracy"]["threshold"]).float()
    correct = (output == label).float().sum()
    return correct / len(label)


def MSE(output, label, reduction="mean", scaling_factor=1):
    """
    Calculate the mean square error between the output variable from the network and the target

    Parameters
    ----------
    output : torch.Tensor
        The output generated usually by the network
    target : torch.Tensor
        The label for the corresponding Tensor for which the output was generated
    reduction : string, optional
        DESCRIPTION. The default is 'mean'.
    scaling_factor : integer, optional
        The scaling factor to multiply the label with

    Returns
    -------
    loss : torch.Tensor
        Computed Mean Squared Error loss for the output and label

    """
    return MSE(output, label, reduction=reduction, scaling_factor=scaling_factor)


def MSE_loss_agg(inp, target, params):
    return MSE_loss(inp, target, params)


def identity(output, label, params):
    """
    Always returns 0

    Parameters
    ----------
    output : Tensor
        Output predicted generally by the network
    label : Tensor
        Required target label to match the output with

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    _, _, _ = output, label, params
    return torch.Tensor(0)


def fetch_metric(metric_name):
    """

    Parameters
    ----------
    metric_name : string
        Should be a name of a metric

    Returns
    -------
    metric_function : function
        The function to compute the metric

    """
    # if dict, only pick the first value
    if isinstance(metric_name, dict):
        metric_name = list(metric_name)[0]

    if (metric_name).lower() == "dice":
        metric_function = multi_class_dice
    elif (metric_name).lower() == "accuracy":
        metric_function = accuracy
    elif (metric_name).lower() == "mse":
        metric_function = MSE_loss_agg
    else:
        print("Metric was undefined")
        metric_function = identity
    return metric_function
