"""
All the metrics are to be called from here
"""
import torch

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
    total_dice = 0
    num_class = params["num_classes"]
    if (
        params["weights"] is not None
    ):  # Reminder to add weights as a possible parameter in config
        weights = params["weights"]
    else:
        weights = None
    for i in range(0, num_class):  # 0 is background
        current_dice = dice(output[:, i, ...], label[:, i, ...])
        # currentDiceLoss = 1 - currentDice # subtract from 1 because this is a loss
        if weights is not None:
            current_dice = current_dice * weights[i]
        total_dice += current_dice
    if weights is None:
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
    # Reminder to add thresholding as a possible parameter in config
    if params["thresh"] is not None:
        thresh = params["thresh"]
    else:
        thresh = 0.5

    if thresh is not None:
        output = (output >= thresh).float()
    correct = (output == label).float().sum()
    return correct / len(label)


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
    if (metric_name).lower() == "dice":
        metric_function = multi_class_dice
    elif (metric_name).lower() == "acc":
        metric_function = accuracy
    else:
        print("Metric was undefined")
        metric_function = identity
    return metric_function
