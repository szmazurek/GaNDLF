import torch
import torchmetrics as tm
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union, Literal
from .gan_utils.lpip import LPIPSGandlf
from .gan_utils.fid import FrechetInceptionDistance


def _calculator_ssim(
    generated_images: torch.Tensor,
    real_images: torch.Tensor,
    params: Dict[str, Any],
) -> torch.Tensor:
    """
    This function computes the SSIM between the generated images and the real
    images. Except for the params specified below, the rest of the params are
    default from torchmetrics. Works both for 2D and 3D images.
    Args:
        generated_images (torch.Tensor): The generated images.
        real_images (torch.Tensor): The real images.
        params (dict): The parameter dictionary containing training and data
    Returns:
        torch.Tensor: The SSIM score.
    """
    if "ssim" in params["metrics_config"].keys():
        reduction = (
            params["metrics_config"]["ssim"]["reduction"]
            if "reduction" in params["metrics_config"]["ssim"]
            else "mean"
        )
    # print(real_images.shape)
    if params["model"]["dimension"] == 2:
        real_images = real_images.squeeze()
        generated_images = generated_images.squeeze()
    ssim = tm.image.StructuralSimilarityIndexMeasure(reduction=reduction)
    return ssim(generated_images, real_images)


def _calculator_FID(
    generated_images: torch.Tensor,
    real_images: torch.Tensor,
    params: Dict[str, Any],
) -> torch.Tensor:
    """This function computes the FID between the generated images and the
    real images. Except for the params specified below, the rest of the params
    are default from torchmetrics.
    Args:
        generated_images (torch.Tensor): The generated images.
        real_images (torch.Tensor): The real images.
        n_input_channels (int): The number of input channels.
    Returns:
        torch.Tensor: The FID score.
    """
    if params["model"]["dimension"] != 2:
        raise ValueError("FID is only supported for 2D images")
    fid_metric = FrechetInceptionDistance(
        feature=2048,
        normalize=True,
    )
    n_input_channels = params["model"]["num_channels"]
    if n_input_channels == 1:
        # need manual patching for single channel data
        fid_metric.get_submodule("inception")._modules[
            "Conv2d_1a_3x3"
        ]._modules["conv"] = torch.nn.Conv2d(
            1, 32, kernel_size=(3, 3), stride=(2, 2), bias=False
        )
    # check input dtype
    if generated_images.dtype != torch.float32:
        generated_images = generated_images.float()
    if real_images.dtype != torch.float32:
        real_images = real_images.float()
    if generated_images.max() > 1:
        warnings.warn(
            "Input generated images are not in [0, 1] range. "
            "This may lead to incorrect results. "
            "FID expects input images to be in [0, 1] range."
            "Dividing the images by 255 for metric calculation."
        )
        generated_images = generated_images / 255.0
    if real_images.max() > 1:
        warnings.warn(
            "Input real images are not in [0, 1] range. "
            "This may lead to incorrect results. "
            "FID expects input images to be in [0, 1] range."
            "Dividing the images by 255 for metric calculation."
        )
        real_images = real_images / 255.0
    fid_metric.update(generated_images, real=False)
    fid_metric.update(real_images, real=True)
    return fid_metric.compute()


def _calculator_LPIPS(
    generated_images: torch.Tensor,
    real_images: torch.Tensor,
    params: Dict[str, Any],
) -> torch.Tensor:
    """This function computes the LPIPS between the generated images and the
    real images. Except for the params specified below, the rest of the params
    are default from torchmetrics.
    Args:
        generated_images (torch.Tensor): The generated images.
        real_images (torch.Tensor): The real images.
        n_input_channels (int): The number of input channels.
        n_dim (int): The number of dimensions.
        net_type (Literal["alex", "squeeze", "vgg"], optional): The network type.
    Defaults to "squeeze".
        reduction (Literal["mean", "sum"], optional): The reduction type.
    Defaults to "mean".
        converter_type (Literal["soft", "acs", "conv3d], optional): The converter
    type from ACS. Defaults to "soft".
    Returns:
        torch.Tensor: The LPIP score.
    """

    def _get_metric_params(
        params: Dict[str, Any]
    ) -> Tuple[int, int, str, str, str]:
        """This function returns the metric parameters from config."""
        n_input_channels = params["model"]["num_channels"]
        n_dim = params["model"]["dimension"]
        net_type = (
            params["metrics"]["lpips"]["net_type"]
            if "net_type" in params["metrics"]["lpips"]
            else "squeeze"
        )
        reduction = (
            params["metrics"]["lpips"]["reduction"]
            if "reduction" in params["metrics"]["lpips"]
            else "mean"
        )
        converter_type = (
            params["metrics"]["lpips"]["converter_type"]
            if "converter_type" in params["metrics"]["lpips"]
            else "soft"
        )
        return n_input_channels, n_dim, net_type, reduction, converter_type

    n_input_channels, n_dim, net_type, reduction, converter_type = (
        _get_metric_params(params)
    )
    lpips_metric = LPIPSGandlf(
        net_type=net_type,  # type: ignore
        normalize=True,
        reduction=reduction,  # type: ignore
        n_dim=n_dim,
        n_channels=n_input_channels,
        converter_type=converter_type,  # type: ignore
    )

    # check input dtype
    if generated_images.dtype != torch.float32:
        generated_images = generated_images.float()
    if real_images.dtype != torch.float32:
        real_images = real_images.float()
    if generated_images.max() > 1:
        warnings.warn(
            "Input generated images are not in [0, 1] range. "
            "This may lead to incorrect results. "
            "LPIPS expects input images to be in [0, 1] range."
            "Dividing the images by 255 for metric calculation."
        )
        generated_images = generated_images / 255.0
    if real_images.max() > 1:
        warnings.warn(
            "Input real images are not in [0, 1] range. "
            "This may lead to incorrect results. "
            "LPIPS expects input images to be in [0, 1] range."
            "Dividing the images by 255 for metric calculation."
        )
        real_images = real_images / 255.0
    return lpips_metric(generated_images, real_images)


def SSIM(
    generated_images: torch.Tensor,
    real_images: torch.Tensor,
    params: Dict[str, Any],
) -> torch.Tensor:
    """
    This function computes the SSIM between the generated images and the real
    images. Except for the params specified below, the rest of the params are
    default from torchmetrics.
    Args:
        generated_images (torch.Tensor): The generated images.
        real_images (torch.Tensor): The real images.
        params (dict): The parameter dictionary containing training and data
    information.
    Returns:
        torch.Tensor: The SSIM score.
    """
    return _calculator_ssim(generated_images, real_images, params)


def FID(
    generated_images: torch.Tensor,
    real_images: torch.Tensor,
    params: Dict[str, Any],
) -> torch.Tensor:
    """This function computes the FID between the generated images and the
    real images. Except for the params specified below, the rest of the params
    are default from torchmetrics.
    Args:
        generated_images (torch.Tensor): The generated images.
        real_images (torch.Tensor): The real images.
        n_input_channels (int): The number of input channels.
    Returns:
        torch.Tensor: The FID score.
    """
    return _calculator_FID(generated_images, real_images, params)


def LPIPS(
    generated_images: torch.Tensor,
    real_images: torch.Tensor,
    params: Dict[str, Any],
) -> torch.Tensor:
    """This function computes the LPIPS between the generated images and the
    real images. Except for the params specified below, the rest of the params
    are default from torchmetrics.
    Args:
        generated_images (torch.Tensor): The generated images.
        real_images (torch.Tensor): The real images.
        n_input_channels (int): The number of input channels.
        n_dim (int): The number of dimensions.
        net_type (Literal["alex", "squeeze", "vgg"], optional): The network type.
    Defaults to "squeeze".
        reduction (Literal["mean", "sum"], optional): The reduction type.
    Defaults to "mean".
        converter_type (Literal["soft", "acs", "conv3d], optional): The converter
    type from ACS. Defaults to "soft".
    Returns:
        torch.Tensor: The LPIP score.
    """
    return _calculator_LPIPS(generated_images, real_images, params)
