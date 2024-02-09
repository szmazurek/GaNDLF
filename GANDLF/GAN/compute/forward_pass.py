import os
import pathlib

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import torchio
from GANDLF.GAN.compute.loss_and_metric import get_loss_and_metrics_gans
from GANDLF.GAN.compute.step import step_gan
from GANDLF.data.post_process import global_postprocessing_dict
from GANDLF.utils import (
    get_date_time,
    get_filename_extension_sanitized,
    get_unique_timestamp,
    resample_image,
    reverse_one_hot,
    get_ground_truths_and_predictions_tensor,
    print_and_format_metrics,
)
from GANDLF.metrics import overall_stats
from tqdm import tqdm
from typing import Union, Tuple
from GANDLF.models.modelBase import ModelBase
from .generic import get_fixed_latent_vector


def validate_network_gan(
    model: ModelBase,
    dataloader: torch.utils.data.DataLoader,
    scheduler_d: torch.optim.lr_scheduler,
    scheduler_g: torch.optim.lr_scheduler,
    params: dict,
    epoch: int = 0,
    mode: str = "validation",
) -> Tuple[float, float, dict]:
    """
    Function to validate the network for a single epoch for GANs.

    Parameters
    ----------
        model (ModelBase): The model to validate. If params["model"]["type"]
    is torch, then this is a torch.nn.Module wrapped in a ModelBase class.
    Otherwise this is OV exec_net.
        dataloader (torch.utils.data.DataLoader): The dataloader to use.
        scheduler_d (torch.optim.lr_scheduler): The scheduler for the discriminator.
        scheduler_g (torch.optim.lr_scheduler): The scheduler for the generator.
        params (dict): The parameters for the run.
        epoch (int): The epoch number.
        mode (str): The mode of operation, either 'validation' or 'inference'.
    used to write outputs if requested.
    Returns
    ----------
        average_epoch_generator_loss (float): The average loss for the generator.
        average_epoch_discriminator_loss (float): The average loss for the discriminator.
        average_epoch_metrics (dict): The average metrics for the epoch.
    """
    assert mode in [
        "validation",
        "inference",
    ], "Mode should be 'validation' or 'inference' "

    print("*" * 20)
    print("Starting " + mode + " : ")
    print("*" * 20)
    total_epoch_discriminator_fake_loss = 0.0
    total_epoch_discriminator_real_loss = 0.0
    total_epoch_metrics = {}
    for metric in params["metrics"]:
        total_epoch_metrics[metric] = 0.0
    subject_id_list = []
    is_inference = mode == "inference"
    if params["verbose"]:
        if params["model"]["amp"]:
            print("Using Automatic mixed precision", flush=True)
    if scheduler_d is None or scheduler_g is None:
        current_output_dir = params["output_dir"]  # this is in inference mode
    else:  # this is useful for inference
        current_output_dir = os.path.join(
            params["output_dir"], "output_" + mode
        )
    pathlib.Path(current_output_dir).mkdir(parents=True, exist_ok=True)

    # I really do not get it
    if ((scheduler_d is None) and (scheduler_g is None)) or is_inference:
        current_output_dir = params["output_dir"]  # this is in inference mode
    else:  # this is useful for inference
        current_output_dir = os.path.join(
            params["output_dir"], "output_" + mode
        )

    if not is_inference:
        current_output_dir = os.path.join(current_output_dir, str(epoch))

    # Set the model to valid
    if params["model"]["type"] == "torch":
        model.eval()

    for batch_idx, (subject) in enumerate(
        tqdm(dataloader, desc="Looping over " + mode + " data")
    ):

        if params["verbose"]:
            print("== Current subject:", subject["subject_id"], flush=True)

        # ensure spacing is always present in params and is always subject-specific
        params["subject_spacing"] = None
        if "spacing" in subject:
            params["subject_spacing"] = subject["spacing"]

        # constructing a new dict because torchio.GridSampler requires torchio.Subject,
        # which requires torchio.Image to be present in initial dict, which the loader does not provide
        subject_dict = {}

        for key in params["channel_keys"]:
            subject_dict[key] = torchio.ScalarImage(
                path=subject[key]["path"],
                tensor=subject[key]["data"].squeeze(0),
                affine=subject[key]["affine"].squeeze(0),
            )

        grid_sampler = torchio.inference.GridSampler(
            torchio.Subject(subject_dict),
            params["patch_size"],
            patch_overlap=params["inference_mechanism"]["patch_overlap"],
        )
        patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=1)
        # aggregator = torchio.inference.GridAggregator(
        #     grid_sampler,
        #     overlap_mode=params["inference_mechanism"][
        #         "grid_aggregator_overlap"
        #     ],
        # )

        # if params["medcam_enabled"]:
        #     attention_map_aggregator = torchio.inference.GridAggregator(
        #         grid_sampler,
        #         overlap_mode=params["inference_mechanism"][
        #             "grid_aggregator_overlap"
        #         ],
        #     )

        current_patch = 0
        for patches_batch in patch_loader:
            if params["verbose"]:
                print(
                    "=== Current patch:",
                    current_patch,
                    ", time : ",
                    get_date_time(),
                    ", location :",
                    patches_batch[torchio.LOCATION],
                    flush=True,
                )
            current_patch += 1
            image = (
                torch.cat(
                    [
                        patches_batch[key][torchio.DATA]
                        for key in params["channel_keys"]
                    ],
                    dim=1,
                )
                .float()
                .to(params["device"])
            )

            current_batch_size = image.shape[0]

            label_real = torch.full(
                size=(current_batch_size,),
                fill_value=1,
                dtype=torch.float,
                device=params["device"],
            )
            label_fake = torch.full(
                size=(current_batch_size,),
                fill_value=0,
                dtype=torch.float,
                device=params["device"],
            )
            with torch.no_grad():
                if (
                    batch_idx == 0
                ):  # genereate the fake images only ONCE, as they are fixed
                    original_batch_size = params["batch_size"]
                    params["batch_size"] = (
                        1  # set this for patch-wise inference
                    )
                    fake_images = model.generator(
                        get_fixed_latent_vector(params, mode)
                    )
                    params["batch_size"] = (
                        original_batch_size  # restore the param
                    )
                    loss_fake, _, output_disc_fake, _ = step_gan(
                        model,
                        fake_images,
                        label_fake,
                        params,
                        secondary_images=None,
                    )

                loss_disc_real, metrics_output, output_disc_real, _ = step_gan(
                    model,
                    image,
                    label_real,
                    params,
                    secondary_images=fake_images,
                )

            for metric in params["metrics"]:
                # average over all patches for the current subject
                total_epoch_metrics[metric] += metrics_output[metric] / len(
                    patch_loader
                )
            total_epoch_discriminator_real_loss += loss_disc_real.cpu().item()

    average_epoch_metrics = {
        metric_name: total_epoch_metrics[metric_name] / len(dataloader)
        for metric_name in total_epoch_metrics
    }

    total_epoch_discriminator_fake_loss += (
        loss_fake.cpu().item()
    )  # fake loss is constant
    total_epoch_discriminator_real_loss /= len(
        dataloader
    )  # average over all batches

    # TODO is it valid?
    # aggregator.add_batch(fake_images.cpu(), patches_batch[torchio.LOCATION])
    # TODO do we use medcam in this case ever?
    # if params["medcam_enabled"]:
    #     _, _, output, attention_map = result
    #     attention_map_aggregator.add_batch(
    #         attention_map, patches_batch[torchio.LOCATION]
    #     )
    # output_prediction = aggregator.get_output_tensor()
    # output_prediction = output_prediction.unsqueeze(0)

    if params["save_output"]:
        img_for_metadata = torchio.ScalarImage(
            tensor=subject["1"]["data"].squeeze(0),
            affine=subject["1"]["affine"].squeeze(0),
        ).as_sitk()
        fake_images_batch = fake_images.cpu().numpy()
        # perform postprocessing before reverse one-hot encoding here

        # if jpg detected, convert to 8-bit arrays
        ext = get_filename_extension_sanitized(subject["1"]["path"][0])
        if ext in [
            ".jpg",
            ".jpeg",
            ".png",
        ]:
            fake_images_batch = fake_images_batch.astype(np.uint8)

        ## special case for 2D
        if image.shape[-1] > 1:
            result_image = sitk.GetImageFromArray(fake_images_batch)
        else:
            result_image = sitk.GetImageFromArray(fake_images_batch.squeeze(0))
        # result_image.CopyInformation(img_for_metadata)

        # this handles cases that need resampling/resizing
        if "resample" in params["data_preprocessing"]:
            result_image = resample_image(
                result_image,
                img_for_metadata.GetSpacing(),
                interpolator=sitk.sitkNearestNeighbor,
            )
        # Create the subject directory if it doesn't exist in the
        # current_output_dir directory
        os.makedirs(
            os.path.join(current_output_dir, "testing"),
            exist_ok=True,
        )
        os.makedirs(
            os.path.join(
                current_output_dir,
                "testing",
                subject["subject_id"][0],
            ),
            exist_ok=True,
        )

        path_to_save = os.path.join(
            current_output_dir,
            "testing",
            subject["subject_id"][0],
            subject["subject_id"][0] + "_gen" + ext,
        )
        sitk.WriteImage(
            result_image,
            path_to_save,
        )

    if scheduler_d is not None:
        if params["scheduler_d"]["type"] in [
            "reduce_on_plateau",
            "reduce-on-plateau",
            "plateau",
            "reduceonplateau",
        ]:
            # scheduler_d.step(average_epoch_discriminator_loss)
            raise NotImplementedError(
                "Reduce on plateau scheduler not implemented for GAN"
            )
        else:
            scheduler_d.step()
    if scheduler_g is not None:
        if params["scheduler_g"]["type"] in [
            "reduce_on_plateau",
            "reduce-on-plateau",
            "plateau",
            "reduceonplateau",
        ]:
            # scheduler_g.step(average_epoch_generator_loss)
            raise NotImplementedError(
                "Reduce on plateau scheduler not implemented for GAN"
            )
        else:
            scheduler_g.step()
    return (
        total_epoch_discriminator_fake_loss,
        total_epoch_discriminator_real_loss,
        average_epoch_metrics,
    )
