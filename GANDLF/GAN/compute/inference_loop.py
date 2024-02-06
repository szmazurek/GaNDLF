"""Inference loop for GANs. The inference is understood as the generation 
of images from the generator network. For now, user only provides the
number of images to generate. In the future, there can be an option to pass
the test data to compute metrics (also can be useful in conditional generation
or for explainability)."""

from .forward_pass import validate_network_gan
from .generic import create_pytorch_objects_gan, generate_latent_vector
import os, sys
from pathlib import Path

# hides torchio citation request, see https://github.com/fepegar/torchio/issues/235
os.environ["TORCHIO_HIDE_CITATION_PROMPT"] = "1"

import torch
import cv2
import numpy as np
from skimage.io import imsave
from tqdm import tqdm
from torch.cuda.amp import autocast
import tiffslide as openslide
from GANDLF.utils import (
    best_model_path_end,
    latest_model_path_end,
    print_model_summary,
)


def inference_loop(
    device: str, parameters: dict, modelDir: str, outputDir: str = None
):
    """
    The main training loop.

    Args:
        device (str): The device to perform computations on.
        parameters (dict): The parameters dictionary.
        modelDir (str): The path to the directory containing the model to be used for inference.
        outputDir (str): The path to the directory where the output of the inference session will be stored.
    """
    # Defining our model here according to parameters mentioned in the configuration file
    print("Current model type : ", parameters["model"]["type"])
    print("Number of dims     : ", parameters["model"]["dimension"])
    if "num_channels" in parameters["model"]:
        print("Number of channels : ", parameters["model"]["num_channels"])
    # ensure outputs are saved properly
    parameters["save_output"] = True

    assert (
        parameters["model"]["type"].lower() == "torch"
    ), f"The model type is not recognized: {parameters['model']['type']}"
    if parameters["model"]["type"].lower() == "openvino":
        raise NotImplementedError("OpenVINO inference is not yet implemented")
    pytorch_objects = create_pytorch_objects_gan(parameters, device=device)
    model, parameters = pytorch_objects[0], pytorch_objects[-1]
    main_dict = None
    if parameters["model"]["type"].lower() == "torch":
        # Loading the weights into the model
        if os.path.isdir(modelDir):
            files_to_check = [
                os.path.join(
                    modelDir,
                    str(parameters["model"]["architecture"])
                    + best_model_path_end,
                ),
                os.path.join(
                    modelDir,
                    str(parameters["model"]["architecture"])
                    + latest_model_path_end,
                ),
            ]

            file_to_load = None
            for best_file in files_to_check:
                if os.path.isfile(best_file):
                    file_to_load = best_file
                    break

            assert file_to_load != None, "The 'best_file' was not found"

        main_dict = torch.load(file_to_load, map_location=parameters["device"])
        model.load_state_dict(main_dict["model_state_dict"])
        parameters["previous_parameters"] = main_dict.get("parameters", None)
        model.eval()
    elif parameters["model"]["type"].lower() == "openvino":
        raise NotImplementedError("OpenVINO inference is not yet implemented")

    n_generated_samples = parameters["inference_config"]["n_generated_samples"]
    latent_vector_size = parameters["model"]["latent_vector_size"]
    batch_size = parameters["inference_config"]["batch_size"]
    n_iterations = (
        n_generated_samples // batch_size
    )  # how many iterations to run
    remaining_samples = (
        n_generated_samples % batch_size
    )  # remaining samples for last iteration

    print(
        f"Running {n_iterations} generator iterations to generate {n_generated_samples} samples with batch size {batch_size}."
    )

    if os.environ.get("HOSTNAME") is not None:
        print("\nHostname     :" + str(os.environ.get("HOSTNAME")), flush=True)

    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    for iteration in tqdm(range(n_iterations)):
        with torch.no_grad():
            latent_vector = generate_latent_vector(
                batch_size if iteration < n_iterations else remaining_samples,
                latent_vector_size,
                parameters["model"]["dimension"],
                device,
            )
            if parameters["model"]["amp"]:
                with autocast():
                    generated_images = model(latent_vector)
            else:
                generated_images = model(latent_vector)
            generated_images = generated_images.cpu()
            for i in range(generated_images.shape[0]):

                if parameters["model"]["dimension"] == 2:
                    image_to_save = (
                        generated_images[i].permute(1, 2, 0).numpy()
                    )
                    imsave(
                        os.path.join(
                            outputDir, f"batch_num_{iteration}_image_{i}.png"
                        ),
                        image_to_save,
                    )
                elif parameters["model"]["dimension"] == 3:
                    raise NotImplementedError(
                        "3D generation saving is not yet implemented"
                    )
