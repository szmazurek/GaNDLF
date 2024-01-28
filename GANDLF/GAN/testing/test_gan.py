from pathlib import Path
import os, shutil, yaml


from GANDLF.GAN.parseConfigGAN import parseConfigGAN
from GANDLF.GAN.training_manager_gan import TrainingManagerGAN
from GANDLF.utils import populate_header_in_parameters, parseTrainingCSV

device = "cpu"

all_models_generation = [
    "dcgan",
]

all_clip_modes = ["norm", "value", "agc"]
all_norm_types = ["batch", "instance"]

all_model_type = ["torch", "openvino"]

patch_size = {"2D": [128, 128, 1], "3D": [32, 32, 32]}

testingDir = Path(__file__).parent.absolute().__str__()
baseConfigDir = os.path.join(testingDir, os.pardir, "samples")
inputDir = os.path.join(testingDir, "data")
outputDir = os.path.join(testingDir, "data_output")
Path(outputDir).mkdir(parents=True, exist_ok=True)
gandlfRootDir = Path(__file__).parent.parent.absolute().__str__()


# # these are helper functions to be used in other tests
def sanitize_outputDir():
    print("02_1: Sanitizing outputDir")
    if os.path.isdir(outputDir):
        shutil.rmtree(outputDir)  # overwrite previous results
    Path(outputDir).mkdir(parents=True, exist_ok=True)


def write_temp_config_path(parameters_to_write):
    print("02_2: Creating path for temporary config file")
    temp_config_path = os.path.join(outputDir, "config_temp.yaml")
    # if found in previous run, discard.
    if os.path.exists(temp_config_path):
        os.remove(temp_config_path)
    if parameters_to_write is not None:
        with open(temp_config_path, "w") as file:
            yaml.dump(parameters_to_write, file)
    return temp_config_path


# these are helper functions to be used in other tests


def test_train_segmentation_rad_2d(device):
    print("03: Starting 2D Rad segmentation tests")
    # read and parse csv
    parameters = parseConfigGAN(
        testingDir + "/config_generation.yaml", version_check_flag=False
    )
    training_data, parameters["headers"] = parseTrainingCSV(
        inputDir + "/train_2d_rad_segmentation.csv"
    )
    parameters["modality"] = "rad"
    parameters["patch_size"] = patch_size["2D"]
    parameters["model"]["dimension"] = 2
    parameters["model"]["class_list"] = [0, 255]
    parameters["model"]["amp"] = True
    parameters["model"]["num_channels"] = 3
    parameters["model"]["onnx_export"] = False
    parameters["model"]["print_summary"] = False
    parameters["data_preprocessing"]["resize_image"] = [224, 224]
    parameters = populate_header_in_parameters(
        parameters, parameters["headers"]
    )
    # read and initialize parameters for specific data dimension
    for model in all_models_generation:
        parameters["model"]["architecture"] = model
        parameters["nested_training"]["testing"] = -5
        parameters["nested_training"]["validation"] = -5
        sanitize_outputDir()
        TrainingManagerGAN(
            dataframe=training_data,
            outputDir=outputDir,
            parameters=parameters,
            device=device,
            resume=False,
            reset=True,
        )

    sanitize_outputDir()

    print("passed")
