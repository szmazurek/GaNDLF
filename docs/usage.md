# Usage

For any DL pipeline, the following flow needs to be performed:

1. Data preparation
2. Split data into training, validation, and testing
3. Customize the training parameters

GaNDLF tackles all of these and the details are split in the manner explained in [the following section](#table-of-contents).
## Table of Contents
- [Preparing the Data](#preparing-the-data)
- [Constructing the Data CSV](#constructing-the-data-csv)
- [Customize the Training](#customize-the-training)
- [Running GaNDLF](#running-gandlf-traininginference)
- [Plot the final results](#plot-the-final-results)
- [Multi-GPU systems](#multi-gpu-systems)

## Preparing the Data

It is **highly** recommended that the dataset you want to train/infer on has been harmonized:

- Registration
  - Within-modality co-registration [[1](https://doi.org/10.1109/TMI.2014.2377694), [2](https://doi.org/10.1038/sdata.2017.117), [3](https://arxiv.org/abs/1811.02629)]
  - **OPTIONAL**: Registration of all datasets to patient atlas, if applicable [[1](https://doi.org/10.1109/TMI.2014.2377694), [2](https://doi.org/10.1038/sdata.2017.117), [3](https://arxiv.org/abs/1811.02629)]
- Size harmonization: Same physical definition of all images (see https://upenn.box.com/v/spacingsIssue for a presentation on how voxel resolutions affects downstream analyses). This is available via [GaNDLF's preprocessing module](#customize-the-training).
- Intensity harmonization: Same intensity profile, i.e., normalization [[4](https://doi.org/10.1016/j.nicl.2014.08.008), [5](https://visualstudiomagazine.com/articles/2020/08/04/ml-data-prep-normalization.aspx), [6](https://developers.google.com/machine-learning/data-prep/transform/normalization), [7](https://towardsdatascience.com/understand-data-normalization-in-machine-learning-8ff3062101f0)]. Z-scoring is available via [GaNDLF's preprocessing module](#customize-the-training).

Recommended tool for tackling all aforementioned preprocessing tasks: https://github.com/CBICA/CaPTk

**For Histopathology Only:**
- Convert WSI/label map to patches with OPM: [See using OPM](https://github.com/CBICA/OPM/blob/master/README.md)

[Back To Top &uarr;](#table-of-contents)


## Constructing the Data CSV

This application can leverage multiple channels/modalities for training while using a multi-class segmentation file. The expected format is shown as an example in [samples/sample_train.csv](https://github.com/CBICA/GaNDLF/blob/master/samples/sample_train.csv) and needs to be structured with the following header format (which shows a CSV with `N` subjects, each having `X` channels/modalities that need to be processed):

```csv
SubjectID,Channel_0,Channel_1,...,Channel_X,Label
001,/full/path/001/0.nii.gz,/full/path/001/1.nii.gz,...,/full/path/001/X.nii.gz,/full/path/001/segmentation.nii.gz
002,/full/path/002/0.nii.gz,/full/path/002/1.nii.gz,...,/full/path/002/X.nii.gz,/full/path/002/segmentation.nii.gz
...
N,/full/path/N/0.nii.gz,/full/path/N/1.nii.gz,...,/full/path/N/X.nii.gz,/full/path/N/segmentation.nii.gz
```

- `Channel` can be substituted with `Modality` or `Image`
- `Label` can be substituted with `Mask` or `Segmentation`and is used to specify the annotation file for segmentation models
- `ValueToPredict` is used for regression/classification models
- Only a single `Label` header should be passed (multiple segmentation classes should be in a single file with unique label numbers)

The [gandlf_constructCSV](https://github.com/CBICA/GaNDLF/blob/master/gandlf_constructCSV) can be used to make this easier:

```bash
# continue from previous shell
python gandlf_constructCSV \
  -inputDir ./experiment_0/data_dir/ # this is the main data directory
  -channelsID _t1.nii.gz,_t1ce.nii.gz,_t2.nii.gz,_flair.nii.gz \ # 4 structural brain MR images
  -labelID _seg.nii.gz # label identifier - not needed for regression/classification
  -outputFile ./experiment_0/train_data.csv \ # output CSV to be used for training
```

This assumes the data is in the following format:
```
./experiment_0/data_dir/
  │   │
  │   └───Patient_001 # this is used to construct the "SubjectID" header of the CSV
  │   │   │ Patient_001_brain_t1.nii.gz
  │   │   │ Patient_001_brain_t1ce.nii.gz
  │   │   │ Patient_001_brain_t2.nii.gz
  │   │   │ Patient_001_brain_flair.nii.gz
  │   │   │ Patient_001_brain_seg.nii.gz
  │   │   
  │   └───Patient_002 # this is used to construct the "Subject_ID" header of the CSV
  │   │   │ ...
  │
```

Notes:
- For classification/regression, add a column called `ValueToPredict`. Currently, we are supporting only a single value prediction per model.
- `SubjectID` or `PatientName` is used to ensure that the randomized split is done per-subject rather than per-image.

[Back To Top &uarr;](#table-of-contents)

## Customize the Training

GaNDLF requires a YAML-based configuration that controls various aspects of the training/inference process, such as:

- Model
  - Architecture
    - Segmentation: unet, resunet, uinc, fcn
    - Classification/Regression: 
      - DenseNet configurations: densenet121, densenet161, densenet169, densenet201, densenet264 
      - VGG configurations: vgg11, vgg13, vgg16, vgg19
  - Dimensionality of computations 
  - Final layer of model
  - Mixed precision
  - Class list
- Various training parameters:
  - Patch size
  - Number of epochs and patience parameter
  - Learning rate
  - Scheduler 
  - Loss function
  - Optimizer
- Data Augmentation
- Data preprocessing
- Nested data splits
  - Testing 
  - Validation 

Please see a [sample](https://github.com/CBICA/GaNDLF/blob/master/samples/config_all_options.yaml) for detailed guide and comments.

- [Segmentation example](https://github.com/CBICA/GaNDLF/blob/master/samples/config_segmentation_brats.yaml)
- [Regression example](https://github.com/CBICA/GaNDLF/blob/master/samples/config_regression.yaml)
- [Classification example](https://github.com/CBICA/GaNDLF/blob/master/samples/config_classification.yaml)

**Note**: Ensure that the configuration has valid syntax by checking the file using any YAML validator such as https://yamlchecker.com/ or https://yamlvalidator.com/ **before** trying to train.

[Back To Top &uarr;](#table-of-contents)

## Running GaNDLF (Training/Inference)

```bash
# continue from previous shell
python gandlf_run \
  -config ./experiment_0/model.yaml \ # model configuration - needs to be a valid YAML (check syntax using https://yamlchecker.com/)
  -data ./experiment_0/train.csv \ # data in CSV format 
  -output ./experiment_0/output_dir/ \ # output directory
  -train 1 \ # 1 == train, 0 == inference
  -device cuda # ensure CUDA_VISIBLE_DEVICES env variable is set for GPU device, use 'cpu' for CPU workloads
```

[Back To Top &uarr;](#table-of-contents)

## Plot the final results

After the testing/validation training is finished, GaNDLF makes it possible to collect all the statistics from the final models for testing and validation datasets and plot them. The [gandlf_collectStats](https://github.com/CBICA/GaNDLF/blob/master/gandlf_collectStats) can be used for this:

```bash
# continue from previous shell
python gandlf_collectStats \
  -inputDir /path/to/input/data 
  -channeslID _t1.nii.gz,_t2.nii.gz,_t1ce.nii.gz,_flair.nii.gz # comma-separated strings to compare the filenames from inputDir
  -labelID _seg.nii.gz # Label/mask identifier string to compare the filenames from inputDir
  -output ./experiment_0/output_dir_stats/ \ # output directory
```

[Back To Top &uarr;](#table-of-contents)

### Multi-GPU systems

Please ensure that the environment variable `CUDA_VISIBLE_DEVICES` is set [[ref](https://developer.nvidia.com/blog/cuda-pro-tip-control-gpu-visibility-cuda_visible_devices/)].

For an example how this is set, see [sge_wrapper](https://github.com/CBICA/GaNDLF/blob/master/samples/sge_wrapper).

[Back To Top &uarr;](#table-of-contents)

## M3D-CAM usage

The integration of the [M3D-CAM library](https://arxiv.org/abs/2007.00453) into GaNDLF enables the generation of attention maps for 3D/2D images in the validation epoch for classification and segmentation tasks.
To activate M3D-CAM one simply needs to add the following parameter to the config:

```yaml
medcam: 
{
  backend: "gcam",
  layer: "auto"
}
```

One can choose from the following backends:

- Grad-CAM (gcam)
- Guided Backpropagation (gbp)
- Guided Grad-CAM (ggcam)
- Grad-CAM++ (gcampp)

Optionally one can also change the name of the layer for which the attention maps should be generated.
The default behavior is "auto" which chooses the last convolutional layer.

All generated attention maps can be found in the experiment output_dir.
Link to the original repository: https://github.com/MECLabTUDA/M3d-Cam
