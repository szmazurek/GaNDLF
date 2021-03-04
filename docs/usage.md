# Usage

## Preparing the Data

It is **highly** recommended that the dataset you want to train/infer on has been harmonized:

- Registration
  - Within-modality co-registration [[1](https://doi.org/10.1109/TMI.2014.2377694), [2](https://doi.org/10.1038/sdata.2017.117), [3](https://arxiv.org/abs/1811.02629)]
  - **OPTIONAL**: Registration of all datasets to patient atlas, if applicable [[1](https://doi.org/10.1109/TMI.2014.2377694), [2](https://doi.org/10.1038/sdata.2017.117), [3](https://arxiv.org/abs/1811.02629)]
- Size harmonization: Same physical definition of all images (see https://upenn.box.com/v/spacingsIssue for a presentation on how voxel resolutions affects downstream analyses). This is available via [GANDLF's preprocessing module](#customize-the-training).
- Intensity harmonization: Same intensity profile, i.e., normalization [[4](https://doi.org/10.1016/j.nicl.2014.08.008), [5](https://visualstudiomagazine.com/articles/2020/08/04/ml-data-prep-normalization.aspx), [6](https://developers.google.com/machine-learning/data-prep/transform/normalization), [7](https://towardsdatascience.com/understand-data-normalization-in-machine-learning-8ff3062101f0)]. Z-scoring is available via [GANDLF's preprocessing module](#customize-the-training).

Recommended tool for tackling all aforementioned preprocessing tasks: https://github.com/CBICA/CaPTk

**For Histopathology Only:**
- Convert WSI/label map to patches with OPM: [See using OPM](./GANDLF/OPM/README.md)

## Constructing the Data CSV

This application can leverage multiple channels/modalities for training while using a multi-class segmentation file. The expected format is shown as an example in [samples/sample_train.csv](https://github.com/CBICA/GaNDLF/blob/master/samples/sample_train.csv) and needs to be structured with the following header format:

```csv
SubjectID,Channel_0,Channel_1,...,Channel_X,Label
001,/full/path/0.nii.gz,/full/path/1.nii.gz,...,/full/path/X.nii.gz,/full/path/segmentation.nii.gz
```

- `Channel` can be substituted with `Modality` or `Image`
- `Label` can be substituted with `Mask` or `Segmentation`
- Only a single `Label` header should be passed (multiple segmentation classes should be in a single file with unique label numbers)

The [gandlf_constructCSV](https://github.com/CBICA/GaNDLF/blob/master/gandlf_constructCSV) can be used to make this easier:

```bash
# continue from previous shell
python gandlf_constructCSV \
  -inputDir ./experiment_0/output_dir/ # this is the main output directory of training step
  -output ./experiment_0/output_dir_stats/ \ # output directory
  -channelsID _t1.nii.gz,_t1ce.nii.gz,_t2.nii.gz,_flair.nii.gz \ # 4 structural brain MR images
  -labelID _seg.nii.gz # label identifier - not needed for regression/classification
```

Notes:
- For classification/regression, add a column called `ValueToPredict`. Currently, we are supporting only a single value prediction per model.
- `SubjectID` or `PatientName` is used to ensure that the randomized split is done per-subject rather than per-image.

## Customize the Training

GANDLF requires a YAML-based configuration that controls various aspects of the training/inference process, such as:

- Model
  - Architecture
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

Please see a [sample](https://github.com/CBICA/GaNDLF/blob/master/samples/sample_training.yaml) for detailed guide and comments.
## Running GANDLF (Training/Inference)

```bash
# continue from previous shell
python gandlf_run \
  -config ./experiment_0/model.yaml \ # model configuration - needs to be a valid YAML (check syntax using https://yamlchecker.com/)
  -data ./experiment_0/train.csv \ # data in CSV format 
  -output ./experiment_0/output_dir/ \ # output directory
  -train 1 \ # 1 == train, 0 == inference
  -device cuda # ensure CUDA_VISIBLE_DEVICES env variable is set for GPU device, -1 for CPU
  # -modelDir /path/to/model/weights # used in inference mode
```

## Plot the final results

After the testing/validation training is finished, GANDLF makes it possible to collect all the statistics from the final models for testing and validation datasets and plot them. The [gandlf_collectStats](https://github.com/CBICA/GaNDLF/blob/master/gandlf_collectStats) can be used for this:

```bash
# continue from previous shell
python gandlf_collectStats \
  -inputDir /path/to/input/data 
  -channeslID _t1.nii.gz,_t2.nii.gz,_t1ce.nii.gz,_flair.nii.gz # comma-separated strings to compare the filenames from inputDir
  -labelID _seg.nii.gz # Label/mask identifier string to compare the filenames from inputDir
  -output ./experiment_0/output_dir_stats/ \ # output directory
```

### Multi-GPU systems

Please ensure that the environment variable `CUDA_VISIBLE_DEVICES` is set [[ref](https://developer.nvidia.com/blog/cuda-pro-tip-control-gpu-visibility-cuda_visible_devices/)].

For an example how this is set, see [sge_wrapper](https://github.com/CBICA/GaNDLF/blob/master/samples/sge_wrapper).
