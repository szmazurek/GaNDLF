import math
import sys
from pathlib import Path
import requests, zipfile, io, os, csv, random, copy, shutil

from GANDLF.utils import *
from GANDLF.parseConfig import parseConfig
from GANDLF.training_manager import TrainingManager
from GANDLF.inference_manager import InferenceManager 

## global defines
# all_models_segmentation = ['unet', 'resunet', 'fcn', 'uinc'] # pre-defined segmentation model types for testing
all_models_segmentation = ['unet', 'fcn', 'uinc'] # pre-defined segmentation model types for testing
# all_models_regression = ['densenet121', 'densenet161', 'densenet169', 'densenet201', 'vgg16'] # populate once it becomes available
all_models_regression = ['densenet121', 'vgg16']

inputDir = os.path.abspath(os.path.normpath('./testing/data'))
outputDir = os.path.abspath(os.path.normpath('./testing/data_output'))
Path(outputDir).mkdir(parents=True, exist_ok=True)

'''
steps to follow to write tests:
[x] download sample data
[x] construct the training csv
[x] for each dir (application type) and sub-dir (image dimension), run training for a single epoch on cpu
  [x] separate tests for 2D and 3D segmentation
  [x] read default parameters from yaml config
  [x] for each type, iterate through all available segmentation model archs
  [x] call training manager with default parameters + current segmentation model arch
[ ] for each dir (application type) and sub-dir (image dimension), run inference for a single trained model per testing/validation split for a single subject on cpu
4. hopefully the various sys.exit messages throughout the code will catch issues
'''

def test_download_data():
  '''
  This function downloads the sample data, which is the first step towards getting everything ready
  '''
  urlToDownload = 'https://github.com/sarthakpati/tempDownloads/raw/main/data.zip'
  # do not download data again
  if not Path(os.getcwd() + '/testing/data/test/3d_rad_segmentation/001/image.nii.gz').exists():
    print('Downloading and extracting sample data')
    r = requests.get(urlToDownload)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall('./testing')

def test_constructTrainingCSV():
  '''
  This function constructs training csv
  '''
  # inputDir = os.path.normpath('./testing/data')
  # delete previous csv files
  files = os.listdir(inputDir)
  for item in files:
    if item.endswith(".csv"):
      os.remove(os.path.join(inputDir, item))

  for application_data in os.listdir(inputDir):
    currentApplicationDir = os.path.join(inputDir, application_data)

    if '2d_rad_segmentation' in application_data:
      channelsID = '_blue.png,_red.png,_green.png'
      labelID = 'mask.png'
    elif '3d_rad_segmentation' in application_data:
      channelsID = 'image'
      labelID = 'mask'
    writeTrainingCSV(currentApplicationDir, channelsID, labelID, inputDir + '/train_' + application_data + '.csv')

    # write regression and classification files
    application_data_regression = application_data.replace('segmentation', 'regression')
    application_data_classification = application_data.replace('segmentation', 'classification')
    with open(inputDir + '/train_' + application_data + '.csv', 'r') as read_f, \
    open(inputDir + '/train_' + application_data_regression + '.csv', 'w', newline='') as write_reg, \
    open(inputDir + '/train_' + application_data_classification + '.csv', 'w', newline='') as write_class:
      csv_reader = csv.reader(read_f)
      csv_writer_1 = csv.writer(write_reg)
      csv_writer_2 = csv.writer(write_class)
      i = 0
      for row in csv_reader:
        if i == 0:
          row.append('ValueToPredict')
          csv_writer_1.writerow(row)
          csv_writer_2.writerow(row)
        else:
          row_regression = copy.deepcopy(row)
          row_classification = copy.deepcopy(row)
          row_regression.append(str(random.uniform(0, 1)))
          row_classification.append(str(random.randint(0,2)))
          csv_writer_1.writerow(row_regression)
          csv_writer_2.writerow(row_classification)
        i += 1 

def test_train_segmentation_rad_2d():
  print('Starting 2D Rad segmentation tests')
  application_data = '2d_rad_segmentation'
  parameters = parseConfig(inputDir + '/' + application_data + '/sample_training.yaml', version_check = False)
  parameters['modality'] = 'rad'
  training_data, headers = parseTrainingCSV(inputDir + '/train_' + application_data + '.csv')
  for model in all_models_segmentation:
    parameters['model']['architecture'] = model 
    shutil.rmtree(outputDir) # overwrite previous results
    Path(outputDir).mkdir(parents=True, exist_ok=True)
    TrainingManager(dataframe=training_data, headers = headers, outputDir=outputDir, parameters=parameters, device='cpu', reset_prev=True)

  print('passed')

def test_train_segmentation_rad_3d():
  print('Starting 3D Rad segmentation tests')
  application_data = '3d_rad_segmentation'
  parameters = parseConfig(inputDir + '/' + application_data + '/sample_training.yaml', version_check = False)
  parameters['modality'] = 'rad'
  training_data, headers = parseTrainingCSV(inputDir + '/train_' + application_data + '.csv')
  for model in all_models_segmentation:
    parameters['model']['architecture'] = model 
    shutil.rmtree(outputDir) # overwrite previous results
    Path(outputDir).mkdir(parents=True, exist_ok=True)
    TrainingManager(dataframe=training_data, headers = headers, outputDir=outputDir, parameters=parameters, device='cpu', reset_prev=True)

  print('passed')

def test_regression_rad_2d():
  application_data = '2d_rad_segmentation'
  parameters = parseConfig(inputDir + '/' + application_data + '/sample_training_regression.yaml')
  training_data, headers = parseTrainingCSV(inputDir + '/train_2d_rad_regression.csv')
  for model in all_models_regression:
    parameters['model']['architecture'] = model 
    shutil.rmtree(outputDir) # overwrite previous results
    Path(outputDir).mkdir(parents=True, exist_ok=True)
    TrainingManager(dataframe=training_data, headers = headers, outputDir=outputDir, parameters=parameters, device='cpu', reset_prev=True)

  print('passed')

def test_regression_rad_3d():
  application_data = '3d_rad_segmentation'
  parameters = parseConfig(inputDir + '/' + application_data + '/sample_training_regression.yaml')
  training_data, headers = parseTrainingCSV(inputDir + '/train_3d_rad_regression.csv')
  for model in all_models_regression:
    parameters['model']['architecture'] = model 
    shutil.rmtree(outputDir) # overwrite previous results
    Path(outputDir).mkdir(parents=True, exist_ok=True)
    TrainingManager(dataframe=training_data, headers = headers, outputDir=outputDir, parameters=parameters, device='cpu', reset_prev=True)

  print('passed')

def test_classification_rad_2d():
  application_data = '2d_rad_segmentation'
  parameters = parseConfig(inputDir + '/' + application_data + '/sample_training_classification.yaml')
  training_data, headers = parseTrainingCSV(inputDir + '/train_2d_rad_classification.csv')
  shutil.rmtree(outputDir) # overwrite previous results
  Path(outputDir).mkdir(parents=True, exist_ok=True)
  for model in all_models_regression:
    parameters['model']['architecture'] = model 
    shutil.rmtree(outputDir) # overwrite previous results
    Path(outputDir).mkdir(parents=True, exist_ok=True)
    TrainingManager(dataframe=training_data, headers = headers, outputDir=outputDir, parameters=parameters, device='cpu', reset_prev=True)

  print('passed')

def test_classification_rad_3d():
  application_data = '3d_rad_segmentation'
  parameters = parseConfig(inputDir + '/' + application_data + '/sample_training_classification.yaml')
  training_data, headers = parseTrainingCSV(inputDir + '/train_3d_rad_classification.csv')
  for model in all_models_regression:
    parameters['model']['architecture'] = model 
    shutil.rmtree(outputDir) # overwrite previous results
    Path(outputDir).mkdir(parents=True, exist_ok=True)
    TrainingManager(dataframe=training_data, headers = headers, outputDir=outputDir, parameters=parameters, device='cpu', reset_prev=True)

  print('passed')