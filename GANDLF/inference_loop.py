import os
os.environ['TORCHIO_HIDE_CITATION_PROMPT'] = '1' # hides torchio citation request, see https://github.com/fepegar/torchio/issues/235

from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
from torch.utils.data.dataset import Dataset
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import random
import torchio
from torchio import Image, Subject
from torchio.transforms import *
from torchio import Image, Subject
from sklearn.model_selection import KFold
from shutil import copyfile
import time
import sys
import ast 
import pickle
from pathlib import Path
import argparse
import datetime
import SimpleITK as sitk
from GANDLF.data.ImagesFromDataFrame import ImagesFromDataFrame
if os.name != 'nt':
    from GANDLF.inference_dataloader_histopath import InferTumorSegDataset
from GANDLF.schd import *
from GANDLF.losses import *
from GANDLF.utils import *
from .parameterParsing import *

def inferenceLoopRad(inferenceDataFromPickle, headers, device, parameters, outputDir):
    '''
    This is the main inference loop
    '''
    # extract variables form parameters dict
    psize = parameters['psize']
    q_max_length = parameters['q_max_length']
    q_samples_per_volume = parameters['q_samples_per_volume']
    q_num_workers = parameters['q_num_workers']
    q_verbose = parameters['q_verbose']
    augmentations = parameters['data_augmentation']
    preprocessing = parameters['data_preprocessing']
    which_model = parameters['model']['architecture']
    class_list = parameters['class_list']
    base_filters = parameters['base_filters']
    batch_size = parameters['batch_size']
    loss_function = parameters['loss_function']
    
    n_channels = len(headers['channelHeaders'])
    n_classList = len(class_list)

    if len(psize) == 2:
        psize.append(1) # ensuring same size during torchio processing

    # Setting up the inference loader
    inferenceDataForTorch = ImagesFromDataFrame(inferenceDataFromPickle, psize, headers, q_max_length, q_samples_per_volume, q_num_workers, q_verbose, train = False, augmentations = augmentations, preprocessing = preprocessing)
    inference_loader = DataLoader(inferenceDataForTorch, batch_size=batch_size)

    # Defining our model here according to parameters mentioned in the configuration file
    model = get_model(which_model, parameters['dimension'], n_channels, n_classList, base_filters, final_convolution_layer = parameters['model']['final_layer'], psize = psize)
    
    # Loading the weights into the model
    main_dict = torch.load(os.path.join(outputDir, str(which_model) + "_best.pth.tar"))
    model.load_state_dict(main_dict['model_state_dict'])
    
    if not(os.environ.get('HOSTNAME') is None):
        print("\nHostname         :" + str(os.environ.get('HOSTNAME')))
        sys.stdout.flush()

    # get the channel keys for concatenation later (exclude non numeric channel keys)
    batch = next(iter(inference_loader))
    channel_keys = list(batch.keys())
    channel_keys_new = []
    for item in channel_keys:
        if item.isnumeric():
            channel_keys_new.append(item)
    channel_keys = channel_keys_new

    print("Data Samples: ", len(inference_loader.dataset))
    sys.stdout.flush()
    model, amp, device = send_model_to_device(model, amp, device, optimizer=None)
    
    # print stats
    print('Using device:', device)
    sys.stdout.flush()

    # get loss function
    loss_fn, MSE_requested = get_loss(loss_function)

    model.eval()
    average_dice, average_loss = get_metrics_save_mask(model, inference_loader, psize, channel_keys, class_list, loss_fn,
                                                       weights=None, save_mask=True)
    print(average_dice, average_loss)



if os.name != 'nt':
    def inferenceLoopPath(inferenceDataFromPickle, headers, device, parameters, outputDir):
        '''
        This is the main inference loop for histopathology
        '''
        # extract variables form parameters dict
        psize = parameters['psize']
        augmentations = parameters['data_augmentation']
        preprocessing = parameters['data_preprocessing']
        which_model = parameters['model']['architecture']
        class_list = parameters['class_list']
        base_filters = parameters['base_filters']
        batch_size = parameters['batch_size']
        n_channels = len(headers['channelHeaders'])
        n_classList = len(class_list)
        # Report the time stamp
        training_start_time = time.asctime()
        startstamp = time.time()
        print("\nHostname   :" + str(os.getenv("HOSTNAME")))
        print("\nStart Time :" + str(training_start_time))
        print("\nStart Stamp:" + str(startstamp))
        sys.stdout.flush()

        # PRINT PARSED ARGS
        print("\n\n")
        print("Output directory        :", outputDir)
        print("Number of channels      :", parameters['num_channels'])
        print("Modalities              :", parameters['modality'])
        print("Number of classes       :", parameters['num_classes'])
        print("Batch Size              :", parameters['batch_size'])
        print("Patch Size              :", parameters['patch_size'])
        print("Sampling Stride         :", parameters['stride_size'])
        print("Base Filters            :", parameters['base_filters'])
        print("Load Weights            :", parameters['load_weights'])
        sys.stdout.flush()
        # We generate CSV for training if not provided
        print("Reading CSV Files")

        test_csv = parameters['test_csv']

        # Defining our model here according to parameters mentioned in the configuration file
        model = get_model(which_model, parameters['dimension'], n_channels, n_classList, base_filters,
                        final_convolution_layer=parameters['model']['final_layer'], psize=psize)

        # Loading the weights into the model
        main_dict = torch.load(os.path.join(outputDir, str(which_model) + "_best.pth.tar"))
        model.load_state_dict(main_dict['model_state_dict'])
        print('Loaded Weights successfully.')
        sys.stdout.flush()

        model, amp, device = send_model_to_device(model, amp, device, optimizer=None)

        model.eval()
    # print stats
        print('Using device:', device)
        sys.stdout.flush()

        test_df = pd.read_csv(test_csv)
        # Patch blocks

        for index, row in test_df.iterrows():
            subject_name = row[headers['subjectIDHeader']]
            print("Patient Slide       : ", row[headers['subjectIDHeader']])
            print("Patient Location    : ", row[headers['channelHeaders']])
            os_image = OpenSlide(row[headers['channelHeaders']])
            level_width, level_height = os_image.level_dimensions[int(parameters['slide_level'])]
            subject_dest_dir = os.path.join(outputDir, subject_name)
            os.makedirs(subject_dest_dir, exist_ok=True)

            probs_map = np.zeros((level_height, level_width), dtype=np.float16)
            count_map = np.zeros((level_height, level_width), dtype=np.uint8)

            patient_dataset_obj = InferTumorSegDataset(row[headers['channelHeaders']],
                                                    patch_size=psize,
                                                    stride_size=stride,
                                                    selected_level=parameters['slide_level'],
                                                    mask_level=4)

            dataloader = DataLoader(patient_dataset_obj,
                                    batch_size=int(parameters['batch_size']),
                                    shuffle=False, num_workers=2)
            for image_patches, (x_coords, y_coords) in tqdm(dataloader):
                x_coords, y_coords = y_coords.numpy(), x_coords.numpy()
                with autocast():
                    output = model(image_patches.half().cuda())
                output = output.cpu().detach().numpy()
                for i in range(int(output.shape[0])):
                    count_map[x_coords[i]:x_coords[i]+psize,
                            y_coords[i]:y_coords[i]+psize] += 1
                    probs_map[x_coords[i]:x_coords[i]+psize,
                            y_coords[i]:y_coords[i]+psize] += output[i][0].unsqueeze(-1)
            probs_map = probs_map/count_map
            count_map = (count_map/count_map.max())
            out = count_map*probs_map
            count_map = np.array(count_map*255, dtype=np.uint16)
            out_thresh = np.array((out > 0.5)*255, dtype=np.uint16)
            imsave(os.path.join(subject_dest_dir, row[headers['subjectIDHeader']]+'_prob.png'), out)
            imsave(os.path.join(subject_dest_dir, row[headers['subjectIDHeader']]+'_seg.png'), out_thresh)
            imsave(os.path.join(subject_dest_dir, row[headers['subjectIDHeader']]+'_count.png'), count_map)

if __name__ == "__main__":

    # parse the cli arguments here
    parser = argparse.ArgumentParser(description = "Inference Loop of GANDLF")
    parser.add_argument('-inference_loader_pickle', type=str, help = 'Inference loader pickle', required=True)
    parser.add_argument('-parameter_pickle', type=str, help = 'Parameters pickle', required=True)
    parser.add_argument('-headers_pickle', type=str, help = 'Header pickle', required=True)
    parser.add_argument('-outputDir', type=str, help = 'Output directory', required=True)
    parser.add_argument('-device', type=str, help = 'Device to train on', required=True)
    
    args = parser.parse_args()

    # # write parameters to pickle - this should not change for the different folds, so keeping is independent
    psize = pickle.load(open(args.psize_pickle,"rb"))
    headers = pickle.load(open(args.headers_pickle,"rb"))
    label_header = pickle.load(open(args.label_header_pickle,"rb"))
    parameters = pickle.load(open(args.parameter_pickle,"rb"))
    inferenceDataFromPickle = pd.read_pickle(args.inference_loader_pickle)

    if parameters['modalities'] == 'rad':
        inferenceLoopRad(inference_loader_pickle=inferenceDataFromPickle, 
                         headers=headers, 
                         parameters=parameters,
                         outputDir=args.outputDir,
                         device=args.device)
    elif parameters['modalities'] == 'path':
        if os.name != 'nt':
            inferenceLoopPath(inference_loader_pickle=inferenceDataFromPickle, 
                            headers=headers, 
                            parameters=parameters,
                            outputDir=args.outputDir,
                            device=args.device)
    else:
        print('Please Select a modality between rad and path. Correct the option in the config file.')
        sys.exit(0)
