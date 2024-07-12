# https://docs.ultralytics.com/tasks/pose/#models

import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models.detection.keypoint_rcnn import keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights
import os
import numpy as np
from torch.utils.data import Subset
from tqdm import tqdm
import pytz
import logging
import datetime
from collections import defaultdict
import pickle
import sys
# change this based on you file system, append "utils" folder where "keypoints2d_utils.py" is stored
sys.path.append('/content/Master-s-thesis---2D-Keypoints-extraction-experiments/utils')
from keypoints2d_utils import compute_MPJPE, visualize_keypoints2d

from ultralytics import YOLO

from functools import wraps

def capture_arguments(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        wrapper.captured_args = args
        wrapper.captured_kwargs = kwargs
        print("Positional arguments:", args)
        print("Keyword arguments:", kwargs)
        return func(self, *args, **kwargs)
    wrapper.captured_args = ()
    wrapper.captured_kwargs = {}
    return wrapper

class YOLO_Pose:
    
    def __init__(self, weights=None) -> None:
        self.weights = weights
        self.num_classes = 2
        self.num_keypoints = 21
        self.model_name = 'YOLO_pose'
    
    # https://docs.ultralytics.com/modes/train for train() parameters explanation
    @capture_arguments
    def train(self, dataset_config, model_config,
              num_epochs=10, 
              time=None, # Maximum training time in hours. If set, this overrides the epochs argument, allowing training to automatically stop after the specified duration. Useful for time-constrained training scenarios.
              batch_size=1, 
              image_size=640, # Target image size for training. All images are resized to this dimension before being fed into the model. Affects model accuracy and computational complexity.
              save=True, # Enables saving of training checkpoints and final model weights. Useful for resuming training or model deployment.
              checkpoint_step = -1, # Frequency of saving model checkpoints, specified in epochs. A value of -1 disables this feature. Useful for saving interim models during long training sessions.
              num_workers=8, # Number of worker threads for data loading (per RANK if Multi-GPU training). Influences the speed of data preprocessing and feeding into the model, especially useful in multi-GPU setups.
              output_folder=None, # Name of the project directory where training outputs are saved. Allows for organized storage of different experiments.
              training_run_folder_name=None, # Name of the training run. Used for creating a subdirectory within the project folder, where training logs and outputs are stored.
              verbose=False, # Enables verbose output during training, providing detailed logs and progress updates. Useful for debugging and closely monitoring the training process.
              use_autocast=True, # Enables Automatic Mixed Precision (AMP) training, reducing memory usage and possibly speeding up training with minimal impact on accuracy.
              lr=0.01, # Initial learning rate (i.e. SGD=1E-2, Adam=1E-3) . Adjusting this value is crucial for the optimization process, influencing how rapidly model weights are updated.
              lrf=0.01, # Final learning rate as a fraction of the initial rate = (lr0 * lrf), used in conjunction with schedulers to adjust the learning rate over time.
              generate_plots=False # Generates and saves plots of training and validation metrics, as well as prediction examples, providing visual insights into model performance and learning progression.
              ):
        
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        #     os.makedirs(os.path.join(output_folder, 'checkpoints'))
            
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        """ Configure a log """
        # Create a new log file
        filename_log = os.path.join(output_folder, f'log_{output_folder.rpartition(os.sep)[-1]}.txt')
        # Configure the logging
        log_format = '%(message)s'
        logging.basicConfig(
            filename=filename_log,
            level=logging.INFO,
            format=log_format,
            filemode='a'  
        )
        fh = logging.FileHandler(filename_log, mode='a')  # Use 'a' mode to append to the log file
        fh.setFormatter(logging.Formatter(log_format))
        logger = logging.getLogger(__name__)
        logger.addHandler(fh)
        logger.setLevel(logging.INFO)

        logging.info('args:')
        for k, v in self.train.captured_kwargs.items():
            logging.info(f'\t{k}: {v}')
        logging.info(f'\tdevice: {device}')
        logging.info('--'*50) 
        print('\n')

        print(f'🟢 Logging info in "{filename_log}"\n')
        
        """ Model Loading """
        
        scale = {
            'small': 's',
            'nano': 'n',
            'medium': 'm',
            'large': 'l',
            'extra-large': 'x',
        }

        # Load model: https://github.com/orgs/ultralytics/discussions/10211#discussioncomment-9182568
        config_folder = '/content/Master-s-thesis---2D-Keypoints-extraction-experiments/YOLO_Pose/config
        model = YOLO(os.path.join(config_folder, f'yolov8{scale[model_type]}-pose.yaml')) # TODO
        print(model) # DEBUG

        # Train the model
        results = model.train(data="coco8.yaml", epochs=100, imgsz=640, device=device)
        

##### DEBUG #####

current_timestamp = datetime.datetime.now(pytz.timezone("Europe/Rome")).strftime("%d-%m-%Y_%H-%M")
training_run_folder_name = f'Training-DEBUG--{current_timestamp}'
output_folder = '/content/drive/MyDrive/Thesis/Keypoints2d_extraction/YOLO_Pose'

##### DEBUG #####

YOLO_Pose().train(
    dataset_config='/content/Master-s-thesis---2D-Keypoints-extraction-experiments/YOLO_Pose/utils/config_povsurgery.yaml',
    model_config='small',
    num_epochs=10,
    batch_size=1,
    image_size=1920,
    save=True,
    num_workers=2,
    training_run_folder_name=training_run_folder_name,
    verbose=True,
    generate_plots=True,
    lr=00.01,
    checkpoint_step=-1,
    output_folder='/content/drive/MyDrive/Thesis/Keypoints2d_extraction/YOLO_Pose',
    use_autocast=False
)
'''
YOLO_Pose().evaluate(
    dataset_root='/content/drive/MyDrive/Thesis/POV_Surgery_data',
    annot_root='/content/drive/MyDrive/Thesis/THOR-Net_based_work/povsurgery/object_False',
    model_path='/content/drive/MyDrive/Thesis/Keypoints2d_extraction/KeypointRCNN/Training-DEBUG--08-07-2024_15-58/checkpoints/model_best-1',
    batch_size=1,
    seq='',
    output_results='/content/drive/MyDrive/Thesis/Keypoints2d_extraction/KeypointRCNN/Training-DEBUG--08-07-2024_15-58/output_results',
    visualize=False
)
'''

##### DEBUG #####