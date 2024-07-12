# https://docs.ultralytics.com/tasks/pose/#models

import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models.detection.keypoint_rcnn import keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights
import os
import numpy as np
from utils.dataset import Dataset
from torch.utils.data import Subset
from tqdm import tqdm
import pytz
import logging
import datetime
from collections import defaultdict
import pickle
from utils.keypoints2d_utils import compute_MPJPE, visualize_keypoints2d

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

def povsurgery_collate_fn(batch):
    """
    Custom collate function for DataLoader.
    Args:
        batch (list): List of tuples (image, target_dict).
                      image: Tensor containing an image.
                      target_dict: Dictionary containing target annotations.

    Returns:
        tuple: Tuple containing:
            - images (Tensor): Tensor containing batch_size images.
            - targets (list): List of length batch_size, each element is a target dictionary.
    """
    images = [item[0] for item in batch]  # Extract images from each tuple
    targets = [item[1] for item in batch]  # Extract targets from each tuple

    # Stack images into a single tensor
    images = torch.stack(images, dim=0)

    return images, targets

class YOLO_Pose:
    
    def __init__(self, weights=None) -> None:
        self.weights = weights
        self.num_classes = 2
        self.num_keypoints = 21
        self.model_name = 'YOLO_pose'
    
    # https://docs.ultralytics.com/modes/train for train() parameters explanation
    @capture_arguments
    def train(self, dataset_config, num_epochs=10, 
              time=None, # Maximum training time in hours. If set, this overrides the epochs argument, allowing training to automatically stop after the specified duration. Useful for time-constrained training scenarios.
              batch_size=1, 
              image_size=640, # Target image size for training. All images are resized to this dimension before being fed into the model. Affects model accuracy and computational complexity.
              save=True, # Enables saving of training checkpoints and final model weights. Useful for resuming training or model deployment.
              checkpoint_step = -1, # Frequency of saving model checkpoints, specified in epochs. A value of -1 disables this feature. Useful for saving interim models during long training sessions.
              device=None, # 	Specifies the computational device(s) for training: a single GPU (device=0), multiple GPUs (device=0,1), CPU (device=cpu), or MPS for Apple silicon (device=mps).
              num_workers=8, # Number of worker threads for data loading (per RANK if Multi-GPU training). Influences the speed of data preprocessing and feeding into the model, especially useful in multi-GPU setups.
              output_results=None, # Name of the project directory where training outputs are saved. Allows for organized storage of different experiments.
              training_run_folder_name=None, # Name of the training run. Used for creating a subdirectory within the project folder, where training logs and outputs are stored.
              verbose=False, # Enables verbose output during training, providing detailed logs and progress updates. Useful for debugging and closely monitoring the training process.
              use_autocast=True, # Enables Automatic Mixed Precision (AMP) training, reducing memory usage and possibly speeding up training with minimal impact on accuracy.
              lr=0.01, # Initial learning rate (i.e. SGD=1E-2, Adam=1E-3) . Adjusting this value is crucial for the optimization process, influencing how rapidly model weights are updated.
              lrf=0.01, # Final learning rate as a fraction of the initial rate = (lr0 * lrf), used in conjunction with schedulers to adjust the learning rate over time.
              generate_plots=False # Generates and saves plots of training and validation metrics, as well as prediction examples, providing visual insights into model performance and learning progression.
              ):
        
        # Load a model
model = YOLO("path/to/last.pt")  # load a partially trained model   