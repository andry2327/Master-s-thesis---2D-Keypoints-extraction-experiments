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
        
    @capture_arguments
    def train(self, dataset_root, annot_root, num_epochs=10, batch_size=1, lr=0.02, step_size=120000, lr_step_gamma=0.1,
              log_train_step=1, val_step=1, checkpoint_step=1, output_folder='/content/drive/MyDrive/Thesis/Keypoints2d_extraction',
              use_autocast=False):
        pass