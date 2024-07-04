# https://pytorch.org/vision/main/models/generated/torchvision.models.detection.keypointrcnn_resnet50_fpn.html

import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models.detection.keypoint_rcnn import keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights
import os
import numpy as np
from utils.dataset import Dataset
from tqdm import tqdm
import pytz
import logging
import datetime

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

class KeypointRCNN:
    
    def __init__(self, weights=None) -> None:
        self.weights = weights
        self.num_classes = 2
        self.num_keypoints = 21
        self.model_name = 'KeypointRCNN'
        
    @capture_arguments
    def train(self, dataset_root, annot_root, num_epochs=10, batch_size=1, lr=0.02, step_size=120000, lr_step_gamma=0.1,
              log_train_step=1, val_step=1, output_folder='/content/drive/MyDrive/Thesis/Keypoints2d_extraction'):

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
    
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
        logging.info('--'*50) 
        print('\n')

        print(f'🟢 Logging info in "{filename_log}"')
    
        """ load dataset """
        
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        transform = transforms.Compose([transforms.ToTensor()])
        
        trainset = Dataset(root=annot_root, model_name=self.model_name, load_set='train', transforms=transform)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=povsurgery_collate_fn)
        
        valset = Dataset(root=annot_root, model_name=self.model_name, load_set='val', transforms=transform)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=povsurgery_collate_fn)
        
        torch.cuda.empty_cache()
        model = keypointrcnn_resnet50_fpn(weights=self.weights, num_classes=self.num_classes, num_keypoints=self.num_keypoints)
        model.to(device)
        print('Keypoint RCNN model is loaded')
        
        if torch.cuda.is_available():
            model = nn.DataParallel(model, device_ids=[device])
            
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size//len(train_loader), gamma=lr_step_gamma)
        # scheduler.last_epoch = start
        
        # Training loop
        for epoch in range(num_epochs):
            model.train()
            pbar = tqdm(desc=f'Epoch {epoch+1} - train: ', total=len(train_loader))
            for i, (images, targets) in enumerate(train_loader):
                
                images = list(image.to(device) for image in images)
                # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
                # Forward pass
                loss_dict = model(images, targets)
                
                # Compute total loss
                losses = sum(loss for loss in loss_dict.values())
                
                # Backward pass and optimization
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
                
                if i % log_train_step == 0:
                    logging.info(f'[Epoch {epoch+1}/{num_epochs}, Processed data {i+1}/{len(train_loader)}] Loss: {losses.item()}') # for log file
                pbar.update(1)

            if (epoch+1) % val_step == 0:
                
                pbar = tqdm(desc=f'Epoch {epoch+1} - val: ', total=len(val_loader))
                for i, (images, targets) in enumerate(val_loader):
                    
                    images = list(image.to(device) for image in images)
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                    
                    # Forward pass
                    loss_dict = model(images, targets)
                    
                    # Compute total loss
                    val_losses = sum(loss for loss in loss_dict.values())
                    
                    line_str = f'Epoch {epoch+1}/{num_epochs} - val Loss: {val_losses.item()}'
                    logging.info(line_str)
                    print(line_str)
                    pbar.update(1)
                                 
            # Decay Learning Rate
            scheduler.step()
                    
        print("Training complete!")

##### DEBUG #####
current_timestamp = datetime.datetime.now(pytz.timezone("Europe/Rome")).strftime("%d-%m-%Y_%H-%M")
folder = f'Training-DEBUG--{current_timestamp}'
output_folder = f'/content/drive/MyDrive/Thesis/Keypoints2d_extraction/KeypointRCNN/{folder}'
##### DEBUG #####

KeypointRCNN().train(
    dataset_root='/content/drive/MyDrive/Thesis/POV_Surgery_data',
    annot_root='/content/drive/MyDrive/Thesis/THOR-Net_based_work/povsurgery/object_False',
    num_epochs=10, batch_size=2, lr=0.02, step_size=120000, lr_step_gamma=0.1, log_train_step=1, val_step=1,
    output_folder=output_folder
)

##### DEBUG #####

