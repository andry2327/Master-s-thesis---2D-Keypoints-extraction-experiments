# https://pytorch.org/vision/main/models/generated/torchvision.models.detection.keypointrcnn_resnet50_fpn.html

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
import sys
# change this based on you file system, append "utils" folder where "keypoints2d_utils.py" is stored
sys.path.append('/content/Master-s-thesis---2D-Keypoints-extraction-experiments/utils') 
from keypoints2d_utils import compute_MPJPE, visualize_keypoints2d

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
              log_train_step=1, val_step=1, checkpoint_step=1, output_folder='/content/drive/MyDrive/Thesis/Keypoints2d_extraction',
              use_autocast=False):

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            os.makedirs(os.path.join(output_folder, 'checkpoints'))
        elif not os.path.exists(os.path.join(output_folder, 'checkpoints')):
            os.makedirs(os.path.join(output_folder, 'checkpoints'))
            
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

        """ load dataset """
        
        transform = transforms.Compose([transforms.ToTensor()])
        
        trainset = Dataset(root=annot_root, model_name=self.model_name, load_set='train', transforms=transform)
        # trainset = Subset(trainset, list(range(5))) # DEBUG
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, 
                                                collate_fn=povsurgery_collate_fn, pin_memory=True, persistent_workers=True)
        
        valset = Dataset(root=annot_root, model_name=self.model_name, load_set='val', transforms=transform)
        # valset = Subset(valset, list(range(2))) # DEBUG
        val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=True, num_workers=2, 
                                                collate_fn=povsurgery_collate_fn, pin_memory=True, persistent_workers=True)
        
        torch.cuda.empty_cache()
        model = keypointrcnn_resnet50_fpn(weights=self.weights, num_classes=self.num_classes, num_keypoints=self.num_keypoints)
        model.to(device)
        print('Keypoint RCNN model is loaded')
        
        if torch.cuda.is_available():
            model = nn.DataParallel(model, device_ids=[device])
            
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size//len(train_loader), gamma=lr_step_gamma)

        if use_autocast:
            scaler = torch.cuda.amp.GradScaler()

        # Training loop
        min_total_loss = float('inf')
        for epoch in range(num_epochs):
            model.train()
            pbar = tqdm(desc=f'Epoch {epoch+1} - train: ', total=len(train_loader))
            for i, (images, targets) in enumerate(train_loader):

                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items() if not isinstance(v, str)} for t in targets]
                
                # Forward pass
                if use_autocast:
                    with torch.cuda.amp.autocast():
                        loss_dict = model(images, targets)
                        losses = sum(loss for loss in loss_dict.values())
                    
                    # Backward pass and optimization
                    optimizer.zero_grad()
                    scaler.scale(losses).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Forward pass
                    loss_dict = model(images, targets)
                    
                    # Compute total loss
                    losses = sum(loss for loss in loss_dict.values())
                    
                    # Backward pass and optimization
                    optimizer.zero_grad()
                    losses.backward()
                    optimizer.step()
                
                if i % log_train_step == 0:
                    losses_str = ', '.join([f"{k}: {v.item():.4f}" for k, v in loss_dict.items()])
                    logger.info(f'[Epoch {epoch+1}/{num_epochs}, Processed data {i+1}/{len(train_loader)}] Losses: {losses_str}') # for log file
                pbar.update(1)
                
                # Clear unused variables
                del images, targets, loss_dict
                torch.cuda.empty_cache()
            pbar.close()
            
            # Save best and last model checkpoints
            if (epoch+1) % checkpoint_step == 0:
                folder_model = os.path.join(output_folder, 'checkpoints')
                torch.save(model.state_dict(), os.path.join(folder_model, f'model_last-{epoch+1}'))
                if epoch+1 > 1:
                    file_to_delete = [x for x in os.listdir(folder_model) if f'model_last-' in x and f'model_last-{epoch+1}' not in x][0]
                    try:
                        os.remove(os.path.join(folder_model, file_to_delete))
                    except:
                        pass
                if losses.data < min_total_loss: 
                    # Save best checkpoint
                    torch.save(model.state_dict(), os.path.join(folder_model, f'model_best-{epoch+1}'))
                    if epoch+1 > 1:
                        file_to_delete = [x for x in os.listdir(folder_model) if f'model_best-' in x and f'model_best-{epoch+1}' not in x][0]
                        try:
                            os.remove(os.path.join(folder_model, file_to_delete))
                        except:
                            pass
                    

            if (epoch+1) % val_step == 0:
                
                val_losses = {}
                pbar = tqdm(desc=f'Epoch {epoch+1} - val: ', total=len(val_loader))
                for i, (images, targets) in enumerate(val_loader):
                    
                    images = list(image.to(device) for image in images)
                    targets = [{k: v.to(device) for k, v in t.items() if not isinstance(v, str)} for t in targets]
                    
                    # Forward pass
                    if use_autocast:
                        with torch.cuda.amp.autocast():
                            loss_dict = model(images, targets)
                    else:
                        loss_dict = model(images, targets)
                    
                    # Accumulate losses
                    for k, v in loss_dict.items():
                        if k in val_losses.keys():
                            val_losses[k] += v.item()
                        else:
                            val_losses[k] = v.item()

                    pbar.update(1)
                pbar.close()
                
                for k in val_losses:
                    val_losses[k] /= len(val_loader)            
                losses_str = ', '.join([f"{k}: {v:.4f}" for k, v in val_losses.items()])
                line_str = f'Epoch {epoch+1}/{num_epochs} - val Losses: {losses_str}'
                logging.info(line_str)
                print(line_str)
            
            # Decay Learning Rate
            scheduler.step()       

        print("\n🟢 Training complete!")

    def evaluate(self, dataset_root, annot_root='', model_path='', batch_size=1, seq='', output_results='', visualize=False):
        
        if not os.path.exists(output_results):
            os.makedirs(output_results)
        
        """ load dataset """
        
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        transform = transforms.Compose([transforms.ToTensor()])
        
        testset = Dataset(root=annot_root, model_name=self.model_name, load_set='test', transforms=transform)
        # testset = Subset(testset, list(range(100))) # DEBUG
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=povsurgery_collate_fn)
        
        ### Load model
        torch.cuda.empty_cache()
        model = keypointrcnn_resnet50_fpn(weights=self.weights, num_classes=self.num_classes, num_keypoints=self.num_keypoints)
        print('Keypoint RCNN model is loaded')
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model = model.eval()
        print(f'🟢 Model "{model_path.split(os.sep)[-2]}{os.sep}{model_path.split(os.sep)[-1]}" loaded')
        
        """ Evaluation """
        
        print(f'Results are saved in: \n{os.path.join(output_results, "results")}')
        if visualize:
            print(f'Visualizations are saved in: \n{os.path.join(output_results, "visualize")}')
        
        if seq != 'NO_SEQ':
            print(f'🟢 Searching for sequence "{seq}" to evaluate ...')
            
        # results_dict = {} # DEBUG
        mpjpe_results = []
                
        for i, (images, targets) in tqdm(enumerate(test_loader), total=len(test_loader), desc='Evaluation: '):
            
            # select specific sequence
            if seq != 'NO_SEQ':
                if not any(seq in t['path'] for t in targets):
                    continue
            
            images = list(image.to(device) for image in images)
            
            results = model(images)
             
            # Save results and compute MPJPE
            for res, t in zip(results, targets):
                # results_dict[t['path']] = res # DEBUG
                sequence, frame = t['path'].split(os.sep)[-2:]
                frame = frame.split('.')[0]
                path_to_save_results = os.path.join(output_results, 'results', sequence)
                if not os.path.exists(path_to_save_results):
                    os.makedirs(path_to_save_results)
                with open(os.path.join(path_to_save_results, f'{frame}.pkl'), 'wb') as f:
                    pickle.dump(res, f)
                # compute MPJPE               
                print(f'pred.shape = {res["keypoints"].shape}: ', end = '') # DEBUG
                avg_mpjpe, best_mpjpe = compute_MPJPE(res['keypoints'], t['keypoints'])
                print(f'avg_mpjpe {avg_mpjpe}, best_mpjpe {best_mpjpe}') # DEBUG
                mpjpe_results.append(avg_mpjpe)
                
                if visualize:
                    path_to_save_visual = os.path.join(output_results, 'visualize')
                    if not os.path.exists(path_to_save_visual):
                        os.makedirs(path_to_save_visual)
                    visualize_keypoints2d(t['path'], res['keypoints'], 
                                          dataset_root=dataset_root, 
                                          output_results=path_to_save_visual)
        
        avg_rpjpe_result = np.ma.mean(np.ma.masked_array(mpjpe_results, np.isinf(mpjpe_results)))
        
        if seq != 'NO_SEQ': 
            print(f'🟢 Average MPJPE on sequence "{seq}": {avg_rpjpe_result:.4f}')
        else:
            print(f'🟢 Average MPJPE on Test set: {avg_rpjpe_result:.4f}')
        
        return avg_rpjpe_result
        

##### DEBUG #####
'''
current_timestamp = datetime.datetime.now(pytz.timezone("Europe/Rome")).strftime("%d-%m-%Y_%H-%M")
folder = f'Training-DEBUG--{current_timestamp}'
output_folder = f'/content/drive/MyDrive/Thesis/Keypoints2d_extraction/KeypointRCNN/{folder}'
'''
##### DEBUG #####

'''KeypointRCNN().train(
    dataset_root='/content/drive/MyDrive/Thesis/POV_Surgery_data',
    annot_root='/content/drive/MyDrive/Thesis/THOR-Net_based_work/povsurgery/object_False',
    num_epochs=10, batch_size=1, lr=2e-5, step_size=120000, lr_step_gamma=0.1, log_train_step=1, val_step=1,
    checkpoint_step=1,
    output_folder=output_folder
)'''
'''
KeypointRCNN().evaluate(
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

