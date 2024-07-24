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
from datetime import datetime
from collections import defaultdict
import pickle
import sys
# change this based on you file system, append "utils" folder where "keypoints2d_utils.py" is stored
sys.path.append('/content/Master-s-thesis---2D-Keypoints-extraction-experiments/utils')
from keypoints2d_utils import compute_MPJPE, visualize_keypoints2d

sys.path.append('/content/Master-s-thesis---2D-Keypoints-extraction-experiments/YOLO_Pose/ultralytics')
from ultralytics.models import YOLO

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
    def train(self, dataset_config, model_config_folder, model_config,
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
              fraction_sample_dtataset = 1, # Specifies the fraction of the dataset to use for training. Allows for training on a subset of the full dataset, useful for experiments or when resources are limited.
              lr=0.01, # Initial learning rate (i.e. SGD=1E-2, Adam=1E-3) . Adjusting this value is crucial for the optimization process, influencing how rapidly model weights are updated.
              lrf=0.01, # Final learning rate as a fraction of the initial rate = (lr0 * lrf), used in conjunction with schedulers to adjust the learning rate over time.
              generate_plots=False # Generates and saves plots of training and validation metrics, as well as prediction examples, providing visual insights into model performance and learning progression.
              ):
        
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        #     os.makedirs(os.path.join(output_folder, 'checkpoints'))
            
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        """ Model Loading """
        
        # https://docs.ultralytics.com/tasks/pose/ Models section
        model_scale = {
            'small': 's',
            'nano': 'n',
            'medium': 'm',
            'large': 'l',
            'extra-large': 'x',
        }

        # Load model: https://github.com/orgs/ultralytics/discussions/10211#discussioncomment-9182568
        model = YOLO(os.path.join(model_config_folder, f'yolov8{model_scale[model_config]}-pose.yaml'))

        # Train the model
        results = model.train(data=dataset_config, 
                              epochs=num_epochs, 
                              time=time,
                              batch=batch_size,
                              imgsz=image_size,
                              save=save,
                              save_period=checkpoint_step,
                              device=device,
                              workers=num_workers,
                              project=output_folder,
                              name=training_run_folder_name,
                              verbose=verbose,
                              amp=use_autocast,
                              fraction=fraction_sample_dtataset,
                              lr0=lr,
                              lrf=lrf,
                              plots=generate_plots,
                              val=False # DEBUG # disable validation
                              ) 

    # https://docs.ultralytics.com/modes/predict
    def evaluate(self, dataset_root, annot_root, model_path='', batch_size=1, seq='', output_results='', visualize=False):
        
        if not os.path.exists(output_results):
            os.makedirs(output_results)
        
        """ load dataset """
        
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        transform = transforms.Compose([transforms.ToTensor()])
        
        # testset = np.load(os.path.join(annot_root, f'images-test.npy'), allow_pickle=True)
        testset = Dataset(root=annot_root, load_set='test', transforms=transform)
        # testset = Subset(testset, list(range(386, 400))) # DEBUG
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=povsurgery_collate_fn) # DEBUG shuffle
        
        ### Load model
        torch.cuda.empty_cache()
        model = YOLO(model_path)  # pretrained YOLOv8n model
        print('YOLOv8-pose model is loaded')
        model.to(device)
        print(f'🟢 Model "{model_path.split(os.sep)[-2]}{os.sep}{model_path.split(os.sep)[-1]}" loaded')
        
        """ Evaluation """
        
        print(f'Results are saved in: \n{os.path.join(output_results, "results")}')
        if visualize:
            print(f'Visualizations are saved in: \n{os.path.join(output_results, "visualize")}')
        
        if seq != 'NO_SEQ':
            print(f'🟢 Searching for sequence "{seq}" to evaluate ...')
            
        mpjpe_results = []
        for i, (images, targets) in tqdm(enumerate(test_loader), total=len(test_loader), desc='Evaluation: '):
            # select specific sequence
            if seq != 'NO_SEQ':
                if not any(seq in t['path'] for t in targets):
                    continue

            images = images.to(device)

            results = model.predict(images, imgsz=images.shape[-2:], device=device) # #1
            # results = model(targets[0]['path'], imgsz=1920) # #2
    
            # Save results and compute MPJPE
            for res, t in zip(results, targets):
                sequence, frame = t['path'].split(os.sep)[-2:]
                frame = frame.split('.')[0]
                path_to_save_results = os.path.join(output_results, 'results', sequence)
                # re-formatting results
                res_dict = {
                    'boxes': res.boxes.xyxy,
                    'labels': res.boxes.cls,
                    'scores': res.boxes.conf,
                    'keypoints': res.keypoints.xy
                }
                print(res_dict['keypoints'].shape, res.feature_maps.shape) # DEBUG
                if not os.path.exists(path_to_save_results):
                    os.makedirs(path_to_save_results)
                with open(os.path.join(path_to_save_results, f'{frame}.pkl'), 'wb') as f:
                    pickle.dump(res_dict, f)
                # compute MPJPE               
                avg_mpjpe, best_mpjpe = compute_MPJPE(res_dict['keypoints'], t['keypoints'])
                mpjpe_results.append(avg_mpjpe)
                
                if visualize:
                    path_to_save_visual = os.path.join(output_results, 'visualize')
                    if not os.path.exists(path_to_save_visual):
                        os.makedirs(path_to_save_visual)
                    visualize_keypoints2d(t['path'], res_dict['keypoints'], 
                                          dataset_root=dataset_root, 
                                          output_results=path_to_save_visual)
        
        avg_rpjpe_result = np.ma.mean(np.ma.masked_array(mpjpe_results, np.isinf(mpjpe_results)))
        
        if seq != 'NO_SEQ': 
            print(f'🟢 Average MPJPE on sequence "{seq}": {avg_rpjpe_result:.4f}')
        else:
            print(f'🟢 Average MPJPE on Test set: {avg_rpjpe_result:.4f}')
        
        return avg_rpjpe_result

##### DEBUG #####

current_timestamp = datetime.now(pytz.timezone("Europe/Rome")).strftime("%d-%m-%Y_%H-%M")
training_run_folder_name = f'Training-DEBUG--{current_timestamp}'
output_folder = '/content/drive/MyDrive/Thesis/Keypoints2d_extraction/YOLO_Pose'

##### DEBUG #####


# YOLO_Pose().train(
#     dataset_config='/content/Master-s-thesis---2D-Keypoints-extraction-experiments/YOLO_Pose/utils/config_povsurgery.yaml',
#     model_config_folder='/content/Master-s-thesis---2D-Keypoints-extraction-experiments/YOLO_Pose/config',
#     model_config='small',
#     num_epochs=10,
#     batch_size=1,
#     image_size=1920,
#     save=True,
#     num_workers=2,
#     training_run_folder_name=training_run_folder_name,
#     verbose=True,
#     generate_plots=True,
#     lr=0.01,
#     lrf=0.01,
#     checkpoint_step=1,
#     output_folder='/content/drive/MyDrive/Thesis/Keypoints2d_extraction/YOLO_Pose',
#     use_autocast=False,
#     fraction_sample_dtataset=0.01 # DEBUG
# )

'''
YOLO_Pose().evaluate(
    dataset_root='/content/drive/MyDrive/Thesis/POV_Surgery_data',
    annot_root='/content/drive/MyDrive/Thesis/THOR-Net_based_work/povsurgery/object_False',
    model_path='/content/drive/MyDrive/Thesis/Keypoints2d_extraction/YOLO_Pose/Training-DEBUG--16-07-2024_09-46/weights/best.pt',
    batch_size=1,
    seq='NO_SEQ',
    output_results='/content/drive/MyDrive/Thesis/Keypoints2d_extraction/YOLO_Pose/Training-DEBUG--16-07-2024_09-46/output_results',
    visualize=False
)'''


##### DEBUG #####