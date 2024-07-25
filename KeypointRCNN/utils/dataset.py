import torch
import torch.utils.data as data
import numpy as np
import os
import cv2
import sys
import re
# change this based on you file system, append "utils" folder where "keypoints2d_utils.py" is stored
sys.path.append('/content/Master-s-thesis---2D-Keypoints-extraction-experiments/utils')
from keypoints2d_utils import get_keypoints2d_from_frame, get_bbox_from_frame

def check_valid_bbox(bbox):
    x1, y1, x2, y2 = tuple(bbox)
    if ((x2 - x1) <= 0) or ((y2 - y1) <= 0):
        return False
    return True

class Dataset(data.Dataset):
    def __init__(self, root, model_name='', load_set='train', transforms=None):
        self.root = root
        self.transforms = transforms
        self.images = np.load(os.path.join(root, f'images-{load_set}.npy'), allow_pickle=True)
        self.model_name = 'KeypointRCNN'

    def __getitem__(self, idx):
        img_path = self.images[idx]
        pattern = r' \(\d+\)' # fix 
        img_path = re.sub(pattern, '', img_path) # fix 
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        keypoints2d = get_keypoints2d_from_frame(img_path, add_visibility=True) # format [y, x, visibility]
        bbox = get_bbox_from_frame(img_path, list_as_out_format=True)
        labels = torch.tensor([1], dtype=torch.int64)
        
        # apply transformations to img and targets
        if self.model_name == 'KeypointRCNN':
            img = img / 255 # normalize values     
            img_height, img_width, _ = img.shape
            if bbox == [None]*4 or (not check_valid_bbox(bbox)):
                # hand not visible/present -> empty targets
                bbox = torch.empty((0, 4), dtype=torch.float)
                keypoints2d = torch.empty((0, 21, 3), dtype=torch.float)
                labels = torch.tensor([0], dtype=torch.int64)
            else:
                keypoints2d = torch.tensor(keypoints2d, dtype=torch.float).unsqueeze(0)
                bbox = torch.tensor(bbox, dtype=torch.float).unsqueeze(0)
                # switch x, y position and normalize
                keypoints2d[:, :, [0, 1]] = keypoints2d[:, :, [1, 0]]
                keypoints2d[:, :, 0] = keypoints2d[:, :, 0] #/ img_width
                keypoints2d[:, :, 1] = keypoints2d[:, :, 1] #/ img_height 
        
        if self.transforms is not None:
            img = self.transforms(img) # to tensor, from shape (H, W, C) -> (C, H, W)
        img = img.to(torch.float)
        
        target = {
            'path': img_path,
            'boxes': bbox,
            'keypoints': keypoints2d,
            'labels': labels
        }
        return img, target

    def __len__(self):
        return len(self.images)