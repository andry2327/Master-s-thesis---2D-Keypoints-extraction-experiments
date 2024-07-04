import torch
import torch.utils.data as data
import numpy as np
import os
import cv2
from utils.keypoints2d_utils import get_keypoints2d_from_frame, get_bbox_from_frame
class Dataset(data.Dataset):
    def __init__(self, root, model_name='', load_set='train', transforms=None):
        self.root = root
        self.transforms = transforms
        self.images = np.load(os.path.join(root, f'images-{load_set}.npy'), allow_pickle=True)
        self.model_name = 'KeypointRCNN'

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        keypoints2d = get_keypoints2d_from_frame(img_path, add_visibility=True)
        bbox = get_bbox_from_frame(img_path, list_as_out_format=True)
        
        # apply transformations to img and targets
        if self.model_name == 'KeypointRCNN':
            img = img / 255 # normalize values
            img_height, img_width, _ = img.shape
            keypoints2d[:, 0] = keypoints2d[:, 0] / pass
        
        if self.transforms is not None:
            img = self.transforms(img)
        
        target = {
            'boxes': bbox,
            'keypoints2d': keypoints2d,
            'labels': torch.tensor([1], dtype=torch.int64)
        }
        return img, target

    def __len__(self):
        return len(self.images)