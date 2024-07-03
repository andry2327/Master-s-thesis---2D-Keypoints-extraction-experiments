import torch
from torch.utils.data import Dataset
import numpy as np
import os
import cv2
from utils.keypoints2d_utils import get_keypoints2d_from_frame, get_bbox_from_frame


class Dataset(Dataset):
    def __init__(self, root, load_set='train', transforms=None):
        self.root = root
        self.transforms = transforms
        self.images = np.load(os.path.join(root, f'images-{load_set}.npy'), allow_pickle=True)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        img = img / 255.0 # normalize values
        keypoints2d = get_keypoints2d_from_frame(img_path, add_visibility=True)
        bbox = get_bbox_from_frame(img_path, list_as_out_format=True)
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