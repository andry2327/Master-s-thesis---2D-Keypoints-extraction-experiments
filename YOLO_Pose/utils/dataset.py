import torch
import torch.utils.data as data
import numpy as np
import os
import cv2

class Dataset(data.Dataset):
    def __init__(self, root, load_set='test', transforms=None):
        self.root = root
        self.transforms = transforms
        self.images = np.load(os.path.join(root, f'images-{load_set}.npy'), allow_pickle=True)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = cv2.imread(img_path)
        keypoints2d = get_keypoints2d_from_frame(img_path, add_visibility=True) # format [y, x, visibility]
        bbox = get_bbox_from_frame(img_path, list_as_out_format=True)
        labels = torch.tensor([1], dtype=torch.int64)
        
        # apply transformations to img and targets
        if bbox == [None]*4:
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