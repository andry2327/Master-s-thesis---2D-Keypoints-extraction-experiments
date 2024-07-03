# https://pytorch.org/vision/main/models/generated/torchvision.models.detection.keypointrcnn_resnet50_fpn.html

import torch
import torchvision
from torchvision.models.detection.keypoint_rcnn import keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights
import os
import numpy as np

model = keypointrcnn_resnet50_fpn()
model.eval()
x = [torch.rand(3, 3, 4)]
predictions = model(x)
print(predictions)

class KeypointRCNN:
    def __init__(self, weights=None) -> None:
        self.weights = weights
        self.num_classes = 2
        self.num_keypoints = 21
        