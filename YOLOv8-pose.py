import numpy as np
import os
import cv2
from utils.keypoints2d_utils import get_keypoints2d_from_frame, get_bbox_from_frame

''' UTILS to create annotations in YOLO format, for POV-Surgery dataset'''

# YOLO requires annotations in .txt, in a specific formatting:
# https://docs.ultralytics.com/it/datasets/pose/
def img_path_to_txt_annot(img_path, img_dimensions=(1920, 1080)):
    
    OBJ_CLASS = 1 # 1: hand object
    BACKGROUND_CLASS = 0 # 1: background
    
    img_width, img_height = img_dimensions
    
    sequence, frame = img_path.split(os.sep)[-2:]
    frame = frame.split('.')[0]
    
    line_str = ''
    
    # img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    keypoints2d = get_keypoints2d_from_frame(img_path, add_visibility=False)
    bbox = get_bbox_from_frame(img_path, list_as_out_format=False)

    if bbox == [None]*4:
        line_str += f'{BACKGROUND_CLASS}'
    else:
        x_center, y_center = (bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2
        x_center, y_center = x_center/img_width, y_center/img_height
        
        obj_width, obj_height = abs(bbox[0]-bbox[2])/img_width, abs(bbox[1]-bbox[3])/img_height 
        
        line_str += f'{OBJ_CLASS} {x_center} {y_center} {obj_width} {obj_height}'
        
        # Add 2D keypoints
        for py, px in keypoints2d:
            line_str += f' {px/img_width} {py/img_height}'

    return line_str
    
    

##### DEBUG #####
img_path = '/content/drive/MyDrive/Thesis/POV_Surgery_data/color/d_diskplacer_1/00145.jpg'
##### DEBUG #####

res = img_path_to_txt_annot(img_path)
print(res)
##### DEBUG #####