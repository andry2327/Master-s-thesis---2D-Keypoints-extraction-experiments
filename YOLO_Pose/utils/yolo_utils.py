import numpy as np
import pandas as pd
import os
import cv2
from keypoints2d_utils import get_keypoints2d_from_frame, get_bbox_from_frame
from tqdm import tqdm
import shutil

''' UTILS to create annotations in YOLO format, for POV-Surgery dataset '''

TRAIN_SEQUENCES = ['r_diskplacer_1', 'm_diskplacer_2', 'r_diskplacer_2', 'r_diskplacer_3', 'r_diskplacer_4', 
                   's_diskplacer_1', 'm_friem_1', 'm_friem_2', 'r_friem_1', 'r_friem_2', 's_friem_1', 's_friem_2', 
                   'm_scalpel_1', 'm_scalpel_2', 'r_scalpel_1', 'r_scalpel_2', 's_scalpel_1', 's_scalpel_2', 
                   'd_diskplacer_1', 'i_diskplacer_1', 'r_diskplacer_5', 'd_scalpel_1', 'r_scalpel_3', 
                   's_scalpel_3', 'd_diskplacer_2', 'i_diskplacer_2', 'r_diskplacer_6', 's_diskplacer_2', 
                   'd_friem_2', 'i_friem_2', 'r_friem_4', 's_friem_3', 'd_scalpel_2', 'i_scalpel_2', 'r_scalpel_4']

VAL_SEQUENCES = ['d_scalpel_1', 'r_scalpel_3', 'r_diskplacer_5', 's_friem_2', 's_scalpel_3']

TRAIN_SEQUENCES = list(set(TRAIN_SEQUENCES) - set(VAL_SEQUENCES))

# YOLO requires annotations in .txt, in a specific formatting:
# https://docs.ultralytics.com/it/datasets/pose/
def img_path_to_str_annot(img_path, img_dimensions=(1920, 1080)):
    
    OBJ_CLASS = 1 # 1: hand object
    BACKGROUND_CLASS = 0 # 1: background
    
    img_width, img_height = img_dimensions
    
    line_str = ''
    
    # img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    keypoints2d = get_keypoints2d_from_frame(img_path, add_visibility=False)
    bbox = get_bbox_from_frame(img_path, list_as_out_format=True)

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

def create_annotations(dataset_root, annot_path):
    
    INFO_SHEET_PATH = os.path.join(dataset_root, 'POV_Surgery_info.csv')
    INFO_SHEET = pd.read_csv(INFO_SHEET_PATH)
        
    TRAIN_ANNOT_PATH = os.path.join(annot_path, 'train')
    VAL_ANNOT_PATH = os.path.join(annot_path, 'val')
    
    IMAGES_PATH = os.path.join(dataset_root, 'color')
    
    # Create TRAINING annotations
    print('🟢 Creating TRAINING annotations ...')
    for i_seq, sequence in enumerate(TRAIN_SEQUENCES):
        
        frame_start = int(INFO_SHEET.loc[INFO_SHEET['OUT_seq'] == sequence, 'start_frame'].values)
        frame_end = int(INFO_SHEET.loc[INFO_SHEET['OUT_seq'] == sequence, 'end_frame'].values)
        
        pbar = tqdm(desc=f'Seq {i_seq+1}/{len(TRAIN_SEQUENCES)} - {sequence}: ', total=len(list(range(frame_start, frame_end+1))))
        for i in range(frame_start, frame_end+1):
            frame_id = str(i).zfill(5)
            annot_txt_name = f'{sequence}_{frame_id}.txt'
            if os.path.exists(os.path.join(TRAIN_ANNOT_PATH, annot_txt_name)):
                continue # skip annotations already created
            else:
                frame_path = os.path.join(IMAGES_PATH, sequence, f'{frame_id}.jpg')
                line_to_write = img_path_to_str_annot(frame_path)
                with open(os.path.join(TRAIN_ANNOT_PATH, annot_txt_name), 'w') as f:
                    f.write(line_to_write)
                pbar.update(1)
        pbar.close()
    print()
    # Create VALIDATION annotations
    print('🟢 Creating VALIDATION annotations ...')
    for i_seq, sequence in enumerate(VAL_SEQUENCES):
        
        frame_start = int(INFO_SHEET.loc[INFO_SHEET['OUT_seq'] == sequence, 'start_frame'].values)
        frame_end = int(INFO_SHEET.loc[INFO_SHEET['OUT_seq'] == sequence, 'end_frame'].values)
        
        pbar = tqdm(desc=f'Seq {i_seq+1}/{len(VAL_SEQUENCES)} - {sequence}: ', total=len(list(range(frame_start, frame_end+1))))
        for i in range(frame_start, frame_end+1):
            frame_id = str(i).zfill(5)
            annot_txt_name = f'{sequence}_{frame_id}.txt'
            if os.path.exists(os.path.join(VAL_ANNOT_PATH, annot_txt_name)):
                continue # skip annotations already created
            else:
                frame_path = os.path.join(IMAGES_PATH, sequence, f'{frame_id}.jpg')
                line_to_write = img_path_to_str_annot(frame_path)
                with open(os.path.join(VAL_ANNOT_PATH, annot_txt_name), 'w') as f:
                    f.write(line_to_write)
            pbar.update(1)
        pbar.close()
    print(f'\n🟢 Annotations saved in {annot_path}')
    
def copy_images(dataset_root, images_yolo_path):
 
    INFO_SHEET_PATH = os.path.join(dataset_root, 'POV_Surgery_info.csv')
    INFO_SHEET = pd.read_csv(INFO_SHEET_PATH)
        
    TRAIN_IMAGES_PATH = os.path.join(images_yolo_path, 'train')
    VAL_IMAGES_PATH = os.path.join(images_yolo_path, 'val')
    
    IMAGES_PATH = os.path.join(dataset_root, 'color')
    
    # Create TRAINING images shortcuts
    print('🟢 Creating TRAINING images shortcuts ...')
    for i_seq, sequence in enumerate(TRAIN_SEQUENCES):
        
        frame_start = int(INFO_SHEET.loc[INFO_SHEET['OUT_seq'] == sequence, 'start_frame'].values)
        frame_end = int(INFO_SHEET.loc[INFO_SHEET['OUT_seq'] == sequence, 'end_frame'].values)
        
        pbar = tqdm(desc=f'Copying seq {i_seq+1}/{len(TRAIN_SEQUENCES)} - {sequence}: ', total=len(list(range(frame_start, frame_end+1))))
        for i in range(frame_start, frame_end+1):
            frame_id = str(i).zfill(5)
            source_image_path = os.path.join(IMAGES_PATH, sequence, f'{frame_id}.jpg')
            frame_yolo_path = os.path.join(TRAIN_IMAGES_PATH, f'{sequence}_{frame_id}.jpg')
            if os.path.exists(frame_yolo_path):
                continue # skip if already created
            else:
                shutil.copy(source_image_path, frame_yolo_path)
            pbar.update(1)
        pbar.close()
    print()
    # Create VALIDATION images shortcuts
    print('🟢 Creating VALIDATION images shortcuts ...')
    for i_seq, sequence in enumerate(VAL_SEQUENCES):
        
        frame_start = int(INFO_SHEET.loc[INFO_SHEET['OUT_seq'] == sequence, 'start_frame'].values)
        frame_end = int(INFO_SHEET.loc[INFO_SHEET['OUT_seq'] == sequence, 'end_frame'].values)
        
        pbar = tqdm(desc=f'Copying seq {i_seq+1}/{len(VAL_SEQUENCES)} - {sequence}: ', total=len(list(range(frame_start, frame_end+1))))
        for i in range(frame_start, frame_end+1):
            frame_id = str(i).zfill(5)
            source_image_path = os.path.join(IMAGES_PATH, sequence, f'{frame_id}.jpg')
            frame_yolo_path = os.path.join(VAL_IMAGES_PATH, f'{sequence}_{frame_id}.jpg')
            if os.path.exists(frame_yolo_path):
                continue # skip if already created
            else:
                shutil.copy(source_image_path, frame_yolo_path)
            pbar.update(1)
        pbar.close()
    print(f'\n🟢 Images for YOLO format saved in {annot_path}')

##### DEBUG #####
img_path = '/content/drive/MyDrive/Thesis/POV_Surgery_data/color/d_diskplacer_1/00145.jpg'
dataset_root = '/content/drive/MyDrive/Thesis/POV_Surgery_data'
annot_path = '/content/drive/MyDrive/Thesis/POV_Surgery_data-YOLO_format/labels'
images_yolo_path = '/content/drive/MyDrive/Thesis/POV_Surgery_data-YOLO_format/images'

##### DEBUG #####

# res = img_path_to_str_annot(img_path)
# create_annotations(dataset_root, annot_path)
# copy_images(dataset_root, images_yolo_path)

##### DEBUG #####