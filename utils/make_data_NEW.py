from tqdm import tqdm
import os
import torch
import re
import numpy as np
import pickle
from keypoints2d_utils import get_bbox_from_frame, get_keypoints2d_from_frame, get_keypoints3d_from_frame, get_mesh3d_from_frame

ANNOT_PATH = '/home/aidara/Desktop/Thesis_Andrea/data/annotations_POV-Surgey_object_False_NEW_NEW'
if not os.path.exists(ANNOT_PATH):
    os.makedirs(ANNOT_PATH)

OLD_ANNOT_PATH = '/home/aidara/Desktop/Thesis_Andrea/data/annotations_POV-Surgey_object_False'

SPLITS = ['train', 'val', 'test']
EXTENSION = 'pkl'
IMG_SIZE = (1920, 1080)

# Function to check if a bounding box is valid
def check_valid_bbox(bbox):
    x1, y1, x2, y2 = tuple(bbox)
    return (x2 - x1) > 0 and (y2 - y1) > 0

# Load images
IMAGES = {}
name = 'images'
for split in SPLITS:
    file_path = os.path.join(OLD_ANNOT_PATH, f'{name}-{split}.npy')
    IMAGES[split] = np.load(file_path, allow_pickle=True)

# Remove bbox=None images and invalid boxes from dataset
# Bounding boxes

name = 'boxes'
invalid_boxes_count = 0
none_boxes_count = 0
error_count = 0

for split in SPLITS:
    invalid_boxes_count = 0
    none_boxes_count = 0
    error_count = 0
    images_split = IMAGES[split]
    num_images = len(images_split)
    boxes = []
    images = []

    pbar = tqdm(desc=f'{name}-{split}: ', total=num_images)

    for i, fp in enumerate(images_split):
        try:
            # Get bounding box from frame
            bbox = get_bbox_from_frame(fp, list_as_out_format=True)  # bbox = (x1, y1, x2, y2)

            # Check if bbox is valid (not [None, None, None, None])
            if bbox is not None and np.all(bbox != [None]*4):
                # Normalize the bounding box by image size
                bbox = np.array(bbox) / np.array([IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[0], IMG_SIZE[1]])

                # Validate the normalized bounding box
                if check_valid_bbox(bbox):
                    images.append(fp)
                    boxes.append(bbox)
                else:
                    invalid_boxes_count += 1
                    # print(f"Invalid bounding box for {fp}: {bbox}")
            else:
                none_boxes_count += 1
                # print(f"Invalid bounding box (None values) for {fp}: {bbox}")

        except Exception as e:
            error_count += 1
            print(f"Error processing {fp}: {e}")  # Log the error

        pbar.update(1)

    pbar.close()

    # Summary log
    print(f'Total invalid bounding boxes - {split} split: {invalid_boxes_count}')
    print(f'Total none bounding boxes - {split} split: {none_boxes_count}')
    print(f'Total processing errors - {split} split: {error_count}')

    # Save the valid bounding boxes and images
    boxes_path = os.path.join(ANNOT_PATH, f'{name}-{split}.{EXTENSION}')
    images_path = os.path.join(ANNOT_PATH, f'images-{split}.{EXTENSION}')

    # Save boxes
    boxes = np.array(boxes)
    with open(boxes_path, 'wb') as file:
        pickle.dump(boxes, file, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'✅ File "{name}-{split}.{EXTENSION}" saved, shape={boxes.shape}')

    # Save images
    images = np.array(images)
    with open(images_path, 'wb') as file:
        pickle.dump(images, file, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'✅ File "images-{split}.{EXTENSION}" saved, shape={images.shape}')
    
# exit()

# Load NEW images
IMAGES_NEW = {}
name = 'images'
for split in SPLITS:
    file_path = os.path.join(ANNOT_PATH, f'{name}-{split}.{EXTENSION}')
    with open(file_path, 'rb') as file:
        IMAGES_NEW[split] = pickle.load(file)



# -------------- Created .pkl files for Keypoints2d, Keypoints3d and Mesh3d --------------



for split in SPLITS:
    
    images_split = IMAGES_NEW[split]
    num_images = len(images_split)
    
    # Keypoints2d
    name = 'keypoints2d'
    kps2d_array = np.empty((num_images, 21, 3)) # (x, y, visibility)
    
    pbar = tqdm(desc=f'{name}-{split}: ', total=num_images)
    for i, fp in enumerate(images_split):
        kps2d = get_keypoints2d_from_frame(fp, add_visibility=True)  # returned kps2d = (y, x, visibility)
        kps2d = kps2d.astype(np.float32)
        kps2d[:, [0, 1]] = kps2d[:, [1, 0]] # apply swap x-y: > (y, x, visibility) -> (x, y, visibility)
        kps2d[:, 0] /=  IMG_SIZE[0]  # Normalize keypoints x values
        kps2d[:, 1] /=  IMG_SIZE[1]  # Normalize keypoints y values
        kps2d_array[i] = kps2d
        pbar.update(1)
    pbar.close()
    
    kps2d_path = os.path.join(ANNOT_PATH, f'{name}-{split}.pkl')
    # np.save(kps2d_path, kps2d_array)
    with open(kps2d_path, 'wb') as f:
        pickle.dump(kps2d_array, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush()
        f.close()
    print(f'✅ File "{name}-{split}.pkl" saved, shape={kps2d_array.shape}')
    
    # Keypoints3d
    name = 'keypoints3d'
    kps3d_array = np.empty((num_images, 21, 3)) # (x, y, z)
    
    pbar = tqdm(desc=f'{name}-{split}: ', total=num_images)
    for i, fp in enumerate(images_split):
        kps3d = get_keypoints3d_from_frame(fp, add_visibility=False) # returned kps2d = (y, x, z)
        kps3d_array[i] = kps3d
        pbar.update(1)
    pbar.close()
    
    kps3d_path = os.path.join(ANNOT_PATH, f'{name}-{split}.pkl')
    # np.save(kps2d_path, kps2d_array)
    with open(kps3d_path, 'wb') as f:
        pickle.dump(kps3d_array, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush()
        f.close()
    print(f'✅ File "{name}-{split}.pkl" saved, shape={kps3d_array.shape}')

    # Mesh3d
    name = 'mesh3d'
    mesh3d_array = np.empty((num_images, 778, 3)) 
    
    pbar = tqdm(desc=f'{name}-{split}: ', total=num_images)
    for i, fp in enumerate(images_split):
        mesh3d = get_mesh3d_from_frame(fp, add_visibility=False)
        mesh3d_array[i] = mesh3d
        pbar.update(1)
    pbar.close()
    
    mesh3d_path = os.path.join(ANNOT_PATH, f'{name}-{split}.pkl')
    # np.save(mesh3d_path, kps2d_array)
    with open(mesh3d_path, 'wb') as f:
        pickle.dump(mesh3d_array, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush()
        f.close()
    print(f'✅ File "{name}-{split}.pkl" saved, shape={mesh3d_array.shape}')

    # COnvert .npy files to .pkl
    '''what = ['images', 'boxes']
    for w in what:
        file_name = f'{w}-{split}'
        file_path = os.path.join(ANNOT_PATH, f'{file_name}.npy')
        data = np.load(file_path, allow_pickle=True)
        pkl_file_path = os.path.join(ANNOT_PATH, f'{file_name}.pkl')
        with open(pkl_file_path, 'wb') as f:
            pickle.dump(data, f)

        print(f"✅ Successfully converted {file_name}.npy to .pkl")'''
        


    
    