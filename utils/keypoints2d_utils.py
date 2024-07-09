import pyrender
import numpy as np
import torch
import cv2
from os.path import join, dirname
import os
from tqdm import tqdm
import trimesh
import matplotlib.pyplot as plt
import pandas
import numpy as np
import pickle
import mano
import pickle
from scipy.spatial.transform import Rotation as R
import scipy.io as sio

device = 'cpu' # cuda, cpu
batch_size = 1
TPID = [744, 320, 443, 554, 671]
###################################################################################
DATASET_ROOT = '/content/drive/MyDrive/Thesis/POV_Surgery_data'
INFO_SHEET_PATH = join(DATASET_ROOT, 'POV_Surgery_info.csv')
MANO_PATH = '/content/drive/MyDrive/Thesis/mano_v1_2/models/MANO_RIGHT.pkl'
REPRO_DIR = '/content/drive/MyDrive/Thesis/Keypoints2d_extraction/repro_dir'
###################################################################################
info_sheet = pandas.read_csv(INFO_SHEET_PATH) 
SCALPE_OFFSET = [0.04805371, 0 ,0]
DISKPLACER_OFFSET = [0, 0.34612157 ,0]
FRIEM_OFFSET = [0, 0.1145 ,0]

with torch.no_grad():
    rh_mano = mano.load(model_path=MANO_PATH,
                        model_type='mano',
                        num_pca_comps=45,
                        batch_size=1,
                        emissiveFactor=1,
                        flat_hand_mean=True).to(device)
    
rh_faces = torch.from_numpy(rh_mano.faces.astype(np.int32)).view(1, -1, 3).to(device)

if not os.path.exists(REPRO_DIR):
    os.makedirs(REPRO_DIR)

def vis_keypoints_with_skeleton(image, kp, fname='/home/rui/Downloads/inftest0.png'):
    color = np.ones(shape=(1080, 1920, 3), dtype=np.int16)
    color[:, :, 0] = image[:, :, 0]
    color[:, :, 1] = image[:, :, 1]
    color[:, :, 2] = image[:, :, 2]
    color[kp[:,0], kp[:,1]] = 100
    img = color
    # kp = kp[[0]]
    # kp[:, 0] = 512 - kp[:, 0]
    # kp[:, 1] = 512 - kp[:, 1]
    index_p2d = np.ones((21, 3))
    index_p2d[:, 0] = kp[:, 1]
    index_p2d[:, 1] = kp[:, 0]
    kps = index_p2d.T
    kp_thresh = 0.4
    alpha = 1
    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    kps_lines = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 17],
        [0, 13],
        [13, 14],
        [14, 15],
        [15, 16],
        [0, 4],
        [4, 5],
        [5, 6],
        [6, 18],
        [0, 10],
        [10, 11],
        [11, 12],
        [12, 19],
        [0, 7],
        [7, 8],
        [8, 9],
        [9, 20]
    ]
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)

    # Draw the keypoints.
    for l in range(len(kps_lines)):
        i1 = kps_lines[l][0]
        i2 = kps_lines[l][1]
        p1 = kps[0, i1].astype(np.int32), kps[1, i1].astype(np.int32)
        p2 = kps[0, i2].astype(np.int32), kps[1, i2].astype(np.int32)
        if kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh:
            cv2.line(
                kp_mask, p1, p2,
                color=colors[l], thickness=4, lineType=cv2.LINE_AA)
        if kps[2, i1] > kp_thresh:
            cv2.circle(
                kp_mask, p1,
                radius=6, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
        if kps[2, i2] > kp_thresh:
            cv2.circle(
                kp_mask, p2,
                radius=6, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)

    # Blend the keypoints.
    o_img = cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)
    # cv2.imwrite(fname, o_img)
    
    # if not os.path.exists(join(REPRO_DIR, seq_name)):
    #     os.makedirs(join(REPRO_DIR, seq_name))
    # img_fn_cropped = join(REPRO_DIR, seq_name, f'{frame}_kps2d_visualize.jpg')
    # cv2.imwrite(img_fn_cropped, color)
    
    return o_img
    # def vis_kp(self, image, kp, fname ='/home/rui/Downloads/inftest0.png'):

    #     # cv2.circle(img = color, (index_p2d[:, 0], index_p2d[:, 1]), 5, (0, 255, 0), -1)
    #     for temp_i in range(len(index_p2d[:, 0] )):
    #         cv2.circle(img=color, center=(index_p2d[temp_i,0], index_p2d[temp_i,1]), radius=5, color=(0, 255, 0), thickness=-1)

    #     # color[index_p2d[:, 1], index_p2d[:, 0], :] = 244

    #     cv2.imwrite(fname, color)
    
def get_keypoints2d_from_frame(frame_path='', add_visibility=False):

    # Extract sequence name and frame number
    seq_name, frame = os.path.split(frame_path)[-2:]
    frame = os.path.splitext(frame)[0]

    # Load annotations
    annot_path = join(DATASET_ROOT, 'annotation', seq_name, f'{frame}.pkl')
    if not os.path.exists(annot_path):
        annot_path = annot_path.replace('color', 'annotation')
        assert os.path.exists(annot_path), f"Path {annot_path} does not exist."
        
    with open(annot_path, 'rb') as f:
        frame_anno = pickle.load(f)

    # Prepare hand dictionary for the model
    hand_dict = {k: torch.tensor(v, dtype=torch.float32, device=device) for k, v in frame_anno['mano'].items()}
    
    # Get hand vertices and keypoints
    this_hand = rh_mano(**hand_dict)
    hand_kp_base = this_hand.joints.squeeze(0).cpu().numpy()
    hand_finger = this_hand.vertices.squeeze(0).cpu().numpy()[TPID]
    hand_kp = np.vstack((hand_kp_base, hand_finger))
    hand_kp = hand_kp @ frame_anno['grab2world_R'] + frame_anno['grab2world_T']

    # Apply camera transformation
    cam_pose = np.eye(4)
    cam_pose[:3, :3], cam_pose[:3, 3] = frame_anno['cam_rot'], frame_anno['cam_transl']
    inv_cam_pose = np.linalg.inv(cam_pose)
    hand_kp = hand_kp @ inv_cam_pose[:3, :3].T + inv_cam_pose[:3, 3]

    # Change coordinates and project to 2D
    coord_change_mat = np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32)
    K = np.array([[1198.4395, 0., 960.], [0., 1198.4395, 175.2], [0., 0., 1.]])
    temp_1 = (hand_kp @ coord_change_mat.T) @ K.T
    p2d = temp_1[:, :2] / temp_1[:, [2]]

    # Clip coordinates to image dimensions
    new_p2d = np.zeros_like(p2d)
    new_p2d[:, 1] = np.clip(p2d[:, 0], 0, 1919)  # Image width - 1
    new_p2d[:, 0] = np.clip(p2d[:, 1], 0, 1079)  # Image height - 1

    if add_visibility:
        # add visibility flag
        visibility = np.ones((new_p2d.shape[0], 1))
        new_p2d = np.hstack((new_p2d, visibility))
    
    return new_p2d.astype(np.int32)

def get_bbox_from_frame(frame_path='', list_as_out_format=False):
    
    HAND_COLOR = np.array([100, 100, 100]) # Define the color for the hand mask in POV-Surgery
    
    # boxes (FloatTensor[N, 4]): the predicted boxes in [x1, y1, x2, y2] format, 
    # with 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H. ( (x1, y1) top-left corner, (x2, y2) bottom-right corner)
    bbox = (None, None, None, None) 
    
    # Extract sequence name and frame number
    seq_name, frame = os.path.split(frame_path)[-2:]
    frame = os.path.splitext(frame)[0]
    
    # Load mask
    mask_path = join(DATASET_ROOT, 'mask', seq_name, f'{frame}.png')
    if not os.path.exists(mask_path):
        mask_path = mask_path.replace('color', 'mask')
        assert os.path.exists(mask_path), f"Path {mask_path} does not exist."
    
    mask_image = cv2.imread(mask_path)
    mask = np.all(mask_image == HAND_COLOR, axis=-1) # Find the pixels that match the hand color
    coords = np.argwhere(mask) # Get the coordinates of the matching pixels

    if coords.size == 0:
        pass
        # print(f'No hand mask found in the image "{join(frame_path.split(os.sep)[-2], frame_path.split(os.sep)[-1])}".') # DEBUG
    else:
        # Get the bounding box coordinates
        y1, x1 = coords.min(axis=0)
        y2, x2 = coords.max(axis=0)
        bbox = (x1, y1, x2, y2)
        # # Print the bounding box coordinates (top left and bottom right)
        # print(f"Bounding box: (x1, y1, x2, y2) = ({x1}, {y1}, {x2}, {y2})")

        # # Optionally, draw the bounding box on the image for visualization
        # output_image = cv2.rectangle(mask_image.copy(), (x1, y1), (x2, y2), (0, 255, 0), 2)
        # # Draw points and text indicating the coordinates
        # cv2.circle(output_image, (x1, y1), 5, (0, 0, 255), -1)  # Draw a red circle at (x1, y1)
        # cv2.circle(output_image, (x2, y2), 5, (0, 0, 255), -1)  # Draw a red circle at (x2, y2)
        # cv2.putText(output_image, f'({x1}, {y1})', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        # cv2.putText(output_image, f'({x2}, {y2})', (x2, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        # cv2.imwrite('/content/bbox_on_image.png', output_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    
    return list(bbox) if list_as_out_format else bbox
    
def visualize_keypoints2d(frame_path, keypoints2d, dataset_root='', output_results=''):
    
    seq_name, frame = frame_path.split(os.sep)[-2:]
    frame = os.path.splitext(frame)[0]  
    image = cv2.imread(join(dataset_root, 'color', seq_name, f'{frame}.jpg'))
    
    img_o = vis_keypoints_with_skeleton(image, keypoints2d)
    
    if not os.path.exists(join(output_results, seq_name)):
        os.makedirs(join(output_results, seq_name))
    img_path = join(output_results, seq_name, f'{frame}_kps2d_visual.jpg')
    cv2.imwrite(img_path, img_o)
    
    return img_o

# frame_path = '/content/drive/MyDrive/Thesis/POV_Surgery_data/color/d_diskplacer_1/00145.jpg'
# keypoints2d = get_keypoints2d_from_frame(frame_path=frame_path)
# visualize_keypoints2d(frame_path, keypoints2d)

# print(get_bbox_from_frame(frame_path=frame_path))

# Compute Mean Per Joint Position Error (MPJPE)
# MPJPE measures the average Euclidean distance between predicted joint locations 
# and their corresponding ground truth positions. It provides a quantitative measure 
# of how far off, on average, the predicted keypoints are from their true locations.
def compute_MPJPE(pred, target):
    """
    pred: tensor of shape (N, 21, 3) - N sets of predicted keypoints
    target: tensor of shape (21, 3) - single set of ground truth keypoints
    """
    
    if target.numel() == 0: # if target is empty
        return float('inf'), float('inf') 
        
    N = pred.shape[0]
    
    # Expand target to match pred shape
    target_expanded = target.expand(N, -1, -1)
    
    distances = torch.norm(pred[:, :, :2] - target_expanded[:, :, :2], dim=2)
    avg_mpjpe_per_pred = distances.mean(dim=1)
    best_mpjpe, _ = avg_mpjpe_per_pred.min(dim=0)
    avg_mpjpe = avg_mpjpe_per_pred.mean()

    return avg_mpjpe.detach().cpu().numpy(), best_mpjpe.detach().cpu().numpy()