import pyrender
import numpy as np
import torch
import cv2
from os.path import join, dirname
import os
from tqdm import tqdm
import trimesh
import matplotlib.pyplot as plt
import mano
import pandas
import numpy as np
import pickle
from scipy.spatial.transform import Rotation as R
import scipy.io as sio
import re

device = 'cpu' # cuda, cpu
batch_size = 1
TPID = [744, 320, 443, 554, 671]
###################################################################################
DATASET_ROOT = '/home/aidara/Desktop/Thesis_Andrea/data/POV_Surgery_data'
INFO_SHEET_PATH = join(DATASET_ROOT, 'POV_Surgery_info.csv')
MANO_PATH = '/home/aidara/Desktop/Thesis_Andrea/data/mano_v1_2/models/MANO_RIGHT.pkl'
REPRO_DIR = '/home/aidara/Desktop/Thesis_Andrea/Keypoint_2D_extraction_Experiemnts/repro_dir'
###################################################################################
info_sheet = pandas.read_csv(INFO_SHEET_PATH) 
SCALPE_OFFSET = [0.04805371, 0 ,0]
DISKPLACER_OFFSET = [0, 0.34612157 ,0]
FRIEM_OFFSET = [0, 0.1145 ,0]

''' ---------------------- PARAMETERS ---------------------- '''

RED_DOMINANCE_THRESHOLD = 1.15 # Red should be RED_DOMINANCE_THRESHOLD times greater than green and blue
RED_MINIMUM_VALUE = 100       # Minimum value for red to be considered a "red" shade

''' -------------------------------------------------------- '''

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
    
def get_keypoints2d_from_frame(frame_path='', add_visibility=False, dataset_root_real=None):

    # Extract sequence name and frame number
    seq_name, frame = os.path.split(frame_path)[-2:]
    frame = os.path.splitext(frame)[0]

    # Load annotations
    dataset_root_real = dataset_root_real if dataset_root_real else DATASET_ROOT
    annot_path = os.path.join(dataset_root_real, 'annotation', seq_name, f'{frame}.pkl')

    if not os.path.exists(annot_path):
        annot_path = annot_path.replace('color', 'annotation')
        assert os.path.exists(annot_path), f"Path {annot_path} does not exist."
    
    pattern = r' \(\d+\)' # fix 
    annot_path = re.sub(pattern, '', annot_path) # fix
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
    # print(hand_kp @ coord_change_mat.T) # DEBUG
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
    
    return new_p2d.astype(np.int32) # IT RETURNS (y, x) coordinates

def get_keypoints3d_from_frame(frame_path='', add_visibility=False, dataset_root_real=None):

    # Extract sequence name and frame number
    seq_name, frame = os.path.split(frame_path)[-2:]
    frame = os.path.splitext(frame)[0]

    # Load annotations
    dataset_root_real = dataset_root_real if dataset_root_real else DATASET_ROOT
    annot_path = os.path.join(dataset_root_real, 'annotation', seq_name, f'{frame}.pkl')

    if not os.path.exists(annot_path):
        annot_path = annot_path.replace('color', 'annotation')
        assert os.path.exists(annot_path), f"Path {annot_path} does not exist."
    
    pattern = r' \(\d+\)' # fix 
    annot_path = re.sub(pattern, '', annot_path) # fix
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
    temp_1 = (hand_kp @ coord_change_mat.T)

    new_p2d = temp_1
    if add_visibility:
        # add visibility flag
        visibility = np.ones((temp_1.shape[0], 1))
        new_p2d = np.hstack((temp_1, visibility))
    
    return new_p2d # shape = (21, 3)

def visualize_keypoints3d(kps3d, add_visibility=False):
    K = np.array([[1198.4395, 0., 960.], [0., 1198.4395, 175.2], [0., 0., 1.]])
    
    kps3d = kps3d @ K.T
    p2d = kps3d[:, :2] / kps3d[:, [2]]

    # Clip coordinates to image dimensions
    new_p2d = np.zeros_like(p2d)
    new_p2d[:, 1] = np.clip(p2d[:, 0], 0, 1919)  # Image width - 1
    new_p2d[:, 0] = np.clip(p2d[:, 1], 0, 1079)  # Image height - 1

    if add_visibility:
        # add visibility flag
        visibility = np.ones((new_p2d.shape[0], 1))
        new_p2d = np.hstack((new_p2d, visibility))
    
    return new_p2d.astype(np.int32) # IT RETURNS (y, x) coordinates


def get_mesh3d_from_frame(frame_path='', add_visibility=False, dataset_root_real=None):

    # Extract sequence name and frame number
    seq_name, frame = os.path.split(frame_path)[-2:]
    frame = os.path.splitext(frame)[0]

    # Load annotations
    dataset_root_real = dataset_root_real if dataset_root_real else DATASET_ROOT
    annot_path = os.path.join(dataset_root_real, 'annotation', seq_name, f'{frame}.pkl')

    if not os.path.exists(annot_path):
        annot_path = annot_path.replace('color', 'annotation')
        assert os.path.exists(annot_path), f"Path {annot_path} does not exist."
    
    pattern = r' \(\d+\)' # fix 
    annot_path = re.sub(pattern, '', annot_path) # fix
    with open(annot_path, 'rb') as f:
        frame_anno = pickle.load(f)

    # Prepare hand dictionary for the model
    hand_dict = {k: torch.tensor(v, dtype=torch.float32, device=device) for k, v in frame_anno['mano'].items()}
    
    # Get hand vertices and keypoints
    this_hand = rh_mano(**hand_dict)
    hand_vertices = this_hand.vertices.squeeze(0).cpu().numpy()

    # Apply the global transformations (grab2world_R and grab2world_T)
    hand_vertices = hand_vertices @ frame_anno['grab2world_R'] + frame_anno['grab2world_T']
    
    camera_pose = np.eye(4)
    camera_pose[:3, 3] = frame_anno['cam_transl']
    camera_pose[:3, :3] = frame_anno['cam_rot']
    
    hand_vertices = hand_vertices @ np.linalg.inv(camera_pose)[:3,:3].T + np.linalg.inv(camera_pose)[:3,3]

    return hand_vertices


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
        
    pattern = r' \(\d+\)' # fix 
    mask_path = re.sub(pattern, '', mask_path) # fix
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
    '''
    # Add text annotations for each keypoint
    for (y, x) in keypoints2d:
        x = int(x)
        y = int(y)
        cv2.putText(img_o, f'({x}, {y})', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    '''
    if not os.path.exists(join(output_results, seq_name)):
        os.makedirs(join(output_results, seq_name))
    img_path = join(output_results, seq_name, f'{frame}_kps2d_visual.jpg')
    cv2.imwrite(img_path, img_o)
    
    return img_o

# frame_path = '/content/drive/MyDrive/Thesis/POV_Surgery_data/color/d_diskplacer_1/00145.jpg'
# keypoints2d = get_keypoints2d_from_frame(frame_path=frame_path)
# visualize_keypoints2d(frame_path, keypoints2d)

# print(get_bbox_from_frame(frame_path=frame_path))

def visualize_mesh3d_with_light(frame_path, mesh_vertices, rh_faces_path, dataset_root='',
                                output_results='', light_direction=np.array([0, 0, -1]), light_intensity=0.7,
                                custom_str='', show_vertex=True, show_faces=True, face_color_rgb=(0, 255, 0)):
    """
    Visualizes the 3D mesh by projecting it onto the 2D image and drawing the faces with shading.

    Parameters:
    - frame_path (str): Path to the frame (color image).
    - mesh_vertices (np.ndarray): 3D vertices of the hand mesh.
    - rh_faces_path (str): Path to the file containing the face connectivity (torch tensor).
    - dataset_root (str): Root directory of the dataset.
    - output_results (str): Directory to save the output images.
    - light_direction (np.ndarray): Direction of the light source.
    - light_intensity (float): Intensity of the light source.
    - custom_str (str): A custom string to append to the saved filename.
    - show_vertex (bool): If True, visualizes the blue circles of each vertex.
    - show_faces (bool): If True, visualizes the faces (triangles).
    - face_color_rgb (tuple): RGB color values for the faces (default is green).

    Returns:
    - img_o (np.ndarray): The image with the 3D mesh (vertices and faces) projected onto it.
    """
    
    # Load the face indices (connections between vertices)
    rh_faces = torch.load(rh_faces_path, weights_only=True).cpu().numpy()

    # Extract sequence name and frame number
    seq_name, frame = frame_path.split(os.sep)[-2:]
    frame = os.path.splitext(frame)[0]  
    image = cv2.imread(os.path.join(dataset_root, 'color', seq_name, f'{frame}.jpg'))
    
    # Camera intrinsics (from the original script)
    K = np.array([[1198.4395, 0.0000, 960.0000],
                  [0.0000, 1198.4395, 175.2000],
                  [0.0000, 0.0000, 1.0000]])

    # Apply the camera intrinsic matrix to project the 3D vertices to 2D
    coord_change_mat = np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32)
    transformed_vertices = mesh_vertices @ coord_change_mat.T
    projected_vertices = transformed_vertices @ K.T
    
    # Normalize by depth
    p2d = projected_vertices[:, :2] / projected_vertices[:, 2:3]

    # Clip coordinates to image dimensions
    p2d[:, 0] = np.clip(p2d[:, 0], 0, image.shape[1] - 1)  # X coordinates
    p2d[:, 1] = np.clip(p2d[:, 1], 0, image.shape[0] - 1)  # Y coordinates

    # Create a copy of the image for drawing
    img_o = np.copy(image)

    # Normalize light direction
    light_direction = light_direction / np.linalg.norm(light_direction)

    if show_faces:
        # Calculate shading for each face based on the light direction
        shaded_faces = []
        for face in rh_faces:
            # Get the vertex positions of the face
            v0, v1, v2 = transformed_vertices[face]

            # Compute the face normal
            normal = np.cross(v1 - v0, v2 - v0)
            normal = normal / np.linalg.norm(normal)

            # Compute the shading based on the dot product with the light direction
            shading = np.dot(normal, light_direction) * light_intensity
            shading = np.clip(shading, 0, 1)  # Ensure shading is between 0 and 1

            # Apply shading to the face color
            shaded_color = tuple(int(channel * shading) for channel in face_color_rgb)[::-1]
            shaded_faces.append((face, shaded_color))
        
        # Draw the shaded faces on the image
        for face, color in reversed(shaded_faces):
            pts = p2d[face].astype(np.int32)
            cv2.polylines(img_o, [pts], isClosed=True, color=color, thickness=1)
            cv2.fillPoly(img_o, [pts], color=color + (50,))  # Adding transparency

    if show_vertex:
        # Draw the mesh vertices on the image
        for vertex in p2d:
            cv2.circle(img_o, (int(vertex[0]), int(vertex[1])), radius=2, color=(0, 0, 255), thickness=-1)
    
    # Save the output image
    if not os.path.exists(os.path.join(output_results, seq_name)):
        os.makedirs(os.path.join(output_results, seq_name))
    img_path = os.path.join(output_results, seq_name, f'{frame}_mesh3d_visual_{custom_str}.jpg')
    cv2.imwrite(img_path, img_o)
    
    return img_o

# Compute Mean Per Joint Position Error (MPJPE)
# MPJPE measures the average Euclidean distance between predicted joint locations 
# and their corresponding ground truth positions. It provides a quantitative measure 
# of how far off, on average, the predicted keypoints are from their true locations.
def compute_MPJPE(pred, target):
    """
    pred: tensor of shape (N, 21, 3) - N sets of predicted keypoints
    target: tensor of shape (21, 3) - single set of ground truth keypoints
    """
    
    if pred.numel() == 0 and target.numel() == 0:
        return np.array(0), np.array(0)
    
    if target.numel() == 0 or pred.numel() == 0: # if target is empty
        return float('inf'), float('inf') 
        
    N = pred.shape[0]
    
    # Expand target to match pred shape
    target_expanded = target.expand(N, -1, -1)
    
    distances = torch.norm(pred[:, :, :2] - target_expanded[:, :, :2], dim=2)
    avg_mpjpe_per_pred = distances.mean(dim=1)
    # if avg_mpjpe_per_pred.numel() == 0:
    #     return float('inf'), float('inf') 
    best_mpjpe, _ = avg_mpjpe_per_pred.min(dim=0)
    avg_mpjpe = avg_mpjpe_per_pred.mean()
    
    return avg_mpjpe.detach().cpu().numpy(), best_mpjpe.detach().cpu().numpy()

# For each of the N keypoints, returns a dim x dim box around it
def get_boxes_keypoints(frame_path, dim=50):
    
    boxes = []
    keypoints = get_keypoints2d_from_frame(frame_path)
    
    for keypoint in keypoints:
        y, x = keypoint

        x1 = int(x - dim / 2)
        y1 = int(y - dim / 2)
        x2 = int(x + dim / 2)
        y2 = int(y + dim / 2)
        
        boxes.append([x1, y1, x2, y2])
    
    return boxes

def visualize_boxes_keypoints(image_path, boxes, output_folder):
    
    seq_name, frame = image_path.split(os.sep)[-2:]
    frame = os.path.splitext(frame)[0]  
    
    # Load the image
    image = cv2.imread(image_path)
    # Check if the image was loaded successfully
    if image is None:
        raise FileNotFoundError(f"Image file {image_path} not found or unable to load.")
    
    height, width, _ = image.shape
    
    for box in boxes:
        x1, y1, x2, y2 = box
        
        # Clip the coordinates to ensure they stay within the image boundaries
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(width, int(x2))
        y2 = min(height, int(y2))
        
        # Draw the bounding box (rectangle) on the image
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green color with thickness of 2
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_file_path = os.path.join(output_folder, seq_name, f'{frame}_keypoints_boxes.png')
    
    # Save the image with boxes
    cv2.imwrite(output_file_path, image)
    
    # Display the image (optional)
    cv2.imshow('Image with Boxes', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def is_red(rgb):
    R, G, B = rgb
    if R > RED_MINIMUM_VALUE and (R >= G * RED_DOMINANCE_THRESHOLD and R >= B * RED_DOMINANCE_THRESHOLD):
        return True
    return False

def get_keypoints_bloodiness(frame_path, dim_boxes=50):
    
    boxes = get_boxes_keypoints(frame_path, dim=dim_boxes)
    
    seq_name, frame = frame_path.split(os.sep)[-2:]
    frame = os.path.splitext(frame)[0]  
    
    # Load the image
    image = cv2.imread(frame_path) # image shape  (HEIGHT, WIDTH, CHANNELS)
    # Check if the image was loaded successfully
    if image is None:
        raise FileNotFoundError(f"Image file {frame_path} not found or unable to load.")
    
    bloodiness_values = []
    
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        roi = image[y1: y2, x1: x2]
        # cv2.imshow(f'kps {i} ROI, ({(x1+x2)/2}, {(y1+y2)/2})', roi)
        # cv2.imwrite(f'kps {i} ROI, ({(x1+x2)/2}, {(y1+y2)/2}).png', roi)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        H, W, C = roi.shape
        roi = roi.reshape(H*W, C)
        n_pixels = H*W
        count = sum(is_red((p[2], p[1], p[0])) for p in roi) # BGR to RGB when passed to is_red
        bloodiness = count/n_pixels
        # print(f'bloodiness kp {i} = {bloodiness:.2%}')
        bloodiness_values.append(bloodiness)
    
    return bloodiness_values

# /home/aidara/Desktop/Thesis_Andrea/data/POV_Surgery_data/color/d_diskplacer_1/00145.jpg
fp = '/home/aidara/Desktop/Thesis_Andrea/data/POV_Surgery_data/color/d_diskplacer_1/00689.jpg'#'/home/aidara/Desktop/Thesis_Andrea/Github_repos/Master-s-thesis---2D-Keypoints-extraction-experiments/utils/outs/d_diskplacer_1/00145_kps2d_visual.jpg' #'/home/aidara/Desktop/Thesis_Andrea/Github_repos/Master-s-thesis---2D-Keypoints-extraction-experiments/utils/outs/d_diskplacer_1/00145_kps2d_visual.jpg'
dataset_root = '/home/aidara/Desktop/Thesis_Andrea/data/POV_Surgery_data'
outs = '/home/aidara/Desktop'


# boxes = get_boxes_keypoints(fp, dim=50)
# visualize_boxes_keypoints(fp, boxes, 
#                           output_folder='/home/aidara/Desktop/Thesis_Andrea/Github_repos/Master-s-thesis---2D-Keypoints-extraction-experiments/utils/outs'
#                          )

kps = get_keypoints2d_from_frame(fp, add_visibility=False)
visualize_keypoints2d(fp, kps, dataset_root=dataset_root, output_results=outs)


# bloodiness_values = get_keypoints_bloodiness(fp)

# fp = '/home/aidara/Desktop/Thesis_Andrea/data/POV_Surgery_data/color/r_scalpel_1/01365.jpg'
# fp = '/home/aidara/Desktop/Thesis_Andrea/data/POV_Surgery_data/color/d_diskplacer_1/00145.jpg'
# fp = '/home/aidara/Desktop/Thesis_Andrea/data/POV_Surgery_data/color/R2_d_diskplacer_1/02037.jpg'


# kps = get_keypoints2d_from_frame(fp, add_visibility=False)
# print('-'*5)
# print(kps)

# print('-'*30)

# kps3d = get_keypoints3d_from_frame(fp, add_visibility=False)
# print(kps3d)
# torch.save(kps3d, os.path.join(outs, 'kps3d_test.pt'))
# print('-'*5)
# kps2d_proj = visualize_keypoints3d(kps3d)
# print(kps2d_proj)

# mesh3d = get_mesh3d_from_frame(fp)
# print(type(mesh3d), mesh3d.shape)
# print(mesh3d)
# torch.save(mesh3d, os.path.join(outs, 'mesh3d_test.pt'))


# rh_faces_path = '/home/aidara/Desktop/Thesis_Andrea/right_hand_mesh_faces.pt'

# visualize_mesh3d_with_light(
#     frame_path=fp,
#     mesh_vertices=mesh3d,
#     rh_faces_path=rh_faces_path,
#     dataset_root=DATASET_ROOT,
#     output_results=outs,
#     show_vertex=True,  # Set to True to show vertices
#     show_faces=True,  # Set to True to show faces
#     face_color_rgb=(0, 255, 0),  # Set face color RGB values
#     light_direction=np.array([0, 0, -1]),  #  Light direction
#     light_intensity=0.9,  # Example light intensity
#     custom_str=''  # Custom string for output file
# )