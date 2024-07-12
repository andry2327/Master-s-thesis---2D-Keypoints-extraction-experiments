import argparse
import os
import torch
import datetime
import pytz
from YOLOv8-pose import YOLO_Pose

# Argument parsing
parser = argparse.ArgumentParser(description='Train Keypoint RCNN model')
parser.add_argument('--dataset_config', type=str, required=True, help='Root directory of the .yaml dataset config file')
parser.add_argument('--model_type', type=str, required=True, help="Model's scale to load")
parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs to train')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
parser.add_argument('--image_size', type=int, default=640, help='# Target image size for training. All images are resized to this dimension before being fed into the model. Affects model accuracy and computational complexity')
parser.add_argument('--save', action='store_true', help='Enables saving of training checkpoints and final model weights.')
parser.add_argument('--num_workers', type=int, default=8, help='Number of worker threads for data loading (per RANK if Multi-GPU training). Influences the speed of data preprocessing and feeding into the model, especially useful in multi-GPU setups')
parser.add_argument('--training_run_folder_name', type=str, required=True, help='Name of the training run. Used for creating a subdirectory within the project folder, where training logs and outputs are stored')
parser.add_argument('--verbose', action='store_true', help='Enables verbose output during training, providing detailed logs and progress updates. Useful for debugging and closely monitoring the training process')
parser.add_argument('--use_autocast', action='store_true', help='Enables Automatic Mixed Precision (AMP) training, reducing memory usage and possibly speeding up training with minimal impact on accuracy')
parser.add_argument('--generate_plots', action='store_true', help='Generates and saves plots of training and validation metrics, as well as prediction examples, providing visual insights into model performance and learning progression')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
parser.add_argument('--lr_final', type=int, default=0.01, help='Final learning rate as a fraction of the initial rate = (lr0 * lrf), used in conjunction with schedulers to adjust the learning rate over time.')
parser.add_argument('--checkpoint_step', type=int, default=-1, help='Checkpoint save interval')
parser.add_argument('--output_folder', type=str, required=True, help='Output folder for logs and checkpoints')
parser.add_argument('--use_autocast', action='store_true', help='Flag to enable mixed precision training')
args = parser.parse_args()

# Initialize and train YOLO_Pose
yolo-pose_trainer = YOLO_Pose()

pose_trainer.train(
    dataset_config=args.dataset_config,
    model_type=args.model_type,
    num_epochs=args.num_epochs,
    batch_size=args.batch_size,
    image_size=args.image_size,
    save=args.save,
    num_workers=args.num_workers,
    training_run_folder_name=args.training_run_folder_name,
    verbose=args.verbose,
    generate_plots=args.generate_plots,
    lr=args.lr,
    lrf=args.lr_final,
    checkpoint_step=args.checkpoint_step,
    output_folder=args.output_folder,
    use_autocast=args.use_autocast
)