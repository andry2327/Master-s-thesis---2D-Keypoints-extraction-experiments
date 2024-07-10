import argparse
import os
import torch
import datetime
import pytz
from keypointrcnn_resnet50_fpn import KeypointRCNN

# Argument parsing
parser = argparse.ArgumentParser(description='Train Keypoint RCNN model')
parser.add_argument('--dataset_root', type=str, required=True, help='Root directory of the dataset')
parser.add_argument('--annot_root', type=str, required=True, help='Root directory of annotations')
parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs to train')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
parser.add_argument('--lr', type=float, default=0.02, help='Learning rate')
parser.add_argument('--step_size', type=int, default=120000, help='Step size for LR scheduler')
parser.add_argument('--lr_step_gamma', type=float, default=0.1, help='Gamma factor for LR scheduler')
parser.add_argument('--log_train_step', type=int, default=1, help='Logging interval during training')
parser.add_argument('--val_step', type=int, default=1, help='Validation interval')
parser.add_argument('--checkpoint_step', type=int, default=1, help='Checkpoint save interval')
parser.add_argument('--output_folder', type=str, required=True, help='Output folder for logs and checkpoints')
args = parser.parse_args()

# # Generate a timestamp for logging purposes
# current_timestamp = datetime.datetime.now(pytz.timezone("Europe/Rome")).strftime("%d-%m-%Y_%H-%M")
# folder = f'Training-DEBUG--{current_timestamp}'
# output_folder = os.path.join(args.output_folder, 'KeypointRCNN', folder)

# Initialize and train KeypointRCNN
keypoint_rcnn_trainer = KeypointRCNN()
keypoint_rcnn_trainer.train(
    dataset_root=args.dataset_root,
    annot_root=args.annot_root,
    num_epochs=args.num_epochs,
    batch_size=args.batch_size,
    lr=args.lr,
    step_size=args.step_size,
    lr_step_gamma=args.lr_step_gamma,
    log_train_step=args.log_train_step,
    val_step=args.val_step,
    checkpoint_step=args.checkpoint_step,
    output_folder=args.output_folder
)