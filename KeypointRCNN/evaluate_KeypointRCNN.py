import argparse
import os
import torch
import datetime
import pytz
from keypointrcnn_resnet50_fpn import KeypointRCNN

# Argument parsing
parser = argparse.ArgumentParser(description='Evaluate Keypoint RCNN model')
parser.add_argument('--dataset_root', type=str, required=True, help='Root directory of the dataset')
parser.add_argument('--annot_root', type=str, default='', help='Root directory of annotations')
parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model checkpoint')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size for evaluation')
parser.add_argument('--seq', type=str, default='', help='Sequence name to evaluate (optional)')
parser.add_argument('--output_results', type=str, required=True, help='Output directory to save evaluation results')
parser.add_argument('--visualize', action='store_true', help='Flag to enable visualization')
args = parser.parse_args()

# Initialize and evaluate KeypointRCNN
keypoint_rcnn_evaluator = KeypointRCNN()
keypoint_rcnn_evaluator.evaluate(
    dataset_root=args.dataset_root,
    annot_root=args.annot_root,
    model_path=args.model_path,
    batch_size=args.batch_size,
    seq=args.seq,
    output_results=args.output_results,
    visualize=args.visualize
)