import re
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np

plt.rcParams['figure.dpi'] = 300
FONTSIZE = 18

def parse_logs(log_file):
    epochs_train = {}
    val_epochs = {}
    loss_classifier = {}
    loss_box_reg = {}
    loss_keypoint = {}
    loss_objectness = {}
    loss_rpn_box_reg = {}
    val_loss_classifier = {}
    val_loss_box_reg = {}
    val_loss_keypoint = {}
    val_loss_objectness = {}
    val_loss_rpn_box_reg = {}

    with open(log_file, 'r') as f:
        for line in f:
            if 'Epoch' in line and 'Processed data' in line:  # train loss
                line_splitted = line.split(' ')
                epoch = int(line_splitted[1].split('/')[0])
                loss_cl = float(line_splitted[7].strip(','))
                loss_br = float(line_splitted[9].strip(','))
                loss_kp = float(line_splitted[11].strip(','))
                loss_obj = float(line_splitted[13].strip(','))
                loss_rpn_br = float(line_splitted[15].strip('\n'))
                epochs_train[epoch] = epoch
                if epoch not in loss_classifier.keys():
                    loss_classifier[epoch] = []
                loss_classifier[epoch].append(loss_cl)
                if epoch not in loss_box_reg.keys():
                    loss_box_reg[epoch] = []
                loss_box_reg[epoch].append(loss_br)
                if epoch not in loss_keypoint.keys():
                    loss_keypoint[epoch] = []
                loss_keypoint[epoch].append(loss_kp)
                if epoch not in loss_objectness.keys():
                    loss_objectness[epoch] = []
                loss_objectness[epoch].append(loss_obj)
                if epoch not in loss_rpn_box_reg.keys():
                    loss_rpn_box_reg[epoch] = []
                loss_rpn_box_reg[epoch].append(loss_rpn_br)
            elif 'val Losses' in line:  # val loss
                line_splitted = line.split(' ')
                epoch = int(line_splitted[1].split('/')[0])
                val_loss_cl = float(line_splitted[6].strip(','))
                val_loss_br = float(line_splitted[8].strip(','))
                val_loss_kp = float(line_splitted[10].strip(','))
                val_loss_obj = float(line_splitted[12].strip(','))
                val_loss_rpn_br = float(line_splitted[14].strip('\n'))
                val_epochs[epoch] = epoch
                val_loss_classifier[epoch] = val_loss_cl
                val_loss_box_reg[epoch] = val_loss_br
                val_loss_keypoint[epoch] = val_loss_kp
                val_loss_objectness[epoch] = val_loss_obj
                val_loss_rpn_box_reg[epoch] = val_loss_rpn_br

    # Convert dictionaries to lists sorted by epoch
    epochs_train = sorted(epochs_train.values())
    val_epochs = sorted(val_epochs.values())
    
    loss_classifier = [np.mean(loss_classifier[epoch]) for epoch in epochs_train]
    loss_box_reg = [np.mean(loss_box_reg[epoch]) for epoch in epochs_train]
    loss_keypoint = [np.mean(loss_keypoint[epoch]) for epoch in epochs_train]
    loss_objectness = [np.mean(loss_objectness[epoch]) for epoch in epochs_train]
    loss_rpn_box_reg = [np.mean(loss_rpn_box_reg[epoch]) for epoch in epochs_train]
    
    val_loss_classifier = [val_loss_classifier[epoch] for epoch in val_epochs]
    val_loss_box_reg = [val_loss_box_reg[epoch] for epoch in val_epochs]
    val_loss_keypoint = [val_loss_keypoint[epoch] for epoch in val_epochs]
    val_loss_objectness = [val_loss_objectness[epoch] for epoch in val_epochs]
    val_loss_rpn_box_reg = [val_loss_rpn_box_reg[epoch] for epoch in val_epochs]

    return epochs_train, val_epochs, loss_classifier, loss_box_reg, loss_keypoint, loss_objectness, loss_rpn_box_reg, val_loss_classifier, val_loss_box_reg, val_loss_keypoint, val_loss_objectness, val_loss_rpn_box_reg

def plot_losses(log_file, out_path=''):
    epochs_train, val_epochs, loss_classifier, loss_box_reg, loss_keypoint, loss_objectness, loss_rpn_box_reg, val_loss_classifier, val_loss_box_reg, val_loss_keypoint, val_loss_objectness, val_loss_rpn_box_reg = parse_logs(log_file)
    
    fig, axs = plt.subplots(5, 2, figsize=(30, 40))

    # Define fontsize
    fontsize = FONTSIZE

    # Plot training loss classifier
    axs[0, 0].plot(epochs_train, loss_classifier, label='Loss Classifier', marker='o')
    axs[0, 0].set_xlabel('Epochs', fontsize=fontsize)
    axs[0, 0].set_ylabel('Loss', fontsize=fontsize)
    axs[0, 0].set_title('Training Loss Classifier', fontsize=fontsize)
    axs[0, 0].legend(fontsize=fontsize)
    axs[0, 0].grid(True)
    axs[0, 0].tick_params(axis='x', rotation=45, labelsize=fontsize)
    axs[0, 0].tick_params(axis='y', labelsize=fontsize)

    # Plot training loss box reg
    axs[1, 0].plot(epochs_train, loss_box_reg, label='Loss Box Reg', marker='o')
    axs[1, 0].set_xlabel('Epochs', fontsize=fontsize)
    axs[1, 0].set_ylabel('Loss', fontsize=fontsize)
    axs[1, 0].set_title('Training Loss Box Reg', fontsize=fontsize)
    axs[1, 0].legend(fontsize=fontsize)
    axs[1, 0].grid(True)
    axs[1, 0].tick_params(axis='x', rotation=45, labelsize=fontsize)
    axs[1, 0].tick_params(axis='y', labelsize=fontsize)

    # Plot training loss keypoint
    axs[2, 0].plot(epochs_train, loss_keypoint, label='Loss Keypoint', marker='o')
    axs[2, 0].set_xlabel('Epochs', fontsize=fontsize)
    axs[2, 0].set_ylabel('Loss', fontsize=fontsize)
    axs[2, 0].set_title('Training Loss Keypoint', fontsize=fontsize)
    axs[2, 0].legend(fontsize=fontsize)
    axs[2, 0].grid(True)
    axs[2, 0].tick_params(axis='x', rotation=45, labelsize=fontsize)
    axs[2, 0].tick_params(axis='y', labelsize=fontsize)

    # Plot training loss objectness
    axs[3, 0].plot(epochs_train, loss_objectness, label='Loss Objectness', marker='o')
    axs[3, 0].set_xlabel('Epochs', fontsize=fontsize)
    axs[3, 0].set_ylabel('Loss', fontsize=fontsize)
    axs[3, 0].set_title('Training Loss Objectness', fontsize=fontsize)
    axs[3, 0].legend(fontsize=fontsize)
    axs[3, 0].grid(True)
    axs[3, 0].tick_params(axis='x', rotation=45, labelsize=fontsize)
    axs[3, 0].tick_params(axis='y', labelsize=fontsize)

    # Plot training loss rpn box reg
    axs[4, 0].plot(epochs_train, loss_rpn_box_reg, label='Loss RPN Box Reg', marker='o')
    axs[4, 0].set_xlabel('Epochs', fontsize=fontsize)
    axs[4, 0].set_ylabel('Loss', fontsize=fontsize)
    axs[4, 0].set_title('Training Loss RPN Box Reg', fontsize=fontsize)
    axs[4, 0].legend(fontsize=fontsize)
    axs[4, 0].grid(True)
    axs[4, 0].tick_params(axis='x', rotation=45, labelsize=fontsize)
    axs[4, 0].tick_params(axis='y', labelsize=fontsize)

    # Plot validation loss classifier
    axs[0, 1].plot(val_epochs, val_loss_classifier, label='Validation Loss Classifier', marker='x', color='orange')
    axs[0, 1].set_xlabel('Epochs', fontsize=fontsize)
    axs[0, 1].set_ylabel('Loss', fontsize=fontsize)
    axs[0, 1].set_title('Validation Loss Classifier', fontsize=fontsize)
    axs[0, 1].legend(fontsize=fontsize)
    axs[0, 1].grid(True)
    axs[0, 1].tick_params(axis='x', rotation=45, labelsize=fontsize)
    axs[0, 1].tick_params(axis='y', labelsize=fontsize)
    
    # Plot validation loss box reg
    axs[1, 1].plot(val_epochs, val_loss_box_reg, label='Validation Loss Box Reg', marker='x', color='orange')
    axs[1, 1].set_xlabel('Epochs', fontsize=fontsize)
    axs[1, 1].set_ylabel('Loss', fontsize=fontsize)
    axs[1, 1].set_title('Validation Loss Box Reg', fontsize=fontsize)
    axs[1, 1].legend(fontsize=fontsize)
    axs[1, 1].grid(True)
    axs[1, 1].tick_params(axis='x', rotation=45, labelsize=fontsize)
    axs[1, 1].tick_params(axis='y', labelsize=fontsize)

    # Plot validation loss keypoint
    axs[2, 1].plot(val_epochs, val_loss_keypoint, label='Validation Loss Keypoint', marker='x', color='orange')
    axs[2, 1].set_xlabel('Epochs', fontsize=fontsize)
    axs[2, 1].set_ylabel('Loss', fontsize=fontsize)
    axs[2, 1].set_title('Validation Loss Keypoint', fontsize=fontsize)
    axs[2, 1].legend(fontsize=fontsize)
    axs[2, 1].grid(True)
    axs[2, 1].tick_params(axis='x', rotation=45, labelsize=fontsize)
    axs[2, 1].tick_params(axis='y', labelsize=fontsize)

    # Plot validation loss objectness
    axs[3, 1].plot(val_epochs, val_loss_objectness, label='Validation Loss Objectness', marker='x', color='orange')
    axs[3, 1].set_xlabel('Epochs', fontsize=fontsize)
    axs[3, 1].set_ylabel('Loss', fontsize=fontsize)
    axs[3, 1].set_title('Validation Loss Objectness', fontsize=fontsize)
    axs[3, 1].legend(fontsize=fontsize)
    axs[3, 1].grid(True)
    axs[3, 1].tick_params(axis='x', rotation=45, labelsize=fontsize)
    axs[3, 1].tick_params(axis='y', labelsize=fontsize)

    # Plot validation loss rpn box reg
    axs[4, 1].plot(val_epochs, val_loss_rpn_box_reg, label='Validation Loss RPN Box Reg', marker='x', color='orange')
    axs[4, 1].set_xlabel('Epochs', fontsize=fontsize)
    axs[4, 1].set_ylabel('Loss', fontsize=fontsize)
    axs[4, 1].set_title('Validation Loss RPN Box Reg', fontsize=fontsize)
    axs[4, 1].legend(fontsize=fontsize)
    axs[4, 1].grid(True)
    axs[4, 1].tick_params(axis='x', rotation=45, labelsize=fontsize)
    axs[4, 1].tick_params(axis='y', labelsize=fontsize)

    plt.subplots_adjust(hspace=0.4, wspace=0.4, top=0.9, bottom=0.1)
    
    file_name = f'Loss_plots--{os.path.basename(log_file).strip(".txt")}.png'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    path_out = os.path.join(out_path, file_name)
    print(f'Plots saved in "{path_out}"')
    plt.tight_layout()
    plt.savefig(path_out, dpi=300)
    # plt.show()
    
def main():
    parser = argparse.ArgumentParser(description='Plot training and validation losses from log file.')
    parser.add_argument('--log_file', type=str, help='Path to the log file')
    parser.add_argument('--output_path', required=False, type=str, help='Path where to save plots')
    args = parser.parse_args()
    
    plot_losses(args.log_file, args.output_path)
    # file = '/content/drive/MyDrive/Thesis/Keypoints2d_extraction/KeypointRCNN/Training-DEBUG--09-07-2024_17-09/log_Training-DEBUG--09-07-2024_17-09.txt'
    # out_path = '/content/drive/MyDrive/Thesis/Keypoints2d_extraction/KeypointRCNN/Training-DEBUG--09-07-2024_17-09'
    # plot_losses(file, out_path)

if __name__ == "__main__":
    main()