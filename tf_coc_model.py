import numpy as np
import cv2
import gzip

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
# from torch.utils.tensorboard import SummaryWriter

from pathlib import Path

### ======= Utils.py ======= ###
from tf_utils import dice_coefficient, dice_loss
from tf_utils import COCTrainDataset
from tf_utils import plot_performance
### ======================== ###

### ======= Models.py ======= ###
from tf_models import initialize_model
### ======================== ###

images_path = Path("/data/eurova/multi_annotators_project/LNLMI/oocytes_gent_raw/images/")
masks_path = Path("/data/eurova/multi_annotators_project/LNLMI/oocytes_gent_raw/avrg/")
path_to_save = Path("/data/eurova/multi_annotators_project/LNLMI/Results/coc/coc_tf/")
log_path = Path("/data/eurova/multi_annotators_project/LNLMI/Results/skin/coc_tf/")

IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS = 192, 240, 1
learning_rate = 1e-3
batch_size = 16
val_split = 0.1
epochs = 100
patience = 500

def train_model(images_path:Path, masks_path:Path, path_to_save: Path, log_path:Path):
    path_to_save.mkdir(exist_ok=True)

    # Load the dataset
    dataset = COCTrainDataset(images_path, masks_path, IMG_WIDTH, IMG_HEIGHT)
    print("Dataset was loaded...")
    # print("dataset size: ", dataset.size())

    train_len = int(len(dataset) * (1 - val_split))
    val_len = len(dataset) - train_len
    train_dataset, val_dataset = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model, loss function and optimizer
    model = initialize_model(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS).to('cuda')
    print("Model initialized...")
    criterion = nn.BCEWithLogitsLoss(reduce = 'mean')  # The loss function
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)

    # Setup TensorBoard logging
    # writer = SummaryWriter(log_dir=log_path)

    # Early stopping setup
    patience_counter = 0
    best_val_loss = float('inf')

    train_dice_values = []
    val_dice_values = []
    train_loss_values = []
    val_loss_values = []

    print("Training...")

train_model(images_path, masks_path, path_to_save, log_path)