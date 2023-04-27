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
from tf_utils import plot_performance
### ======================== ###

### ======= Data_Loader.py ======= ###
from tf_dataloaders import COC3TrainDataset
### ======================== ###

### ======= Models.py ======= ###
from tf_models import initialize_model
### ======================== ###

images_path = Path("/data/eurova/multi_annotators_project/LNLMI/oocytes_gent_raw")
masks_path = Path("/data/eurova/multi_annotators_project/LNLMI/oocytes_gent_raw")
path_to_save = Path("/data/eurova/multi_annotators_project/LNLMI/Results/coc/coc3_tf/")
log_path = Path("/data/eurova/multi_annotators_project/LNLMI/Results/coc/coc3_tf/")

IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS = 192, 240, 1
learning_rate = 1e-3
batch_size = 16
val_split = 0.05
epochs = 10
patience = 500

TL = False
weights_path = './tf_coc/coc_Final_dict.pt'

def train_model(images_path:Path, masks_path:Path, path_to_save: Path, log_path:Path):
    path_to_save.mkdir(exist_ok = True)

    print(images_path)
    print(masks_path)
    # Load the dataset
    dataset = COC3TrainDataset(images_path, masks_path, IMG_WIDTH, IMG_HEIGHT)
    print("Dataset was loaded...")

    train_len = int(len(dataset) * (1 - val_split))
    val_len = len(dataset) - train_len
    train_dataset, val_dataset = random_split(dataset, [train_len, val_len])

    print("Train length: ", train_len)
    print("Val length: ", val_len)

    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False)

    # Initialize the model, loss function and optimizer
    model = initialize_model(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS).to('cuda')

    if TL:
        pretrained_weights = torch.load(weights_path)
        model.load_state_dict(pretrained_weights, strict = False)
        model.eval()
        print("Weights have been loaded succesfully...")

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
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_dice = 0.0
        for X, y_AR, y_HS, y_SG, y_avrg in train_loader:

            X, y_AR, y_HS, y_SG, y_avrg = X.to('cuda'), y_AR.to('cuda'), y_HS.to('cuda'), y_SG.to('cuda'), y_avrg.to('cuda')