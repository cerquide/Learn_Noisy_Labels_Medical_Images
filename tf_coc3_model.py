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
from tf_utils import noisy_label_loss
from tf_utils import plot_performance
### ======================== ###

### ======= Data_Loader.py ======= ###
from tf_dataloaders import COC3TrainDataset
### ======================== ###

### ======= Models.py ======= ###
from tf_models import initialize_model3
### ======================== ###

images_path = Path("/data/eurova/multi_annotators_project/LNLMI/oocytes_gent_raw")
masks_path = Path("/data/eurova/multi_annotators_project/LNLMI/oocytes_gent_raw")
path_to_save = Path("/data/eurova/multi_annotators_project/LNLMI/Results/coc/coc3_tf/")
log_path = Path("/data/eurova/multi_annotators_project/LNLMI/Results/coc/coc3_tf/")

IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS = 192, 240, 1
DEVICE = 'cuda'
learning_rate = 1e-3
batch_size = 16
val_split = 0.05
epochs = 10
patience = 500

TL = True
weights_path = './tf_coc/coc_Final_dict.pt'

def train_model(images_path:Path, masks_path:Path, path_to_save: Path, log_path:Path):
    path_to_save.mkdir(exist_ok = True)

    print(images_path)
    print(masks_path)
    # Load the dataset
    dataset = COC3TrainDataset(images_path, IMG_WIDTH, IMG_HEIGHT)
    print("Dataset was loaded...")

    train_len = int(len(dataset) * (1 - val_split))
    val_len = len(dataset) - train_len
    train_dataset, val_dataset = random_split(dataset, [train_len, val_len])

    print("Train length: ", train_len)
    print("Val length: ", val_len)

    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False)

    # Initialize the model, loss function and optimizer
    model = initialize_model3(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS).to(DEVICE)

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
        train_loss_ce = 0.0
        train_loss_trace = 0.0
        train_dice = 0.0

        for X, y_AR, y_HS, y_SG, y_avrg in train_loader:

            X, y_AR, y_HS, y_SG, y_avrg = X.to(DEVICE), y_AR.to(DEVICE), y_HS.to(DEVICE), y_SG.to(DEVICE), y_avrg.to(DEVICE)

            labels_all = []
            labels_all.append(y_AR)
            labels_all.append(y_HS)
            labels_all.append(y_SG)

            optimizer.zero_grad()
            output, output_cms = model(X)

            # Calculate the Loss
            loss = dice_loss(output, y_avrg)
            loss, loss_ce, loss_trace = noisy_label_loss(output, output_cms, labels_all)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_loss_ce += loss_ce.item()
            train_loss_trace += loss_trace.item()

            # Calculate the Dice
            pred = torch.sigmoid(output) > 0.5
            train_dice_ = dice_coefficient(pred.float(), y_avrg)
            train_dice += train_dice_.item()

        train_loss /= len(train_loader)
        train_loss_ce /= len(train_loader)
        train_loss_trace /= len(train_loader)
        train_dice /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        # with torch.no_grad():
        #     for X, y in val_loader:

        #         X, y = X.to('cuda'), y.to('cuda')

        #         # Calculate the Loss 
        #         output = model(X)
        #         # loss = criterion(output, y)
        #         loss = dice_loss(output, y)
        #         val_loss += loss.item()

        #         # Calculate the Dice 
        #         pred = torch.sigmoid(output) > 0.5
        #         dice = dice_coefficient(pred.float(), y)
        #         val_dice += dice.item()

        # val_loss /= len(val_loader)
        # val_dice /= len(val_loader)

        # train_loss_values.append(train_loss)
        # val_loss_values.append(val_loss)
        # train_dice_values.append(train_dice)
        # val_dice_values.append(val_dice)

        print(f'Epoch: {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Train Loss CE: {train_loss_ce:.4f}, Train Dice: {train_dice:.4f}, Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}')
    print(len(output_cms))

train_model(images_path, masks_path, path_to_save, log_path)