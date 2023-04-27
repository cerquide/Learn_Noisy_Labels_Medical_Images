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
from tf_dataloaders import SkinTrainDataset
### ======================== ###

### ======= Models.py ======= ###
from tf_models import initialize_model
### ======================== ###

images_path = Path("/data/eurova/cumulus_database/numpy/melanoma/imgs_train.npy.gz")
masks_path = Path("/data/eurova/cumulus_database/numpy/melanoma/imgs_mask_train.npy.gz")
path_to_save = Path("/data/eurova/multi_annotators_project/LNLMI/Results/skin/skin_tf/")
log_path = Path("/data/eurova/multi_annotators_project/LNLMI/Results/skin/skin_tf/")

IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS = 192, 240, 1
learning_rate = 1e-3
batch_size = 16
val_split = 0.1
epochs = 100
patience = 500

### ======================== ###
### ======= Train.py ======= ###
### ======================== ###

def train_model(images_path:Path, masks_path:Path, path_to_save: Path, log_path:Path):
    path_to_save.mkdir(exist_ok=True)

    # Load the dataset
    dataset = SkinTrainDataset(images_path, masks_path, IMG_WIDTH, IMG_HEIGHT)
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
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_dice = 0.0
        for X, y in train_loader:

            X, y = X.to('cuda'), y.to('cuda')

            optimizer.zero_grad()
            output = model(X)

            # Calculate the Loss
            loss = criterion(output, y)
            # loss = dice_loss(output, y)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # Calculate the Dice
            pred = torch.sigmoid(output) > 0.5
            train_dice_ = dice_coefficient(pred.float(), y)
            train_dice += train_dice_.item()

        train_loss /= len(train_loader)
        train_dice /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        with torch.no_grad():
            for X, y in val_loader:

                X, y = X.to('cuda'), y.to('cuda')

                # Calculate the Loss 
                output = model(X)
                loss = criterion(output, y)
                # loss = dice_loss(output, y)
                val_loss += loss.item()

                # Calculate the Dice 
                pred = torch.sigmoid(output) > 0.5
                dice = dice_coefficient(pred.float(), y)
                val_dice += dice.item()

        val_loss /= len(val_loader)
        val_dice /= len(val_loader)

        train_loss_values.append(train_loss)
        val_loss_values.append(val_loss)
        train_dice_values.append(train_dice)
        val_dice_values.append(val_dice)

        print(f'Epoch: {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}, Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}')

        # TensorBoard logging
        # writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)

        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), path_to_save / 'melanoma_base_weights.pth')
        else:
            patience_counter += 1

            if patience_counter >= patience:
                print("Early stopping")
                break
        
    save_path = './tf_skin'
    plot_performance(train_loss_values, val_loss_values, train_dice_values, val_dice_values, save_path)
    print("Figures were saved.")

    torch.save(model.state_dict(), save_path + '/skin_Final_dict.pt')

    # Save the training history
    # np.save(str(path_to_save / 'melanoma_base_history_.npy'), {'train_loss': train_loss, 'val_loss': val_loss})

### ======================== ###
### ======================== ###
### ======================== ###

train_model(images_path, masks_path, path_to_save, log_path)