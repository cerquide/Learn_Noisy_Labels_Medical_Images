import numpy as np
import cv2
import gzip

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Dataset
# from torch.utils.tensorboard import SummaryWriter

from pathlib import Path

images_path = Path("/data/eurova/cumulus_database/numpy/melanoma/imgs_train.npy.gz")
masks_path = Path("/data/eurova/cumulus_database/numpy/melanoma/imgs_mask_train.npy.gz")
path_to_save = Path("/data/eurova/multi_annotators_project/LNLMI/Results/skin/skin_tf/")
log_path = Path("/data/eurova/multi_annotators_project/LNLMI/Results/skin/skin_tf/")

IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS = 192, 240, 1
learning_rate = 1e-3
batch_size = 16
val_split = 0.1
epochs = 10
patience = 500


### ======================== ###
### ======= Utils.py ======= ###
### ======================== ###

def dice_coefficient(pred, target):

    smooth = 1e-6
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()

    return (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)

def preprocessor(input_img, img_rows, img_cols):
    """
    Resize input images to constants sizes
    :param input_img: numpy array of images
    :return: numpy array of preprocessed images
    """
    input_img = np.swapaxes(input_img, 2, 3)
    input_img = np.swapaxes(input_img, 1, 2)

    output_img = np.ndarray((input_img.shape[0], input_img.shape[1], img_rows, img_cols), dtype = np.uint8)

    for i in range(input_img.shape[0]):
        output_img[i, 0] = cv2.resize(input_img[i, 0], (img_cols, img_rows), interpolation = cv2.INTER_AREA)

    output_img = np.swapaxes(output_img, 1, 2)
    output_img = np.swapaxes(output_img, 2, 3)

    return output_img

def load_skin_train_data(imgs_path, masks_path, img_width, img_height):
    X_train = np.load(gzip.open(imgs_path))
    y_train = np.load(gzip.open(masks_path))

    X_train = preprocessor(X_train, img_width, img_height)
    y_train = preprocessor(y_train, img_width, img_height)

    X_train = X_train.astype('float32') / 255.
    mean = np.mean(X_train)
    std = np.std(X_train)

    X_train -= mean
    X_train /= std

    y_train = y_train.astype('float32')
    y_train /= 255.

    return X_train, y_train

class SkinTrainDataset(Dataset):
    def __init__(self, imgs_path, masks_path, img_width, img_height, transform=None):
        self.imgs, self.masks = load_skin_train_data(imgs_path, masks_path, img_width, img_height)
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        mask = self.masks[idx]

        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)

        img = torch.from_numpy(img).permute(2, 0, 1).float()  # Move the channel dimension to the front
        mask = torch.from_numpy(mask).permute(2, 0, 1).float()  # Move the channel dimension to the front

        return img, mask

### ======================== ###
### ======================== ###
### ======================== ###

### ======================== ###
### ======= Train.py ======= ###
### ======================== ###

class UNet(nn.Module):
    def __init__(self, img_channels):
        super(UNet, self).__init__()

        def conv_block(in_channels, out_channels, dropout=0.0):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.ReLU(inplace=True)
            )

        def upconv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
            )

        self.enc1 = conv_block(img_channels, 16, dropout=0.1)
        self.enc2 = conv_block(16, 32, dropout=0.1)
        self.enc3 = conv_block(32, 64, dropout=0.2)
        self.enc4 = conv_block(64, 128, dropout=0.2)

        self.middle = conv_block(128, 256, dropout=0.3)

        self.dec4 = conv_block(256, 128, dropout=0.2)
        self.dec3 = conv_block(128 , 64, dropout=0.2)
        self.dec2 = conv_block(64, 32, dropout=0.1)
        self.dec1 = conv_block(32, 16, dropout=0.1)

        self.upconv4 = upconv_block(256, 128)
        self.upconv3 = upconv_block(128, 64)
        self.upconv2 = upconv_block(64, 32)
        self.upconv1 = upconv_block(32, 16)

        self.output = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        enc3 = self.enc3(F.max_pool2d(enc2, 2))
        enc4 = self.enc4(F.max_pool2d(enc3, 2))

        middle = self.middle(F.max_pool2d(enc4, 2))

        up4 = self.upconv4(middle)
        dec4 = self.dec4(torch.cat([up4, enc4], 1))
        up3 = self.upconv3(dec4)
        dec3 = self.dec3(torch.cat([up3, enc3], 1))
        up2 = self.upconv2(dec3)
        dec2 = self.dec2(torch.cat([up2, enc2], 1))
        up1 = self.upconv1(dec2)
        dec1 = self.dec1(torch.cat([up1, enc1], 1))

        return torch.sigmoid(self.output(dec1))

def initialize_model(img_width, img_height, img_channels):
    model = UNet(img_channels)
    return model

### ======================== ###
### ======================== ###
### ======================== ###


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
    criterion = nn.BCEWithLogitsLoss()  # The loss function
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)

    # Setup TensorBoard logging
    # writer = SummaryWriter(log_dir=log_path)

    # Early stopping setup
    patience_counter = 0
    best_val_loss = float('inf')

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
                val_loss += loss.item()

                # Calculate the Dice 
                pred = torch.sigmoid(output) > 0.5
                dice = dice_coefficient(pred.float(), y)
                val_dice += dice.item()

        val_loss /= len(val_loader)
        val_dice /= len(val_loader)

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

    # Save the training history
    # np.save(str(path_to_save / 'melanoma_base_history_.npy'), {'train_loss': train_loss, 'val_loss': val_loss})

### ======================== ###
### ======================== ###
### ======================== ###

train_model(images_path, masks_path, path_to_save, log_path)