import numpy as np
import cv2
import gzip

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from torch.optim.lr_scheduler import StepLR
# from torch.utils.tensorboard import SummaryWriter

from pathlib import Path

### ======= Utils.py ======= ###
from tf_utils import dice_coefficient, dice_loss
from tf_utils import noisy_label_loss_GCM, noisy_label_loss_lCM, combined_loss
from tf_utils import plot_performance
from tf_utils import test_lGM
from tf_utils import calculate_cm, evaluate_cm
### ======================== ###

### ======= Data_Loader.py ======= ###
from tf_dataloaders import COC3TrainDataset
### ======================== ###

### ======= Models.py ======= ###
from tf_models import initialize_model_GCM
from tf_models import initialize_model_lCM
### ======================== ###

images_path = Path("/data/eurova/multi_annotators_project/LNLMI/oocytes_gent_raw")
masks_path = Path("/data/eurova/multi_annotators_project/LNLMI/oocytes_gent_raw")
path_to_save = Path("/data/eurova/multi_annotators_project/LNLMI/Results/coc/coc3_tf/")
log_path = Path("/data/eurova/multi_annotators_project/LNLMI/Results/coc/coc3_tf/")

IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS = 192, 240, 1
DEVICE = 'cuda'
ALPHA = 0.5
learning_rate = 1e-3
batch_size = 16
val_split = 0.05
test_split = 0.05
epochs = 10
patience = 500

GCM = False  # for using Global CM, else local CM.
TL = True   # for using transfer learning
weights_path = './tf_coc/coc_Final_dict.pt'
# weights_path = './tf_skin/skin_Final_dict.pt'

def train_model(images_path:Path, masks_path:Path, path_to_save: Path, log_path:Path):
    path_to_save.mkdir(exist_ok = True)

    print(images_path)
    print(masks_path)
    # Load the dataset
    dataset = COC3TrainDataset(images_path, IMG_WIDTH, IMG_HEIGHT)
    print("Dataset was loaded...")

    train_len = int(len(dataset) * (1 - val_split - test_split))
    val_len = int(len(dataset) * val_split)
    test_len = len(dataset) - train_len - val_len
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_len, val_len, test_len])

    print("Train length: ", train_len)
    print("Val length: ", val_len)
    print("Test length: ", test_len)

    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle = False)

    # Initialize the model, loss function and optimizer
    if GCM:
        model = initialize_model_GCM(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS).to(DEVICE)
        noisy_label_loss = noisy_label_loss_GCM
    else:
        model = initialize_model_lCM(IMG_CHANNELS).to(DEVICE)
        noisy_label_loss = noisy_label_loss_lCM
        test = test_lGM
    if TL:
        pretrained_weights = torch.load(weights_path)
        
        ### print names of layers ###
        model_dict = model.state_dict()
        # for name, param in model.named_children():
        #     print(name)
        layers = ['enc1', 'enc2', 'enc3', 'enc4',
                  'middle',
                  'dec4', 'dec3', 'dec2', 'dec1',
                  'upconv4', 'upconv3', 'upconv2', 'upconv1',
                  'cms_output',
                  'output']
        for name, param in model.named_parameters():
            if 'cms_output' in name:
                param.requires_grad = True
            else:
                param.requires_grad = True
        ### ===================== ###

        model.load_state_dict(pretrained_weights, strict = False)
        model.eval()
        print("Weights have been loaded succesfully...")
    
    total_params = sum(p.numel() for p in model.parameters())
    print("Total number of params: ", total_params)
    total_params_grad  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total number of params with grad: ", total_params_grad)

    print("Model initialized...")
    criterion = nn.BCEWithLogitsLoss(reduce = 'mean')  # The loss function
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)

    scheduler = StepLR(optimizer, step_size = 30, gamma = 1.)

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
        train_loss_dice = 0.0
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
            # loss = dice_loss(output, y_avrg)
            loss, loss_dice, loss_trace = noisy_label_loss(output, output_cms, labels_all, alpha = ALPHA)
            
            # loss, loss_dice, loss_cm = combined_loss(pred = output, cms = output_cms, ys = [y_AR, y_HS, y_SG, y_avrg])

            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_loss_dice += loss_dice.item()
            train_loss_trace += loss_trace.item()

            # Calculate the Dice
            pred = torch.sigmoid(output) > 0.5
            train_dice_ = dice_coefficient(pred.float(), y_avrg)
            train_dice += train_dice_.item()
        
        train_loss /= len(train_loader)
        train_loss_dice /= len(train_loader)
        train_loss_trace /= len(train_loader)
        train_dice /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0.0
        val_loss_dice = 0.0
        val_loss_trace = 0.0
        val_dice = 0.0
        with torch.no_grad():
            for X, y_AR, y_HS, y_SG, y_avrg in val_loader:

                X, y_AR, y_HS, y_SG, y_avrg = X.to(DEVICE), y_AR.to(DEVICE), y_HS.to(DEVICE), y_SG.to(DEVICE), y_avrg.to(DEVICE)

                labels_all = []
                labels_all.append(y_AR)
                labels_all.append(y_HS)
                labels_all.append(y_SG)

                if GCM == False:
                    cm_all_true = []
                    cm_AR_true = calculate_cm(pred = y_AR, true = y_avrg)
                    cm_HS_true = calculate_cm(pred = y_HS, true = y_avrg)
                    cm_SG_true = calculate_cm(pred = y_SG, true = y_avrg)
                    # print("Confusion Matrix AR: ", cm_AR_true)
                    # print("Confusion Matrix HS: ", cm_HS_true)
                    # print("Confusion Matrix SG: ", cm_SG_true)
                    
                    cm_all_true.append(cm_AR_true)
                    cm_all_true.append(cm_HS_true)
                    cm_all_true.append(cm_SG_true)

                # Calculate the Loss 
                output, output_cms = model(X)
                # loss = criterion(output, y)
                loss, loss_dice, loss_trace = noisy_label_loss(output, output_cms, labels_all)
                val_loss += loss.item()
                val_loss_dice += loss_dice.item()
                val_loss_trace += loss_trace.item()

                # insert evaluate_cms(pred = torch.sigmoid(output), ...)
                mse_outputs, mses = evaluate_cm(pred = torch.sigmoid(output), pred_cm = output_cms, true_cm = cm_all_true)

                # Calculate the Dice 
                pred = torch.sigmoid(output) > 0.5
                dice = dice_coefficient(pred.float(), y_avrg)
                val_dice += dice.item()

            val_loss /= len(val_loader)
            val_loss_dice /= len(val_loader)
            val_loss_trace /= len(val_loader)
            val_dice /= len(val_loader)

        train_loss_values.append(train_loss)
        val_loss_values.append(val_loss)
        train_dice_values.append(train_dice)
        val_dice_values.append(val_dice)

        print(f'Epoch: {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Train Loss Dice: {train_loss_dice:.4f}, Train Dice: {train_dice:.4f}, Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}, Val MSE AR: {mses[0]:.4f}')

        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), path_to_save / 'coc3_base_weights.pt')
        else:
            patience_counter += 1

            if patience_counter >= patience:
                print("Early stopping")
                break

    with torch.no_grad():
        for X, y_AR, y_HS, y_SG, y_avrg in val_loader:

            X, y_AR, y_HS, y_SG, y_avrg = X.to(DEVICE), y_AR.to(DEVICE), y_HS.to(DEVICE), y_SG.to(DEVICE), y_avrg.to(DEVICE)

            labels_all = []
            labels_all.append(y_AR)
            labels_all.append(y_HS)
            labels_all.append(y_SG)

            # Calculate the Loss 
            output, output_cms = model(X)
            # loss = criterion(output, y)
            loss, loss_dice, loss_trace = noisy_label_loss(output, output_cms, labels_all)
            val_loss += loss.item()
            val_loss_dice += loss_dice.item()
            val_loss_trace += loss_trace.item()

            # Calculate the Dice 
            pred = torch.sigmoid(output) > 0.5
            dice = dice_coefficient(pred.float(), y_avrg)
            val_dice += dice.item()

    if GCM:
        save_path = './tf_coc3'
        if TL:
            # save_path = save_path + '/wtTL'
            save_path = save_path + '/wtTLskin'
        else:
            save_path = save_path + '/noTL'
    else:
        save_path = './tf_coc3_lcm'
        if TL:
            save_path = save_path + '/wtTL'
            # save_path = save_path + '/wtTLskin'
        else:
            save_path = save_path + '/noTL'
        test(model, test_loader, noisy_label_loss, save_path, DEVICE)

    plot_performance(train_loss_values, val_loss_values, train_dice_values, val_dice_values, save_path)
    print("Figures were saved.")

train_model(images_path, masks_path, path_to_save, log_path)