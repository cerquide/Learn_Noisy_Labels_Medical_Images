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
from tf_utils import noisy_label_loss_GCM, noisy_label_loss_lCM, noisy_loss, noisy_loss2, noisy_loss3, compute_behavior
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
ALPHA = 0.0
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
def print_matrices(model):
    for name, param in model.named_parameters():
        if 'cms_output' in name:
            print(param)

def print_matrices_on_borders(borders):

    cmA = []
    cmB = []
    cmC = []
    cmDL = []
    for item in borders:
        cmA.append(item[0])
        cmB.append(item[1])
        cmC.append(item[2])
        cmDL.append(item[3])
    
    total_cm = [cmA, cmB, cmC, cmDL]
    
    for i, cm in enumerate(total_cm):

        sum_cm = torch.zeros(2, 2)
        for tensor in cm:
            sum_cm += tensor
        avrg_cm = sum_cm / len(cm)

        if i != 3:
            print("Confusion Matrix of Annotator", (i + 1))
            print(avrg_cm)
        else:
            print("Confusion Matrix of DL model")
            print(avrg_cm)

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
                param.requires_grad = False
        ### ===================== ###

        model.load_state_dict(pretrained_weights, strict = False)
        model.eval()
        print("Weights have been loaded succesfully...")
    
    total_params = sum(p.numel() for p in model.parameters())
    print("Total number of params: ", total_params)
    total_params_grad  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total number of params with grad: ", total_params_grad)

    

    print("Model initialized...")
    # print_matrices(model)
    criterion = nn.BCEWithLogitsLoss(reduction = 'mean')  # The loss function
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
    train_lossAR_values = []
    train_lossHS_values = []
    train_lossSG_values = []
    val_loss_values = []
    val_lossAR_values = []
    val_lossHS_values = []
    val_lossSG_values = []

    print("Training...")
    lossAR, lossHS, lossSG = 0, 0, 0
    borders = []

    for epoch in range(epochs):
        # Train

        #print("Before train")
        #print_matrices(model)

        model.train()
        #print("After train")
        #print_matrices(model)

        train_loss = 0.0
        train_lossAR = 0.0
        train_lossHS = 0.0
        train_lossSG = 0.0
        train_loss_dice = 0.0
        train_loss_trace = 0.0
        train_dice = 0.0
        cm_AR_true = torch.zeros(2,2)
        cm_HS_true = torch.zeros(2,2)
        cm_SG_true = torch.zeros(2,2)
        nb=0
        for name, X, y_AR, y_HS, y_SG, y_avrg in train_loader:

            X, y_AR, y_HS, y_SG, y_avrg = X.to(DEVICE), y_AR.to(DEVICE), y_HS.to(DEVICE), y_SG.to(DEVICE), y_avrg.to(DEVICE)

            labels_all = []
            labels_all.append(y_AR)
            labels_all.append(y_HS)
            labels_all.append(y_SG)

            labels_avrg_all = []
            labels_avrg_all.append(y_avrg)
            
            names = name

            optimizer.zero_grad()

            output, output_cms = model(X)

            borders_cm = compute_behavior(pred = output, labels = labels_all, labels_avrg = labels_avrg_all)
            borders.append(borders_cm)
            
            # Calculate the Loss
            # loss = dice_loss(output, y_avrg)
            # loss, loss_dice, loss_trace = noisy_loss2(output, output_cms, labels_all, names)
            # loss, loss_dice, loss_trace, lossAR, lossHS, lossSG = noisy_loss3(output, output_cms, labels_all, labels_avrg_all, names)
            
            loss, loss_dice, loss_trace = noisy_label_loss(output, output_cms, labels_all, names, alpha = ALPHA)

            # loss, loss_dice, loss_cm = combined_loss(pred = output, cms = output_cms, ys = [y_AR, y_HS, y_SG, y_avrg])

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_lossAR += lossAR
            train_lossHS += lossHS
            train_lossSG += lossSG
            train_loss_dice += loss_dice.item()
            train_loss_trace += loss_trace.item()

            # Calculate the Dice
            pred = torch.sigmoid(output) > 0.5
            train_dice_ = dice_coefficient(pred.float(), y_avrg)
            train_dice += train_dice_.item()

            # # print CMs Real
            # cm_AR_true += calculate_cm(y_pred = y_AR, y_true = y_avrg)
            # cm_HS_true += calculate_cm(y_pred = y_HS, y_true = y_avrg)
            # cm_SG_true += calculate_cm(y_pred = y_SG, y_true = y_avrg)
            # nb+=1
            # # Real CMs #
            # # ======== #
            # print("==== Real CMs ====")
            # print("AR CM: ", cm_AR_true/nb)
            # print("HS CM: ", cm_HS_true/nb)
            # print("SG CM: ", cm_SG_true/nb)
        
        # for cm in output_cms:
        #     print("CM :", cm[0, :, :, 0, 0])

        train_loss /= len(train_loader)
        train_lossAR /= len(train_loader)
        train_lossHS /= len(train_loader)
        train_lossSG /= len(train_loader)
        train_loss_dice /= len(train_loader)
        train_loss_trace /= len(train_loader)
        train_dice /= len(train_loader)

        
        # Validate
        model.eval()
        val_loss = 0.0
        val_lossAR = 0.0
        val_lossHS = 0.0
        val_lossSG = 0.0
        val_loss_dice = 0.0
        val_loss_trace = 0.0
        val_dice = 0.0
        with torch.no_grad():
            for name, X, y_AR, y_HS, y_SG, y_avrg in val_loader:

                X, y_AR, y_HS, y_SG, y_avrg = X.to(DEVICE), y_AR.to(DEVICE), y_HS.to(DEVICE), y_SG.to(DEVICE), y_avrg.to(DEVICE)

                labels_all = []
                labels_all.append(y_AR)
                labels_all.append(y_HS)
                labels_all.append(y_SG)

                names = name

                # if GCM == False:
                #     cm_all_true = []
                #     cm_AR_true = calculate_cm(pred = y_AR, true = y_avrg)
                #     cm_HS_true = calculate_cm(pred = y_HS, true = y_avrg)
                #     cm_SG_true = calculate_cm(pred = y_SG, true = y_avrg)
                    
                #     cm_all_true.append(cm_AR_true)
                #     cm_all_true.append(cm_HS_true)
                #     cm_all_true.append(cm_SG_true)

                # Calculate the Loss 
                output, output_cms = model(X)
                # loss = criterion(output, y)
                # loss, loss_dice, loss_trace = noisy_loss2(output, output_cms, labels_all, names)
                # loss, loss_dice, loss_trace, lossAR, lossHS, lossSG = noisy_loss3(output, output_cms, labels_all, names)
                loss, loss_dice, loss_trace = noisy_label_loss(output, output_cms, labels_all, names, alpha = ALPHA)

                val_loss += loss.item()
                val_lossAR += lossAR
                val_lossHS += lossHS
                val_lossSG += lossSG
                val_loss_dice += loss_dice.item()
                val_loss_trace += loss_trace.item()

                # insert evaluate_cms(pred = torch.sigmoid(output), ...)
                # mse_outputs, mses = evaluate_cm(pred = torch.sigmoid(output), pred_cm = output_cms, true_cm = cm_all_true)

                # Calculate the Dice 
                pred = torch.sigmoid(output) > 0.5
                dice = dice_coefficient(pred.float(), y_avrg)
                val_dice += dice.item()

            val_loss /= len(val_loader)
            val_lossAR /= len(val_loader)
            val_lossHS /= len(val_loader)
            val_lossSG /= len(val_loader)
            val_loss_dice /= len(val_loader)
            val_loss_trace /= len(val_loader)
            val_dice /= len(val_loader)

        train_loss_values.append(train_loss)
        train_lossAR_values.append(train_lossAR)
        train_lossHS_values.append(train_lossHS)
        train_lossSG_values.append(train_lossSG)
        val_loss_values.append(val_loss)
        val_lossAR_values.append(val_lossAR)
        val_lossHS_values.append(val_lossHS)
        val_lossSG_values.append(val_lossSG)
        train_dice_values.append(train_dice)
        val_dice_values.append(val_dice)

        print(f'Epoch: {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Train Loss Dice: {train_loss_dice:.4f}, Train Dice: {train_dice:.4f}, Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}')
        
        # numpy_array = output_cms[0][0, :, :, 0, 0].cpu().numpy()
        # formatted_array = np.array2string(numpy_array, precision=4, separator=', ')
        # print(f'Annotator 1: {formatted_array}')
        # numpy_array = output_cms[1][0, :, :, 0, 0].cpu().numpy()
        # formatted_array = np.array2string(numpy_array, precision=4, separator=', ')
        # print(f'Annotator 2: {formatted_array}')
        # numpy_array = output_cms[2][0, :, :, 0, 0].cpu().numpy()
        # formatted_array = np.array2string(numpy_array, precision=4, separator=', ')
        # print(f'Annotator 3: {formatted_array}')

        # print_matrices(model)
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
    print(len(borders))
    print(len(borders[0]))
    print(borders[0][0].size())

    print_matrices_on_borders(borders = borders)

    # Test
    with torch.no_grad():
        for name, X, y_AR, y_HS, y_SG, y_avrg in test_loader:

            X, y_AR, y_HS, y_SG, y_avrg = X.to(DEVICE), y_AR.to(DEVICE), y_HS.to(DEVICE), y_SG.to(DEVICE), y_avrg.to(DEVICE)

            labels_all = []
            labels_all.append(y_AR)
            labels_all.append(y_HS)
            labels_all.append(y_SG)

            names = name

            # Calculate the Loss 
            output, output_cms = model(X)
            # loss = criterion(output, y)
            # loss, loss_dice, loss_trace = noisy_loss2(output, output_cms, labels_all, names)
            # loss, loss_dice, loss_trace, _, _, _ = noisy_loss3(output, output_cms, labels_all, names)
            loss, loss_dice, loss_trace = noisy_label_loss(output, output_cms, labels_all, names, alpha = ALPHA)

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
    plot_performance(train_lossAR_values, val_lossAR_values, train_dice_values, val_dice_values, save_path, 'AR')
    plot_performance(train_lossHS_values, val_lossHS_values, train_dice_values, val_dice_values, save_path, 'HS')
    plot_performance(train_lossSG_values, val_lossSG_values, train_dice_values, val_dice_values, save_path, 'SG')
    print("Figures were saved.")

train_model(images_path, masks_path, path_to_save, log_path)