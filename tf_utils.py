import numpy as np
import os

import matplotlib.pyplot as plt

import torch
import torchvision.utils as vutils

from sklearn.metrics import confusion_matrix

def dice_coefficient(pred, target):

    smooth = 1e-6
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()

    return (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)

def dice_loss(pred, target):
    
    pred = torch.sigmoid(pred)

    return 1 - dice_coefficient(pred, target)

def dice_loss2(pred, target):
    # print(pred.size())
    # print(target.size())
    # target = target.unsqueeze(1)

    return 1 - dice_coefficient(pred, target)

### GCM ###

def noisy_label_loss_GCM(pred, cms, labels, alpha = 0.1):

    main_loss = 0.0
    regularisation = 0.0

    pred_norm = torch.sigmoid(pred)

    mask_prob = pred_norm
    back_prob = 1 - pred_norm

    pred_norm = torch.cat([mask_prob, back_prob], dim = 1)
    b, c, h, w = pred_norm.size()
   
    pred_norm = pred_norm.view(b, c, h*w).permute(0, 2, 1).contiguous().view(b*h*w, c, 1)

    for cm, label_noisy in zip(cms, labels):

        #print("CM :", cm[0, :, :, 0, 0])

        cm = cm.view(b, c ** 2, h * w).permute(0, 2, 1).contiguous().view(b * h * w, c * c).view(b * h * w, c, c)

        # normalisation along the rows:
        cm = cm / cm.sum(1, keepdim = True)

        # matrix multiplication to calculate the predicted noisy segmentation:
        # cm: b*h*w x c x c
        # pred_noisy: b*h*w x c x 1
        
        pred_noisy = torch.bmm(cm, pred_norm) #.view(b*h*w, c)
        
        pred_noisy = pred_noisy.view(b, h*w, c).permute(0, 2, 1).contiguous().view(b, c, h, w)
        pred_noisy_mask = pred_noisy[:, 0, :, :]
        pred_noisy = pred_noisy_mask.unsqueeze(1)

        loss_current = dice_loss(pred_noisy, label_noisy.view(b, h, w).long())
        main_loss += loss_current
        regularisation += torch.trace(torch.transpose(torch.sum(cm, dim = 0), 0, 1)).sum() / (b * h * w)
    #print("=====================")
    

    regularisation = alpha * regularisation
    loss = main_loss + regularisation

    return loss, main_loss, regularisation

### lCM ###

def print_cms(cms):

    b, c, w, h = cms[0].size()

    cms_ar = cms[0].view(b, 2, 2, w, h)
    cms_ar = torch.nn.Softmax(dim = 1)(cms_ar)
    print("AR CM: ", cms_ar[0, :, :, 0, 0])
    

def noisy_label_loss_lCM(pred, cms, labels, alpha = 0.1):

    main_loss = 0.0
    regularisation = 0.0

    pred_norm = torch.sigmoid(pred)

    # mask_prob = pred_norm
    # back_prob = 1 - pred_norm

    # pred_norm = torch.cat([mask_prob, back_prob], dim = 1)
    b, c, w, h = pred_norm.size()
    # print(pred_norm.size())
    pred_norm = pred_norm.view(b, c, h*w).permute(0, 2, 1).contiguous().view(b*h*w, c, 1)
    
    # print_cms(cms)
    for cm, label_noisy in zip(cms, labels):
        # print(cm.size())
        # print(cm[0, :, 0, 0])
        cm = cm[:, 0, :, :].unsqueeze(1)
        cm = cm.view(b, c ** 2, h * w).permute(0, 2, 1).contiguous().view(b * h * w, c * c).view(b * h * w, c, c)

        # normalisation along the rows:
        cm = cm / cm.sum(1, keepdim = True)

        # matrix multiplication to calculate the predicted noisy segmentation:
        # cm: b*h*w x c x c
        # pred_noisy: b*h*w x c x 1
        
        pred_noisy = torch.bmm(cm, pred_norm) #.view(b*h*w, c)
        
        pred_noisy = pred_noisy.view(b, h*w, c).permute(0, 2, 1).contiguous().view(b, c, w, h)
        # pred_noisy_mask = pred_noisy[:, 0, :, :]
        # pred_noisy = pred_noisy_mask.unsqueeze(1)

        loss_current = dice_loss(pred_noisy, label_noisy.view(b, h, w).long())
        main_loss += loss_current
        regularisation += torch.trace(torch.transpose(torch.sum(cm, dim = 0), 0, 1)).sum() / (b * h * w)

    regularisation = alpha * regularisation
    loss = main_loss + regularisation

    return loss, main_loss, regularisation

def combined_loss(pred, cms, ys):

    dice_loss = 0.
    cms_loss = 0.
    total_loss = 0.

    pred_norm = torch.sigmoid(pred)

    b, c, w, h = pred_norm.size()

    y_AR, y_HS, y_SG, y_avrg = ys[0], ys[1], ys[2], ys[3]

    cm_AR = torch.tensor(calculate_cm(pred = y_AR, true = y_avrg))
    cm_HS = torch.tensor(calculate_cm(pred = y_HS, true = y_avrg))
    cm_SG = torch.tensor(calculate_cm(pred = y_SG, true = y_avrg))

    print("CM size: ", cm_AR.size())

    cm_AR_reshaped = cm_AR.unsqueeze(0).repeat(b, 1, 1).unsqueeze(-1).repeat(1, 1, 1, w).unsqueeze(-1).repeat(1, 1, 1, 1, h)
    print("CM resize: ", cm_AR_reshaped.size())   
    print("CM pred size: ", cms[0].size())

    

    # pred_norm = pred_norm.view(b, c, h*w).permute(0, 2, 1).contiguous().view(b*h*w, c, 1)


    return total_loss, dice_loss, cms_loss

def calculate_cm(pred, true):
   
    pred = pred.view(-1)
    true = true.view(-1)

    pred = pred.cpu().detach().numpy()
    true = true.cpu().detach().numpy()

    confusion_matrices = confusion_matrix(y_true = np.round(true).astype(int), y_pred = np.round(pred).astype(int), normalize = 'all')
    
    return confusion_matrices

def evaluate_cm(pred, pred_cm, true_cm):

    # print("pred: ", pred.size())
    # print("pred_cm len: ", len(pred_cm))
    # print("pred_cm: ", pred_cm[0].size())
    # print("true_cm len: ", len(true_cm))
    # print("true_cm: ", torch.from_numpy(true_cm[0]).size())

    b, c, w, h = pred.size()
    nnn = 1
    
    pred = pred.reshape(b, c, h * w)
    pred = pred.permute(0, 2, 1).contiguous()
    pred = pred.view(b * h * w, c).view(b * h * w, c, 1)
    # mean squared error
    mse = 0
    outputs = []
    mses = []

    for j, cm in enumerate(pred_cm):
        
        cm = cm[:, 0, :, :].unsqueeze(1)
        cm = cm.view(b, c ** 2, h * w).permute(0, 2, 1).contiguous().view(b * h * w, c * c).view(b * h * w, c, c)
        cm = cm / cm.sum(1, keepdim = True)
        if j < len(true_cm):

            cm_pred_ = cm.sum(0) / (b * h * w)
            cm_pred_ = cm_pred_.cpu().detach().numpy()
            # print(cm_pred_)
            cm_true_ = true_cm[j]
            # print(cm_true_)
            
            diff = cm_pred_ - cm_true_
            diff_squared = diff ** 2

            mse += diff_squared.mean()
            # print(mse)
        
        mses.append(mse)

        output = torch.bmm(cm, pred).view(b * h * w, c)
        output = output.view(b, h*w, c).permute(0, 2, 1).contiguous().view(b, c, h, w)
        
        output = output > 0.5
        # print("output shape: ", output.shape)
        outputs.append(output)

    return outputs, mses

### Testing ###
def test_lGM(model, test_loader, noisy_label_loss, save_path, device = 'cuda'):

    model.eval()

    test_loss = 0.0
    test_loss_dice = 0.0
    test_loss_trace = 0.0
    test_dice = 0.0

    with torch.no_grad():
        for i, (X, y_AR, y_HS, y_SG, y_avrg) in enumerate(test_loader):

            X, y_AR, y_HS, y_SG, y_avrg = X.to(device), y_AR.to(device), y_HS.to(device), y_SG.to(device), y_avrg.to(device)

            labels_all = []
            labels_all.append(y_AR)
            labels_all.append(y_HS)
            labels_all.append(y_SG)

            output, output_cms = model(X)

            for j in range(len(output)):

                image_path = os.path.join(save_path, "batch{}_image{}.png".format(i, j))
                mask_path = os.path.join(save_path, "batch{}_mask{}.png".format(i, j))
                vutils.save_image(X[j], image_path)
                vutils.save_image(output[j], mask_path)

            loss, loss_dice, loss_trace = noisy_label_loss(output, output_cms, labels_all)

            test_loss += loss.item()
            test_loss_dice += loss_dice.item()
            test_loss_trace += loss_trace.item()

            # Calculate the Dice 
            pred = torch.sigmoid(output) > 0.5
            dice = dice_coefficient(pred.float(), y_avrg)
            test_dice += dice.item()

        test_loss /= len(test_loader)
        test_loss_dice /= len(test_loader)
        test_loss_trace /= len(test_loader)
        test_dice /= len(test_loader)

    print(f'Test data size: {len(test_loader)}, Test Loss: {test_loss:.4f}, Test Loss Dice: {test_loss_dice:.4f}, Test Dice: {test_dice:.4f}')

### Plotting ###
def plot_performance(train_losses, val_losses, train_dices, val_dices, fig_path):
    epochs = range(1, len(train_losses) + 1)

    # Plot losses
    plt.plot(epochs, train_losses, 'b', label='Training loss')
    plt.plot(epochs, val_losses, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.ylim([0., 1.])
    plt.yticks([0.1 * i for i in range(11)])
    plt.grid(True)
    plt.savefig(fig_path + '/losses.png')
    plt.close()

    # Plot dices
    plt.plot(epochs, train_dices, 'b', label='Training dice')
    plt.plot(epochs, val_dices, 'r', label='Validation dice')
    plt.title('Training and validation dice')
    plt.xlabel('Epochs')
    plt.ylabel('Dice')
    plt.legend()
    plt.ylim([0., 1.])
    plt.yticks([0.1 * i for i in range(11)])
    plt.grid(True)
    plt.savefig(fig_path + '/dices.png')
    plt.close()