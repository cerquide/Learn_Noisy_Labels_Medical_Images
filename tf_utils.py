import numpy as np

import matplotlib.pyplot as plt

import torch

def dice_coefficient(pred, target):

    smooth = 1e-6
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()

    return (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)

def dice_loss(pred, target):

    pred = torch.sigmoid(pred)

    return 1 - dice_coefficient(pred, target)


### GCM ###

def noisy_label_loss(pred, cms, labels, alpha = 0.1):

    main_loss = 0.0
    regularisation = 0.0

    pred_norm = torch.sigmoid(pred)

    mask_prob = pred_norm
    back_prob = 1 - pred_norm

    pred_norm = torch.cat([mask_prob, back_prob], dim = 1)
    b, c, h, w = pred_norm.size()
   
    pred_norm = pred_norm.view(b, c, h*w).permute(0, 2, 1).contiguous().view(b*h*w, c, 1)

    for cm, label_noisy in zip(cms, labels):
        
        cm = cm.view(b, c ** 2, h * w).permute(0, 2, 1).contiguous().view(b * h * w, c * c).view(b * h * w, c, c)

        # normalisation along the rows:
        cm = cm / cm.sum(1, keepdim = True)

        # matrix multiplication to calculate the predicted noisy segmentation:
        # cm: b*h*w x c x c
        # pred_noisy: b*h*w x c x 1
        
        pred_noisy = torch.bmm(cm, pred_norm) #.view(b*h*w, c)
        print("pred_noisy size: ", pred_noisy.size())
        break
        pred_noisy = pred_noisy.view(b, h*w, c).permute(0, 2, 1).contiguous().view(b, c, h, w)

        loss_current = dice_loss(pred_noisy, label_noisy.view(b, h, w).long())
        main_loss += loss_current
        regularisation += torch.trace(torch.transpose(torch.sum(cm, dim = 0), 0, 1)).sum() / (b * h * w)

    regularisation = alpha * regularisation
    loss = main_loss + regularisation

    return loss, main_loss, regularisation


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