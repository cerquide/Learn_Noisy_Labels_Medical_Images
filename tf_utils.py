import numpy as np
import os

import matplotlib.pyplot as plt

import torch
import torchvision.utils as vutils

from sklearn.metrics import confusion_matrix

DEVICE = 'cuda'

def dice_coefficient(pred, target):

    smooth = 1e-6
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()

    return (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)

def dice_coefficient2(pred, target):

    smooth = 1e-6
    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)
    intersection = (pred_flat * target_flat).sum()

    return (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)

def dice_loss(pred, target):
    
    pred = torch.sigmoid(pred)

    return 1 - dice_coefficient(pred, target)

def dice_loss2(pred, target):
    # print(pred.size())
    # print(target.size())
    # target = target.unsqueeze(1)

    return 1 - dice_coefficient2(pred, target)

### GCM ###
def save_histogram(tensor):

    tensor = tensor.cpu()
    for i in range(tensor.shape[0]):
        bins = np.arange(0.0, 1.1, 0.1)
        hist, _ = np.histogram(tensor[i].numpy(), bins = bins)

        plt.bar(bins[:-1], hist, width = 0.1)

        plt.savefig(f'./tf_coc3/wtTL/histograms/histogram_{i}.png')
        plt.clf()
    
def save_borders(boolean, tensor, annotator = 1, names = []):

    print(names)
    borders = boolean * tensor
    borders = borders.cpu()

    for i in range(tensor.shape[0]):

        plt.imshow(borders[i].detach().numpy())
        plt.savefig(f'./tf_coc3/wtTL/borders/border_{names[i]}_annotator_{annotator}.png')
        plt.clf()

def clear_cms(tensor, label, threshold = 1e-3):

    x = tensor
    y = label

    indices = (x >= threshold).nonzero()

    cleared_tensor = x[indices].squeeze()
    cleared_label = y[indices].squeeze()

def clear_pred(pred, increment = 0.05):

    # Boolean #
    increment = 0.05
    clear_tensor = torch.logical_or(pred >= (1 - increment), pred <= increment)
    dirty_tensor = torch.logical_and(pred < (1 - increment), pred > increment)

    # Count number of True and False values
    num_true_c = clear_tensor.sum().item()
    num_false_c = (clear_tensor.numel() - num_true_c)
    num_true_u = dirty_tensor.sum().item()
    num_false_u = (dirty_tensor.numel() - num_true_u)

    # prints
    
    return clear_tensor, dirty_tensor

def round_to_01(tensor, threshold):
    rounded_tensor = torch.where(tensor >= threshold, torch.ones_like(tensor), torch.zeros_like(tensor))
    return rounded_tensor

def noisy_loss(pred, cms, labels, names):

    main_loss = 0.0

    pred_norm = torch.sigmoid(pred)
    pred_init = pred_norm

    b, c, h, w = pred_norm.size()
    pred_flat = pred_norm.view(b, c * h * w)
    # print(pred_flat.size())
    labels_flat_list = []
    labels_part = []
    for labels_list in labels:
        for label in labels_list:
            label_flat = label.view(1, h * w)
            labels_part.append(label_flat)
        labels_tensor = torch.cat(labels_part, dim = 0)
        labels_flat_list.append(labels_tensor)
        labels_part = []
    # print(len(labels_flat_list))
    # print(len(labels_flat_list[0]))
    # print(labels_flat_list[0][0].size())

    threshold = 0.05
    indices = []
    focus_pred = []
    focus_labels = []
    focus_labels1 = []
    focus_labels2 = []
    focus_labels3 = []
    for i in range(b):

        mask = (pred_flat[i] > threshold) & (pred_flat[i] < (1 - threshold))
        indices.append(torch.nonzero(mask))
        # print(indices[i].size())
        new_tensor = torch.zeros(1, indices[i].size(0))
        new_label1 = torch.zeros(1, indices[i].size(0))
        new_label2 = torch.zeros(1, indices[i].size(0))
        new_label3 = torch.zeros(1, indices[i].size(0))
        # print(new_tensor.size())
        # print(new_tensor)

        position = 0
        for j in range(pred_flat[i].size(-1)):
            index = torch.where(indices[i] == j)[0]
            if len(index) > 0:
                # print(j)
                new_tensor[0, position] = pred_flat[i, j]
                new_label1[0, position] = labels_flat_list[0][i][j]
                new_label2[0, position] = labels_flat_list[1][i][j]
                new_label3[0, position] = labels_flat_list[2][i][j]
                position += 1

        mask_prob = new_tensor.unsqueeze(1)
        back_prob = (1 - new_tensor).unsqueeze(1)
        new_tensor = torch.cat([mask_prob, back_prob], dim = 1)
        focus_pred.append(new_tensor)
        focus_labels1.append(new_label1)
        focus_labels2.append(new_label2)
        focus_labels3.append(new_label3)
    focus_labels.append(focus_labels1)
    focus_labels.append(focus_labels2)
    focus_labels.append(focus_labels3)

    # print(len(focus_labels))
    # print(len(focus_labels1))
    # print(focus_labels1[0].size())
    # print(focus_pred[0].size())

    enum = 0
    total_loss = 0
    total_loss3 = 0
    annotators_loss = []
    
    for cm, label in zip(cms, focus_labels):
        enum += 1

        batch_loss = 0
        # print(len(focus_pred)) 
        for i in range(len(focus_pred)):

            cm_simple = cm[i, :, :, 0, 0].unsqueeze(0).unsqueeze(-1).repeat(1, 1, 1, focus_pred[i].size(2)).to('cuda')
            # print(cm_simple.size())
            
            a1, a2, a3, a4 = cm_simple.size()

            cm_simple = cm_simple.view(a1, a2 * a3, a4).view(a1 * a4, a2 * a3).view(a1 * a4, a2, a3)
            focus_pred_ = focus_pred[i].permute(2, 1, 0)

            # print(cm_simple.size())
            # print(focus_pred_.size())
            pred_noisy = torch.bmm(cm_simple.to(DEVICE), focus_pred_.to(DEVICE))

            pred_noisy = pred_noisy.view(a1, a4, a2).permute(0, 2, 1).contiguous().view(a1, a2, a4)
            pred_noisy_mask = pred_noisy[:, 0, :]

            # print(pred_noisy_mask.size())
            # print(label[i].size())

            loss_current = dice_loss2(pred_noisy_mask.to(DEVICE), label[i].to(DEVICE))

            batch_loss += loss_current
        batch_loss = batch_loss / (i + 1)
        total_loss += batch_loss
        total_loss3 += total_loss
        # print("Annotator", enum)
        # print("Loss: ", total_loss.item())
    # print("Annotator 1:", cms[0][0, :, :, 0, 0])
    # print("Annotator 2:", cms[1][0, :, :, 0, 0])
    # print("Annotator 3:", cms[2][0, :, :, 0, 0])
    # print("Total Loss: ", total_loss3.item())

    return total_loss3, total_loss3, total_loss3 * 0

def noisy_loss2(pred, cms, labels, names):

    total_loss = 0.0
    pred_norm = torch.sigmoid(pred)

    b, c, h, w = pred_norm.size()
    pred_flat = pred_norm.view(b, c * h * w)

    labels_flat_list = []
    for labels_list in labels:
        labels_tensor = torch.cat([label.view(1, h * w) for label in labels_list], dim=0)
        labels_flat_list.append(labels_tensor)

    # print("Pred_flat size: ", pred_flat.size())
    # print("Pred_norm size: ", pred_norm.size())
    # print("Len labels_flat: ", len(labels_flat_list))
    # print("Size labels_flat[0]: ", labels_flat_list[0].size())
    # print("labels_flat[0][0]: ", labels_flat_list[0][0])
    # print("Zero count: ", torch.count_nonzero(torch.eq(labels_flat_list[0][0], 0)))
    # print("One count: ", torch.count_nonzero(torch.eq(labels_flat_list[0][0], 1)))
    # print("Non 0-1 count: ", labels_flat_list[0][0].size(0)
    #        - torch.count_nonzero(torch.eq(labels_flat_list[0][0], 0)) 
    #        - torch.count_nonzero(torch.eq(labels_flat_list[0][0], 1))
    #        )
    
    threshold = 0.05
    focus_pred = []
    focus_labels = []
    
    for i in range(b):
        # print(i)
        mask = (pred_flat[i] > threshold) & (pred_flat[i] < (1 - threshold))
        # for j in range(len(pred_flat[i])):
            # if (pred_flat[i, j] > threshold) & (pred_flat[i, j] < (1 - threshold)):
                # print(pred_flat[i, j])
   
        indices = torch.nonzero(mask)
        
        new_tensor = round_to_01(pred_flat[i, indices[:, 0]], 0.5)
        # print("new tensor size: ", new_tensor.size())
        # print("new tensor: ", new_tensor)
        # print("Number of zero elements:", torch.eq(new_tensor, 0).sum().item())
        # print("Number of one elements:", new_tensor.numel() - torch.eq(new_tensor, 0).sum().item())

        new_labels = [labels_flat_list[j][i, indices[:, 0]] for j in range(len(labels_flat_list))]
        new_labels[0] = round_to_01(new_labels[0], 0.5)
        new_labels[1] = round_to_01(new_labels[1], 0.5)
        new_labels[2] = round_to_01(new_labels[2], 0.5)
        
        # print("len new labels: ", len(new_labels))
        # print("new_labels[0]", new_labels[0])
        # print("Number of zero elements:", torch.eq(new_labels[0], 0).sum().item())
        # print("Number of one elements:", new_labels[0].numel() - torch.eq(new_labels[0], 0).sum().item())
        # print("new_labels[1]", new_labels[1])
        # print("Number of zero elements:", torch.eq(new_labels[1], 0).sum().item())
        # print("Number of one elements:", new_labels[1].numel() - torch.eq(new_labels[1], 0).sum().item())
        # print("new_labels[2]", new_labels[2])
        # print("Number of zero elements:", torch.eq(new_labels[1], 0).sum().item())
        # print("Number of one elements:", new_labels[2].numel() - torch.eq(new_labels[1], 0).sum().item())

        mask_prob = new_tensor.unsqueeze(0)
        back_prob = (1 - new_tensor).unsqueeze(0)
        new_tensor = torch.cat([mask_prob, back_prob], dim = 0)
        focus_pred.append(new_tensor)
        focus_labels.append(new_labels)
    
    # print("Len focus_pred: ", len(focus_pred))
    # print("focus_pred: ", focus_pred[0])
    # print("Size focus_pred[0]: ", focus_pred[0].size())
    # print("Len focus_labels: ", len(focus_labels))
    # print("Size focus_labels[0][0]: ", focus_labels[0][0].size())
    # print("focus_labels[0][0]", focus_labels[0][0])

     # Reorganize focus labels:
    new_labels_list = []
    listA = []
    listB = []
    listC = []

    # Create 3 lists of 16 lists each
    for labels_list in focus_labels:
        listA.append(labels_list[0])
        listB.append(labels_list[1])
        listC.append(labels_list[2])
    
    new_labels_list.append(listA)
    new_labels_list.append(listB)
    new_labels_list.append(listC)

    lossAR = 0.0
    lossHS = 0.0
    lossSG = 0.0

    enum = 0
    for cm, label in zip(cms, new_labels_list):
        enum += 1
        batch_loss = 0
        print("Annotator ", enum)
        mean_confusion_matrix = []

        for i, focus_pred_i in enumerate(focus_pred):
            
            # print("CM size: ", cm.size())
            # print("CM[i, :, :, 0, 0] size: ", cm[i, :, :, 0, 0].size())
            cm_simple = cm[i, :, :, 0, 0].unsqueeze(0).unsqueeze(-1).repeat(1, 1, 1, focus_pred_i.size(1)).to('cuda')
            # print("CM_simple size: ", cm_simple.size())
            a1, a2, a3, a4 = cm_simple.size()
            # print("a1 =", a1)
            # print("a2 =", a2)
            # print("a3 =", a3)
            # print("a4 =", a4)
            cm_simple = cm_simple.view(a1 * a4, a2 * a3).view(a1 * a4, a2, a3)
            # print("CM_simple size: ", cm_simple.size())
            # print("Size focus_pred: ", focus_pred_i.unsqueeze(-1).permute(1, 0, 2).size())
            # print("focus_pred: ", focus_pred_i.unsqueeze(-1).permute(1, 0, 2)[:10, 0, :])
            # print("cm_simple: ", cm_simple)
            pred_noisy = torch.bmm(cm_simple.to(DEVICE), focus_pred_i.unsqueeze(-1).permute(1, 0, 2).to(DEVICE))
            # print("Pred_noisy size: ", pred_noisy.size())
            pred_noisy = pred_noisy.view(a1, a4, a2).permute(0, 2, 1).contiguous().view(a1, a2, a4)
            # print("Pred_noisy size: ", pred_noisy.size())
            pred_noisy_mask = pred_noisy[:, 0, :]
            # print("Pred_noisy: ", pred_noisy.size())
            # print("Label: ", label[i].size())
            # print("pred_noisy: ", pred_noisy[0, 0, :].size())
            # print("Label: ", label[i][:10])
            # print("Confusion matrix: ")
            confusion_matrix_np = confusion_matrix( np.round(label[i].cpu().detach().numpy()).astype(int), 
                                                    np.round(pred_noisy[0, 0, :].cpu().detach().numpy()).astype(int))
            # print(torch.nn.functional.normalize(torch.from_numpy(confusion_matrix_np).float(), p = 1, dim = 0))
            mean_confusion_matrix.append(torch.nn.functional.normalize(torch.from_numpy(confusion_matrix_np).float(), p = 1, dim = 0))

            loss_current = dice_loss2(pred_noisy_mask.to(DEVICE), label[i].to(DEVICE))
            # print("Step loss: ", loss_current.item())
            batch_loss += loss_current
        
        if enum == 1:
            lossAR += batch_loss.item() / len(focus_pred)
            print("Confusion matrix AR: ")
            print(torch.mean(torch.stack(mean_confusion_matrix, dim = 0), dim = 0))
        elif enum == 2:
            lossHS += batch_loss.item() / len(focus_pred)
            print("Confusion matrix HS: ")
            print(torch.mean(torch.stack(mean_confusion_matrix, dim = 0), dim = 0))
        elif enum == 3:
            lossSG += batch_loss.item() / len(focus_pred)
            print("Confusion matrix SG: ")
            print(torch.mean(torch.stack(mean_confusion_matrix, dim = 0), dim = 0))
        else:
            print("loss not assigned...")

        # print("Annotator's loss: ", batch_loss.item() / len(focus_pred))
        
        batch_loss /= len(focus_pred)
        total_loss += batch_loss
    
    print(f'Total loss: {total_loss.item():.4f}, AR Loss: {lossAR:.4f}, HS Loss: {lossHS:.4f}, SG Loss: {lossSG:.4f}')

    # print("Total loss: ", total_loss.item())
    
    return total_loss, total_loss, total_loss * 0

def noisy_loss3(pred, cms, labels, names):

    total_loss = 0.0
    pred_norm = torch.sigmoid(pred)

    b, c, h, w = pred_norm.size()
    pred_flat = pred_norm.view(b, c * h * w)

    labels_flat_list = []
    for labels_list in labels:
        labels_tensor = torch.cat([label.view(1, h * w) for label in labels_list], dim = 0)
        labels_flat_list.append(labels_tensor)

    # print("Pred_flat size: ", pred_flat.size())
    # print("Pred_norm size: ", pred_norm.size())
    # print("Len labels_flat: ", len(labels_flat_list))
    # print("Size labels_flat[0]: ", labels_flat_list[0].size())
    # print("labels_flat[0][0]: ", labels_flat_list[0][0])
    # print("Zero count: ", torch.count_nonzero(torch.eq(labels_flat_list[0][0], 0)))
    # print("One count: ", torch.count_nonzero(torch.eq(labels_flat_list[0][0], 1)))
    # print("Non 0-1 count: ", labels_flat_list[0][0].size(0)
    #        - torch.count_nonzero(torch.eq(labels_flat_list[0][0], 0)) 
    #        - torch.count_nonzero(torch.eq(labels_flat_list[0][0], 1))
    #        )

    cms_flat_list = []
    for cms_list in cms:
        cms_tensor = torch.cat([cm.view(cm.size(0), h * w) for cm in cms_list], dim = 0)
        cms_flat_list.append(cms_tensor)

    # print("Len cms_flat: ", len(cms_flat_list))
    # print("Size cms_flat[0]: ", cms_flat_list[0].size())
    # print("cms_flat[0][0]: ", cms_flat_list[0][0])
    
    threshold = 0.05
    focus_pred = []
    focus_labels = []
    focus_cms = []
    
    for i in range(b):
        # print(i)
        mask = (pred_flat[i] > threshold) & (pred_flat[i] < (1 - threshold))
        # for j in range(len(pred_flat[i])):
            # if (pred_flat[i, j] > threshold) & (pred_flat[i, j] < (1 - threshold)):
                # print(pred_flat[i, j])
        
        indices = torch.nonzero(mask)
        
        new_tensor = round_to_01(pred_flat[i, indices[:, 0]], 0.5)
        # print("new tensor size: ", new_tensor.size())
        # print("new tensor: ", new_tensor)
        # print("Number of zero elements:", torch.eq(new_tensor, 0).sum().item())
        # print("Number of one elements:", new_tensor.numel() - torch.eq(new_tensor, 0).sum().item())

        new_labels = [labels_flat_list[j][i, indices[:, 0]] for j in range(len(labels_flat_list))]
        new_labels[0] = round_to_01(new_labels[0], 0.5)
        new_labels[1] = round_to_01(new_labels[1], 0.5)
        new_labels[2] = round_to_01(new_labels[2], 0.5)

        new_cms1 = [cms_flat_list[j][i, indices[:, 0]] for j in range(len(cms_flat_list))]
        new_cms2 = [cms_flat_list[j][i + b, indices[:, 0]] for j in range(len(cms_flat_list))]
        new_cms3 = [cms_flat_list[j][i + b * 2, indices[:, 0]] for j in range(len(cms_flat_list))]
        new_cms4 = [cms_flat_list[j][i + b * 3, indices[:, 0]] for j in range(len(cms_flat_list))]
        
        # print("len new labels: ", len(new_labels))
        # print("len new cms1: ", len(new_cms1))
        # print("len new cms2: ", len(new_cms2))
        # print("len new cms3: ", len(new_cms3))
        # print("len new cms4: ", len(new_cms4))
        # print("new_labels[0]", new_labels[0].size())
        # print("new_cms1[0]", new_cms1[0].size())
        # print("new_cms2[0]", new_cms2[0].size())
        # print("new_cms3[0]", new_cms3[0].size())
        # print("new_cms4[0]", new_cms4[0].size())

        # print("Number of zero elements:", torch.eq(new_labels[0], 0).sum().item())
        # print("Number of one elements:", new_labels[0].numel() - torch.eq(new_labels[0], 0).sum().item())
        # print("new_labels[1]", new_labels[1].size())
        # print("new_cms[1]", new_cms1[1].size())
        # print("Number of zero elements:", torch.eq(new_labels[1], 0).sum().item())
        # print("Number of one elements:", new_labels[1].numel() - torch.eq(new_labels[1], 0).sum().item())
        # print("new_labels[2]", new_labels[2])
        # print("new_cms[2]", new_cms1[2])
        # print("Number of zero elements:", torch.eq(new_labels[1], 0).sum().item())
        # print("Number of one elements:", new_labels[2].numel() - torch.eq(new_labels[1], 0).sum().item())

        mask_prob = new_tensor.unsqueeze(0)
        back_prob = (1 - new_tensor).unsqueeze(0)
        new_tensor = torch.cat([mask_prob, back_prob], dim = 0)
        focus_pred.append(new_tensor)
        focus_labels.append(new_labels)
        new_cmsA = torch.cat((torch.cat((torch.cat((new_cms1[0], new_cms2[0]), dim = 0), new_cms3[0]), dim = 0), new_cms4[0]), dim = 0)
        new_cmsB = torch.cat((torch.cat((torch.cat((new_cms1[1], new_cms2[1]), dim = 0), new_cms3[1]), dim = 0), new_cms4[1]), dim = 0)
        new_cmsC = torch.cat((torch.cat((torch.cat((new_cms1[2], new_cms2[2]), dim = 0), new_cms3[2]), dim = 0), new_cms4[2]), dim = 0)
        focus_cms.append([new_cmsA, new_cmsB, new_cmsC])
        # print(len(focus_labels[0][0]))
        # print(len(focus_cms))
        # print(len(focus_cms[0]))

    # print("Len focus_pred: ", len(focus_pred))
    # print("focus_pred: ", focus_pred[0])
    # print("Size focus_pred[0]: ", focus_pred[0].size())
    # print("Len focus_labels: ", len(focus_labels))
    # print("Len focus_cms: ", len(focus_cms))
    # print("Size focus_labels[0][0]: ", focus_labels[0][0].size())
    # print("Size focus_cms[0][0]: ", focus_cms[0][0].size())
    # print("focus_cms[0][0]: ", focus_cms[0][0])
    # print("focus_labels[0][0]", focus_labels[0][0])
    
     # Reorganize focus labels:
    new_labels_list = []
    listA = []
    listB = []
    listC = []

    # Create 3 lists of 16 lists each
    for labels_list in focus_labels:
        listA.append(labels_list[0])
        listB.append(labels_list[1])
        listC.append(labels_list[2])
    
    new_labels_list.append(listA)
    new_labels_list.append(listB)
    new_labels_list.append(listC)

    # Reorganize focus cms:
    new_cms_list = []
    listCMA = []
    listCMB = []
    listCMC = []

    # Create 3 lists of 64 lists each
    for cms_list in focus_cms:
        listCMA.append(cms_list[0])
        listCMB.append(cms_list[1])
        listCMC.append(cms_list[2])
    
    new_cms_list.append(listCMA)
    new_cms_list.append(listCMB)
    new_cms_list.append(listCMC)

    # print("labels:", len(new_labels_list[0]))
    # print("cms:", len(new_cms_list[0]))
    # print("labels:", new_labels_list[0][0].size())
    # print("cms:", new_cms_list[0][0].size())

    lossAR = 0.0
    lossHS = 0.0
    lossSG = 0.0

    enum = 0
    for cm, label in zip(new_cms_list, new_labels_list):
        enum += 1
        batch_loss = 0
        # print("Annotator ", enum)
        # print("label:", len(label))
        # print("label:", len(label[0]))
        # print("label:", label[0].size())
        # print("cm:", len(cm))
        # print("cm:", len(cm[0]))
        # print("cm:", cm[0].size())

        # mean_confusion_matrix = []

        for i, focus_pred_i in enumerate(focus_pred):

            # print(cm[i])
            # print(cm[i].size(0))
            # print(torch.reshape(cm[i], (int(cm[i].size(0) / 4), 2, 2) ))
            cm_2x2 = torch.reshape(cm[i], (int(cm[i].size(0) / 4), 2, 2))
            # print(cm_2x2.size())
            # print("Size focus_pred: ", focus_pred_i.unsqueeze(-1).permute(1, 0, 2).size())
            # print("focus_pred: ", focus_pred_i.unsqueeze(-1).permute(1, 0, 2)[:10, 0, :])
            pred_noisy = torch.bmm(cm_2x2.to(DEVICE), focus_pred_i.unsqueeze(-1).permute(1, 0, 2).to(DEVICE))
            # print("Pred_noisy size: ", pred_noisy.size())
            a1, a2, a4 = pred_noisy.size()
            pred_noisy = pred_noisy.view(a1, a4, a2).permute(0, 2, 1).contiguous().view(a1, a2, a4)
            # print("Pred_noisy size: ", pred_noisy.size())
            
            pred_noisy_mask = pred_noisy[:, 0, :]
            # print("Pred_noisy: ", pred_noisy.size())
            # print("Label: ", label[i].size())
            # print("pred_noisy: ", pred_noisy[0, 0, :].size())
            # print("Label: ", label[i][:10])
            # print("Confusion matrix: ")
            # confusion_matrix_np = confusion_matrix( np.round(label[i].cpu().detach().numpy()).astype(int), 
            #                                         np.round(pred_noisy[0, 0, :].cpu().detach().numpy()).astype(int))
            
            # print(torch.nn.functional.normalize(torch.from_numpy(confusion_matrix_np).float(), p = 1, dim = 0))
            # mean_confusion_matrix.append(torch.nn.functional.normalize(torch.from_numpy(confusion_matrix_np).float(), p = 1, dim = 0))

            loss_current = dice_loss2(pred_noisy_mask.to(DEVICE), label[i].to(DEVICE))
            # print("Step loss: ", loss_current.item())
            batch_loss += loss_current
            
        if enum == 1:
            lossAR += batch_loss.item() / len(focus_pred)
            # print("Confusion matrix AR: ")
            # print(torch.mean(torch.stack(mean_confusion_matrix, dim = 0), dim = 0))
        elif enum == 2:
            lossHS += batch_loss.item() / len(focus_pred)
            # print("Confusion matrix HS: ")
            # print(torch.mean(torch.stack(mean_confusion_matrix, dim = 0), dim = 0))
        elif enum == 3:
            lossSG += batch_loss.item() / len(focus_pred)
            # print("Confusion matrix SG: ")
            # print(torch.mean(torch.stack(mean_confusion_matrix, dim = 0), dim = 0))
        else:
            print("loss not assigned...")

        # print("Annotator's loss: ", batch_loss.item() / len(focus_pred))
        
        batch_loss /= len(focus_pred)
        total_loss += batch_loss
    
    print(f'Total loss: {total_loss.item():.4f}, AR Loss: {lossAR:.4f}, HS Loss: {lossHS:.4f}, SG Loss: {lossSG:.4f}')

    # print("Total loss: ", total_loss.item())
    
    return total_loss, total_loss, total_loss * 0

def noisy_label_loss_GCM(pred, cms, labels, names, alpha = 0.1):

    main_loss = 0.0
    regularisation = 0.0

    # print("Pred:",pred)
    pred_norm = torch.sigmoid(pred)
    save_histogram(pred_norm)
    pred_init = pred_norm
    
    clear_tensor, unclear_tensor = clear_pred(pred_norm)

    # print("Pred norm:",pred_norm)
    mask_prob = pred_norm
    back_prob = 1 - pred_norm

    pred_norm = torch.cat([mask_prob, back_prob], dim = 1)
    b, c, h, w = pred_norm.size()
   
    pred_norm = pred_norm.view(b, c, h*w).permute(0, 2, 1).contiguous().view(b*h*w, c, 1)

    enum = 0

    for cm, label_noisy in zip(cms, labels):
        # print("cm size: ", cm.size())
        # print("labels len: ", len(label_noisy))
        enum += 1

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
        
        save_borders(unclear_tensor.squeeze(1), pred_noisy_mask, enum, names)

        # pred_noisy = pred_noisy_mask.unsqueeze(1)
        # pred_noisy = pred_noisy_mask
        pred_noisy = clear_tensor.squeeze(1) * pred_init.squeeze(1) + unclear_tensor.squeeze(1) * pred_noisy_mask


        criterion = torch.nn.BCEWithLogitsLoss(reduce = 'mean')  # The loss function
        # loss_current = dice_loss(pred_noisy, label_noisy.view(b, h, w).long())
        loss_current = dice_loss2(pred_noisy, label_noisy.view(b, h, w).long())
        # Calculate the Loss
        # loss_current = criterion(pred_noisy, label_noisy.view(b, h, w))
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

def calculate_cm(y_pred, y_true):

    # flatten the tensors into a 1D array
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)

    # compute the number of true positives, false positives, true negatives, and false negatives
    tp = torch.sum((y_pred == 1) & (y_true == 1)).item()
    fp = torch.sum((y_pred == 1) & (y_true == 0)).item()
    tn = torch.sum((y_pred == 0) & (y_true == 0)).item()
    fn = torch.sum((y_pred == 0) & (y_true == 1)).item()

    # total_pixels = y_pred.numel()

    # create the confusion matrix
    confusion_matrix = torch.tensor([[tn, fp], [fn, tp]])

    col_sums = confusion_matrix.sum(dim = 0)
    confusion_matrix = confusion_matrix / col_sums

    return torch.round(confusion_matrix * 10000) / 10000
   
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
