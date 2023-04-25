import os
import glob
import random
import torch
import imageio
import errno
import numpy as np
import tifffile as tiff
import gzip

import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils import data
# import torchmetrics

from sklearn.metrics import confusion_matrix
# =============================================


class CustomDataset(torch.utils.data.Dataset):

    def __init__(self, imgs_folder, labels_folder, augmentation):

        # 1. Initialize file paths or a list of file names.
        self.imgs_folder = imgs_folder
        self.labels_folder = labels_folder
        self.data_augmentation = augmentation
        self.skin_np = False
        # self.transform = transforms

    def __getitem__(self, index):
        # 1. Read one data from file (e.g. using num py.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).

        # all_images = glob.glob(os.path.join(self.imgs_folder, '*.npy'))
        # all_labels = glob.glob(os.path.join(self.labels_folder, '*.npy'))
        # # sort all in the same order
        # all_labels.sort()
        # all_images.sort()

        if self.skin_np:
            all_images = glob.glob(os.path.join(self.imgs_folder, '*.npy.gz'))
            all_images.sort()

            all_labels = glob.glob(os.path.join(self.labels_folder, '*.npy.gz'))
            all_labels.sort()

            label = np.load(gzip.open(all_labels[index]))
            image = np.load(gzip.open(all_images[index]))
        
        else:
            all_images = glob.glob(os.path.join(self.imgs_folder, '*.tif'))
            all_images.sort()

            all_labels = glob.glob(os.path.join(self.labels_folder, '*.tif'))
            all_labels.sort()

            label = tiff.imread(all_labels[index])
            image = tiff.imread(all_images[index])

            label = np.array(label, dtype='float32') / 255.
            image = np.array(image, dtype='float32') / 255.
        # print(image.max())
        # print(label.max())
        #
        # # label = Image.open(all_labels[index])
        # # label = tiff.imread(all_labels[index])
        # label = np.load(all_labels[index])
        # label = np.array(label, dtype='float32')
        # # image = tiff.imread(all_images[index])
        # image = np.load(all_images[index])
        # image = np.array(image, dtype='float32')
        #
        labelname = all_labels[index]
        path_label, labelname = os.path.split(labelname)
        labelname, labelext = os.path.splitext(labelname)
        #
        c_amount = len(np.shape(label))
        #

        #
        # Reshaping everyting to make sure the order: channel x height x width
        if c_amount == 3:
            d1, d2, d3 = np.shape(label)
            if d1 != min(d1, d2, d3):
                label = np.reshape(label, (d3, d1, d2))
                #
        elif c_amount == 2:
            h, w = np.shape(label)
            label = np.reshape(label, (1, h, w))
        #

        if len(np.shape(image)) == 3:
            d1, d2, d3 = np.shape(image)
            if d1 != min(d1, d2, d3):
                image = np.reshape(image, (d3, d1, d2))
        elif len(np.shape(image)) == 2:
            h, w = np.shape(image)
            image = np.reshape(image, (1, h, w))

        d1, d2, d3 = np.shape(image)
        #
        # if d1 != min(d1, d2, d3):
        #     #
        #     image = np.reshape(image, (d3, d1, d2))
        #
        if self.data_augmentation == 'full':
            # augmentation:
            augmentation = random.uniform(0, 1)
            #
            if augmentation < 0.25:
                #
                c, h, w = np.shape(image)
                #
                for channel in range(c):
                    #
                    image[channel, :, :] = np.flip(image[channel, :, :], axis=0).copy()
                    image[channel, :, :] = np.flip(image[channel, :, :], axis=1).copy()
                    #
                label = np.flip(label, axis=1).copy()
                label = np.flip(label, axis=2).copy()

            elif augmentation < 0.5:
                #
                mean = 0.0
                sigma = 0.15
                noise = np.random.normal(mean, sigma, image.shape)
                mask_overflow_upper = image + noise >= 1.0
                mask_overflow_lower = image + noise < 0.0
                noise[mask_overflow_upper] = 1.0
                noise[mask_overflow_lower] = 0.0
                image += noise

            elif augmentation < 0.75:
                #
                c, h, w = np.shape(image)
                #
                for channel in range(c):
                    #
                    channel_ratio = random.uniform(0, 1)
                    #
                    image[channel, :, :] = image[channel, :, :] * channel_ratio

        elif self.data_augmentation == 'flip':
            # augmentation:
            augmentation = random.uniform(0, 1)
            #
            if augmentation > 0.5 or augmentation == 0.5:
                #
                c, h, w = np.shape(image)
                #
                for channel in range(c):
                    #
                    image[channel, :, :] = np.flip(image[channel, :, :], axis=0).copy()
                    #
                label = np.flip(label, axis=1).copy()

        elif self.data_augmentation == 'all_flip':
            # augmentation:
            augmentation = random.uniform(0, 1)
            #
            if augmentation > 0.5 or augmentation == 0.5:
                #
                c, h, w = np.shape(image)
                #
                for channel in range(c):
                    #
                    image[channel, :, :] = np.flip(image[channel, :, :], axis=0).copy()
                    image[channel, :, :] = np.flip(image[channel, :, :], axis=1).copy()
                    #
                label = np.flip(label, axis=1).copy()
                label = np.flip(label, axis=2).copy()

        return image, label, labelname

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        #print("Len: ", len(glob.glob(os.path.join(self.imgs_folder, '*.tif'))))
        if self.skin_np:
            return len(glob.glob(os.path.join(self.imgs_folder, '*.npy.gz')))
        else:
            return len(glob.glob(os.path.join(self.imgs_folder, '*.tif')))


# ============================================================================================

def evaluate_noisy_label(data, model1, model2, class_no):

    """

    Args:
        data:
        model1:
        model2:
        class_no:

    Returns:

    """

    model1.eval()
    model2.eval()
    #
    test_dice = 0
    test_dice_all = []
    #
    for i, (v_images, v_labels_over, v_labels_under, v_labels_wrong, v_labels_true, v_imagename) in enumerate(data):
        #
        v_images = v_images.to(device='cuda', dtype=torch.float32)
        v_outputs_logits = model1(v_images)
        v_outputs_logits_noisy = model2(v_images)
        #
        _, v_output = torch.max(v_outputs_logits, dim=1)
        v_outputs_noisy = []
        #
        for v_noisy_logit in v_outputs_logits_noisy:
            #
            _, v_noisy_output = torch.max(v_noisy_logit, dim=1)
            v_outputs_noisy.append(v_noisy_output.cpu().detach().numpy())
        #
        v_dice_ = segmentation_scores(v_labels_true, v_output.cpu().detach().numpy(), class_no)
        #
        epoch_noisy_labels = [v_labels_over.cpu().detach().numpy(), v_labels_under.cpu().detach().numpy(), v_labels_wrong.cpu().detach().numpy(), v_labels_true.cpu().detach().numpy()]
        v_ged = generalized_energy_distance(epoch_noisy_labels, v_outputs_noisy, class_no)
        test_dice += v_dice_
        test_dice_all.append(test_dice)
        #
    # print(i)
    # print(test_dice)
    # print(test_dice / (i + 1))

    return test_dice / (i + 1), v_ged


def evaluate_noisy_label_2(data, model1, model2, class_no):

    """

    Args:
        data:
        model1:
        model2:
        class_no:

    Returns:

    """
    model1.eval()
    model2.eval()

    test_dice = 0
    test_dice_all = []

    for i, (v_images, v_labels_over, v_labels_under, v_labels_wrong, v_labels_true, v_imagename) in enumerate(data):
        #
        v_images = v_images.to(device='cuda', dtype=torch.float32)
        v_outputs_logits = model1(v_images)
        b, c, h, w = v_outputs_logits.size()
        v_outputs_logits = nn.Softmax(dim=1)(v_outputs_logits)
        cms = model2(v_images)
        #
        _, v_output = torch.max(v_outputs_logits, dim=1)
        v_outputs_noisy = []
        #
        for cm in cms:
            #
            cm = cm.reshape(b * h * w, c, c)
            cm = cm / cm.sum(1, keepdim=True)
            v_noisy_logit = torch.bmm(cm, v_outputs_logits.reshape(b * h * w, c, 1)).reshape(b, c, h, w)
            _, v_noisy_output = torch.max(v_noisy_logit, dim=1)
            v_outputs_noisy.append(v_noisy_output.cpu().detach().numpy())
        #
        v_dice_ = segmentation_scores(v_labels_true, v_output.cpu().detach().numpy(), class_no)
        #
        epoch_noisy_labels = [v_labels_over.cpu().detach().numpy(), v_labels_under.cpu().detach().numpy(), v_labels_wrong.cpu().detach().numpy(), v_labels_true.cpu().detach().numpy()]
        v_ged = generalized_energy_distance(epoch_noisy_labels, v_outputs_noisy, class_no)
        test_dice += v_dice_
        test_dice_all.append(test_dice)
        #
    # print(i)
    # print(test_dice)
    # print(test_dice / (i + 1))

    return test_dice / (i + 1), v_ged


def evaluate_noisy_label_3(data, model1, class_no):
    """

    Args:
        data:
        model1:
        class_no:

    Returns:

    """
    model1.eval()
    # model2.eval()
    #
    test_dice = 0
    test_dice_all = []
    #
    for i, (v_images, v_labels_over, v_labels_under, v_labels_wrong, v_labels_good, v_imagename) in enumerate(data):
        #
        # print(i)
        #
        v_images = v_images.to(device='cuda', dtype=torch.float32)
        v_outputs_logits, cms = model1(v_images)
        b, c, h, w = v_outputs_logits.size()
        v_outputs_logits = nn.Softmax(dim=1)(v_outputs_logits)
        # cms = model2(v_images)
        #
        _, v_output = torch.max(v_outputs_logits, dim=1)
        v_outputs_noisy = []
        #
        # v_outputs_logits = v_outputs_logits.permute(0, 2, 3, 1).contiguous()
        # v_outputs_logits = v_outputs_logits.reshape(b * h * w, c, 1)
        #
        for cm in cms:
            #
            cm = cm.reshape(b * h * w, c, c)
            cm = cm / cm.sum(1, keepdim=True)
            v_noisy_output = torch.bmm(cm, v_outputs_logits.reshape(b * h * w, c, 1)).reshape(b, c, h, w)
            # cm = cm.permute(0, 2, 3, 1).contiguous().view(b * h * w, c, c)
            # cm = cm / cm.sum(1, keepdim=True)
            # v_noisy_output = torch.bmm(cm, v_outputs_logits)
            # v_noisy_output = v_noisy_output.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
            _, v_noisy_output = torch.max(v_noisy_output, dim=1)
            v_outputs_noisy.append(v_noisy_output.cpu().detach().numpy())
        #
        v_dice_ = segmentation_scores(v_labels_good, v_output.cpu().detach().numpy(), class_no)
        #
        epoch_noisy_labels = [v_labels_over.cpu().detach().numpy(), v_labels_under.cpu().detach().numpy(), v_labels_wrong.cpu().detach().numpy(), v_labels_good.cpu().detach().numpy()]
        v_ged = generalized_energy_distance(epoch_noisy_labels, v_outputs_noisy, class_no)
        test_dice += v_dice_
        test_dice_all.append(test_dice)
        #
    # print(i)
    # print(test_dice)
    # print(test_dice / (i + 1))
    #
    return test_dice / (i + 1), v_ged


# def evaluate_noisy_label_4(data, model1, class_no):
#     """

#     Args:
#         data:
#         model1:
#         class_no:

#     Returns:

#     """
#     model1.eval()
#     # model2.eval()
#     #
#     test_dice = 0
#     test_dice_all = []
#     #
#     for i, (v_images, v_labels_over, v_labels_under, v_labels_wrong, #v_labels_good,
#             v_imagename) in enumerate(data):
#         #
#         # print(i)
#         #
#         v_images = v_images.to(device='cuda', dtype=torch.float32)
#         v_outputs_logits, cms = model1(v_images)
#         b, c, h, w = v_outputs_logits.size()
#         v_outputs_logits = nn.Softmax(dim=1)(v_outputs_logits)
#         # cms = model2(v_images)
#         #
#         _, v_output = torch.max(v_outputs_logits, dim=1)
#         v_outputs_noisy = []
#         #
#         v_outputs_logits = v_outputs_logits.view(b, c, h*w)
#         v_outputs_logits = v_outputs_logits.permute(0, 2, 1).contiguous().view(b*h*w, c)
#         v_outputs_logits = v_outputs_logits.view(b * h * w, c, 1)
#         #
#         for cm in cms:
#             #
#             cm = cm.reshape(b, c**2, h*w).permute(0, 2, 1).contiguous().view(b*h*w, c*c).view(b*h*w, c, c)
#             cm = cm / cm.sum(1, keepdim=True)
#             v_noisy_output = torch.bmm(cm, v_outputs_logits).view(b*h*w, c)
#             v_noisy_output = v_noisy_output.view(b, h*w, c).permute(0, 2, 1).contiguous().view(b, c, h, w)
#             _, v_noisy_output = torch.max(v_noisy_output, dim=1)
#             v_outputs_noisy.append(v_noisy_output.cpu().detach().numpy())
#         #
#         v_dice_ = segmentation_scores(v_labels_good, v_output.cpu().detach().numpy(), class_no)
#         #
#         epoch_noisy_labels = [v_labels_over.cpu().detach().numpy(), v_labels_under.cpu().detach().numpy(), v_labels_wrong.cpu().detach().numpy(), v_labels_good.cpu().detach().numpy()]
#         v_ged = generalized_energy_distance(epoch_noisy_labels, v_outputs_noisy, class_no)
#         test_dice += v_dice_
#         test_dice_all.append(test_dice)
#         #
#     # print(i)
#     # print(test_dice)
#     # print(test_dice / (i + 1))
#     #
#     return test_dice / (i + 1), v_ged


def evaluate_noisy_label_6(data, model1, class_no):
    """

    Args:
        data:
        model1:
        class_no:

    Returns:

    """
    model1.eval()
    # model2.eval()
    #
    test_dice = 0
    test_dice_all = []
    #
    for i, (v_images, v_labels_over, v_labels_under, v_labels_wrong, v_labels_good, v_imagename) in enumerate(data):
        #
        # print(i)
        #
        v_images = v_images.to(device='cuda', dtype=torch.float32)
        v_outputs_logits, cms = model1(v_images)
        b, c, h, w = v_outputs_logits.size()
        v_outputs_logits = nn.Softmax(dim=1)(v_outputs_logits)
        # cms = model2(v_images)
        #
        _, v_output = torch.max(v_outputs_logits, dim=1)
        v_outputs_noisy = []
        #
        v_outputs_logits = v_outputs_logits.view(b, c, h*w)
        v_outputs_logits = v_outputs_logits.permute(0, 2, 1).contiguous().view(b*h*w, c)
        v_outputs_logits = v_outputs_logits.view(b * h * w, c, 1)
        #
        for cm in cms:
            #
            b, c_r_d, h, w = cm.size()
            r = c_r_d // c // 2
            cm1 = cm[:, 0:r * c, :, :]
            if r == 1:
                cm2 = cm[:, r * c:c_r_d-1, :, :]
            else:
                cm2 = cm[:, r * c:c_r_d-1, :, :]
            cm1_reshape = cm1.view(b, c_r_d // 2, h * w).permute(0, 2, 1).contiguous().view(b * h * w, r * c).view(b * h * w, r, c)
            cm2_reshape = cm2.view(b, c_r_d // 2, h * w).permute(0, 2, 1).contiguous().view(b * h * w, r * c).view(b * h * w, c, r)
            #
            cm1_reshape = cm1_reshape / cm1_reshape.sum(1, keepdim=True)
            cm2_reshape = cm2_reshape / cm2_reshape.sum(1, keepdim=True)
            #
            v_noisy_output = torch.bmm(cm1_reshape, v_outputs_logits)
            v_noisy_output = torch.bmm(cm2_reshape, v_noisy_output).view(b * h * w, c)
            v_noisy_output = v_noisy_output.view(b, h * w, c).permute(0, 2, 1).contiguous().view(b, c, h, w)
            #
            # v_noisy_output = torch.bmm(cm, v_outputs_logits).view(b*h*w, c)
            # v_noisy_output = v_noisy_output.view(b, h*w, c).permute(0, 2, 1).contiguous().view(b, c, h, w)
            _, v_noisy_output = torch.max(v_noisy_output, dim=1)
            v_outputs_noisy.append(v_noisy_output.cpu().detach().numpy())
        #
        v_dice_ = segmentation_scores(v_labels_good, v_output.cpu().detach().numpy(), class_no)
        #
        epoch_noisy_labels = [v_labels_over.cpu().detach().numpy(), v_labels_under.cpu().detach().numpy(), v_labels_wrong.cpu().detach().numpy(), v_labels_good.cpu().detach().numpy()]
        v_ged = generalized_energy_distance(epoch_noisy_labels, v_outputs_noisy, class_no)
        test_dice += v_dice_
        test_dice_all.append(test_dice)
        #
    # print(i)
    # print(test_dice)
    # print(test_dice / (i + 1))
    #
    return test_dice / (i + 1), v_ged


def evaluate_noisy_label_7(data, model1, model2, class_no, low_rank):
    """

    Args:
        data:
        model1:
        model2:
        class_no:
        low_rank:

    Returns:

    """
    model1.eval()
    model2.eval()
    #
    test_dice = 0
    test_dice_all = []
    #
    for i, (v_images, v_labels_over, v_labels_under, v_labels_wrong, v_labels_good, v_imagename) in enumerate(data):
        #
        # print(i)
        #
        v_images = v_images.to(device='cuda', dtype=torch.float32)
        v_outputs_logits = model1(v_images)
        b, c, h, w = v_outputs_logits.size()
        v_outputs_logits = nn.Softmax(dim=1)(v_outputs_logits)
        cms = model2(v_images)
        #
        _, v_output = torch.max(v_outputs_logits, dim=1)
        v_outputs_noisy = []
        #
        v_outputs_logits = v_outputs_logits.view(b, c, h*w)
        v_outputs_logits = v_outputs_logits.permute(0, 2, 1).contiguous().view(b*h*w, c)
        v_outputs_logits = v_outputs_logits.view(b * h * w, c, 1)
        #
        for cm in cms:
            #
            if low_rank is False:
                #
                cm = cm.reshape(b, c**2, h*w).permute(0, 2, 1).contiguous().view(b*h*w, c*c).view(b*h*w, c, c)
                cm = cm / cm.sum(1, keepdim=True)
                v_noisy_output = torch.bmm(cm, v_outputs_logits).view(b*h*w, c)
                v_noisy_output = v_noisy_output.view(b, h*w, c).permute(0, 2, 1).contiguous().view(b, c, h, w)
                #
            else:
                #
                b, c_r_d, h, w = cm.size()
                r = c_r_d // c // 2
                cm1 = cm[:, 0:r * c, :, :]
                cm2 = cm[:, r * c:c_r_d, :, :]
                cm1_reshape = cm1.view(b, c_r_d // 2, h * w).permute(0, 2, 1).contiguous().view(b * h * w, r * c).view(b * h * w, r, c)
                cm2_reshape = cm2.view(b, c_r_d // 2, h * w).permute(0, 2, 1).contiguous().view(b * h * w, r * c).view(b * h * w, c, r)
                #
                cm1_reshape = cm1_reshape / cm1_reshape.sum(1, keepdim=True)
                cm2_reshape = cm2_reshape / cm2_reshape.sum(1, keepdim=True)
                #
                v_noisy_output = torch.bmm(cm1_reshape, v_outputs_logits)
                v_noisy_output = torch.bmm(cm2_reshape, v_noisy_output).view(b * h * w, c)
                v_noisy_output = v_noisy_output.view(b, h * w, c).permute(0, 2, 1).contiguous().view(b, c, h, w)
                #
            _, v_noisy_output = torch.max(v_noisy_output, dim=1)
            v_outputs_noisy.append(v_noisy_output.cpu().detach().numpy())
        #
        v_dice_ = segmentation_scores(v_labels_good, v_output.cpu().detach().numpy(), class_no)
        #
        epoch_noisy_labels = [v_labels_over.cpu().detach().numpy(), v_labels_under.cpu().detach().numpy(), v_labels_wrong.cpu().detach().numpy(), v_labels_good.cpu().detach().numpy()]
        v_ged = generalized_energy_distance(epoch_noisy_labels, v_outputs_noisy, class_no)
        test_dice += v_dice_
        test_dice_all.append(test_dice)
        #
    # print(i)
    # print(test_dice)
    # print(test_dice / (i + 1))
    #
    return test_dice / (i + 1), v_ged


def evaluate_noisy_label_5(data, model1, class_no):
    """

    Args:
        data:
        model1:
        class_no:

    Returns:

    """
    model1.eval()
    # model2.eval()
    #
    test_dice = 0
    test_dice_all = []
    #
    for i, (v_images, v_labels_over, v_labels_under, v_labels_wrong, v_labels_good, v_labels_true, v_imagename) in enumerate(data):
        #
        # print(i)
        #
        v_images = v_images.to(device='cuda', dtype=torch.float32)
        v_outputs_logits, cms = model1(v_images)
        b, c, h, w = v_outputs_logits.size()
        v_outputs_logits = nn.Softmax(dim=1)(v_outputs_logits)
        # cms = model2(v_images)
        #
        _, v_output = torch.max(v_outputs_logits, dim=1)
        v_outputs_noisy = []
        #
        v_outputs_logits = v_outputs_logits.view(b, c, h*w)
        v_outputs_logits = v_outputs_logits.permute(0, 2, 1).contiguous().view(b*h*w, c)
        v_outputs_logits = v_outputs_logits.view(b * h * w, c, 1)
        #
        for cm in cms:
            #
            # cm = cm.reshape(b * h * w, c, c)
            # cm = cm / cm.sum(1, keepdim=True)
            # v_noisy_output = torch.bmm(cm, v_outputs_logits.reshape(b * h * w, c, 1)).reshape(b, c, h, w)
            # cm = cm.permute(0, 2, 3, 1).contiguous().view(b * h * w, c, c)
            cm = cm.reshape(b, c**2, h*w).permute(0, 2, 1).contiguous().view(b*h*w, c*c).view(b*h*w, c, c)
            cm = cm / cm.sum(1, keepdim=True)
            v_noisy_output = torch.bmm(cm, v_outputs_logits).view(b*h*w, c)
            v_noisy_output = v_noisy_output.view(b, h*w, c).permute(0, 2, 1).contiguous().view(b, c, h, w)
            _, v_noisy_output = torch.max(v_noisy_output, dim=1)
            v_outputs_noisy.append(v_noisy_output.cpu().detach().numpy())
        #
        v_dice_ = segmentation_scores(v_labels_true, v_output.cpu().detach().numpy(), class_no)
        #
        epoch_noisy_labels = [v_labels_over.cpu().detach().numpy(), v_labels_under.cpu().detach().numpy(), v_labels_wrong.cpu().detach().numpy(), v_labels_true.cpu().detach().numpy(), v_labels_good.cpu().detach().numpy()]
        v_ged = generalized_energy_distance(epoch_noisy_labels, v_outputs_noisy, class_no)
        test_dice += v_dice_
        test_dice_all.append(test_dice)
        #
    # print(i)
    # print(test_dice)
    # print(test_dice / (i + 1))
    #
    return test_dice / (i + 1), v_ged


def evaluate(evaluatedata, model, device, class_no):
    """

    Args:
        evaluatedata:
        model:
        device:
        class_no:

    Returns:

    """
    model.eval()
    #
    with torch.no_grad():
        #
        test_iou = 0
        testloss = 0
        #
        for j, (testimg, testlabel, testname) in enumerate(evaluatedata):
            #
            testimg = testimg.to(device=device, dtype=torch.float32)
            testlabel = testlabel.to(device=device, dtype=torch.float32)
            #
            # print("val image: ", testname)
            # print("val image size: ", testimg.size())
            # print("val label size: ", testlabel.size())
            testoutput = model(testimg)
            #
            testloss = nn.BCEWithLogitsLoss(reduction='mean')(testoutput, testlabel)
            # print("val out size: ", testoutput.size())
            if class_no == 2:
                testoutput = torch.sigmoid(testoutput)
                # testoutput = (testoutput > 0.5).float()
                # _, testoutput = torch.max(testoutput, dim=1)
            else:
                _, testoutput = torch.max(testoutput, dim=1)
            # print("val max size: ", testoutput.size())
            #
            # Loss
            #
            # mean_iu_ = segmentation_scores(testlabel.cpu().detach().numpy(), testoutput.cpu().detach().numpy(), class_no)
            # plt.imsave('./test_results/' + testname[0] + '_GT.png', np.swapaxes(np.swapaxes(testimg[0].cpu().detach().numpy(), 0, 1), 1, 2))
            # plt.imsave('./test_results/' + testname[0] + '_segmented_max_0.png', testoutput[0].cpu().detach().numpy(), cmap = 'gray')
            # plt.imsave('./test_results/' + testname[0] + '_label_0.png', testlabel[0, 0].cpu().detach().numpy(), cmap = 'gray')
            # plt.imsave('./test_results/' + testname[1] + '_segmented_max_1.png', testoutput[1].cpu().detach().numpy(), cmap = 'gray')
            # plt.imsave('./test_results/' + testname[1] + '_label_1.png', testlabel[1, 0].cpu().detach().numpy(), cmap = 'gray')
            mean_iu_ = dice_coef_simplified(testoutput, testlabel)
            # mean_iu_ = dice_coef_torchmetrics(testoutput, testlabel, 2, 'cuda')
            test_iou += mean_iu_
        #
        return test_iou / (j+1), testloss


def test(testdata,
         model,
         device,
         class_no,
         save_path):
    """

    Args:
        testdata:
        model:
        device:
        class_no:
        save_path:

    Returns:

    """
    model.eval()

    with torch.no_grad():
        #
        test_iou = 0
        #
        for j, (testimg, testlabel, testname) in enumerate(testdata):
            #
            testimg = testimg.to(device=device, dtype=torch.float32)
            testlabel = testlabel.to(device=device, dtype=torch.float32)
            #
            testoutput = model(testimg)
            if class_no == 2:
                testoutput = torch.sigmoid(testoutput)
                #testoutput = (testoutput > 0.5).float()
                _, testoutput = torch.max(testoutput, dim=1)
            else:
                _, testoutput = torch.max(testoutput, dim=1)
            #
            # mean_iu_ = segmentation_scores(testlabel.cpu().detach().numpy(), testoutput.cpu().detach().numpy(), class_no)
            mean_iu_ = dice_coef_simplified(testoutput, testlabel)
            test_iou += mean_iu_
            h = 192
            w = 256
            # print("image size: ", testimg.size())
            # print("y size: ", testoutput.size())
            # plt.imsave('./test_results/' + testname[0] + '_GT.png', np.swapaxes(np.swapaxes(testimg[0].cpu().detach().numpy(), 0, 1), 1, 2))
            # plt.imsave('./test_results/' + testname[0] + '_segmented_original_1.png', testoutput[0].reshape(h, w).cpu().detach().numpy(), cmap = 'gray')
            
            #
            # ========================================================
            # # Plotting segmentation:
            # ========================================================
            prediction_map_path = save_path + '/' + 'Visual_results'
            #
            try:
                os.mkdir(prediction_map_path)
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise
                pass

            b, c, h, w = np.shape(testlabel)
            testoutput_original = np.asarray(testoutput.cpu().detach().numpy(), dtype=np.uint8)
            testoutput_original = np.squeeze(testoutput_original, axis=0)
            testoutput_original = np.repeat(testoutput_original, 3, axis=0)
            #
            save_bool = False
            if save_bool:
                if class_no == 2:
                    segmentation_map = np.zeros((3, h, w), dtype=np.uint8)
                    #
                    segmentation_map[0, :, :][np.logical_and(testoutput_original[0, :, :] == 1, np.logical_and(testoutput_original[1, :, :] == 1, testoutput_original[2, :, :] == 1))] = 255
                    segmentation_map[1, :, :][np.logical_and(testoutput_original[0, :, :] == 1, np.logical_and(testoutput_original[1, :, :] == 1, testoutput_original[2, :, :] == 1))] = 0
                    segmentation_map[2, :, :][np.logical_and(testoutput_original[0, :, :] == 1, np.logical_and(testoutput_original[1, :, :] == 1, testoutput_original[2, :, :] == 1))] = 0
                    #
                else:
                    segmentation_map = np.zeros((h, w, 3), dtype=np.uint8)
                    if class_no == 4:
                        # multi class for brats 2018
                        segmentation_map[:, :, 0][np.logical_and(testoutput_original[:, :, 0] == 1, testoutput_original[:, :, 1] == 1, testoutput_original[:, :, 2] == 1)] = 255
                        segmentation_map[:, :, 1][np.logical_and(testoutput_original[:, :, 0] == 1, testoutput_original[:, :, 1] == 1, testoutput_original[:, :, 2] == 1)] = 0
                        segmentation_map[:, :, 2][np.logical_and(testoutput_original[:, :, 0] == 1, testoutput_original[:, :, 1] == 1, testoutput_original[:, :, 2] == 1)] = 0
                        #
                        segmentation_map[:, :, 0][np.logical_and(testoutput_original[:, :, 0] == 2, testoutput_original[:, :, 1] == 2, testoutput_original[:, :, 2] == 2)] = 0
                        segmentation_map[:, :, 1][np.logical_and(testoutput_original[:, :, 0] == 2, testoutput_original[:, :, 1] == 2, testoutput_original[:, :, 2] == 2)] = 255
                        segmentation_map[:, :, 2][np.logical_and(testoutput_original[:, :, 0] == 2, testoutput_original[:, :, 1] == 2, testoutput_original[:, :, 2] == 2)] = 0
                        #
                        segmentation_map[:, :, 0][np.logical_and(testoutput_original[:, :, 0] == 3, testoutput_original[:, :, 1] == 3, testoutput_original[:, :, 2] == 3)] = 0
                        segmentation_map[:, :, 1][np.logical_and(testoutput_original[:, :, 0] == 3, testoutput_original[:, :, 1] == 3, testoutput_original[:, :, 2] == 3)] = 0
                        segmentation_map[:, :, 2][np.logical_and(testoutput_original[:, :, 0] == 3, testoutput_original[:, :, 1] == 3, testoutput_original[:, :, 2] == 3)] = 255
                        #
                    elif class_no == 8:
                        # multi class for cityscapes
                        segmentation_map[:, :, 0][np.logical_and(testoutput_original[:, :, 0] == 0, testoutput_original[:, :, 1] == 0, testoutput_original[:, :, 2] == 0)] = 255
                        segmentation_map[:, :, 1][np.logical_and(testoutput_original[:, :, 0] == 0, testoutput_original[:, :, 1] == 0, testoutput_original[:, :, 2] == 0)] = 0
                        segmentation_map[:, :, 2][np.logical_and(testoutput_original[:, :, 0] == 0, testoutput_original[:, :, 1] == 0, testoutput_original[:, :, 2] == 0)] = 0
                        #
                        segmentation_map[:, :, 0][np.logical_and(testoutput_original[:, :, 0] == 1, testoutput_original[:, :, 1] == 1, testoutput_original[:, :, 2] == 1)] = 0
                        segmentation_map[:, :, 1][np.logical_and(testoutput_original[:, :, 0] == 1, testoutput_original[:, :, 1] == 1, testoutput_original[:, :, 2] == 1)] = 255
                        segmentation_map[:, :, 2][np.logical_and(testoutput_original[:, :, 0] == 1, testoutput_original[:, :, 1] == 1, testoutput_original[:, :, 2] == 1)] = 0
                        #
                        segmentation_map[:, :, 0][np.logical_and(testoutput_original[:, :, 0] == 2, testoutput_original[:, :, 1] == 2, testoutput_original[:, :, 2] == 2)] = 0
                        segmentation_map[:, :, 1][np.logical_and(testoutput_original[:, :, 0] == 2, testoutput_original[:, :, 1] == 2, testoutput_original[:, :, 2] == 2)] = 0
                        segmentation_map[:, :, 2][np.logical_and(testoutput_original[:, :, 0] == 2, testoutput_original[:, :, 1] == 2, testoutput_original[:, :, 2] == 2)] = 255
                        #
                        segmentation_map[:, :, 0][np.logical_and(testoutput_original[:, :, 0] == 3, testoutput_original[:, :, 1] == 3, testoutput_original[:, :, 2] == 3)] = 255
                        segmentation_map[:, :, 1][np.logical_and(testoutput_original[:, :, 0] == 3, testoutput_original[:, :, 1] == 3, testoutput_original[:, :, 2] == 3)] = 255
                        segmentation_map[:, :, 2][np.logical_and(testoutput_original[:, :, 0] == 3, testoutput_original[:, :, 1] == 3, testoutput_original[:, :, 2] == 3)] = 0
                        #
                        segmentation_map[:, :, 0][np.logical_and(testoutput_original[:, :, 0] == 4, testoutput_original[:, :, 1] == 4, testoutput_original[:, :, 2] == 4)] = 153
                        segmentation_map[:, :, 1][np.logical_and(testoutput_original[:, :, 0] == 4, testoutput_original[:, :, 1] == 4, testoutput_original[:, :, 2] == 4)] = 51
                        segmentation_map[:, :, 2][np.logical_and(testoutput_original[:, :, 0] == 4, testoutput_original[:, :, 1] == 4, testoutput_original[:, :, 2] == 4)] = 255
                        #
                        segmentation_map[:, :, 0][np.logical_and(testoutput_original[:, :, 0] == 5, testoutput_original[:, :, 1] == 5, testoutput_original[:, :, 2] == 5)] = 255
                        segmentation_map[:, :, 1][np.logical_and(testoutput_original[:, :, 0] == 5, testoutput_original[:, :, 1] == 5, testoutput_original[:, :, 2] == 5)] = 102
                        segmentation_map[:, :, 2][np.logical_and(testoutput_original[:, :, 0] == 5, testoutput_original[:, :, 1] == 5, testoutput_original[:, :, 2] == 5)] = 178
                        #
                        segmentation_map[:, :, 0][np.logical_and(testoutput_original[:, :, 0] == 6, testoutput_original[:, :, 1] == 6, testoutput_original[:, :, 2] == 6)] = 102
                        segmentation_map[:, :, 1][np.logical_and(testoutput_original[:, :, 0] == 6, testoutput_original[:, :, 1] == 6, testoutput_original[:, :, 2] == 6)] = 255
                        segmentation_map[:, :, 2][np.logical_and(testoutput_original[:, :, 0] == 6, testoutput_original[:, :, 1] == 6, testoutput_original[:, :, 2] == 6)] = 102
                        #
                prediction_name = 'seg_' + testname[0] + '.png'
                full_error_map_name = os.path.join(prediction_map_path, prediction_name)
                segmentation_map = np.swapaxes(segmentation_map, axis1 = 0, axis2 = 1)
                segmentation_map = np.swapaxes(segmentation_map, axis1 = 1, axis2 = 2)
                imageio.imsave(full_error_map_name, segmentation_map)

        #
        prediction_result_path = save_path + '/Quantitative_Results'
        #
        try:
            #
            os.mkdir(prediction_result_path)
            #
        except OSError as exc:
            #
            if exc.errno != errno.EEXIST:
                #
                raise
            #
            pass
        #
        result_dictionary = {'Test dice': str(test_iou / len(testdata))}
        #
        ff_path = prediction_result_path + '/test_result_data.txt'
        ff = open(ff_path, 'w')
        ff.write(str(result_dictionary))
        ff.close()

        print('Test iou: {:.4f}, '.format(test_iou / len(testdata)))


class CustomDataset_punet(torch.utils.data.Dataset):

    def __init__(self, dataset_location, dataset_tag, noisylabel, augmentation=False):

        #
        self.label_mode = noisylabel
        self.dataset_tag = dataset_tag
        #
        if noisylabel == 'multi':
            #
            if dataset_tag == 'mnist':
                self.label_over_folder = dataset_location + '/Over'
                self.label_under_folder = dataset_location + '/Under'
                self.label_wrong_folder = dataset_location + '/Wrong'
                self.label_good_folder = dataset_location + '/GT'
                self.image_folder = dataset_location + '/Gaussian'
            elif dataset_tag == 'brats':
                self.label_over_folder = dataset_location + '/Over'
                self.label_under_folder = dataset_location + '/Under'
                self.label_wrong_folder = dataset_location + '/Wrong'
                self.label_good_folder = dataset_location + '/Good'
                self.image_folder = dataset_location + '/Image'
            elif dataset_tag == 'lidc':
                self.label_over_folder = dataset_location + '/Annotator_1'
                self.label_under_folder = dataset_location + '/Annotator_2'
                self.label_wrong_folder = dataset_location + '/Annotator_3'
                self.label_good_folder = dataset_location + '/Annotator_4'
                self.label_true_folder = dataset_location + '/Annotator_5'
                self.image_folder = dataset_location + '/Image'
            elif dataset_tag == 'oocytes':
                self.label_over_folder = dataset_location + '/Over'
                self.label_under_folder = dataset_location + '/Under'
                self.label_wrong_folder = dataset_location + '/Wrong'
                self.label_good_folder = dataset_location + '/GT'
                self.image_folder = dataset_location + '/Gaussian'
            elif dataset_tag == 'oocytes_gent':
                self.label_AR_folder = dataset_location + '/AR'
                self.label_HS_folder = dataset_location + '/HS'
                self.label_SG_folder = dataset_location + '/SG'
                self.label_avrg_folder = dataset_location + '/avrg'
                self.image_folder = dataset_location + '/images'
                print("Gent data loaded from Utilis.py as .tif format ...")
            elif dataset_tag == 'oocytes_gent_np':
                self.label_AR_folder = dataset_location + '/AR'
                self.label_HS_folder = dataset_location + '/HS'
                self.label_SG_folder = dataset_location + '/SG'
                self.label_avrg_folder = dataset_location + '/avrg'
                self.image_folder = dataset_location + '/images'
                print("Gent data loaded from Utilis.py as .npy.gz format ...")
                #
        elif noisylabel == 'binary':
            if dataset_tag == 'mnist':
                self.label_folder = dataset_location + '/Mean'
                self.image_folder = dataset_location + '/Gaussian'
                self.true_label_folder = dataset_location + '/GT'

        elif noisylabel == 'normal':
            if dataset_tag == 'mnist':
                self.label_folder = dataset_location + '/GT'
                self.image_folder = dataset_location + '/Gaussian'

        elif noisylabel == 'p_unet':
            if dataset_tag == 'mnist':
                self.label_folder = dataset_location + '/All'
                self.image_folder = dataset_location + '/Gaussian'

        self.data_aug = augmentation

    def __getitem__(self, index):

        if self.label_mode == 'multi':
            #
            if self.dataset_tag == 'mnist' or self.dataset_tag == 'brats' or self.dataset_tag == 'oocytes':
                #
                all_labels_over = glob.glob(os.path.join(self.label_over_folder, '*.tif'))
                all_labels_over.sort()
                #
                all_labels_under = glob.glob(os.path.join(self.label_under_folder, '*.tif'))
                all_labels_under.sort()
                #
                all_labels_wrong = glob.glob(os.path.join(self.label_wrong_folder, '*.tif'))
                all_labels_wrong.sort()
                #
                all_labels_good = glob.glob(os.path.join(self.label_good_folder, '*.tif'))
                all_labels_good.sort()
                #
                all_images = glob.glob(os.path.join(self.image_folder, '*.tif'))
                all_images.sort()
                #
                label_over = tiff.imread(all_labels_over[index])
                label_over = np.array(label_over, dtype='float32')
                #
                label_under = tiff.imread(all_labels_under[index])
                label_under = np.array(label_under, dtype='float32')
                #
                label_wrong = tiff.imread(all_labels_wrong[index])
                label_wrong = np.array(label_wrong, dtype='float32')
                #
                label_good = tiff.imread(all_labels_good[index])
                label_good = np.array(label_good, dtype='float32')
                #
                image = tiff.imread(all_images[index])
                image = np.array(image, dtype='float32')
                #
                # dim_length = len(np.shape(label_over))

                label_over[label_over == 4.0] = 3.0
                label_wrong[label_wrong == 4.0] = 3.0
                label_good[label_good == 4.0] = 3.0
                label_under[label_under == 4.0] = 3.0

                if self.dataset_tag == 'mnist' or self.dataset_tag == 'oocytes':
                    label_over = np.where(label_over > 0.5, 1.0, 0.0)
                    label_under = np.where(label_under > 0.5, 1.0, 0.0)
                    label_wrong = np.where(label_wrong > 0.5, 1.0, 0.0)

                    if np.amax(label_good) != 1.0:
                        # sometimes, some preprocessing might give it as 0 - 255 range
                        label_good = np.where(label_good > 10.0, 1.0, 0.0)
                    else:
                        assert np.amax(label_good) == 1.0
                        label_good = np.where(label_good > 0.5, 1.0, 0.0)

                # print(np.unique(label_over))
                # label_over: h x w
                # image: h x w x c
                c_amount = len(np.shape(label_over))

                # Reshaping everyting to make sure the order: channel x height x width
                if c_amount == 3:
                    #
                    d1, d2, d3 = np.shape(label_over)
                    #
                    if d1 != min(d1, d2, d3):
                        #
                        assert d3 == min(d1, d2, d3)
                        #
                        label_over = np.transpose(label_over, (2, 0, 1))
                        label_under = np.transpose(label_under, (2, 0, 1))
                        label_wrong = np.transpose(label_wrong, (2, 0, 1))
                        label_good = np.transpose(label_good, (2, 0, 1))
                    #
                elif c_amount == 2:
                    #
                    label_over = np.expand_dims(label_over, axis=0)
                    label_under = np.expand_dims(label_under, axis=0)
                    label_wrong = np.expand_dims(label_wrong, axis=0)
                    label_good = np.expand_dims(label_good, axis=0)
                #
                c_amount = len(np.shape(image))
                #
                if c_amount == 3:
                    #
                    d1, d2, d3 = np.shape(image)
                    #
                    if d1 != min(d1, d2, d3):
                        #
                        image = np.transpose(image, (2, 0, 1))
                        #
                elif c_amount == 2:
                    #
                    image = np.expand_dims(image, axis=0)
                #
                imagename = all_images[index]
                path_image, imagename = os.path.split(imagename)
                imagename, imageext = os.path.splitext(imagename)
                #
                if self.data_aug is True:
                    #
                    augmentation = random.uniform(0, 1)
                    #
                    if augmentation > 0.5:
                        #
                        c, h, w = np.shape(image)
                        #
                        for channel in range(c):
                            #
                            image[channel, :, :] = np.flip(image[channel, :, :], axis=0).copy()
                            image[channel, :, :] = np.flip(image[channel, :, :], axis=1).copy()
                            #
                        label_over = np.flip(label_over, axis=1).copy()
                        label_over = np.flip(label_over, axis=2).copy()
                        label_under = np.flip(label_under, axis=1).copy()
                        label_under = np.flip(label_under, axis=2).copy()
                        label_wrong = np.flip(label_wrong, axis=1).copy()
                        label_wrong = np.flip(label_wrong, axis=2).copy()
                        label_good = np.flip(label_good, axis=1).copy()
                        label_good = np.flip(label_good, axis=2).copy()
                        #
                return image, label_over, label_under, label_wrong, label_good, imagename

            elif self.dataset_tag == 'oocytes_gent':
                #
                all_labels_AR = glob.glob(os.path.join(self.label_AR_folder, '*.tif'))
                all_labels_AR.sort()
                #
                all_labels_HS = glob.glob(os.path.join(self.label_HS_folder, '*.tif'))
                all_labels_HS.sort()
                #
                all_labels_SG = glob.glob(os.path.join(self.label_SG_folder, '*.tif'))
                all_labels_SG.sort()

                # -missing -> think of using majority-vote or STAPLE
                #all_labels_good = glob.glob(os.path.join(self.label_good_folder, '*.tif'))
                #all_labels_good.sort()
                #

                # using majority-vote
                all_labels_avrg = glob.glob(os.path.join(self.label_avrg_folder, '*.tif'))
                all_labels_avrg.sort()
                #

                all_images = glob.glob(os.path.join(self.image_folder, '*.tif'))
                all_images.sort()
                #
                label_AR = tiff.imread(all_labels_AR[index])
                label_AR = np.array(label_AR, dtype='float32') / 255.
                #
                label_HS = tiff.imread(all_labels_HS[index])
                label_HS = np.array(label_HS, dtype='float32') / 255.
                #
                label_SG = tiff.imread(all_labels_SG[index])
                label_SG = np.array(label_SG, dtype='float32') / 255.
                #
                label_avrg = tiff.imread(all_labels_avrg[index])
                label_avrg = np.array(label_avrg, dtype='float32') / 255.
                #
                image = tiff.imread(all_images[index])
                image = np.array(image, dtype='float32') / 255.

                # print("Image max: ", image.max())
                # print("Mask max: ", label_HS.max())

                label_AR[label_AR == 4.0] = 3.0
                label_SG[label_SG == 4.0] = 3.0
                label_avrg[label_avrg == 4.0] = 3.0
                label_HS[label_HS == 4.0] = 3.0

                if self.dataset_tag == 'oocytes_gent':
                    label_AR = np.where(label_AR > 0.5, 1.0, 0.0)
                    label_HS = np.where(label_HS > 0.5, 1.0, 0.0)
                    label_SG = np.where(label_SG > 0.5, 1.0, 0.0)

                    if np.amax(label_avrg) != 1.0:
                        # sometimes, some preprocessing might give it as 0 - 255 range
                        label_avrg = np.where(label_avrg > 10.0, 1.0, 0.0)
                    else:
                        assert np.amax(label_avrg) == 1.0
                        label_avrg = np.where(label_avrg > 0.5, 1.0, 0.0)

                # print(np.unique(label_over))
                # label_over: h x w
                # image: h x w x c
                c_amount = len(np.shape(label_AR))

                if c_amount == 3:
                    #
                    d1, d2, d3 = np.shape(label_AR)
                    #
                    if d1 != min(d1, d2, d3):
                        #
                        assert d3 == min(d1, d2, d3)
                        #
                        label_AR = np.transpose(label_AR, (2, 0, 1))
                        label_HS = np.transpose(label_HS, (2, 0, 1))
                        label_SG = np.transpose(label_SG, (2, 0, 1))
                        label_avrg = np.transpose(label_avrg, (2, 0, 1))
                    #
                elif c_amount == 2:
                    #
                    label_AR = np.expand_dims(label_AR, axis=0)
                    label_HS = np.expand_dims(label_HS, axis=0)
                    label_SG = np.expand_dims(label_SG, axis=0)
                    label_avrg = np.expand_dims(label_avrg, axis=0)
                #
                c_amount = len(np.shape(image))
                #
                if c_amount == 3:
                    #
                    d1, d2, d3 = np.shape(image)
                    #
                    if d1 != min(d1, d2, d3):
                        #
                        image = np.transpose(image, (2, 0, 1))
                        #
                elif c_amount == 2:
                    #
                    image = np.expand_dims(image, axis=0)
                #
                imagename = all_images[index]
                path_image, imagename = os.path.split(imagename)
                imagename, imageext = os.path.splitext(imagename)
                #
                if self.data_aug is True:
                    #
                    augmentation = random.uniform(0, 1)
                    #
                    if augmentation > 0.5:
                        #
                        c, h, w = np.shape(image)
                        #
                        for channel in range(c):
                            #
                            image[channel, :, :] = np.flip(image[channel, :, :], axis=0).copy()
                            image[channel, :, :] = np.flip(image[channel, :, :], axis=1).copy()
                            #
                        label_AR = np.flip(label_AR, axis=1).copy()
                        label_AR = np.flip(label_AR, axis=2).copy()
                        label_HS = np.flip(label_HS, axis=1).copy()
                        label_HS = np.flip(label_HS, axis=2).copy()
                        label_SG = np.flip(label_SG, axis=1).copy()
                        label_SG = np.flip(label_SG, axis=2).copy()
                        label_avrg = np.flip(label_avrg, axis=1).copy()
                        label_avrg = np.flip(label_avrg, axis=2).copy()
                        #

                return image, label_AR, label_HS, label_SG, label_avrg, imagename

            elif self.dataset_tag == 'oocytes_gent_np':
                #
                all_labels_AR = glob.glob(os.path.join(self.label_AR_folder, '*.npy.gz'))
                all_labels_AR.sort()
                #
                all_labels_HS = glob.glob(os.path.join(self.label_HS_folder, '*.npy.gz'))
                all_labels_HS.sort()
                #
                all_labels_SG = glob.glob(os.path.join(self.label_SG_folder, '*.npy.gz'))
                all_labels_SG.sort()
                #
                # using majority-vote
                all_labels_avrg = glob.glob(os.path.join(self.label_avrg_folder, '*.npy.gz'))
                all_labels_avrg.sort()
                #
                all_images = glob.glob(os.path.join(self.image_folder, '*.npy.gz'))
                all_images.sort()
                #
                label_AR = np.load(gzip.open(all_labels_AR[index])) / 255.
                #
                label_HS = np.load(gzip.open(all_labels_HS[index])) / 255.
                #
                label_SG = np.load(gzip.open(all_labels_SG[index])) / 255.
                #
                label_avrg = np.load(gzip.open(all_labels_avrg[index])) / 255.
                #
                image = np.load(gzip.open(all_images[index])) / 255.

                # print("Image max: ", image.max())
                # print("Mask max: ", label_HS.max())

                label_AR[label_AR == 4.0] = 3.0
                label_SG[label_SG == 4.0] = 3.0
                label_avrg[label_avrg == 4.0] = 3.0
                label_HS[label_HS == 4.0] = 3.0

                if self.dataset_tag == 'oocytes_gent_np':
                    label_AR = np.where(label_AR > 0.5, 1.0, 0.0)
                    label_HS = np.where(label_HS > 0.5, 1.0, 0.0)
                    label_SG = np.where(label_SG > 0.5, 1.0, 0.0)

                    if np.amax(label_avrg) != 1.0:
                        # sometimes, some preprocessing might give it as 0 - 255 range
                        label_avrg = np.where(label_avrg > 10.0, 1.0, 0.0)
                    else:
                        assert np.amax(label_avrg) == 1.0
                        label_avrg = np.where(label_avrg > 0.5, 1.0, 0.0)

                # print(np.unique(label_over))
                # label_over: h x w
                # image: h x w x c
                c_amount = len(np.shape(label_AR))

                if c_amount == 3:
                    #
                    d1, d2, d3 = np.shape(label_AR)
                    #
                    if d1 != min(d1, d2, d3):
                        #
                        assert d3 == min(d1, d2, d3)
                        #
                        label_AR = np.transpose(label_AR, (2, 0, 1))
                        label_HS = np.transpose(label_HS, (2, 0, 1))
                        label_SG = np.transpose(label_SG, (2, 0, 1))
                        label_avrg = np.transpose(label_avrg, (2, 0, 1))
                    #
                elif c_amount == 2:
                    #
                    label_AR = np.expand_dims(label_AR, axis=0)
                    label_HS = np.expand_dims(label_HS, axis=0)
                    label_SG = np.expand_dims(label_SG, axis=0)
                    label_avrg = np.expand_dims(label_avrg, axis=0)
                #
                c_amount = len(np.shape(image))
                #
                if c_amount == 3:
                    #
                    d1, d2, d3 = np.shape(image)
                    #
                    if d1 != min(d1, d2, d3):
                        #
                        image = np.transpose(image, (2, 0, 1))
                        #
                elif c_amount == 2:
                    #
                    image = np.expand_dims(image, axis=0)
                #
                imagename = all_images[index]
                path_image, imagename = os.path.split(imagename)
                imagename, imageext = os.path.splitext(imagename)
                #
                if self.data_aug is True:
                    #
                    augmentation = random.uniform(0, 1)
                    #
                    if augmentation > 0.5:
                        #
                        c, h, w = np.shape(image)
                        #
                        for channel in range(c):
                            #
                            image[channel, :, :] = np.flip(image[channel, :, :], axis=0).copy()
                            image[channel, :, :] = np.flip(image[channel, :, :], axis=1).copy()
                            #
                        label_AR = np.flip(label_AR, axis=1).copy()
                        label_AR = np.flip(label_AR, axis=2).copy()
                        label_HS = np.flip(label_HS, axis=1).copy()
                        label_HS = np.flip(label_HS, axis=2).copy()
                        label_SG = np.flip(label_SG, axis=1).copy()
                        label_SG = np.flip(label_SG, axis=2).copy()
                        label_avrg = np.flip(label_avrg, axis=1).copy()
                        label_avrg = np.flip(label_avrg, axis=2).copy()
                        #

                return image, label_AR, label_HS, label_SG, label_avrg, imagename

            elif self.dataset_tag == 'lidc':
                #
                all_labels_over = glob.glob(os.path.join(self.label_over_folder, '*.tif'))
                all_labels_over.sort()
                #
                all_labels_under = glob.glob(os.path.join(self.label_under_folder, '*.tif'))
                all_labels_under.sort()
                #
                all_labels_wrong = glob.glob(os.path.join(self.label_wrong_folder, '*.tif'))
                all_labels_wrong.sort()
                #
                all_labels_good = glob.glob(os.path.join(self.label_good_folder, '*.tif'))
                all_labels_good.sort()
                #
                all_labels_true = glob.glob(os.path.join(self.label_true_folder, '*.tif'))
                all_labels_true.sort()
                #
                all_images = glob.glob(os.path.join(self.image_folder, '*.tif'))
                all_images.sort()
                #
                label_over = tiff.imread(all_labels_over[index])
                label_over = np.array(label_over, dtype='float32')
                #
                label_under = tiff.imread(all_labels_under[index])
                label_under = np.array(label_under, dtype='float32')
                #
                label_wrong = tiff.imread(all_labels_wrong[index])
                label_wrong = np.array(label_wrong, dtype='float32')
                #
                label_good = tiff.imread(all_labels_good[index])
                label_good = np.array(label_good, dtype='float32')
                #
                label_true = tiff.imread(all_labels_true[index])
                label_true = np.array(label_true, dtype='float32')
                #
                image = tiff.imread(all_images[index])
                image = np.array(image, dtype='float32')
                #
                # dim_length = len(np.shape(label_over))

                # label_over[label_over == 4.0] = 3.0
                # label_wrong[label_wrong == 4.0] = 3.0
                # label_good[label_good == 4.0] = 3.0
                # label_under[label_under == 4.0] = 3.0
                # label_true[label_true == 4.0] = 3.0
                # print(np.unique(label_over))
                # label_over: h x w
                # image: h x w x c
                c_amount = len(np.shape(label_over))
                # Reshaping everyting to make sure the order: channel x height x width
                if c_amount == 3:
                    #
                    d1, d2, d3 = np.shape(label_over)
                    #
                    if d1 != min(d1, d2, d3):
                        #
                        assert d3 == min(d1, d2, d3)
                        #
                        label_over = np.transpose(label_over, (2, 0, 1))
                        label_under = np.transpose(label_under, (2, 0, 1))
                        label_wrong = np.transpose(label_wrong, (2, 0, 1))
                        label_good = np.transpose(label_good, (2, 0, 1))
                        label_true = np.transpose(label_true, (2, 0, 1))
                    #
                elif c_amount == 2:
                    #
                    label_over = np.expand_dims(label_over, axis=0)
                    label_under = np.expand_dims(label_under, axis=0)
                    label_wrong = np.expand_dims(label_wrong, axis=0)
                    label_good = np.expand_dims(label_good, axis=0)
                    label_true = np.expand_dims(label_true, axis=0)
                #
                c_amount = len(np.shape(image))
                #
                if c_amount == 3:
                    #
                    d1, d2, d3 = np.shape(image)
                    #
                    if d1 != min(d1, d2, d3):
                        #
                        image = np.transpose(image, (2, 0, 1))
                        #
                elif c_amount == 2:
                    #
                    image = np.expand_dims(image, axis=0)
                #
                imagename = all_images[index]
                path_image, imagename = os.path.split(imagename)
                imagename, imageext = os.path.splitext(imagename)
                #
                if self.data_aug is True:
                    #
                    augmentation = random.uniform(0, 1)
                    #
                    if augmentation > 0.5:
                        #
                        c, h, w = np.shape(image)
                        #
                        for channel in range(c):
                            #
                            image[channel, :, :] = np.flip(image[channel, :, :], axis=0).copy()
                            image[channel, :, :] = np.flip(image[channel, :, :], axis=1).copy()
                            #
                        label_over = np.flip(label_over, axis=1).copy()
                        label_over = np.flip(label_over, axis=2).copy()
                        label_under = np.flip(label_under, axis=1).copy()
                        label_under = np.flip(label_under, axis=2).copy()
                        label_wrong = np.flip(label_wrong, axis=1).copy()
                        label_wrong = np.flip(label_wrong, axis=2).copy()
                        label_good = np.flip(label_good, axis=1).copy()
                        label_good = np.flip(label_good, axis=2).copy()
                        label_true = np.flip(label_true, axis=1).copy()
                        label_true = np.flip(label_true, axis=2).copy()
                        #
                return image, label_over, label_under, label_wrong, label_good, label_true, imagename
        #
        elif self.label_mode == 'binary':

            all_true_labels = glob.glob(os.path.join(self.true_label_folder, '*.tif'))
            all_true_labels.sort()
            all_labels = glob.glob(os.path.join(self.label_folder, '*.tif'))
            all_labels.sort()
            all_images = glob.glob(os.path.join(self.image_folder, '*.tif'))
            all_images.sort()
            #
            image = tiff.imread(all_images[index])
            image = np.array(image, dtype='float32')
            #
            label = tiff.imread(all_labels[index])
            label = np.array(label, dtype='float32')
            #
            true_label = tiff.imread(all_true_labels[index])
            true_label = np.array(true_label, dtype='float32')
            #
            d1, d2, d3 = np.shape(label)
            image = np.reshape(image, (d3, d1, d2))
            label = np.reshape(label, (d3, d1, d2))
            true_label = np.reshape(true_label, (d3, d1, d2))
            #
            imagename = all_images[index]
            path_image, imagename = os.path.split(imagename)
            imagename, imageext = os.path.splitext(imagename)
            #
            if self.data_aug is True:
                #
                augmentation = random.uniform(0, 1)
                #
                if augmentation < 0.25:
                    #
                    c, h, w = np.shape(image)
                    #
                    for channel in range(c):
                        #
                        image[channel, :, :] = np.flip(image[channel, :, :], axis=0).copy()
                        image[channel, :, :] = np.flip(image[channel, :, :], axis=1).copy()
                        #
                    label = np.flip(label, axis=1).copy()
                    label = np.flip(label, axis=2).copy()
                    #
                    true_label = np.flip(true_label, axis=1).copy()
                    true_label = np.flip(true_label, axis=2).copy()
                    #
                elif augmentation < 0.5:
                    #
                    mean = 0.0
                    sigma = 0.15
                    noise = np.random.normal(mean, sigma, image.shape)
                    mask_overflow_upper = image + noise >= 1.0
                    mask_overflow_lower = image + noise < 0.0
                    noise[mask_overflow_upper] = 1.0
                    noise[mask_overflow_lower] = 0.0
                    image += noise

                elif augmentation < 0.75:
                    #
                    c, h, w = np.shape(image)
                    #
                    for channel in range(c):
                        #
                        channel_ratio = random.uniform(0, 1)
                        #
                        image[channel, :, :] = image[channel, :, :] * channel_ratio

            return image, label, true_label, imagename

        elif self.label_mode == 'p_unet':

            all_labels = glob.glob(os.path.join(self.label_folder, '*.tif'))
            all_labels.sort()
            all_images = glob.glob(os.path.join(self.image_folder, '*.tif'))
            all_images.sort()
            #
            image = tiff.imread(all_images[index])
            image = np.array(image, dtype='float32')
            #
            label = tiff.imread(all_labels[index])
            label = np.array(label, dtype='float32')
            #
            d1, d2, d3 = np.shape(image)
            image = np.reshape(image, (d3, d1, d2))
            label = np.reshape(label, (1, d1, d2))
            #
            imagename = all_images[index]
            path_image, imagename = os.path.split(imagename)
            imagename, imageext = os.path.splitext(imagename)
            #
            if self.data_aug is True:
                #
                augmentation = random.uniform(0, 1)
                #
                if augmentation > 0.5:
                    #
                    c, h, w = np.shape(image)
                    #
                    for channel in range(c):
                        #
                        image[channel, :, :] = np.flip(image[channel, :, :], axis=0).copy()
                        image[channel, :, :] = np.flip(image[channel, :, :], axis=1).copy()
                        #
                    label = np.flip(label, axis=1).copy()
                    label = np.flip(label, axis=2).copy()
                    #
                # elif augmentation < 0.5:
                #     #
                #     mean = 0.0
                #     sigma = 0.15
                #     noise = np.random.normal(mean, sigma, image.shape)
                #     mask_overflow_upper = image + noise >= 1.0
                #     mask_overflow_lower = image + noise < 0.0
                #     noise[mask_overflow_upper] = 1.0
                #     noise[mask_overflow_lower] = 0.0
                #     image += noise
                #
                # elif augmentation < 0.75:
                #     #
                #     c, h, w = np.shape(image)
                #     #
                #     for channel in range(c):
                #         #
                #         channel_ratio = random.uniform(0, 1)
                #         #
                #         image[channel, :, :] = image[channel, :, :] * channel_ratio

            return image, label, imagename

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        #print("Len: ", len(glob.glob(os.path.join(self.image_folder, '*.tif'))))
        if self.dataset_tag == 'oocytes_gent_np':
            return len(glob.glob(os.path.join(self.image_folder, '*.npy.gz')))
        else:
            return len(glob.glob(os.path.join(self.image_folder, '*.tif')))


def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)


def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        # nn.init.normal_(m.weight, std=0.001)
        # nn.init.normal_(m.bias, std=0.001)
        # truncated_normal_(m.bias, mean=0, std=0.001)


def init_weights_orthogonal_normal(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.orthogonal_(m.weight)
        # truncated_normal_(m.bias, mean=0, std=0.001)
        #nn.init.normal_(m.bias, std=0.001)


def l2_regularisation(m):
    l2_reg = None

    for W in m.parameters():
        if l2_reg is None:
            l2_reg = W.norm(2)
        else:
            l2_reg = l2_reg + W.norm(2)
    return l2_reg


def save_mask_prediction_example(mask, pred, iter):
	plt.imshow(pred[0,:,:],cmap='Greys')
	plt.savefig('images/'+str(iter)+"_prediction.png")
	plt.imshow(mask[0,:,:],cmap='Greys')
	plt.savefig('images/'+str(iter)+"_mask.png")


def test_punet(net, testdata, save_path, sampling_times):
    #
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.eval()
    test_iou = 0
    test_generalized_energy_distance = 0
    epoch_noisy_labels = []
    epoch_noisy_segs = []
    # sampling_times = 10
    # save_path = '../../projects_data/Exp_Results'
    #
    try:
        os.mkdir(save_path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass
    #
    save_path = save_path + '/Visual_segmentation'
    #
    try:
        os.mkdir(save_path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass
    #
    for no_eval, (patch_eval, mask_eval_over, mask_eval_under, mask_eval_wrong, mask_eval_good, mask_name_eval) in enumerate(testdata):
        #
        if no_eval < 30:
            #
            patch_eval = patch_eval.to(device)
            mask_eval_over = mask_eval_over.to(device)
            mask_eval_under = mask_eval_under.to(device)
            mask_eval_wrong = mask_eval_wrong.to(device)
            mask_eval_good = mask_eval_good.to(device)
            #
            for j in range(sampling_times):
                #
                net.eval()
                # segm input doesn't matter
                net.forward(patch_eval, mask_eval_good, training=False)
                seg_sample = net.sample(testing=True)
                seg_sample = (torch.sigmoid(seg_sample) > 0.5).float()
                (b, c, h, w) = seg_sample.shape
                #
                if j == 0:
                    seg_evaluate = seg_sample
                else:
                    seg_evaluate += seg_sample
                    #
                epoch_noisy_segs.append(seg_sample.cpu().detach().numpy())
                #
                if no_eval < 10:
                    #
                    save_name = save_path + '/test_' + str(no_eval) + '_sample_' + str(j) + '_seg.png'
                    #
                    plt.imsave(save_name, seg_sample.reshape(h, w).cpu().detach().numpy(), cmap='gray')
            #
            seg_evaluate = seg_evaluate / sampling_times
        #
        if no_eval < 10:
            #
            gt_save_name = save_path + '/gt_' + str(no_eval) + '.png'
            #
            plt.imsave(gt_save_name, mask_eval_good.reshape(h, w).cpu().detach().numpy(), cmap='gray')
        #
        val_iou = segmentation_scores(mask_eval_good.cpu().detach().numpy(), seg_evaluate.cpu().detach().numpy(), 2)
        test_iou += val_iou
        epoch_noisy_labels = [mask_eval_good.cpu().detach().numpy(), mask_eval_over.cpu().detach().numpy(), mask_eval_under.cpu().detach().numpy(), mask_eval_wrong.cpu().detach().numpy()]
        ged = generalized_energy_distance(epoch_noisy_labels, epoch_noisy_segs, 2)
        test_generalized_energy_distance += ged
        #
    test_iou = test_iou / no_eval
    test_generalized_energy_distance = test_generalized_energy_distance / no_eval
    #
    result_dictionary = {'Test IoU': str(test_iou), 'Test GED': str(test_generalized_energy_distance)}
    ff_path = save_path + '/test_result_data.txt'
    ff = open(ff_path, 'w')
    ff.write(str(result_dictionary))
    ff.close()
    #
    print('Test iou: ' + str(test_iou))
    print('Test generalised energy distance: ' + str(test_generalized_energy_distance))


def evaluate_punet(net, val_data, class_no, sampling_no):
    #
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    validate_iou = 0
    generalized_energy_distance_epoch = 0
    #
    for no_eval, (patch_eval, mask_eval_over, mask_eval_under, mask_eval_wrong, mask_eval_true, mask_name_eval) in enumerate(val_data):
        #
        patch_eval = patch_eval.to(device)
        mask_eval_over = mask_eval_over.to(device)
        mask_eval_under = mask_eval_under.to(device)
        mask_eval_wrong = mask_eval_wrong.to(device)
        mask_eval_true = mask_eval_true.to(device)
        epoch_noisy_segs = []
        #
        for j in range(sampling_no):
            net.eval()
            # segm input doesn't matter
            net.forward(patch_eval, mask_eval_wrong, training=False)
            seg_sample = net.sample(testing=True)
            seg_sample = (torch.sigmoid(seg_sample) > 0.5).float()
            #
            if j == 0:
                #
                seg_evaluate = seg_sample
                #
            else:
                #
                seg_evaluate += seg_sample
                #
            epoch_noisy_segs.append(seg_sample.cpu().detach().numpy())
            #
        seg_evaluate = seg_evaluate / sampling_no
        #
        val_iou = segmentation_scores(mask_eval_true.cpu().detach().numpy(), seg_evaluate.cpu().detach().numpy(), class_no)
        epoch_noisy_labels = [mask_eval_true.cpu().detach().numpy(), mask_eval_over.cpu().detach().numpy(), mask_eval_under.cpu().detach().numpy(), mask_eval_wrong.cpu().detach().numpy()]
        # epoch_noisy_segs = [seg_good.cpu().detach().numpy(), seg_over.cpu().detach().numpy(), seg_under.cpu().detach().numpy(), seg_wrong.cpu().detach().numpy()]
        ged = generalized_energy_distance(epoch_noisy_labels, epoch_noisy_segs, class_no)
        validate_iou += val_iou
        generalized_energy_distance_epoch += ged
        #
    return validate_iou / (no_eval), generalized_energy_distance_epoch / (no_eval)

def segmentation_scores_v2(label_trues, label_preds, n_class):
    # Ensure label_trues and label_preds have the same length
    assert len(label_trues) == len(label_preds)

    # If there are only two classes (binary segmentation), convert predictions to binary using a threshold of 0.5
    if n_class == 2:
        output_zeros = np.zeros_like(label_preds)
        output_ones = np.ones_like(label_preds)
        label_preds = np.where((label_preds > 0.5), output_ones, output_zeros)

    # Convert labels to integer type
    label_preds = np.asarray(label_preds, dtype='int8').copy()
    label_trues = np.asarray(label_trues, dtype='int8').copy()

    # Initialize arrays for storing the intersection and union values for each class
    area_intersection = np.zeros(n_class - 1)
    area_union = np.zeros(n_class - 1)

    # Iterate over each class and calculate intersection and union
    for i in range(1, n_class):
        # Calculate the intersection: the count of pixels where both predicted and true labels are equal for the current class
        intersection = np.logical_and(label_preds == i, label_trues == i).sum()

        # Calculate the union: the count of pixels where either the predicted or true labels are equal to the current class
        union = np.logical_or(label_preds == i, label_trues == i).sum()

        # Store the intersection and union values for the current class
        area_intersection[i - 1] = intersection
        area_union[i - 1] = union

    # Calculate the Dice coefficient for each class and return the mean value
    return ((2 * area_intersection + 1e-6) / (area_union + 1e-6)).mean()


def segmentation_scores(label_trues, label_preds, n_class):
    '''
    :param label_trues:
    :param label_preds:
    :param n_class:
    :return:
    '''
    # Ensure ground truth and predicted labels have the same length
    # print("trues:", len(label_trues))
    # print("preds:", len(label_preds))
    assert len(label_trues) == len(label_preds)

    # Threshold predictions if only two classes (binary segmentation)
    if n_class == 2:
        #
        output_zeros = np.zeros_like(label_preds)
        output_ones = np.ones_like(label_preds)
        label_preds = np.where((label_preds > 0.5), output_ones, output_zeros)

    # Shift ground truth and predicted labels by 1 to exclude the background class from the calculations
    label_trues += 1
    label_preds += 1

    # Convert ground truth and predicted labels to 'int8' numpy arrays
    label_preds = np.asarray(label_preds, dtype='int8').copy()
    label_trues = np.asarray(label_trues, dtype='int8').copy()

    # Apply a boolean mask to only consider predictions where ground truth is not the background class (0)
    #label_preds = label_preds * (label_trues > 0)
    label_preds = np.where(label_trues > 0, label_preds, 0)

    # Compute the element-wise intersection of ground truth and predicted labels
    intersection = label_preds * (label_preds == label_trues)
    # print("label_trues shape:", label_trues.shape)
    # print("label_preds shape:", label_preds.shape)
    # intersection = 2 * label_trues * (label_preds == label_trues)    

    # Compute the number of true positives, false positives, and false negatives
    (area_intersection, _) = np.histogram(intersection, bins=n_class, range=(1, n_class))
    (area_pred, _) = np.histogram(label_preds, bins=n_class, range=(1, n_class))
    (area_lab, _) = np.histogram(label_trues, bins=n_class, range=(1, n_class))

    # Calculate the area of the union by summing true positives, false positives, and false negatives
    area_union = area_pred + area_lab
    
    # Compute the Dice coefficient
    dice = ((2 * area_intersection + 1e-6) / (area_union + 1e-6))
    # print("dice: ", dice)
    return dice.mean()

def binary_dice_coefficient(target, pred, threshold = 0.5):

    dice = 0
    smooth = 1e-6
    # print("Target shape: ", target.shape)
    # print("Preds shape: ", pred.shape)
    # Select the positive class channel from the predicted probabilities
    pred_positive_class = pred[:, 1, :, :]
    
    # Threshold the probabilities to create binary predictions
    pred_binary = (pred_positive_class > threshold).astype(np.float32)
    
    # Remove the channel dimension from target array
    target = np.squeeze(target, axis = 1)
    
    # Calculate intersection and union
    intersection = np.sum(pred_binary * target)
    union = np.sum(pred_binary) + np.sum(target)
    
    # Calculate Dice coefficient
    dice = (2 * intersection + smooth) / (union + smooth)

    return dice

class DiceScore(nn.Module):
    def __init__(self, weight = None, size_average = True):
        super(DiceScore, self).__init__()

    def forward(self, inputs, targets, smooth = 1.):

        inputs = torch.sigmoid(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return dice

def generalized_energy_distance(all_gts, all_segs, class_no):
    '''
    :param all_gts: a list of all noisy labels
    :param all_segs: a list of all noisy segmentation
    :param class_no: class number
    :return:
    '''
    # This is slightly different from the original paper:
    # We didn't take the distance to the power of 2
    #
    gt_gt_dist = [segmentation_scores(gt_1, gt_2, class_no) for i1, gt_1 in enumerate(all_gts) for i2, gt_2 in enumerate(all_gts) if i1 != i2]
    seg_seg_dist = [segmentation_scores(seg_1, seg_2, class_no) for i1, seg_1 in enumerate(all_segs) for i2, seg_2 in enumerate(all_segs) if i1 != i2]
    seg_gt_list = [segmentation_scores(seg_, gt_, class_no) for i, seg_ in enumerate(all_segs) for j, gt_ in enumerate(all_gts)]
    ged_metric = sum(gt_gt_dist) / len(gt_gt_dist) + sum(seg_seg_dist) / len(seg_seg_dist) + 2 * sum(seg_gt_list) / len(seg_gt_list)
    return ged_metric


def preprocessing_accuracy(label_true, label_pred, n_class):
    #
    if n_class == 2:
        output_zeros = np.zeros_like(label_pred)
        output_ones = np.ones_like(label_pred)
        label_pred = np.where((label_pred > 0.5), output_ones, output_zeros)
    #
    label_pred = np.asarray(label_pred, dtype='int8')
    label_true = np.asarray(label_true, dtype='int8')

    mask = (label_true >= 0) & (label_true < n_class) & (label_true != 8)

    label_true = label_true[mask].astype(int)
    label_pred = label_pred[mask].astype(int)

    return label_true, label_pred


def calculate_cm(pred, true):
    #
    pred = pred.view(-1)
    true = true.view(-1)
    #
    pred = pred.cpu().detach().numpy()
    true = true.cpu().detach().numpy()
    #
    confusion_matrices = confusion_matrix(y_true=true, y_pred=pred, normalize='all')
    #
    # if tag == 'brats':
    #     confusion_matrices = confusion_matrix(y_true=true, y_pred=pred, normalize='all', labels=[0, 1, 2, 3])
    # else:
    #     confusion_matrices = confusion_matrix(y_true=true, y_pred=pred, normalize='all', labels=[0, 1])
    #
    #
    return confusion_matrices

# ================================
# Evaluation
# ================================
# def dice_coef_torchmetrics(preds, targets, class_no, device):

#     dice = 0

#     # print("Preds size: ", preds.size())
#     # print("Targets size: ", targets.size())

#     dice_score = torchmetrics.Dice(zero_division = 1e-6,
#                                    num_classes = class_no,
#                                    threshold = 0.5,
#                                    average = 'macro').to(device)

#     probs = torch.sigmoid(preds)
#     targets_int = (targets > 0.5).long()

#     ### Sanity Check ###
#     ### 1. Perfect P ###
#     # probs = torch.tensor([[[[1, 0],
#     #                         [0, 1]],
#     #                        [[0, 1],
#     #                         [1, 0]]],
#     #                       [[[1, 0],
#     #                         [0, 1]],
#     #                        [[0, 1],
#     #                         [1, 0]]]], dtype = torch.float32, device = device)
#     # targets_int = torch.tensor([[[[0, 1],
#     #                               [1, 0]]],
#     #                             [[[0, 1],
#     #                               [1, 0]]]], dtype = torch.long, device = device)
 
#     # print("Probs size: ", probs.size())
#     # print("Targets_int size: ", targets_int.size())
#     ### 2. Worst Pre ###
#     # pass
#     ### ------------ ###

#     dice = dice_score(probs, targets_int)

#     # print("Dice score (metrics) = ", dice.item())

#     return dice.item()

def dice_coef_custom(preds, targets):
    """ This is a normal dice coef function for binary segmentation.
    1. Dice score for the each sample in the batch.
    2. Logits directly.

    Args:
        preds: output of the segmentation network
        targets: ground truth label

    Returns:
        dice
    """
    ### Sanity Check ###
    ### 1. Perfect P ###
    # preds = torch.tensor([[[[1, 0],
    #                         [0, 1]],
    #                        [[0, 1],
    #                         [1, 0]]],
    #                       [[[1, 0],
    #                         [0, 1]],
    #                        [[0, 1],
    #                         [1, 0]]]], dtype = torch.float32, device = device)
    # targets = torch.tensor([[[[0, 1],
    #                               [1, 0]]],
    #                             [[[0, 1],
    #                               [1, 0]]]], dtype = torch.long, device = device)
 
    # print("Probs size: ", preds.size())
    # print("Targets_int size: ", targets.size())
    #probs = probs.unsqueeze(1)
    ### 2. Worst Pre ###
    # pass
    ### ------------ ###

    batch_size, _, h, w = targets.size()

    _, preds = torch.max(preds, dim = 1, keepdim = True)

    preds_flat = preds.view(batch_size, -1)
    targets_flat = targets.view(batch_size, -1)

    intersection = (preds_flat * targets_flat).sum(dim = 1)

    preds_count = preds_flat.sum(dim = 1)
    targets_count = targets_flat.sum(dim = 1)

    dice = (2 * intersection + 1e-6) / (preds_count + targets_count + 1e-6)

    # print("Dice score (custom) = ", dice.mean().item())

    return dice.mean().item()

def dice_coef_default(input, target):
    """ This is a normal dice coef function for binary segmentation.
    1. Dice score for the entire batch.
    2. Sigmoid of the logits.

    Args:
        input: output of the segmentation network
        target: ground truth label

    Returns:
        dice
    """
    smooth = 1e-6
    b, c, h, w = input.size()
    
    # input_sig = torch.sigmoid(input)
    input_sig = input[:, 0, :, :]
    target = target

    iflat = input_sig[:, 1, :, :].contiguous().view(-1).float()
    tflat = target.view(-1).float()
    intersection = (iflat * tflat).sum()
    union = iflat.sum() + tflat.sum()
    dice = (2. * intersection + smooth) / (union + smooth)

    # print("Dice score (default) = ", dice.item())

    return dice

def dice_coef_simplified(pred, target):
    
    smooth = 1.

    if pred.size()[1] == 2:
        pred = pred[:, 1, :, :].unsqueeze(1)

    iflat = pred.contiguous().view(-1).to('cuda')
    tflat = target.contiguous().view(-1).to('cuda')

    intersection = (iflat * tflat).sum()

    A_sum = torch.sum(iflat * iflat)
    B_sum = torch.sum(tflat * tflat)

    dice = (2. * intersection + smooth) / (A_sum + B_sum + smooth)

    return dice

def evaluate_noisy_label_4(data, model1, class_no):
    """

    Args:
        data:
        model1:
        class_no:

    Returns:

    """
    model1.eval()
    # model2.eval()
    #
    #
    with torch.no_grad():
    
        test_dice = 0
        test_dice_all = []
        
        for i, (v_images, v_labels_AR, v_labels_HS, v_labels_SG, v_labels_avrg, v_imagename) in enumerate(data):
            #
            #print(v_imagename)
            #
            v_images = v_images.to(device='cuda', dtype=torch.float32)

            v_outputs_logits, cms = model1(v_images)
            b, c, h, w = v_outputs_logits.size()
            
            if class_no == 2:
                v_outputs_logits = torch.sigmoid(v_outputs_logits)
                #v_outputs_logits = (v_outputs_logits > 0.5).float()
            else:
                v_outputs_logits = nn.Softmax(dim=1)(v_outputs_logits)
            # cms = model2(v_images)
            #
            #_, v_output = torch.max(v_outputs_logits, dim=1)
            # v_output = (v_outputs_logits > 0.5).float()
            _, v_output = torch.max(v_outputs_logits, dim=1)
            ###_, v_output = torch.max(cms[0], dim=1)

            v_outputs_noisy = []
            #
            v_outputs_logits = v_outputs_logits.view(b, c, h*w)
            v_outputs_logits = v_outputs_logits.permute(0, 2, 1).contiguous().view(b*h*w, c)
            v_outputs_logits = v_outputs_logits.view(b * h * w, c, 1)
            #
            for cm in cms:
                #
                cm = cm.reshape(b, c**2, h*w).permute(0, 2, 1).contiguous().view(b*h*w, c*c).view(b*h*w, c, c)
                cm = cm / cm.sum(1, keepdim=True)
                v_noisy_output = torch.bmm(cm, v_outputs_logits).view(b*h*w, c)
                v_noisy_output = v_noisy_output.view(b, h*w, c).permute(0, 2, 1).contiguous().view(b, c, h, w)
                _, v_noisy_output = torch.max(v_noisy_output, dim=1)
                v_outputs_noisy.append(v_noisy_output.cpu().detach().numpy())
            #
            # print("labels: ", v_labels_avrg.size())
            # print("preds: ", v_output.size())
            # v_dice_ = segmentation_scores(v_labels_avrg.cpu().detach().numpy(), v_outputs_logits.cpu().detach().numpy(), class_no)
            v_dice_ = segmentation_scores(v_labels_avrg.cpu().detach().numpy(), v_output.cpu().detach().numpy(), class_no)
            v_dice_ = dice_coef_simplified(v_output, v_labels_avrg)
            # v_dice_ = binary_dice_coefficient(v_labels_avrg.cpu().detach().numpy(), v_output.cpu().detach().numpy())
            # v_dice_ = dice_coef_default(v_output.to(device = 'cuda'), v_labels_avrg.to(device = 'cuda'))
            #v_dice_ = dice_coef_default(model1(v_images)[0].to(device = 'cuda'), v_labels_avrg.to(device = 'cuda'))
            #v_dice_ = dice_coef_default(v_output.unsqueeze(0).repeat(1, 2, 1, 1).to(device = 'cuda'), v_labels_avrg.to(device = 'cuda'))
            #v_dice_ = segmentation_scores(v_labels_AR, v_output.cpu().detach().numpy(), class_no)
            #v_dice_ += segmentation_scores(v_labels_HS, v_output.cpu().detach().numpy(), class_no)
            #v_dice_ += segmentation_scores(v_labels_SG, v_output.cpu().detach().numpy(), class_no)
            #v_dice_ /= 3
            #
            #epoch_noisy_labels = [v_labels_over.cpu().detach().numpy(), v_labels_under.cpu().detach().numpy(), v_labels_wrong.cpu().detach().numpy(), v_labels_good.cpu().detach().numpy()]
            epoch_noisy_labels = [v_labels_AR.cpu().detach().numpy(), v_labels_HS.cpu().detach().numpy(), v_labels_SG.cpu().detach().numpy(), v_labels_avrg.cpu().detach().numpy()]
            # v_ged = generalized_energy_distance(epoch_noisy_labels, v_outputs_noisy, class_no)
            v_ged = 0.
            test_dice += v_dice_
            test_dice_all.append(test_dice)
            #
        # print(i)
        # print(test_dice)
        # print(test_dice / (i + 1))
        #
        return test_dice / (i + 1), v_ged


def evaluate_noisy_label_5(data, model1, class_no):
    """

    Args:
        data:
        model1:
        class_no:

    Returns:

    """
    model1.eval()
    # model2.eval()
    #
    test_dice = 0
    test_dice_all = []
    #
    for i, (v_images, v_labels_over, v_labels_under, v_labels_wrong, v_labels_good, v_labels_true, v_imagename) in enumerate(data):
        #
        # print(i)
        #
        v_images = v_images.to(device='cuda', dtype=torch.float32)
        v_outputs_logits, cms = model1(v_images)
        b, c, h, w = v_outputs_logits.size()
        v_outputs_logits = nn.Softmax(dim=1)(v_outputs_logits)
        # cms = model2(v_images)
        #
        _, v_output = torch.max(v_outputs_logits, dim=1)
        v_outputs_noisy = []
        #
        v_outputs_logits = v_outputs_logits.view(b, c, h*w)
        v_outputs_logits = v_outputs_logits.permute(0, 2, 1).contiguous().view(b*h*w, c)
        v_outputs_logits = v_outputs_logits.view(b * h * w, c, 1)
        #
        for cm in cms:
            #
            # cm = cm.reshape(b * h * w, c, c)
            # cm = cm / cm.sum(1, keepdim=True)
            # v_noisy_output = torch.bmm(cm, v_outputs_logits.reshape(b * h * w, c, 1)).reshape(b, c, h, w)
            # cm = cm.permute(0, 2, 3, 1).contiguous().view(b * h * w, c, c)
            cm = cm.reshape(b, c**2, h*w).permute(0, 2, 1).contiguous().view(b*h*w, c*c).view(b*h*w, c, c)
            cm = cm / cm.sum(1, keepdim=True)
            v_noisy_output = torch.bmm(cm, v_outputs_logits).view(b*h*w, c)
            v_noisy_output = v_noisy_output.view(b, h*w, c).permute(0, 2, 1).contiguous().view(b, c, h, w)
            _, v_noisy_output = torch.max(v_noisy_output, dim=1)
            v_outputs_noisy.append(v_noisy_output.cpu().detach().numpy())
        #
        v_dice_ = segmentation_scores(v_labels_true, v_output.cpu().detach().numpy(), class_no)
        #
        epoch_noisy_labels = [v_labels_over.cpu().detach().numpy(), v_labels_under.cpu().detach().numpy(), v_labels_wrong.cpu().detach().numpy(), v_labels_true.cpu().detach().numpy(), v_labels_good.cpu().detach().numpy()]
        v_ged = generalized_energy_distance(epoch_noisy_labels, v_outputs_noisy, class_no)
        test_dice += v_dice_
        test_dice_all.append(test_dice)
        #
    # print(i)
    # print(test_dice)
    # print(test_dice / (i + 1))
    #
    return test_dice / (i + 1), v_ged


def evaluate_noisy_label_6(data, model1, class_no):
    """

    Args:
        data:
        model1:
        class_no:

    Returns:

    """
    model1.eval()
    # model2.eval()
    #
    test_dice = 0
    test_dice_all = []
    #
    for i, (v_images, v_labels_over, v_labels_under, v_labels_wrong, v_labels_good, v_imagename) in enumerate(data):
        #
        # print(i)
        #
        v_images = v_images.to(device='cuda', dtype=torch.float32)
        v_outputs_logits, cms = model1(v_images)
        b, c, h, w = v_outputs_logits.size()
        v_outputs_logits = nn.Softmax(dim=1)(v_outputs_logits)
        # cms = model2(v_images)
        #
        _, v_output = torch.max(v_outputs_logits, dim=1)
        v_outputs_noisy = []
        #
        v_outputs_logits = v_outputs_logits.view(b, c, h*w)
        v_outputs_logits = v_outputs_logits.permute(0, 2, 1).contiguous().view(b*h*w, c)
        v_outputs_logits = v_outputs_logits.view(b * h * w, c, 1)
        #
        for cm in cms:
            #
            b, c_r_d, h, w = cm.size()
            r = c_r_d // c // 2
            cm1 = cm[:, 0:r * c, :, :]
            if r == 1:
                cm2 = cm[:, r * c:c_r_d-1, :, :]
            else:
                cm2 = cm[:, r * c:c_r_d-1, :, :]
            cm1_reshape = cm1.view(b, c_r_d // 2, h * w).permute(0, 2, 1).contiguous().view(b * h * w, r * c).view(b * h * w, r, c)
            cm2_reshape = cm2.view(b, c_r_d // 2, h * w).permute(0, 2, 1).contiguous().view(b * h * w, r * c).view(b * h * w, c, r)
            #
            cm1_reshape = cm1_reshape / cm1_reshape.sum(1, keepdim=True)
            cm2_reshape = cm2_reshape / cm2_reshape.sum(1, keepdim=True)
            #
            v_noisy_output = torch.bmm(cm1_reshape, v_outputs_logits)
            v_noisy_output = torch.bmm(cm2_reshape, v_noisy_output).view(b * h * w, c)
            v_noisy_output = v_noisy_output.view(b, h * w, c).permute(0, 2, 1).contiguous().view(b, c, h, w)
            #
            # v_noisy_output = torch.bmm(cm, v_outputs_logits).view(b*h*w, c)
            # v_noisy_output = v_noisy_output.view(b, h*w, c).permute(0, 2, 1).contiguous().view(b, c, h, w)
            _, v_noisy_output = torch.max(v_noisy_output, dim=1)
            v_outputs_noisy.append(v_noisy_output.cpu().detach().numpy())
        #
        v_dice_ = segmentation_scores(v_labels_good, v_output.cpu().detach().numpy(), class_no)
        #
        epoch_noisy_labels = [v_labels_over.cpu().detach().numpy(), v_labels_under.cpu().detach().numpy(), v_labels_wrong.cpu().detach().numpy(), v_labels_good.cpu().detach().numpy()]
        v_ged = generalized_energy_distance(epoch_noisy_labels, v_outputs_noisy, class_no)
        test_dice += v_dice_
        test_dice_all.append(test_dice)
        #
    # print(i)
    # print(test_dice)
    # print(test_dice / (i + 1))
    #
    return test_dice / (i + 1), v_ged

def plot_curves(path, train_losses, ce_losses, trace_losses, train_metric, val_metric, metric_name = 'Dice Coef.'):

    print("plot_curves running")

    epochs = range(1, len(train_losses) + 1)

    # Plotting the training and validation loss
    plt.figure()
    plt.plot(epochs, train_losses, label = 'Total loss')
    plt.plot(epochs, ce_losses, label = 'CE loss')
    plt.plot(epochs, trace_losses, label = 'Trace loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Losses')
    plt.savefig(path + '/loss_curves.png')  # Save the loss curves as an image
    #plt.show()

    # Plotting the training and validation metric
    plt.figure()
    plt.plot(epochs, train_metric, label = f'Training {metric_name}')
    plt.plot(epochs, val_metric, label = f'Validation {metric_name}')
    plt.xlabel('Epochs')
    plt.ylabel(metric_name)
    plt.legend()
    plt.title(f'Training and Validation {metric_name}')
    plt.ylim([0.0, 1.0])
    plt.savefig(path + '/' + f'{metric_name.lower()}_curves.png')  # Save the metric curves as an image
    #plt.show()

