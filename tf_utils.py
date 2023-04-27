import numpy as np
import cv2
import gzip
import glob
import tifffile as tiff
import os

import matplotlib.pyplot as plt
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS = 192, 240, 1

def dice_coefficient(pred, target):

    smooth = 1e-6
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()

    return (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)

def dice_loss(pred, target):

    return 1 - dice_coefficient(pred, target)

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

    X_train = X_train.astype('float32')
    mean = np.mean(X_train)
    std = np.std(X_train)

    X_train -= mean
    X_train /= std

    y_train = y_train.astype('float32')
    y_train /= 255.

    return X_train, y_train

class SkinTrainDataset(Dataset):
    def __init__(self, imgs_path, masks_path, img_width, img_height, transform = None):
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
    
def load_coc_train_data(imgs_path, masks_path, img_width = IMG_WIDTH, img_height = IMG_HEIGHT):
    
    IMG_SIZE = (img_width, img_height)
    # Load input image
    input_image = Image.open(imgs_path)
    input_image = transforms.Resize(IMG_SIZE)(input_image)
    input_image = transforms.Grayscale(num_output_channels = IMG_CHANNELS)(input_image)
    input_image = transforms.ToTensor()(input_image)
    # print("img max:", input_image.max())
    # print("img min:", input_image.min())
    # input_image = input_image / 255.
    # input_image = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                     std=[0.229, 0.224, 0.225])(input_image)
    # print("Image size:", input_image.size())

    # Load input mask
    input_mask = Image.open(masks_path)
    input_mask = transforms.Resize(IMG_SIZE)(input_mask)
    # input_mask = transforms.Grayscale(num_output_channels = IMG_CHANNELS)(input_mask)
    input_mask = transforms.ToTensor()(input_mask)
    # print("mask max:", input_mask.max())
    # print("mask min:", input_mask.min())
    input_mask = input_mask / 255.
    # print("Mask size:", input_mask.size())

    return input_image, input_mask

class COCTrainDataset(Dataset):
    def __init__(self, imgs_path, masks_path, img_width, img_height, transform = None):
        # self.imgs, self.masks = load_coc_train_data(imgs_path, masks_path, img_width, img_height)
        # self.transform = transform
        self.imgs_folder = imgs_path
        self.msks_folder = masks_path
        self.transform = transform

    def __len__(self):
        length = len(glob.glob(os.path.join(self.imgs_folder, '*.tif')))
        # print(length)

        return length
    
    def __getitem__(self, idx):
        # img = self.imgs[idx]
        # mask = self.masks[idx]

        # if self.transform:
        #     img = self.transform(img)
        #     mask = self.transform(mask)

        all_images = glob.glob(os.path.join(self.imgs_folder, '*.tif'))
        all_images.sort()

        all_labels = glob.glob(os.path.join(self.msks_folder, '*.tif'))
        all_labels.sort()

        # image = tiff.imread(all_images[idx])
        # label = tiff.imread(all_labels[idx])

        image_path = all_images[idx]
        mask_path = all_labels[idx]
        image, label = load_coc_train_data(image_path, mask_path)

        # image = np.array(image, dtype = 'float32') / 255.
        # label = np.array(label, dtype = 'float32') / 255.
        # label = np.expand_dims(label, axis = -1)
        # print("Image shape:", image.shape)
        # print("Label shape:", label.shape)
        
        # image = torch.from_numpy(image).permute(2, 0, 1).float()
        # label = torch.from_numpy(label).permute(2, 0, 1).float()

        return image, label

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