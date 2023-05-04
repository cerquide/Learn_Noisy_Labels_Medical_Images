import numpy as np
import cv2
import gzip
import glob
# import tifffile as tiff
import os

from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS = 192, 240, 1

### Skin ###

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
    
### COC ###

def load_coc_train_data(imgs_path, masks_path, img_width = IMG_WIDTH, img_height = IMG_HEIGHT):
    
    IMG_SIZE = (img_width, img_height)
    
    # Load input image
    input_image = Image.open(imgs_path)
    input_image = transforms.Resize(IMG_SIZE)(input_image)
    input_image = transforms.Grayscale(num_output_channels = IMG_CHANNELS)(input_image)
    input_image = transforms.ToTensor()(input_image)

    # Load input mask
    input_mask = Image.open(masks_path)
    input_mask = transforms.Resize(IMG_SIZE)(input_mask)
    input_mask = transforms.ToTensor()(input_mask)

    return input_image, input_mask

class COCTrainDataset(Dataset):
    def __init__(self, imgs_path, masks_path, img_width, img_height, transform = None):
       
        self.imgs_folder = imgs_path
        self.msks_folder = masks_path
        self.transform = transform

    def __len__(self):
        
        length = len(glob.glob(os.path.join(self.imgs_folder, '*.tif')))

        return length
    
    def __getitem__(self, idx):

        all_images = glob.glob(os.path.join(self.imgs_folder, '*.tif'))
        all_images.sort()

        all_labels = glob.glob(os.path.join(self.msks_folder, '*.tif'))
        all_labels.sort()

        image_path = all_images[idx]
        mask_path = all_labels[idx]
        image, label = load_coc_train_data(image_path, mask_path)

        image = np.array(image, dtype = 'float32')
        label = np.array(label, dtype = 'float32')
        
        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()

        return image, label
    
### COC with 3 annotators ###

def load_coc_3_train_data(imgs_path, 
                          masks_AR_path, 
                          masks_HS_path,
                          masks_SG_path,
                          masks_avrg_path,
                          img_width = IMG_WIDTH, img_height = IMG_HEIGHT):
    
    IMG_SIZE = (img_width, img_height)
    
    # Load input image
    input_image = Image.open(imgs_path)
    input_image = transforms.Resize(IMG_SIZE)(input_image)
    input_image = transforms.Grayscale(num_output_channels = IMG_CHANNELS)(input_image)
    input_image = transforms.ToTensor()(input_image)

    # Load input mask AR
    input_mask_AR = Image.open(masks_AR_path)
    input_mask_AR = transforms.Resize(IMG_SIZE)(input_mask_AR)
    input_mask_AR = transforms.ToTensor()(input_mask_AR)

    # Load input mask HS
    input_mask_HS = Image.open(masks_HS_path)
    input_mask_HS = transforms.Resize(IMG_SIZE)(input_mask_HS)
    input_mask_HS = transforms.ToTensor()(input_mask_HS)

    # Load input mask SG
    input_mask_SG = Image.open(masks_SG_path)
    input_mask_SG = transforms.Resize(IMG_SIZE)(input_mask_SG)
    input_mask_SG = transforms.ToTensor()(input_mask_SG)

    # Load input mask avrg
    input_mask_avrg = Image.open(masks_avrg_path)
    input_mask_avrg = transforms.Resize(IMG_SIZE)(input_mask_avrg)
    input_mask_avrg = transforms.ToTensor()(input_mask_avrg)

    return input_image, input_mask_AR, input_mask_HS, input_mask_SG, input_mask_avrg

class COC3TrainDataset(Dataset):
    def __init__(self, dataset_location, img_width, img_height, transform = None, seed = 42):

        self.transform = transform

        self.image_folder = str(dataset_location) + '/images'

        self.label_AR_folder = str(dataset_location) + '/AR'
        self.label_HS_folder = str(dataset_location) + '/HS'
        self.label_SG_folder = str(dataset_location) + '/SG'

        self.label_avrg_folder = str(dataset_location) + '/avrg'
        
        print("Gent data loaded from utils.py as .tif format ...")

        # create a list of shuffled indices
        if seed is not None:
            np.random.seed(seed)
        self.indices = np.random.permutation(len(glob.glob(os.path.join(self.image_folder, '*.tif'))))

    def __len__(self):
        
        length = len(glob.glob(os.path.join(self.image_folder, '*.tif')))

        return length
    
    def __getitem__(self, idx):

            all_labels_AR = glob.glob(os.path.join(self.label_AR_folder, '*.tif'))
            all_labels_AR.sort()

            all_labels_HS = glob.glob(os.path.join(self.label_HS_folder, '*.tif'))
            all_labels_HS.sort()

            all_labels_SG = glob.glob(os.path.join(self.label_SG_folder, '*.tif'))
            all_labels_SG.sort()

            all_labels_avrg = glob.glob(os.path.join(self.label_avrg_folder, '*.tif'))
            all_labels_avrg.sort()

            all_images = glob.glob(os.path.join(self.image_folder, '*.tif'))
            all_images.sort()

            rand_idx = self.indices[idx]

            image_path = all_images[rand_idx]
            mask_AR_path = all_labels_AR[rand_idx]
            mask_HS_path = all_labels_HS[rand_idx]
            mask_SG_path = all_labels_SG[rand_idx]
            mask_avrg_path = all_labels_avrg[rand_idx]

            labelname = all_images[rand_idx]
            path_label, labelname = os.path.split(labelname)
            labelname, labelext = os.path.splitext(labelname)
            
            image, label_AR, label_HS, label_SG, label_avrg = load_coc_3_train_data(image_path, 
                                                                                    mask_AR_path,
                                                                                    mask_HS_path,
                                                                                    mask_SG_path,
                                                                                    mask_avrg_path)

            image = np.array(image, dtype = 'float32')
            label_AR = np.array(label_AR, dtype = 'float32')
            label_HS = np.array(label_HS, dtype = 'float32')
            label_SG = np.array(label_SG, dtype = 'float32')
            label_avrg = np.array(label_avrg, dtype = 'float32')
            
            image = torch.from_numpy(image).float()
            label_AR = torch.from_numpy(label_AR).float()
            label_HS = torch.from_numpy(label_HS).float()
            label_SG = torch.from_numpy(label_SG).float()
            label_avrg = torch.from_numpy(label_avrg).float()

            return labelname, image, label_AR, label_HS, label_SG, label_avrg