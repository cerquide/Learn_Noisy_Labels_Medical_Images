
import cv2
import numpy as np
import skimage.morphology
import skimage.filters.rank
import skimage.util
import matplotlib.pyplot as plt
from skimage.transform import rescale, resize
from tensorflow.keras.preprocessing.image import smart_resize
from matplotlib import patches
import os
import json
import multiprocessing
import copy
import stat
import shutil
from multiprocessing import freeze_support
import tensorflow as tf
import torchvision.transforms as transforms
import random
import torch
import torch.nn as nn

def compute_roi(image_path):
      input_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

      if input_image is None:
          raise ValueError("Failed to load image.")

      roi, (left, top), (right, bottom) = extract_roi(input_image / 255., min_fratio=0.5, max_sratio=2, kernel_size=7,
                                                       filled=True, border=.01)

      print("ROI for", image_path, "found:", (left, top), (right, bottom))
      return (left, top, right, bottom)

def local_entropy(im, kernel_size=5, normalize=True):
    kernel = skimage.morphology.disk(kernel_size)
    entr_img = skimage.filters.rank.entropy(skimage.util.img_as_ubyte(im), kernel)

    if normalize:
        max_img = np.max(entr_img)
        entr_img = (entr_img * 255 / max_img).astype(np.uint8)

    return entr_img

def calc_dim(contour):
    if len(contour) > 1:
        c_0 = [point[0][0] for point in contour]
        c_1 = [point[0][1] for point in contour]
        return min(c_0), max(c_0), min(c_1), max(c_1)
    
    elif len(contour) == 1:
        point = contour[0][0]
        return point[0], point[0], point[1], point[1]
    
    else:
        return None

def calc_size(dim):
    return (dim[1] - dim[0]) * (dim[3] - dim[2])

def extract_roi(img, threshold=170, kernel_size=5, min_fratio=0.3, max_sratio=1.5, filled=True, border=0.01):
    """
    Extracts the region of interest (ROI) from the input image based on entropy and contour analysis.

    Parameters:
    - img (ndarray): The input image as a NumPy array.
    - threshold (int): The threshold value for binarizing the entropy image. Default is 135.
    - kernel_size (int): The size of the kernel used for local entropy calculation. Default is 5.
    - min_fratio (float): The minimum filled ratio to remove artifacts. Default is 0.3.
    - max_sratio (float): The maximum size ratio to remove artifacts. Default is 5.
    - filled (bool): Flag indicating whether the ROI mask should be filled or outlined. Default is True.
    - border (float): The border fraction to extend the ROI rectangle. Default is 0.01.

    Returns:
    - filled_mask (ndarray): The filled mask representing the ROI as a binary image.
    - origin (tuple): The (x, y) coordinates of the top-left corner of the ROI rectangle.
    - to (tuple): The (x, y) coordinates of the bottom-right corner of the ROI rectangle.
    """

    # Compute the local entropy of the image
    entr_img = local_entropy(img, kernel_size=kernel_size)

    # Threshold the entropy image to create a binary mask
    _, mask = cv2.threshold(entr_img, threshold, 255, cv2.THRESH_BINARY)

    kernel = np.ones((50, 50)) * 255
    dilated_mask = cv2.dilate(mask, kernel, iterations=1)

    interpolated_mask = np.where(dilated_mask > 0, 1, 0) * 255
    mask = np.array(interpolated_mask, np.uint8)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    im2 = copy.copy(img)
    cv2.drawContours(im2, contours, -1, (255, 0, 0), 2)

    contours_d = []
    for c in contours:
        if len(c) > 1:
            contours_d.append(calc_dim(c))
        elif len(c) == 1:
            point = c[0][0]
            contours_d.append((point[0], point[0], point[1], point[1]))

    contours_sizes = [calc_size(c) for c in contours_d]
    sorted_contours_sizes = sorted(contours_sizes)
    mean_contours = np.mean(sorted_contours_sizes)
    std_contours = np.std(sorted_contours_sizes)

    contour_ratios = []
    surface_list = []
    for i, contour in enumerate(contours):
        filled_mask = np.zeros(img.shape, dtype=np.uint8)
        filled_mask = cv2.fillPoly(filled_mask, [contour], 255)
        filled_area = filled_mask.sum() / 255
        surface_list.append(filled_area)
        contour_length = contours_sizes[i]
        ratio = filled_area / contour_length
        contour_ratios.append(ratio)

    meilleur_score = 0
    selected_contour_index = 0
    for i in range(len(contour_ratios)):
        ratio = contour_ratios[i]
        surface = surface_list[i]
        score = surface * ratio
        if score > meilleur_score:
            selected_contour_index = i
            meilleur_score = score

    selected_contour = contours[selected_contour_index]

    filled_mask = np.zeros(img.shape, dtype=np.uint8)
    cdim = contours_d[selected_contour_index]
    extra = (int(img.shape[0] * border), int(img.shape[1] * border))
    origin = (max(0, cdim[0] - extra[1]), max(0, cdim[2] - extra[0]))
    to = (min(img.shape[1] - 1, cdim[1] + extra[1]), min(img.shape[0] - 1, cdim[3] + extra[0]))

    if filled:
        # Fill the ROI rectangle in the mask
        filled_mask = cv2.rectangle(filled_mask, origin, to, 255, -1)
    else:
        # Draw the ROI rectangle on the mask
        filled_mask = cv2.rectangle(filled_mask, origin, to, 255, 2)

    # Return the filled mask, origin, and to coordinates of the ROI
    return filled_mask, origin, to

def process_single_image(image_path, output_size=(192,192)):
    # Read the input image
    path=image_path
    input_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # Apply ROI transformations to image to extract the region of interest
    left, top, right, bottom = compute_roi(path) 
    output_image = input_image[top:bottom, left:right]
    output_image = np.expand_dims(output_image, axis=-1) 
    output_image = tf.image.resize(output_image, [output_size[0],output_size[1]], method=tf.image.ResizeMethod.BILINEAR)
    output_image = output_image.numpy()
    return output_image

import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, img_channels):
        super(UNet, self).__init__()

        def conv_block(in_channels, out_channels, dropout = 0.0):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding = 1),
                nn.ReLU(inplace = True),
                nn.Dropout(dropout),
                nn.Conv2d(out_channels, out_channels, 3, padding = 1),
                nn.ReLU(inplace = True)
            )

        def upconv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, 2, stride = 2)
            )

        self.enc1 = conv_block(img_channels, 16, dropout = 0.1)
        self.enc2 = conv_block(16, 32, dropout = 0.1)
        self.enc3 = conv_block(32, 64, dropout = 0.2)
        self.enc4 = conv_block(64, 128, dropout = 0.2)

        self.middle = conv_block(128, 256, dropout = 0.3)

        self.dec4 = conv_block(256, 128, dropout = 0.2)
        self.dec3 = conv_block(128 , 64, dropout = 0.2)
        self.dec2 = conv_block(64, 32, dropout = 0.1)
        self.dec1 = conv_block(32, 16, dropout = 0.1)

        self.upconv4 = upconv_block(256, 128)
        self.upconv3 = upconv_block(128, 64)
        self.upconv2 = upconv_block(64, 32)
        self.upconv1 = upconv_block(32, 16)

        self.output = nn.Conv2d(16, 1, kernel_size = 1)

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

        return self.output(dec1)

def initialize_model(img_width, img_height, img_channels):
    model = UNet(img_channels)
    return model

### Global CM model ###

def inference(image_path,unet_out,unet_in):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    #apply ROI transformation
    im_test=process_single_image(image_path, output_size=(192,192))
    
    #Applying normalization
    im_test = im_test / 255.0
    im_plot = copy.copy(im_test)
    im_test = transform(im_test)
    im_test = im_test.float().unsqueeze(0)

    # Make a prediction with the models
    with torch.no_grad():
        output_out = unet_out(im_test)

    with torch.no_grad():
        output_in = unet_in(im_test)

    output_out = nn.Sigmoid()(output_out)
    output_in = nn.Sigmoid()(output_in)

    # Apply thresholding
    threshold = 0.5
    output_out_thresholded = torch.where(output_out > threshold, torch.tensor(1), torch.tensor(0))
    output_in_thresholded = torch.where(output_in > threshold, torch.tensor(1), torch.tensor(0))

    output_out_array = output_out_thresholded.squeeze().numpy()
    output_in_array = output_in_thresholded.squeeze().numpy()

    # Display the predicted image
    fig, axs = plt.subplots(1, 3)
    axs[2].imshow(output_out_array)
    axs[1].imshow(output_in_array)
    axs[0].imshow(im_plot)
    axs[0].set_title("Original Image", fontsize=5)
    axs[1].set_title("Inside Segmentation", fontsize=5)
    axs[2].set_title("Outside Segmentation", fontsize=5)
    plt.show()
    
    nb = random.randint(0, 1000)
    fig.savefig(str(nb) + ".jpg")

    return output_in_array, output_out_array