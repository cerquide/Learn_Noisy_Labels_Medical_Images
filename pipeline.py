

import cv2
import numpy as np
import skimage.morphology
import matplotlib.pyplot as plt
import tensorflow as tf
import torchvision.transforms as transforms
import random
import copy
import cv2
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from satelitte import compute_roi,local_entropy,calc_dim,calc_size,extract_roi,process_single_image,initialize_model,inference

unet_out = initialize_model(192, 192, 1)
unet_in = initialize_model(192, 192, 1)
weights_path_out = 'fol_Final_dict.pt'
weights_path_in = 'ooc_Final_dict.pt'

# Load the weights
unet_out.load_state_dict(torch.load(weights_path_out, map_location=torch.device('cpu')))
unet_in.load_state_dict(torch.load(weights_path_in, map_location=torch.device('cpu')))
unet_out.eval()
unet_in.eval()


# Call the inference function
path_image='images/images_batch_2/P1353_TFBS_D00_C1_3.tif'
inner_segmentation, outer_segmentation = inference(path_image,unet_out,unet_in)

# Save the segmentations to a file
cv2.imwrite("inner_segmentation.jpg", inner_segmentation * 255)
cv2.imwrite("outer_segmentation.jpg", outer_segmentation * 255)

# Calculate the ratio of the inner area to the outer area
inner_area = np.sum(inner_segmentation)
outer_area = np.sum(outer_segmentation)
ratio = inner_area / outer_area
print("Ratio of inner area to outer area:", ratio)