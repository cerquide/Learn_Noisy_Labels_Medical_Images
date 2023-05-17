import cv2
import numpy as np
import skimage.morphology
import skimage.filters.rank
import skimage.util
import matplotlib.pyplot as plt
from skimage.transform import rescale, resize
import tensorflow
from tensorflow.keras.preprocessing.image import smart_resize

import os
import json
import multiprocessing

import stat

import shutil
from multiprocessing import freeze_support

path_image = 'Documents/IIIA/git/Learn_Noisy_Labels_Medical_Images/images/image_batch'
path_mask = 'Documents/IIIA/git//Learn_Noisy_Labels_Medical_Images/images/mask_batch'
path_out = 'Documents/IIIA/git//Learn_Noisy_Labels_Medical_Images/images/out_batch'

def compute_roi(params):
    # Unpack the image and mask paths from the params tuple
    image_path, mask_path = params

    try:
        # Read the input image as grayscale
        input_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Check if the image failed to load
        if input_image is None:
            raise ValueError("Failed to load image.")

        # Check if the mask file does not exist
        if not os.path.exists(mask_path):
            print("Mask file not found for", image_path)
            return None, None, None, None

        # Read the input mask as grayscale
        input_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Check if the mask failed to load
        if input_mask is None:
            raise ValueError("Failed to load mask.")

        # Compute the ROI using the extract_roi function
        roi, (left, top), (right, bottom) = extract_roi(input_image / 255., min_fratio = 0.3, max_sratio = 1.5, 
                                                        filled = True, border = .01)

        # Print the found ROI coordinates
        print("ROI for", image_path, "found:", (left, top), (right, bottom))

        # Return the ROI coordinates
        return left, top, right, bottom

    except Exception as e:
        # Print the error message if an exception occurred
        print("Error processing", image_path + ":", str(e))

        # Return None for the ROI coordinates in case of an error
        return None, None, None, None

def local_entropy(im, kernel_size = 5, normalize = True):
    """
    Calculates the local entropy of the input image.

    Parameters:
    - im (ndarray): The input image as a NumPy array.
    - kernel_size (int): The size of the kernel used for entropy calculation. Default is 5.
    - normalize (bool): Flag indicating whether to normalize the entropy image. Default is True.

    Returns:
    - entr_img (ndarray): The local entropy image as a NumPy array.
    """

    # Create a disk-shaped structuring element
    kernel = skimage.morphology.disk(kernel_size)

    # Calculate the entropy of the image using the rank filter
    entr_img = skimage.filters.rank.entropy(skimage.util.img_as_ubyte(im), kernel)

    if normalize:
        # Normalize the entropy image to the range [0, 255]
        max_img = np.max(entr_img)
        entr_img = (entr_img * 255 / max_img).astype(np.uint8)

    return entr_img


def calc_dim(contour):
    """
    Calculates the dimensions (left, right, top, bottom) of a contour.

    Parameters:
    - contour (ndarray): The contour as a NumPy array.

    Returns:
    - dim (tuple): The dimensions (left, right, top, bottom) of the contour.
    """

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
    """
    Calculates the size of a contour given its dimensions.

    Parameters:
    - dim (tuple): The dimensions (left, right, top, bottom) of the contour.

    Returns:
    - size (float): The size of the contour.
    """

    return (dim[1] - dim[0]) * (dim[3] - dim[2])

def extract_roi(img, threshold = 140, kernel_size = 5, min_fratio = .3, max_sratio = 5, filled = True, border = .01): #135 before
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

    # Find the contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    contours_d = []
    # Calculate the dimensions of each contour
    for c in contours:
        if len(c) > 1:
            contours_d.append(calc_dim(c))
        elif len(c) == 1:
            point = c[0][0]
            contours_d.append((point[0], point[0], point[1], point[1]))

    # Calculate the sizes of the contours
    contours_sizes = [calc_size(c) for c in contours_d]

    # Sort the contour indices based on contour sizes in descending order
    contour_indices = np.argsort(contours_sizes)[::-1]

    # Remove artifacts from the contours
    fratio = min_fratio
    sratio = max_sratio
    idx = -1
    while fratio <= min_fratio or sratio >= max_sratio:
        idx += 1
        biggest = contour_indices[idx]
        filled_mask = np.zeros(img.shape, dtype=np.uint8)
        filled_mask = cv2.fillPoly(filled_mask, [contours[biggest]], 255)
        fratio = filled_mask.sum() / 255 / contours_sizes[biggest]
        cdim = contours_d[biggest]
        sratio = (cdim[3] - cdim[2]) / (cdim[1] - cdim[0])
        if sratio < 1:
            sratio = 1 / sratio

    # Generate the final filled mask
    filled_mask = np.zeros(img.shape, dtype=np.uint8)

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

def gen_images(image_directory, mask_directory, output_directory,output_size):
    
    roi_info = {}  # Create an empty dictionary to store ROI information
    params = []  # Create an empty list to store image and mask paths
    pool = multiprocessing.Pool(4)  # Create a multiprocessing Pool with 16 processes

    # Iterate through the image files in the directory
    for filename in os.listdir(image_directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(image_directory, filename)  # Get the full path of the image file
            image_path = image_path.replace("\\" , "/")
            mask_path = os.path.join(mask_directory, filename)  # Get the full path of the corresponding mask file
            mask_path = mask_path.replace("\\" , "/")
            if os.path.isfile(mask_path):  # Check if the mask file exists
                params.append((image_path, mask_path))  # Append a tuple of image and mask paths to the params list

    # Use multiprocessing to compute the ROIs for each image in parallel
    results = pool.map(compute_roi, params)  # Pass the params as a list of tuples
    print("Finished computing ROIs")
    # Iterate through the image and mask paths along with their corresponding ROI results
    for i, (image_path, mask_path) in enumerate(params):
        
        left, top, right, bottom = results[i]  # Get the ROI coordinates from the results
        image_filename = os.path.basename(image_path)  # Get the filename of the image
        mask_filename = os.path.basename(mask_path)  # Get the filename of the mask

        # Prepare the output filenames for the cropped images and masks
        output_image_filename = os.path.join(output_directory, "roi_images", image_filename)
        output_mask_filename = os.path.join(output_directory, "roi_masks", mask_filename)
        

        # Create the output directories for the cropped images and masks
        output_image_dir = os.path.join(output_directory, "roi_images")
        os.makedirs(output_image_dir, exist_ok=True)  # Create the directory if it doesn't exist
        os.chmod(output_image_dir, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)  # Set permissions to rwxrwxrwx

        output_mask_dir = os.path.join(output_directory, "roi_masks")
        os.makedirs(output_mask_dir, exist_ok=True)  # Create the directory if it doesn't exist
        os.chmod(output_mask_dir, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)  # Set permissions to rwxrwxrwx

        # Read the input image
        input_image = cv2.imread(image_path)

        # Apply ROI transformations to image to extract the region of interest
        output_image = input_image[top:bottom, left:right]
        #smart resizing using tensorflow preprocessing (padding + bilinear interpolation)
        output_image = smart_resize(np.array([output_image]),output_size)[0]
        # Save the cropped image
        cv2.imwrite(output_image_filename, output_image)

        # Set the permissions of the saved image file to match the source image file
        shutil.copymode(image_path, output_image_filename)

        # Read the input mask
        input_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Apply ROI transformations to mask to extract the region of interest
        output_mask = input_mask[top:bottom, left:right]
        #smart resizing using tensorflow preprocessing (padding + bilinear interpolation)
        output_mask = smart_resize(np.array([output_mask]),output_size)[0] 
        

        # Save the cropped mask
        cv2.imwrite(output_mask_filename, output_mask)

        # Set the permissions of the saved mask file to match the source mask file
        shutil.copymode(mask_path, output_mask_filename)

        # Store ROI info in a dictionary
        row = {
            "image_filename": image_filename,
            "mask_filename": mask_filename,
            "top": int(top),
            "bottom": int(bottom),
            "left": int(left),
            "right": int(right)
        }
        roi_info[image_filename] = row
    
    # Save ROI info as JSON
    json_dir = os.path.join(output_directory, "json")
    os.makedirs(json_dir, exist_ok=True)
    json_filename = os.path.join(json_dir, "roi_infoss.json")
    with open(json_filename, "w") as f:
        json.dump(roi_info, f)

    print("Finished saving ROIs and ROI info")

if __name__ == '__main__':
        freeze_support()
        gen_images(image_directory = path_image, mask_directory = path_mask, output_directory = path_out,output_size=(40,40))