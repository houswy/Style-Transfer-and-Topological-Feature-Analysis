import os
import numpy as np
from skimage import io, color
from skimage.util import img_as_ubyte  # Ensure that img_as_ubyte is imported from skimage.util

# Folder path
folder_path = r'G:\GAN\data and pth\captcha_3000'

# Set the number of images to calculate
num_images = 3000  # You can modify this number to control how many images to process

# Store the mean values of all images
mean_values = []

# Iterate through the image files in the folder
count = 0
for filename in os.listdir(folder_path):
    if count >= num_images:
        break  # Stop when the specified number of images is reached

    file_path = os.path.join(folder_path, filename)

    # Ensure only image files are processed
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        # Read the image
        image = io.imread(file_path)

        # Convert to grayscale image
        gray_image = color.rgb2gray(image)
        gray_image = img_as_ubyte(gray_image)  # Convert to 8-bit unsigned integer

        # Calculate the mean intensity of the current image
        mean_intensity = np.mean(gray_image)
        mean_values.append(mean_intensity)

        # Increment the count
        count += 1

# If there are any images processed, calculate the overall mean
if mean_values:
    overall_mean = np.mean(mean_values)
    print(f"Overall mean for the selected {num_images} images: {overall_mean}")
else:
    print("Not enough images for calculation.")
