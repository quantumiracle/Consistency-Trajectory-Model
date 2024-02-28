import numpy as np
from PIL import Image
import os

# Load the npz file
data = np.load('plots/eval/sample.npz')
images = data['arr_0']  # Assuming the images are stored in the first array

# Create a directory for the images if it doesn't already exist
output_dir = 'plots/eval/sample_images'
os.makedirs(output_dir, exist_ok=True)

# Loop through the images and save each one
for i, img in enumerate(images):
    # Convert the array to an image
    image = Image.fromarray(img)
    
    # Save the image to the specified directory
    image.save(os.path.join(output_dir, f'image_{i}.jpeg'))


# then measure fid: https://github.com/mseitzer/pytorch-fid
# python -m pytorch_fid plots/eval/sample_images/ author_ckpt/VIRTUAL_imagenet64_labeled.npz