import os
from PIL import Image
import numpy as np

def generate_grids(directory, output_file, grid_size=(7, 7), image_size=(224, 224)):
    """
    Generates a grid of numerical values for images in a specified directory.

    :param directory: Directory containing images.
    :param output_file: Path to save the numpy array with grids.
    :param grid_size: The dimensions of the grid (rows, columns).
    :param image_size: The size to which images are resized.
    """
    # Placeholder for the grids
    grids = []

    # Iterate through each image in the directory
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Open the image
            with Image.open(os.path.join(directory, filename)) as img:
                # Convert image to grayscale
                img = img.convert('L')
                # Resize image if it's not the expected size
                if img.size != image_size:
                    img = img.resize(image_size, Image.ANTIALIAS)
                
                # Convert image to numpy array
                img_array = np.array(img)
                
                # Compute the grid
                grid = []
                step_x = image_size[0] // grid_size[0]
                step_y = image_size[1] // grid_size[1]
                for i in range(0, image_size[0], step_x):
                    for j in range(0, image_size[1], step_y):
                        # Calculate the mean of the section
                        section = img_array[i:i+step_x, j:j+step_y]
                        mean_value = section.mean()
                        grid.append(mean_value)
                # Append the grid to the list of grids
                grids.append(grid)
                print('added to list')

    # Convert list of grids to a numpy array and save
    grids_array = np.array(grids)
    np.save(output_file, grids_array)

# Usage
directory = './dataset'  # Change this to your images directory
output_file = 'grids.npy'  # Path where the grids will be saved
generate_grids(directory, output_file)
