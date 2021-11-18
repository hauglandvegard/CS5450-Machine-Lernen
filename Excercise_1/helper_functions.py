import numpy as np
import math
import random
from matplotlib import pyplot as plt

def random_patches(data, img_height, img_width, n_patches, patch_size):
    # RANDOMPATCHES draw random square-shaped patches from the data.
    #
    # INPUT:
    #   data : ndarray containing the data (i.e. mnist or olivetti)
    #   img_height : height of the images
    #   img_width : width of the images
    #   n_patches : number of patches to be drawn
    #   patch_size : size (both width and height) of the square-shaped patches
    #
    # OUTPUT:
    #   patches : random patches

    # reshape data 2d -> 3d
    n_images = data.shape[0]
    data = data.reshape(n_images, img_height, img_width)

    # extract random patches
    patches = []  # np.zeros((n_patches, patch_size*patch_size))
    for i in range(n_patches):
        patch_top = random.randint(0, img_height - patch_size)
        patch_left = random.randint(0, img_width - patch_size)
        p = data[random.randint(0, n_images - 1), patch_top:patch_top + patch_size, patch_left:patch_left + patch_size]
        patches.append(p)
        # patches[i,:] = p[:]

    patches = np.stack(patches)
    patches = patches.reshape(patches.shape[0], -1)  # flatten each patch

    return patches


def show_in_grid(images, height, width):
    # flatten images if necessary
    images = images.reshape(images.shape[0], -1)

    # normalize patches
    images = images / (np.abs(images).max(axis=1, keepdims=True) + 1e-6)

    # reshape into images
    images = images.reshape(-1, height, width)

    # make images fit into a rectangular area
    grid_width = math.ceil(math.sqrt(images.shape[0]))
    grid_height = math.ceil(images.shape[0] / grid_width)

    empty_cells = grid_width * grid_height - images.shape[0]

    # fill empty cells
    if empty_cells > 0:
        padding = np.zeros((empty_cells, height, width))
        images = np.concatenate((images, padding))

    # rearrange basis into grid and also switch width and height so x and y axis are not switched
    images = images.reshape(grid_height, grid_width, height, width)
    images = images.transpose(0, 3, 1, 2)

    plt.figure(figsize=(10, 10))
    plt.imshow(images.reshape(grid_height * height, grid_width * width), cmap='gray')
    plt.show()