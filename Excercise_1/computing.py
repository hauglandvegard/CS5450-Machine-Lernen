import numpy as np
from sklearn.decomposition import PCA, FastICA, DictionaryLearning
from matplotlib import pyplot as plt
from helper_functions import *

def prepare(dataset, n_patches, patch_size):
    raw_data = np.load(dataset)  # Loading dataset
    # Converting data to floating point (double precision)
    matrix = raw_data['data'].astype(float)

    # Geting number of images (observations) and number of pixels (features)
    n_images, n_pixels = matrix.shape

    # Taking the square root of the amount of pixels to find width and height because
    # we know that the image is squared

    img_height = int(np.sqrt(n_pixels))
    img_width = int(np.sqrt(n_pixels))

    assert isinstance(img_height, int) and isinstance(img_width, int), "Image dimensions need to be integers."

    # Adding a dimention to the matrix such that you can construct an image
    images = np.reshape(matrix, (n_images, img_height, img_width))

    # Enforcing mean-free data vectors
    mean_free_data = images - images.mean(axis=0)

    # Drawing random patches from data
    patches = random_patches(mean_free_data,
                             img_height,
                             img_width,
                             n_patches,
                             patch_size)

    return n_pixels, patches

def applyMethod(method, patches, n_basis, n_pixels):
    if method == 'pca':
        pca = PCA(n_components=min(n_basis, n_pixels))
        pca.fit(patches)
        basis = pca.components_

        per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
        labels = ['PC' + str(x) for x in range(1, len(per_var) + 1)]

        plt.bar(x=range(1, len(per_var) + 1), height=per_var, tick_label=labels)
        plt.ylabel('Precentage of Explained Varriance')
        plt.xlabel('Princial Component')
        plt.title('Scree Plot')
        plt.show()

        return patches

    elif method == 'ica':
        basis = -1
    elif method == 'sc':
        basis = -1

# TODO show the new set of basis vectors (only the first n_basis)
# show_in_grid(basis[:n_basis], patch_size, patch_size)
