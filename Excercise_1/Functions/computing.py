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

    # Adding a dimension to the matrix such that you can construct an image
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

def applyMethod(method, patches, n_basis, n_pixels, n_iterations):
    if method == 'pca':
        pca = PCA(n_components=min(n_basis, n_pixels))
        pca.fit_transform(patches)
        basis = pca.components_

        print(basis)
        return basis, pca

    elif method == 'ica':
        ica = FastICA(n_components=min(n_basis, n_pixels))
        ica.fit_transform(patches)
        basis = ica.components_

        return basis, ica

    elif method == 'sc':
        dic_learning = DictionaryLearning(n_components=min(n_basis, n_pixels),
                                          verbose=True,
                                          max_iter=n_iterations)
        dic_learning.fit_transform(patches)
        basis = dic_learning.components_

        return basis, dic_learning

# TODO show the new set of basis vectors (only the first n_basis)
# show_in_grid(basis[:n_basis], patch_size, patch_size)
