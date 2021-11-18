from computing import *

dataset = "mnist.npz"
# dataset = 'data/olivetti.npz'

n_patches = 10  # number of random patches to extract from the images
patch_size = 28  # size (both width and height) of the square-shaped patches
n_basis = 10  # number of new basis vectors
method = 'pca'  # string describing whether to perform pca, ica, or sc
n_iterations = 50  # number of iterations (only for sparse coding)

n_pixels, patches = prepare(dataset, n_patches, patch_size)
basis = applyMethod(method, patches, n_basis, n_pixels)
