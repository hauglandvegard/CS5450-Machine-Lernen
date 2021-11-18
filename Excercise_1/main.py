from computing import *
from plotting import *
from reconstruction import *

#dataset = "mnist.npz"
dataset = 'olivetti.npz'

n_patches = 20  # number of random patches to extract from the images
patch_size = 28  # size (both width and height) of the square-shaped patches
n_basis = 20  # number of new basis vectors
method = 'pca'  # string describing whether to perform pca, ica, or sc
n_iterations = 1000  # number of iterations (only for sparse coding)

n_pixels, patches = prepare(dataset, n_patches, patch_size)
basis, obj = applyMethod(method, patches, n_basis, n_pixels, n_iterations)

_, new_patches = prepare(dataset, n_patches, patch_size)
re_construct = reconstruction(n_basis, method, obj, new_patches, basis)

ifGray = True
amount = 10

#test_plot(images,         "images",    False, ifGray, amount)
#test_plot(mean_free_data, "Mean_free", False, ifGray, amount)
#test_plot(patches,        "Patches",   True,  ifGray, amount)
#test_plot(basis,          "Basis",     True,  ifGray, amount)

#test_plot2(patches, basis, "Patches and Basis", True, ifGray, amount)

show_in_grid(new_patches,patch_size,patch_size)
#show_in_grid(basis,patch_size,patch_size)
show_in_grid(re_construct, patch_size, patch_size)