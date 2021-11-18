## Use different subsets of the new basis vectors to reconstruct a random patch
from sklearn.decomposition import sparse_encode
from sklearn.decomposition import PCA, FastICA, DictionaryLearning
import numpy as np

def reconstruction(n_basis, method, obj, new_patch, basis):

    # TODO reconstruct the random patch
    reconstructions = []
    for i in range(n_basis):
        sub_basis = basis[0:i+1]

        if method == 'sc':
            pass
        if method == 'pca':
            transformation = obj.fit_transform(new_patch)[:, :i + 1]
            components = obj.components_[:i + 1, :]

            reconstructions.append(np.dot(transformation, components))
        else:
            pass

    return np.stack(reconstructions)

    # TODO show the reconstructed patch
