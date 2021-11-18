import numpy as np
from matplotlib import pyplot as plt

def test_plot(array, label, raw, grayscale=False, amount=1):
    if (raw):
        square = int(np.sqrt(array.shape[1]))
        array = np.reshape(array, (array.shape[0], square, square))

    for i in range(amount):
        # Subplot 1
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.title(label)
        if (grayscale):
            plt.imshow(array[i * 2, :, :].T, cmap=plt.cm.gray)
        else:
            plt.imshow(array[i * 2, :, :].T)

        # Subplot 2
        plt.subplot(1, 2, 2)
        if (grayscale):
            plt.imshow(array[i * 2 + 1, :, :].T, cmap=plt.cm.gray)
        else:
            plt.imshow(array[i * 2 + 1, :, :].T)


def test_plot2(array1, array2, label, raw, grayscale=False, amount=1):
    if (raw):
        square1 = int(np.sqrt(array1.shape[1]))
        array1 = np.reshape(array1, (array1.shape[0], square1, square1))

        square2 = int(np.sqrt(array2.shape[1]))
        array2 = np.reshape(array2, (array2.shape[0], square2, square2))

    for i in range(amount):
        # Subplot 1
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.title(label)
        if (grayscale):
            plt.imshow(array1[i, :, :].T, cmap=plt.cm.gray)
        else:
            plt.imshow(array1[i, :, :].T)

        # Subplot 2
        plt.subplot(1, 2, 2)
        if (grayscale):
            plt.imshow(array2[i, :, :].T, cmap=plt.cm.gray)
        else:
            plt.imshow(array2[i, :, :].T)