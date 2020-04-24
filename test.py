import numpy as np
from matplotlib import pyplot as plt


images = np.load("xtrain.npy")
labels = np.load("ytrain.npy")
print(images.shape)
print(labels.shape)
print(labels)


    