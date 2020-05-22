import os, sys
from IPython.display import display
from IPython.display import Image as _Imgdis
from PIL import Image
import numpy as np
from time import time
from time import sleep
from scipy import ndimage
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

folder = "/home/keets/Documents/PCB-Defect-Classifier/convert_data/defect"

onlyfiles = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]

train_files = []
y_train = []
i=0
for _file in onlyfiles:
    train_files.append(_file)
    # label_in_file = _file.find("_")
    # y_train.append(int(_file[0:label_in_file]))
    
print("Files in train_files: %d" % len(train_files))

# Original Dimensions
image_width = 300
image_height = 300
ratio = 1

image_width = int(image_width / ratio)
image_height = int(image_height / ratio)

channels = 3
nb_classes = 1

dataset = np.ndarray(shape=(len(train_files), channels, image_height, image_width),
                     dtype=np.float32)

i = 0
for _file in train_files:
    img = load_img(folder + "/" + _file)
    img.thumbnail((image_width, image_height))
    x = img_to_array(img)  
    x = x.reshape((3, 300, 300))
    x = (x - 128.0) / 128.0
    dataset[i] = x
    i += 1
    if i % 2660 == 0:
        print("%d images to array" % i)

folder = "/home/keets/Documents/PCB-Defect-Classifier/convert_data/nodefect"

onlyfiles = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]

train_files = []
y_train = []
i=0
for _file in onlyfiles:
    train_files.append(_file)
    # label_in_file = _file.find("_")
    # y_train.append(int(_file[0:label_in_file]))
    
print("Files in train_files: %d" % len(train_files))

# Original Dimensions
image_width = 300
image_height = 300
ratio = 1

image_width = int(image_width / ratio)
image_height = int(image_height / ratio)

channels = 3
nb_classes = 1

dataset1 = np.ndarray(shape=(len(train_files), channels, image_height, image_width),
                     dtype=np.float32)

i = 0
for _file in train_files:
    img = load_img(folder + "/" + _file)
    img.thumbnail((image_width, image_height))
    y = img_to_array(img)  
    y = y.reshape((3, 300, 300))
    y = (y - 128.0) / 128.0
    dataset1[i] = y
    i += 1
    if i % 2695 == 0:
        print("%d images to array" % i)
print("All images to array!")


data = np.concatenate((dataset,dataset1),axis=0)
np.save('xtrain.npy',data)