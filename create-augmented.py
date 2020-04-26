from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

I = np.load('xtrain.npy')
L = np.load('ytrain.npy')
label_count = 0 


for x in I:
    i=0
    x = x.reshape((1,) + x.shape)

    if(L[label_count] == 0):
        for batch in datagen.flow(x, batch_size=1,
                                save_to_dir='preview/defect', save_prefix='defect', save_format='jpeg'):
            i += 1
            if i > 20:
                break
    
    else:
        for batch in datagen.flow(x, batch_size=1,
                                save_to_dir='preview/nodefect', save_prefix='nodefect', save_format='jpeg'):
            i += 1
            if i > 20:
                break
    label_count+=1




