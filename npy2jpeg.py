from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
datagen = ImageDataGenerator(
        )

I = np.load('xtrain.npy')
L = np.load('ytrain.npy')
label_count = 0 
for x in I:
    x = x.reshape((1,) + x.shape)

    if(L[label_count] == 0):
        for batch in datagen.flow(x, batch_size=1,
                                save_to_dir='train/nodefect', save_prefix='defect', save_format='jpeg'):
            break
    
    else:
        for batch in datagen.flow(x, batch_size=1,
                                save_to_dir='train/defect', save_prefix='nodefect', save_format='jpeg'):
            break
    label_count+=1




