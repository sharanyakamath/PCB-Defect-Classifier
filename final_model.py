from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import keras_metrics as km
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import sys
import keras
from keras.models import load_model
import pandas as pd

np.set_printoptions(threshold=sys.maxsize)


x_train0 = np.load("xtrain0.npy")
x_train1 = np.load("xtrain1.npy")
# y_train = np.load("ytrain.npy")


# WRITE NUMBER OF TRAIN IMAGES 
x_train0 = x_train0.reshape(2520, 300,300, 3).astype('float16') / 255
x_train1 = x_train1.reshape(2520, 300,300, 3).astype('float16') / 255

# y_train = y_train.astype('float32')


# WRITE separation here
x_test0 = x_train0[:630]
y_test0 = np.zeros(630)

x_val0 = x_train0[630:1008]
y_val0 = np.zeros(378)

x_train0 = x_train0[1008:2520]
y_train0 = np.zeros(1512)

##########################
x_test1 = x_train1[:630]
y_test1 = np.ones(630)

x_val1 = x_train1[630:1008]
y_val1 = np.ones(378)

x_train1 = x_train1[1008:2520]
y_train1 = np.ones(1512)

print(x_val0.shape)
print(x_test0.shape)
print(x_train0.shape)

print(x_val1.shape)
print(x_test1.shape)
print(x_train1.shape)

x_val = np.append(x_val0, x_val1, axis=0)
x_test = np.append(x_test0, x_test1, axis=0)
x_train = np.append(x_train0, x_train1, axis=0)

y_val = np.append(y_val0, y_val1, axis=0)
y_test = np.append(y_test0, y_test1, axis=0)
y_train = np.append(y_train0, y_train1, axis=0)

y_train = y_train.astype('float16') 

print(x_val.shape)
print(x_test.shape)
print(x_train.shape)

print(y_val.shape)
print(y_test.shape)
print(y_train.shape)

if K.image_data_format() == 'channels_first':
    input_shape = (3, 300, 300)
else:
    input_shape = (300, 300, 3)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer=keras.optimizers.RMSprop(),
              metrics=['accuracy',km.binary_precision(), km.binary_recall()])


model.fit(
    x_train, y_train,
    epochs=30,
    batch_size=64,
    validation_data=(x_val,y_val))

model.save_weights('cnn3.h5')



model.load_weights('cnn3.h5')

print('\n# Evaluate on test data')
results = model.evaluate(x_test, y_test, batch_size=128)
print('test loss, test acc:', results)

print('\n# Generate predictions for test samples')
predictions = model.predict(x_test)
# print(predictions)

y_pred = np.array([])
for x in predictions:
    for a in x:
        if(a>0.5):
            y_pred = np.append(y_pred,1)
        else:
            y_pred = np.append(y_pred,0)

print(y_pred)

print(y_test.shape)
print(y_pred.shape)

print('Confusion Matrix')
print(confusion_matrix(y_test, y_pred))
print('Classification Report')
target_names = ['Defect', 'NoDefect']
print(classification_report(y_test, y_pred, target_names=target_names))

