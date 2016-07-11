'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
import os
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

import subprocess

batch_size = 12#8
nb_classes = 2
nb_epoch = 100

# input image dimensions
img_rows, img_cols = 20, 20
# number of convolutional filters to use
nb_filters = 13#2
# size of pooling area for max pooling
nb_pool = 1
# convolution kernel size
nb_conv = 2

# the data, shuffled and split between train and test sets
data_file = 'july_10_2016-data/data_set.npz'
if os.path.isfile(data_file):
    data = np.load(data_file)
else:
    subprocess.call('python 2d_seg.py', shell=True)
    data = np.load(data_file)

X_train = data['X_train']
X_test = data['X_test'] 
y_train = data['y_train']
y_test = data['y_test']

print(X_train)
print(X_test)
print(y_train)
print(y_test)

X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()

model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                        border_mode='valid',
                        input_shape=(1, img_rows, img_cols)))
#model.add(Activation('relu'))
#model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
#model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.25))

model.add(Flatten())
#model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])
#checkpointer = ModelCheckpoint(filepath="weights.{epoch:02d}-{val_loss:.2f}.hdf5", monitor='val_acc', verbose=1, save_best_only=True)
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(X_test, Y_test))#, callbacks=[checkpointer])
score = model.evaluate(X_test, Y_test)
print(model.metrics_names)
print('Test score:', score[0])
print('Test accuracy:', score[1])
