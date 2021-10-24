# -*- coding: utf-8 -*-
"""
Created on Sun Aug 29 9:48:39 2021

@author: Nishan Kapoor
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.optimizers import Adam
from keras.layers import normalization,BatchNormalization
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar10
from keras import backend as K
from keras import regularizers, optimizers

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train.shape
plt.imshow(x_train[2])
plt.title(y_train[2])


x_train = x_train/255
x_test = x_test/255

x_train.shape

num_classes = 10

y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)

y_train[0]

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(32,32,3))) 
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(128))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer=Adam(1e-4), metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=64, epochs=4, validation_data=(x_test, y_test))

model.evaluate(x_test, y_test, batch_size=64)

img_to_visualize = x_train[0]
plt.imshow(img_to_visualize)

img_to_visualize = np.expand_dims(img_to_visualize, axis=0)


#################       Completed        #######################


