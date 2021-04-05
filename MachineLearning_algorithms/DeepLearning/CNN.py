# -*- coding: utf-8 -*-

#library import
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#---- CNN model build ----
import keras 
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

classifier =  Sequential()

#Convolution layer
classifier.add( Conv2D(filters = 32, kernel_size = 3, 
                       input_shape = (64,64,3), activation = 'relu' ) )
#Max pooling layer
classifier.add( MaxPooling2D(pool_size = 2, strides = 2) )


"""Adding another convolution and maxpooling layers to increase precision
"""
#Convolution layer
classifier.add( Conv2D(filters = 32, kernel_size = 3, 
                       activation = 'relu' ) )
#Max pooling layer
classifier.add( MaxPooling2D(pool_size = 2, strides = 2) )



#Flattening layer
classifier.add( Flatten() )

#Hidden layer
classifier.add( Dense(units = 128, kernel_initializer = 'uniform', activation = 'relu') )

#Output layer
classifier.add( Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid') )

#ANN compile
classifier.compile( optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#---- CNN model training ----
#Image import 
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale = 1./255)

training_dataset = train_datagen.flow_from_directory('dataset/training_set',
                                                    target_size = (64,64),
                                                    color_mode = 'rgb',
                                                    batch_size = 32,
                                                    class_mode = 'binary')
testing_dataset = test_datagen.flow_from_directory('dataset/test_set',
                                                        target_size = (64,64),
                                                        color_mode = 'rgb',
                                                        batch_size = 32,
                                                        class_mode = 'binary')
classifier.fit_generator(training_dataset,
                        steps_per_epoch = 8000,
                        epochs = 25,
                        validation_data = testing_dataset,
                        validation_steps = 2000 )