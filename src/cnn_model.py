# -*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation,Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D #Conv2D for grayscale and for RGB use Conv3D

def buildCNNModel(input_shape, num_classes, num_channels,kernel_size,dropout,pool_size,stride):#channel means number of features kitne chahye
    model = Sequential()
    model.add(Conv2D(num_channels,kernel_size,padding='valid',strides=stride,input_shape=input_shape))
    
    
    #convolutions adding
    convout1 = Activation('relu')#rectified linear unit
    model.add(convout1)
    model.add(Conv2D(num_channels,kernel_size))
    convout2 = Activation('relu')
    model.add(convout2)
    
#    convout3 = Activation('relu')
#    model.add(convout3)
#    model.add(Conv2D(num_channels,kernel_size))
#    convout4 = Activation('relu')
#    model.add(convout4)
#    model.add(Conv2D(num_channels,kernel_size))
#    convout5 = Activation('relu')
#    model.add(convout5)
#    model.add(Conv2D(num_channels,kernel_size))
#    convout6 = Activation('relu')
#    model.add(convout6)
#    model.add(Conv2D(num_channels,kernel_size))
#    convout7 = Activation('relu')
#    model.add(convout7)
#    model.add(Conv2D(num_channels,kernel_size))
    
    
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(dropout))
    model.add(Flatten())
    model.add(Dense(num_classes))
    model.add(Activation("softmax"))
    return model


    
