# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 01:46:05 2020

@author: Hamza
"""
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation,Dropout ,Flatten, Conv2D, MaxPooling2D
import pickle


X=np.asarray(pickle.load(open("X.pickle", "rb")))
y=np.asarray(pickle.load(open("y.pickle", "rb")))

X=X/255.0#normalization

dense_layers=1
layer_sizes=128
conv_layers=3

model=Sequential()
model.add(Conv2D(layer_sizes, (3,3), input_shape=X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

for l in range(conv_layers-1):
    model.add(Conv2D(layer_sizes,(3,3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

for l in range(dense_layers):
    model.add(Dense(layer_sizes))
    model.add(Activation("relu"))
    model.add(Dropout(0.2))

model.add(Dense(1))
model.add(Activation("sigmoid"))


model.compile(loss="binary_crossentropy",optimizer ="adam",metrics=['accuracy'])

model.fit(X, y, batch_size=32, epochs=10, validation_split=0.1)

model.save('doggos_v_cattos.model')

'''

_____free info(muft ka gyaan)its always good :)_____________________
            
steps:conv->pooling->flattening->full connection
         
sequential:lets you create the model layer by layer.It takes in 
only one i/p(image) and gives out one o/p(dog or cat)

conv: uses a 3*3 matrix feature detector matrix to get a feature map. uses convolutional operation. Feature map
acts like a filter

maxpooling: a 2*2 window which selects the max val of the feature map

flatenning: takes in maxpooled matrix and converts in into a col matrix for inputs into the cnns

dense:A Dense layer feeds all outputs from the previous layer to all its neurons, 
each neuron providing one output to the next layer. 
A Dense(10) has ten neurons

dropout:Dropout works by randomly setting the outgoing edges of hidden units 
(neurons that make up hidden layers) to 0 at each update of the training phase

Activation:introduces non linearity.if we just have linear functions, summing em up would just give us another
linear function and acts as a single layer no matter how many layers we have in it 
            
            

'''
            