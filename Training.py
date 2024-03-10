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

X=X/255.0#normalization. The pixel values in the image data are normalized to the range [0, 1].

dense_layers=1
layer_sizes=128
conv_layers=3

model=Sequential() #This initializes a sequential model, which allows you to create a neural network by adding layers sequentially.
model.add(Conv2D(layer_sizes, (3,3), input_shape=X.shape[1:])) #Conv2D: This adds a 2D convolutional layer with layer_sizes filters, each of size 3x3.
model.add(Activation("relu")) #Activation("relu"): ReLU (Rectified Linear Activation) is applied to introduce non-linearity to the model.
model.add(MaxPooling2D(pool_size=(2,2))) #MaxPooling2D: This adds a 2D max pooling layer to downsample the feature maps by taking the maximum value 
                                         #within each 2x2 window.

'''
This loop adds additional convolutional layers with the same configuration as the first convolutional layer. 
It iterates conv_layers-1 times because the first convolutional layer has already been added.

'''

for l in range(conv_layers-1):
    model.add(Conv2D(layer_sizes,(3,3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten()) #This adds a flatten layer to convert the 2D feature maps into a 1D vector, which can be fed into the dense layers.

'''
This loop adds dense layers with layer_sizes neurons each, followed by ReLU activation and dropout regularization with a rate of 0.2.
This loop iterates dense_layers times.
'''
for l in range(dense_layers):
    model.add(Dense(layer_sizes))
    model.add(Activation("relu"))
    model.add(Dropout(0.2))

model.add(Dense(1))
model.add(Activation("sigmoid")) 
'''
This adds the output layer with a single neuron (because it's a binary classification problem), 
followed by a sigmoid activation function to output probabilities between 0 and 1.
'''


'''
Model Architecture:

The model architecture consists of convolutional layers followed by dense layers.
Convolutional layers are designed to extract features from images, and dense layers are used for classification.
Dropout is added to prevent overfitting.

'''

model.compile(loss="binary_crossentropy",optimizer ="adam",metrics=['accuracy'])

model.fit(X, y, batch_size=32, epochs=10, validation_split=0.1)

model.save('doggos_v_cattos.model')

'''
In summary, this code loads image data of cats and dogs from pickle files, builds a convolutional neural network (CNN) model, 
trains it on the data, and saves the trained model for future use in classifying images as either cats or dogs.

dense:A Dense layer feeds all outputs from the previous layer to all its neurons, 
each neuron providing one output to the next layer. 
A Dense(10) has ten neurons

dropout:Dropout works by randomly setting the outgoing edges of hidden units 
(neurons that make up hidden layers) to 0 at each update of the training phase

Activation:introduces non linearity.if we just have linear functions, summing em up would just give us another
linear function and acts as a single layer no matter how many layers we have in it
non-linear activation functions are essential components of neural network models like CNNs, 
enabling them to learn complex patterns, capture hierarchical representations, break symmetry, and facilitate the training process through backpropagation.
            
            

'''
            
