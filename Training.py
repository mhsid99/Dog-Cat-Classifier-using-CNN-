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
  '''
  Activation:introduces non linearity.if we just have linear functions, summing em up would just give us another
linear function and acts as a single layer no matter how many layers we have in it
non-linear activation functions are essential components of neural network models like CNNs, 
enabling them to learn complex patterns, capture hierarchical representations, break symmetry, and facilitate the training process through backpropagation.      
  '''
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

'''
Overfitting is a common problem in machine learning where a model learns the training data too well.
it can capture noise or random fluctuations in the training data as if they were meaningful patterns or
When the model becomes too tailored to the training data, it may fail to generalize to new, unseen data. 
This results in poor performance on test data or real-world data, as the model has learned to memorize the training examples rather 
than learn the underlying patterns that generalize to new data.

Dropout randomly deactivates neurons during training, forcing the model to learn redundant representations and reducing reliance on specific neurons, 
thus preventing overfitting.

0.2 is the dropout rate, indicating that 20% of the neurons in the previous layer will be randomly set to zero during each training iteration
'''

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
'''
The loss function (or objective function) measures how well the model performs on the training data.
'binary_crossentropy' is used as the loss function, which is commonly used for binary classification problems.

The optimizer is responsible for updating the model's parameters (weights and biases)
'''

model.fit(X, y, batch_size=32, epochs=10, validation_split=0.1) #Training is done in batches of size 32, for 10 epochs, with 10% of the data used for validation.

model.save('doggos_v_cattos.model')

'''
In summary, this code loads image data of cats and dogs from pickle files, builds a convolutional neural network (CNN) model, 
trains it on the data, and saves the trained model for future use in classifying images as either cats or dogs.




      

'''
            
