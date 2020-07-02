import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm

DATADIR="D:\Projects\Project Resources\PetImages2"#bigger the data,the better

CATEGORIES=["Dog","Cat"]

for category in CATEGORIES:  # do dogs and cats
    path = os.path.join(DATADIR,category)  # create path to dogs and cats
    for img in os.listdir(path):  # iterate over each image per dogs and cats
        img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
        plt.imshow(img_array, cmap='gray')  # graph it
        plt.show()  # display!

        break  # we just want one for now so break
    break  #...and one more!
    
print(img_array)#raw data,simply displays what an image looks like to a computer.
print(img_array.shape)#height and width

#resizing the images...
IMG_SIZE = 50

new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
plt.imshow(new_array, cmap='gray')
plt.show()
#all that was just basic operations. main part starts now!

training_data = []

def create_training_data():#creates the data to be trained.
    for category in CATEGORIES:  # do dogs and cats

        path = os.path.join(DATADIR,category)  # create path to dogs and cats
        class_num = CATEGORIES.index(category)  # get the classification  (0 or a 1). 0=dog 1=cat

        for img in tqdm(os.listdir(path)):  # iterate over each image per dogs and cats
            try:
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
                new_array = cv2.resize(img_array, (IMG_SIZE,IMG_SIZE))  # resize to normalize data size
                training_data.append([new_array,class_num])  # add this to our training_data
            except Exception: #skips if the image is corrupted
                pass
           
create_training_data()

print(len(training_data))
#shuffling the training data
import random
random.shuffle(training_data)

for sample in training_data[:10]:
    print(sample[1])

#variables we need to feed into the cnn
X=[]
y=[]

for features,label in training_data:
    X.append(features)
    y.append(label)

print(X[0].reshape(-1,IMG_SIZE,IMG_SIZE,1))

X=np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE,1)#converting x into a numpy array

import pickle

pickle_out=open("X.pickle","wb")
pickle.dump(X,pickle_out)
pickle_out.close()

pickle_out=open("y.pickle","wb")
pickle.dump(y,pickle_out)
pickle_out.close()
#Python pickle module is used for serializing and de-serializing a Python object structure.
# Any object in Python can be pickled so that it can be saved on disk
#What pickle does is that it “serializes” the object first before writing it to file
#this is done so that we dont have to do all the image processing over and over again just incase we change the model  
