# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 01:59:57 2020

@author: Hamza
"""

import cv2
import tensorflow as tf
import os


CATEGORIES = ["Dog", "Cat"]


def prepare(filepath):
    IMG_SIZE = 50
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    img_array = img_array/255.0
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


model = tf.keras.models.load_model("doggos_v_cattos.model")
c=0
dir=r'C:\Users\Hamza\Desktop\cats'
pics=os.listdir(dir)
for i in range(200):
    prediction = model.predict([prepare(os.path.join(dir,pics[i]))])
    if prediction>0.51:
        print("Cat")
        c+=1
    else:
        print("Dog")
print(c)
print(c/2)
'''
import cv2
import tensorflow as tf

CATEGORIES = ["Dog", "Cat"]


def prepare(filepath):
    IMG_SIZE = 50
    img_array = cv2.imread(filepath,1)
    img_array = img_array/255.0
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


model = tf.keras.models.load_model("64x3-CNN.model")

prediction = model.predict([prepare(r'C:\Users\Hamza\Desktop\3.jpg')])
print(prediction)  # will be a list in a list.
print(CATEGORIES[int(prediction[0][0])])


'''