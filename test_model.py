# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 23:04:53 2019

@author: Mitangi
"""

import numpy as np
import tensorflow as tf
from keras.preprocessing import image
import cv2
import matplotlib.pyplot as plt
emotions = ["neutral", "anger", "disgust", "fear", "happy", "sadness", "surprise"]

test_image = cv2.imread('test/User.1.12.jpg',cv2.IMREAD_GRAYSCALE)
test_image = cv2.resize(test_image, (48,48))
"""for i in range(len(test_image)):
    for j in range(len(test_image[0])):
        test_image[i][j] = 255 - test_image[i][j] """
#plt.imshow(test_image)

test_image = np.expand_dims(test_image, axis = 0)
test_image = np.expand_dims(test_image, axis = 3)

#print(test_image.shape) 

#loading saved model
new_model = tf.keras.models.load_model('Thresholded_Blurred_Trained')

predictions = new_model.predict(np.array(test_image))
print("Predicted character is :",end = " ")
print(emotions[np.argmax(predictions[0])])

img = cv2.resize(test_image[0], (48,48))
plt.imshow(img)

#def emotion_analysis(emotion):

#y_pos = np.arange(len(emotions))
 
#plt.bar(y_pos, emotion, align='center', alpha=0.5)
#plt.xticks(y_pos, emotions)
#plt.ylabel('percentage')
#plt.title('emotion')
 
#plt.show()