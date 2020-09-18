#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.models import load_model
import pandas as pd
import cv2
import numpy as np
import glob


# In[2]:


def rotate(model,image):
    image_pred = image.reshape(1,64,64,3)
    output = model.predict(image_pred)
    value = output.argmax()          #get the index of the encoded output
    angle = 0
    
    if value  == 1:      #right
        angle = 90

    elif value == 0:    #left
        angle = 270

    elif value == 3:    #down
        angle = 180
        
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


# In[3]:


if __name__ == '__main__':
    model = load_model('model.h5')
    corrected = []
    for file in glob.glob("./train/*.jpg"):
        image_aux = cv2.imread(file,1)
        new_image = rotate(model,image_aux)
        corrected.append(new_image)
    
    corrected = np.array(corrected)
    np.save('np_out',corrected)


# In[ ]:




