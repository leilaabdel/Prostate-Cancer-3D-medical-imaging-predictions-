#!/usr/bin/env python
# coding: utf-8

# # Import the images into directories 

# In[5]:


import os
import zipfile
#Import TensorFlow and Keras
import tensorflow as tf
from tensorflow import keras
from keras.utils.np_utils import to_categorical
import sys
import Generator
print(sys.version)
help(tf)
#Helper libraries
import numpy as np

from PIL import Image 

from medpy.io import load
import sklearn
from sklearn .model_selection import train_test_split


# In[15]:


AS = [];
PR = [];
y_train_frame = []
AS_Header = [];
PR_Header = [];

y_train_movie = [];
x_train_movie = [];

def isFileOfInterest(filename):
    if "dce" in filename or "sag" in filename or "tra" in filename:
        return True 
    else:
        return False

src_directory = "D:/PROSTATE_FINAL_Validation_12_18"
for dirName, subdirList, filenames in os.walk(src_directory):
    for filename in filenames:
        filepath = os.path.join(dirName, filename)
        if "AS0" in dirName:
            if isFileOfInterest(filename):
                    ASimage , header = load(filepath)
    

                    x_train_movie.append(ASimage)
                    
                    imageY = np.zeros((1 , 168))
                    y_train_frame.append(imageY)
                    y_train_movie.append(0)
                
        if "PR0" in dirName:
            if isFileOfInterest(filename):
                    PRimage , header = load(filepath)
                
                    x_train_movie.append(PRimage)   
        
                    imageY = np.ones((1 , 168))
                    y_train_frame.append(imageY)
                    y_train_movie.append(1)
                


# In[17]:




y_frame_train = np.asarray(y_train_frame)
y_movie_train = np.asarray(y_train_movie)


x_movie_train = np.asarray(x_train_movie)
x_movie_train = np.expand_dims(x_movie_train , axis = 5)




# In[21]:


x_frame_train = []

print(x_movie_train.shape)
print(x_movie_train[1][1])

for i in range (0 , len(x_movie_train)):
    for j in range (0,168):
        x_frame_train.append(x_movie_train[i][j])
        
        
y_frame_train = y_frame_train.ravel();
x_frame_train = np.asarray(x_frame_train)

print(y_frame_train.shape)
print(x_frame_train.shape)        

    


# In[22]:


# Generate the permutation index array.
permutation = np.random.permutation(x_frame_train.shape[0])
   # Shuffle the arrays by giving the permutation in the square brackets.
shuffled_x_train_frame = x_frame_train[permutation]
shuffled_y_train_frame = y_frame_train[permutation]
   


# In[23]:




y_movie_train = y_movie_train.astype(np.uint8)

y_movie_train = to_categorical(y_movie_train, 2, dtype='uint8')


x_train_data , x_test_data , y_train_data  , y_test_data = train_test_split(x_movie_train , y_movie_train , test_size=0.33)


# In[24]:



# Settings
seed = 42;
batchsize = 1;
ep = 100;


image_aug = Generator.customImageDataGenerator(rescale = 1./255,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True , fill_mode = 'nearest')

mask_aug = Generator.customImageDataGenerator(
			rotation_range = 20
)


train_datagen = image_aug.flow(x_train_data , y_train_data , batch_size = batchsize , seed = seed , shuffle = True)
test_datagen =  image_aug.flow(x_test_data , y_test_data , batch_size = batchsize , seed = seed , shuffle = True)


# In[25]:


from keras import models
from keras import layers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv3D , MaxPooling3D , Conv2D, MaxPooling2D

model = models.Sequential();
model.add(layers.Conv3D(168 ,(50 , 50 , 50) ,activation='relu', input_shape=(168 , 168 , 168 , 1) ,
                        kernel_initializer = 'glorot_normal'))
model.add(layers.MaxPooling3D((16 , 16 , 16)))
model.add(layers.Conv3D(336, (4 , 4 , 4) , activation='relu' , kernel_initializer = 'glorot_normal'))
model.add(layers.MaxPooling3D((3 , 3 , 3)))
model.add(layers.Dropout(0.5));
model.add(layers.Flatten());
model.add(layers.Dense(64, activation='relu'));
model.add(layers.Dense(32 , activation = 'relu'));
model.add(layers.Dense(2 ,activation = 'softmax'));



# In[ ]:


model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit_generator(train_datagen, epochs=200, validation_data = test_datagen, verbose = 1 , steps_per_epoch = 1 , validation_steps = 1)


# In[ ]:




