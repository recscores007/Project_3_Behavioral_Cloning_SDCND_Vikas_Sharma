
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf

import pandas as pd
import numpy as np
from tqdm import tqdm_notebook
import time
import shutil
import os
import random
import cv2
import math
import json
import csv
import keras
from keras.preprocessing.image import *
from keras.models import Sequential, Model
from keras.layers import Convolution2D, Flatten, MaxPooling2D, Lambda, ELU
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam
from keras.callbacks import Callback
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

#from IPython.display import display # Allows the use of display() for DataFrames
# Visualizations will be shown in the notebook.
 
 # Get randomized datasets for training and validation
 
# shuffle data

from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

 
lines=[]
 
with open('driving_log3.csv') as csvfile:
    reader=csv.reader(csvfile)
    for line in reader:
        lines.append(line)
images=[]
 
seed = 7
np.random.seed(seed)
 
columns = ['center', 'left', 'right', 'steering_angle', 'throttle', 'brake', 'speed']
data = pd.read_csv('driving_log3.csv', names=columns)
 
print("Dataset Columns:", columns, "\n")
print("Shape of the dataset:", data.shape, "\n")
print(data.describe(), "\n")
 
print("Data loaded...")
 
binwidth = 0.025
 
# histogram before image augmentation
plt.hist(data.steering_angle,bins=np.arange(min(data.steering_angle), max(data.steering_angle) + binwidth, binwidth))
plt.title('Number of images per steering angle')
plt.xlabel('Steering Angle')
plt.ylabel('# Frames')
plt.show()
 
#images_center,images_left,images_right=[]


# In[2]:


# Get randomized datasets for training and validation

# shuffle data
data = data.reindex(np.random.permutation(data.index))

num_train = int((len(data) / 10.) * 9.)

X_train = data.iloc[:num_train]
X_validation = data.iloc[num_train:]

print("X_train has {} elements.".format(len(X_train)))
print("X_valid has {} elements.".format(len(X_validation)))
#print("X_valid has {} elements.".format(X_train.center.iloc[53]))
#print("X_valid has {} elements.".format(data))


# In[3]:


# model training variables
NB_EPOCH = 10
BATCH_SIZE = 32


# ## DATA AUGMENTATION

# #### Mirror Image

# In[4]:


def mirrored(img, steering_angle):
    mirrored_image = cv2.flip(img, 1)
    steering_angle = -1 * steering_angle
    return mirrored_image, steering_angle

test_img =cv2.imread(data.center.iloc[53])
print(data.center.iloc[53])

steering_angle_test= data.steering_angle.iloc[53]

print(steering_angle_test)

translated_image, steering_angle_trans = mirrored(test_img,steering_angle_test)
print(steering_angle_trans)
fig, axs = plt.subplots(1,2, figsize=(10, 3))

axs[0].axis('off')
axs[0].imshow(test_img.squeeze(), cmap='gray')
axs[0].set_title('original')

axs[1].axis('off')
axs[1].imshow(translated_image.squeeze(), cmap='gray')
axs[1].set_title('mirrored')

print('shape in/out:', test_img.shape, translated_image.shape)


# ##### Random Brightness

# In[5]:


import cv2
def random_brightness(img, brightness=None):
    
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    
    if brightness:
        img[:,:,2] += brightness
    else:
        random_bright = .25 + np.random.uniform()
        img[:,:,2] = img[:,:,2] * random_bright
    
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    return img

test_img =cv2.imread(data.center.iloc[53])
print(data.center.iloc[53])

test_dst = random_brightness(test_img)


fig, axs = plt.subplots(1,2, figsize=(10, 3))

axs[0].axis('off')
axs[0].imshow(test_img.squeeze())
axs[0].set_title('original')

axs[1].axis('off')
axs[1].imshow(test_dst.squeeze())
axs[1].set_title('brightness adjusted')

print('shape in/out:', test_img.shape, test_dst.shape)


# ##### Cropping and resize

# In[6]:


# crop the top 1/5 of the image to remove the horizon and the bottom 25 pixels to remove the car’s hood
def resize_image(img):
   
    shape = img.shape
    #img = img[math.floor(shape[0]/3):shape[0]-25, 0:shape[1]]
    img=img[0:140, 25:320]
    img = cv2.resize(img, (64,64), interpolation=cv2.INTER_AREA)    
    return img

test_img =cv2.imread(data.center.iloc[53])
print(data.center.iloc[53])
test_dst = resize_image(test_img)

fig, axs = plt.subplots(1,2, figsize=(10, 3))

axs[0].axis('off')
axs[0].imshow(test_img.squeeze())
axs[0].set_title('original')

axs[1].axis('off')
axs[1].imshow(test_dst.squeeze())
axs[1].set_title('Cropped')

print('shape in/out:', test_img.shape, test_dst.shape)


# ##### Random Translate

# In[7]:


import cv2

def random_translate(img,steering_angle):
    rows, cols, channels = img.shape
    
    # Translation
    tx = 100 * np.random.uniform() - 100 / 2
    ty = 40 * np.random.uniform() - 40 / 2
    steering_angle = steering_angle + tx / 40 * 2 * .2
    
    transform_matrix = np.float32([[1, 0, tx],
                                   [0, 1, ty]])
    
    translated_image = cv2.warpAffine(img, transform_matrix, (cols, rows))
    return translated_image, steering_angle

#test_img = X_train[103]
#steering_angle_test= Y_train[103]
test_img =cv2.imread(data.center.iloc[53])
print(data.center.iloc[53])

steering_angle_test= data.steering_angle.iloc[53]

print(steering_angle_test)

translated_image, steering_angle_trans = random_translate(test_img,steering_angle_test)
print(steering_angle_trans)
fig, axs = plt.subplots(1,2, figsize=(10, 3))

axs[0].axis('off')
axs[0].imshow(test_img.squeeze(), cmap='gray')
axs[0].set_title('original')

axs[1].axis('off')
axs[1].imshow(translated_image.squeeze(), cmap='gray')
axs[1].set_title('translated')

print('shape in/out:', test_img.shape, translated_image.shape)


# In[8]:


def apply_random_transformation(img, steering_angle):
    
    transformed_image, steering_angle = random_translate(img, steering_angle)
    transformed_image = random_brightness(transformed_image)
       
    if np.random.random() < 0.5:
        transformed_image, steering_angle = mirrored(transformed_image, steering_angle)
            
    #transformed_image = resize_image(transformed_image)
    
    return transformed_image, steering_angle


# In[9]:


def load_and_augment_image(line_data):
    i = np.random.randint(3)
    
    if (i == 0):
        path_file = line_data['left'][0].strip()
        shift_angle = 0.25
    elif (i == 1):
        path_file = line_data['center'][0].strip()
        shift_angle = 0.
    elif (i == 2):
        path_file = line_data['right'][0].strip()
        shift_angle = -0.25
        
    steering_angle = line_data['steering_angle'][0] + shift_angle
    
    img = cv2.imread(path_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img, steering_angle = apply_random_transformation(img, steering_angle)
    return img, steering_angle


# In[10]:


# generators in multi-threaded applications is not thread-safe. Hence below:
class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self
    
    def __next__(self):
        with self.lock:
            return self.it.__next__()
        
def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g


# In[11]:


import threading


# ### Keras generator ( to train more images without running into memory issue)

# In[12]:


generated_steering_angles = []
threshold = 1

@threadsafe_generator
def generate_batch_data(_data, batch_size = 32):
    
    batch_images = np.zeros((batch_size, 160, 320,3))
    batch_steering = np.zeros(batch_size)
    
    while 1:
        for batch_index in range(batch_size):
            row_index = np.random.randint(len(_data))
            line_data = _data.iloc[[row_index]].reset_index()
            
            # idea borrowed from Vivek Yadav: Sample images such that images with lower angles 
            # have lower probability of getting represented in the dataset. This alleviates 
            # any problems we may ecounter due to model having a bias towards driving straight.
            
            keep = 0
            while keep == 0:
                x, y = load_and_augment_image(line_data)
                if abs(y) < .1:
                    val = np.random.uniform()
                    if val > threshold:
                        keep = 1
                else:
                    keep = 1
            
            batch_images[batch_index] = x
            batch_steering[batch_index] = y
            generated_steering_angles.append(y)
        yield batch_images, batch_steering
        


# ## MODEL ARCHITECTURE
# CNN architecture:
# - Crop image to eliminate hood and horizon
# - Normalize data
# - Use 4 convolution layers with batch normalization and activation ELU,
# - 1 Flatten layer and 2 dense layer with activation ELU and using dropout to not overfit the network,
# - Learning rate 1e-4 instead of default 1e-3 for Adam optimzer

# In[13]:


# from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Lambda
from keras.layers import Cropping2D
from keras.layers import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

# TODO: Build Convolutional Neural Network in Keras Here
model = Sequential()
model.add(Cropping2D(cropping=((75,25), (0,0)), input_shape=(160,320,3)))

model.add(Lambda(lambda x: x/127.5 - 1))

model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same", activation='elu', name='Conv1'))

model.add(BatchNormalization())
model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same", activation='elu', name='Conv2'))

model.add(BatchNormalization())
model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same", activation='elu', name='Conv3'))

model.add(Flatten())

model.add(Dropout(.2))
model.add(ELU())
model.add(Dense(512, activation='elu', name='FC1'))

model.add(Dropout(.5))
model.add(ELU())
model.add(Dense(1, name='output'))

model.summary()

# compile
opt = Adam(lr=0.0001)
model.compile(optimizer=opt, loss='mse', metrics=[])
#model.fit(X_train,Y_train, validation_split=0.2,shuffle=True,nb_epoch=5)
#model.save('modelnv2_0211.h5')


# ## TRAINING and VALIDATION

# In[14]:



class LifecycleCallback(keras.callbacks.Callback):

    def on_epoch_begin(self, epoch, logs={}):
        pass

    def on_epoch_end(self, epoch, logs={}):
        global threshold
        threshold = 1 / (epoch + 1)

    def on_batch_begin(self, batch, logs={}):
        pass

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

    def on_train_begin(self, logs={}):
        print('BEGIN TRAINING')
        self.losses = []

    def on_train_end(self, logs={}):
        print('END TRAINING')
        
# Calculate the correct number of samples per epoch based on batch size
def calc_samples_per_epoch(array_size, batch_size):
    num_batches = array_size / batch_size
    samples_per_epoch = math.ceil(num_batches)
    samples_per_epoch = samples_per_epoch * batch_size
    return samples_per_epoch


# In[15]:


lifecycle_callback = LifecycleCallback()       

train_generator = generate_batch_data(X_train, BATCH_SIZE)
validation_generator = generate_batch_data(X_validation, BATCH_SIZE)

samples_per_epoch = calc_samples_per_epoch((len(X_train)*3), BATCH_SIZE)
nb_val_samples = calc_samples_per_epoch((len(X_validation)*3), BATCH_SIZE)

history = model.fit_generator(train_generator, 
                              validation_data = validation_generator,
                              samples_per_epoch = samples_per_epoch, 
                              nb_val_samples = nb_val_samples,
                              nb_epoch = NB_EPOCH, verbose=1,
                              callbacks=[lifecycle_callback])


# ### SAVE MODEL

# In[16]:


model.save('modelnv2_0921_2.h5')


# #### RESULT OF AUGMENTATION ( Adding more images at larger angles)

# In[17]:


plt.hist(generated_steering_angles, bins=np.arange(min(generated_steering_angles), max(generated_steering_angles) + binwidth, binwidth))
plt.title('Number of augmented images per steering angle')
plt.xlabel('Steering Angle')
plt.ylabel('# Augmented Images')
plt.show()

