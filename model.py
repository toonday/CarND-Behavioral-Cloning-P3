import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Variables for tunning
nn_type = 'nvidia_net'
resize_shape = (32, 32)
resize_images = True
augment_images = True
use_all_cameras = True
use_gradient_filter = False
use_gray_scale = False
use_hsv = False
save_model = True
correction_factor = 0.14

samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        if line[0] != 'center':
            samples.append(line)

import sklearn
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    count = 0
    while 1: # Loop forever so the generator never terminates
        #shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                if use_all_cameras:
                    for i in range(3):
                        source_path = batch_sample[i]
                        filename = source_path.split('/')[-1]
                        current_path = './data/IMG/' + filename
                        image = cv2.imread(current_path)
                        if use_gray_scale:
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        if use_gradient_filter:
                            image = cv2.Canny(image,240,250)
                            #image = cv2.Sobel(image,cv2.CV_32F,0,1,ksize=7)
                        if resize_images:
                            image = cv2.resize(image, resize_shape)
                        if use_hsv:
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                        images.append(image)
                        angle = 0
                        if i == 0:
                            angle = float(batch_sample[3])
                        elif i == 1:
                            angle = float(batch_sample[3]) + correction_factor
                        elif i == 2:
                            angle = float(batch_sample[3]) - correction_factor
                        angles.append(angle)

                        if augment_images:
                            # Flipping Augmentation
                            images.append(cv2.flip(image, 1))
                            angles.append(angle * -1.0)
                else:
                    source_path = batch_sample[0]
                    filename = source_path.split('/')[-1]
                    current_path = './data/IMG/' + filename
                    image = cv2.imread(current_path)
                    if use_gray_scale:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    if use_gradient_filter:
                        image = cv2.Canny(image,240,250)
                        #image = cv2.Sobel(image,cv2.CV_32F,0,1,ksize=7)
                    if resize_images:
                        image = cv2.resize(image, resize_shape)
                    if use_hsv:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                    images.append(image)
                    angle = float(batch_sample[3])
                    angles.append(angle)

                    if augment_images:
                        # Flipping Augmentation
                        images.append(cv2.flip(image, 1))
                        angles.append(angle * -1.0)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            
            if use_gray_scale:
                #X_train = np.average(X_train, axis=3)
                X_train = np.expand_dims(X_train, axis=3)
                
            #'''
            # Display images for debugging
            if offset == 0 and count == 0:
                count += 1
                
                if use_gray_scale:
                    temp_image = X_train[0][:,:,0]
                else:
                    temp_image = X_train[0]
                orig_image = temp_image.copy()
                #orig_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
                #orig_image = cv2.cvtColor(orig_image, cv2.COLOR_HSV2BGR)

                top_crop = 60 if not resize_images else 12
                btm_crop = 20 if not resize_images else 4
                starty, cropy = top_crop, 160-btm_crop
                startx, cropx = 0, 320
                temp_image = temp_image[starty:cropy, startx:cropx]
                    
                edges = cv2.Canny(temp_image,240,250)
                laplacian = cv2.Laplacian(temp_image,cv2.CV_32F)
                sobelx = cv2.Sobel(temp_image,cv2.CV_32F,1,0,ksize=7)
                sobely = cv2.Sobel(temp_image,cv2.CV_32F,0,1,ksize=7)
                plt.subplot(2,2,1), plt.imshow(orig_image, cmap='gray')
                plt.title('Original'), plt.xticks([]), plt.yticks([])
                plt.subplot(2,2,2),plt.imshow(edges,cmap = 'gray')
                plt.title('Canny Edges'), plt.xticks([]), plt.yticks([])
                plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
                plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
                plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
                plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
                plt.show()
            #'''
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)
input_shape = (160, 320, 3) if not resize_images else (resize_shape[0], resize_shape[1], 3)
if use_gray_scale:
    input_shape = (input_shape[0], input_shape[1], 1) 

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Convolution2D, MaxPooling2D, Dropout, Cropping2D

if nn_type == 'le_net':
    # Le Net
    #input_shape = (160, 320, 3)
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=input_shape))
    #if resize_images:
    #    model.add(Cropping2D(cropping=((12,4),(0,0))))
    #else:
    #    model.add(Cropping2D(cropping=((60,20),(0,0))))
    model.add(Convolution2D(15, 5, 5, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(30, 5, 5, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(120, activation='relu'))
    model.add(Dense(84, activation='relu'))
    model.add(Dense(1))
elif nn_type == 'nvidia_net':
    # NVIDIA
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=input_shape))
    model.add(Convolution2D(24, 5, 5, activation='relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(36, 5, 5, activation='relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(48, 5, 5, activation='relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(60, 3, 3, activation='relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(72, 3, 3, activation='relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(1164, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
else:
    # Simple NN
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=input_shape))
    model.add(Flatten())
    #model.add(Flatten(input_shape=(160,320,3)))
    model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
sample_multiplier = 1
if use_all_cameras:
    sample_multiplier += 2
if augment_images:
    sample_multiplier *= 2
len_of_train_samples = len(train_samples) * sample_multiplier
print(len(train_samples), "::", sample_multiplier, "::", len_of_train_samples)
model.fit_generator(train_generator, samples_per_epoch=len_of_train_samples, validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=5)

if save_model:
    model.save('model.h5')