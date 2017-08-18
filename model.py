##########################################################################################
#                                      Libraries                                         #
##########################################################################################


# Importing libraries for reading data.
import csv
import cv2
import numpy as np
from itertools import islice
# Importing libraries for shuffling/splitting/visualizing data.
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import random
import matplotlib.pyplot as plt
# Importing libraries for creating the model.
import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D



##########################################################################################
#                                      Variables                                         #
##########################################################################################


# Define all required variables.
epochs = 10
batch_size = 32
correction = 0.25
csv_path = './data/driving_log.csv'
center_images = []
left_images = []
right_images = []
steering_angles = []
throttles = []
brakes = []
speeds = []


##########################################################################################
#                                      Functions                                         #
##########################################################################################


def read_data(data):
# Reads in data and creates a histogram

    print('Reading data')
    for line in data:
        center_image = './data/IMG/'+line[0].split('/')[-1]
        left_image = './data/IMG/'+line[1].split('/')[-1]
        right_image = './data/IMG/'+line[2].split('/')[-1]
        steering_angle = float(line[3])
        throttle = float(line[4])
        brake = float(line[5])
        speed = float(line[6])
        image = cv2.imread(center_image)
        center_images.append(image)
        image = cv2.imread(left_image)
        left_images.append(image)
        image = cv2.imread(right_image)
        right_images.append(image)
        steering_angles.append(steering_angle)
        throttles.append(throttle)
        brakes.append(brake)
        speeds.append(speed)

    print('Plotting histogram')
    img, vaxis = plt.subplots(2, 2, figsize=(20,20))
    vaxis[0,0].hist(steering_angles, bins=40)
    vaxis[0,0].set_title('Steering Angle')
    vaxis[0,1].hist(throttles, bins=40)
    vaxis[0,1].set_title('Throttle')
    vaxis[1,0].hist(brakes, bins=40)
    vaxis[1,0].set_title('Brake')
    vaxis[1,1].hist(speeds, bins=40)
    vaxis[1,1].set_title('Speed')
    plt.savefig('data.png')
    print('Histogram saved')

def flip_data(image, angle):
# Flips images vertically and inverts the steering angle.

    flip_image = cv2.flip(image,1)
    flip_angle = angle * -1.0
    return flip_image, flip_angle

def visualize_data(center_images, left_images, right_images, steering_angles):
# Visualizes a random original set and flipped set of images and saves them

    print('Plotting data')
    index = random.randint(0, len(center_images))

    center_image = center_images[index]
    left_image = left_images[index]
    right_image = right_images[index]

    center_angle = steering_angles[index]
    left_angle = center_angle + correction
    right_angle = center_angle - correction

    flipped_center_image, flipped_center_angle = flip_data(center_image, center_angle)
    flipped_left_image, flipped_left_angle = flip_data(left_image, left_angle)
    flipped_right_image, flipped_right_angle = flip_data(right_image, right_angle)

    img, vaxis = plt.subplots(1, 3, figsize=(10,5))
    vaxis[0].imshow(left_image)
    vaxis[0].set_title('Original Left')
    vaxis[1].imshow(center_image)
    vaxis[1].set_title('Original Center')   
    vaxis[2].imshow(right_image)
    vaxis[2].set_title('Original Right')
    plt.savefig('Original Image.png')

    img, vaxis = plt.subplots(1, 3, figsize=(10,5))
    vaxis[0].imshow(flipped_left_image)
    vaxis[0].set_title('Flipped Left')
    vaxis[1].imshow(flipped_center_image)
    vaxis[1].set_title('Flipped Center')
    vaxis[2].imshow(flipped_right_image)
    vaxis[2].set_title('Flipped Right')
    plt.savefig('Flipped Image.png')
    print('Images saved')

def split_data(data):
# Splits 20% of the data into a validation set

    training_data, validation_data = train_test_split(data, test_size=0.2)
    print('Training Dataset Size:', len(training_data))
    print('Validation Dataset Size:', len(validation_data))
    return training_data, validation_data


def generator(data, batch_size):
# Generator to reduce strain on memory.  Outputs processed data, labels, and steering angles    

    data_size = len(data)
    while 1: # Loop forever so the generator never terminates
        shuffle(data)
        for offset in range(0, data_size, batch_size):
            batch_data = data[offset:offset+batch_size]

            images = []
            angles = []
            for batch in batch_data:
                center_name = './data/IMG/'+batch[0].split('/')[-1]
                left_name = './data/IMG/'+batch[1].split('/')[-1]
                right_name = './data/IMG/'+batch[2].split('/')[-1]
                center_angle = float(batch[3])

                # Add flipped images
                center_image, center_angle = flip_data(cv2.imread(center_name), center_angle)
                images.append(center_image)
                angles.append(center_angle)

                left_angle = center_angle + correction
                left_image, left_angle = flip_data(cv2.imread(left_name), left_angle)
                images.append(left_image)
                angles.append(left_angle)

                right_angle = center_angle - correction
                right_image, right_angle = flip_data(cv2.imread(right_name), right_angle)
                images.append(right_image)
                angles.append(right_angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


def nvida():
# Based on Nvidia model from the lecture, with added RELUs and Dropouts

    model = Sequential()
    # Cropping layer
    model.add(Cropping2D(cropping=((60,25),(0,0)), input_shape=(160,320,3)))
    # Lambda & Normalization Layer
    model.add(Lambda(lambda x: x/255.0 - 0.5))
    # Convolution Layer 1
    model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
    # Convolution Layer 2
    model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
    # Convolution Layer 3
    model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
    # Convolution Layer 4
    model.add(Convolution2D(64,3,3,activation="relu"))
    # Convolution Layer 5
    model.add(Convolution2D(64,3,3,activation="relu"))
    # Flatten
    model.add(Flatten())
    # Fully-connected Layer 1
    model.add(Dense(100))
    model.add(Dropout(0.4))
    model.add(Activation('relu'))
    # Fully-connected Layer 2
    model.add(Dense(50))
    model.add(Dropout(0.4))
    model.add(Activation('relu'))
    # Fully-connected Layer 3
    model.add(Dense(10))
    model.add(Dropout(0.4))
    model.add(Activation('relu'))
    # Output Layer
    model.add(Dense(1))
    model.add(Activation('linear'))
    return model


def training_model(training_generator, validation_generator, training_samples, validation_samples):
# Trains and saves the model, while also creating a graph to visualize the loss

    print('Training the model')
    model = nvida()
    model.compile(optimizer='adam', loss='mse')
    history_object = model.fit_generator(training_generator, samples_per_epoch= len(training_samples*2), validation_data=validation_generator, nb_val_samples=len(validation_samples*2), nb_epoch=epochs)
    model.save('model.h5')
    model.summary()

    print('Plotting Loss histogram')
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('MSE Loss Graph')
    plt.xlabel('epochs')
    plt.ylabel('MSE Loss')
    plt.legend(['Training Loss', 'Validation Loss'], loc='upper right')
    plt.savefig('MSE_Loss_Graph.png')
    plt.show()
    print('Complete')


##########################################################################################
#                                      Main Program                                      #
##########################################################################################


# Read the csv file.
data = []
with open(csv_path) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:       #//islice(reader, 1, None):
        data.append(line)

#
read_data(data) 
visualize_data(center_images, left_images, right_images, steering_angles) # Visualize Data

# Split data
training_samples, validation_samples = split_data(data)

# Create training and validation generators
training_generator = generator(training_samples, batch_size)
validation_generator = generator(validation_samples, batch_size)

# Train the model
training_model(training_generator, validation_generator, training_samples, validation_samples)