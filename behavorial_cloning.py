import csv
import cv2
import numpy as np
import random
from random import randint
import pickle

lines = []
with open('driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	next(reader,None)
	for line in reader:
		lines.append(line)

images = []
output_steerings = []
for line in lines:
	center_steering = float(line[3])
	correction = 0.25
	imageBGRC = cv2.imread(line[0].strip())
	imageBGRL = cv2.imread(line[1].strip())
	imageBGRR = cv2.imread(line[2].strip())

	# imageBGRC = cv2.cvtColor(imageBGRC, cv2.COLOR_BGR2HSV)
	# h,s,v = cv2.split(imageBGRC)
	# s += abs(randint(-5,5))
	# imageBGRC = cv2.merge((h,s,v))
	# imageBGRC = cv2.cvtColor(imageBGRC, cv2.COLOR_HSV2BGR)

	# imageBGRL = cv2.cvtColor(imageBGRL, cv2.COLOR_BGR2HSV)
	# h,s,v = cv2.split(imageBGRL)
	# s += abs(randint(-5,5))
	# imageBGRL = cv2.merge((h,s,v))
	# imageBGRL = cv2.cvtColor(imageBGRL, cv2.COLOR_HSV2BGR)

	# imageBGRR = cv2.cvtColor(imageBGRR, cv2.COLOR_BGR2HSV)
	# h,s,v = cv2.split(imageBGRR)
	# s += abs(randint(-5,5))
	# imageBGRR = cv2.merge((h,s,v))
	# imageBGRR = cv2.cvtColor(imageBGRR, cv2.COLOR_HSV2BGR)

	imageC = cv2.cvtColor(imageBGRC, cv2.COLOR_BGR2RGB)
	imageL = cv2.cvtColor(imageBGRL, cv2.COLOR_BGR2RGB)
	imageR = cv2.cvtColor(imageBGRR, cv2.COLOR_BGR2RGB)

	images.append(imageC)
	images.append(imageL)
	images.append(imageR)

	output_steerings.append(center_steering)
	output_steerings.append(center_steering + correction)
	output_steerings.append(center_steering - correction)
	output_steerings.append(center_steering - correction)


	if(center_steering > 0.3):
		images += 10 * [imageC]
		output_steerings += 10 * [center_steering]
		#images += 1 * [np.fliplr(imageC)]
		#output_steerings += 1 * [-center_steering]
	if(center_steering < -0.3):
		images += 10 * [imageC]
		output_steerings += 10 * [center_steering]
		#images += 1 * [np.fliplr(imageC)]
		#output_steerings += 1 * [-center_steering]

X_train = np.array(images)
y_train = np.array(output_steerings)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Activation, Dropout, ELU, Reshape
from keras.regularizers import l2
from keras.layers.convolutional import Convolution2D, Cropping2D, MaxPooling2D, AveragePooling2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

model = Sequential()
model.add(Cropping2D(cropping=((60,25), (0,0)),dim_ordering='tf',  input_shape=(160,320,3)))
model.add(Lambda(lambda x: x /255.0 - 0.5))
model.add(AveragePooling2D(pool_size=(2,2)))
model.add(Convolution2D(24, 2, 2, subsample=(2, 2)))
model.add(Activation('relu'))
model.add(Convolution2D(36, 2, 2, subsample=(2, 2)))
model.add(Activation('relu'))
model.add(Dropout(0.7))
model.add(Convolution2D(48, 2, 2, subsample=(2, 2)))
model.add(Activation('relu'))
model.add(Convolution2D(64, 1, 1, subsample=(1, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(48))
model.add(Activation('relu'))
model.add(Dense(24))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(12))
model.add(Activation('relu'))
model.add(Dense(6))
model.add(Activation('relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer=Adam(lr=0.0001))
model.fit(X_train, y_train, validation_split=0.2, nb_epoch=5,shuffle=True)

model.save('model.h6')
