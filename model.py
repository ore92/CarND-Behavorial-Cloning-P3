import csv
import cv2
import numpy as np
import random
from random import randint
import pickle

samples = []
with open('driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	next(reader,None)
	for line in reader:
		samples.append(line)

def generator(samples, batch_size=32):
	num_samples = len(samples)
	while 1:
		shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]
			images = []
			output_steerings = []
			for batch_sample in batch_samples:
				center_steering = float(batch_sample3])
				correction = 0.25
				#center image
				imageBGRC = cv2.imread(batch_sample[0].strip())

				#left image
				imageBGRL = cv2.imread(batch_sample[1].strip())
				#right image
				imageBGRR = cv2.imread(batch_sample[2].strip())

				imageBGRC = cv2.cvtColor(imageBGRC, cv2.COLOR_BGR2HSV)
				h,s,v = cv2.split(imageBGRC)
				s += abs(randint(-5,5))
				imageBGRC = cv2.merge((h,s,v))
				imageBGRC = cv2.cvtColor(imageBGRC, cv2.COLOR_HSV2BGR)

				imageBGRL = cv2.cvtColor(imageBGRL, cv2.COLOR_BGR2HSV)
				h,s,v = cv2.split(imageBGRL)
				s += abs(randint(-5,5))
				imageBGRL = cv2.merge((h,s,v))
				imageBGRL = cv2.cvtColor(imageBGRL, cv2.COLOR_HSV2BGR)

				imageBGRR = cv2.cvtColor(imageBGRR, cv2.COLOR_BGR2HSV)
				h,s,v = cv2.split(imageBGRR)
				s += abs(randint(-5,5))
				imageBGRR = cv2.merge((h,s,v))
				imageBGRR = cv2.cvtColor(imageBGRR, cv2.COLOR_HSV2BGR)

				imageC = cv2.cvtColor(imageBGRC, cv2.COLOR_BGR2RGB)
				imageL = cv2.cvtColor(imageBGRL, cv2.COLOR_BGR2RGB)
				imageR = cv2.cvtColor(imageBGRR, cv2.COLOR_BGR2RGB)

				images.append(imageC)
				images.append(imageL)
				images.append(imageR)

				output_steerings.append(center_steering)
				output_steerings.append(center_steering + correction)
				output_steerings.append(center_steering - correction)


				if(center_steering > 0.3):
					images += 10 * [imageC]
					output_steerings += 10 * [center_steering]
					images += 1 * [np.fliplr(imageC)]
					output_steerings += 1 * [-center_steering]
					if(center_steering < -0.3):
						images += 10 * [imageC]
						output_steerings += 10 * [center_steering]
						images += 1 * [np.fliplr(imageC)]
						output_steerings += 1 * [-center_steering]

		X_train = np.array(images)
		y_train = np.array(output_steerings)
		yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Activation, Dropout, ELU, Reshape
from keras.regularizers import l2
from keras.layers.convolutional import Convolution2D, Cropping2D, MaxPooling2D, AveragePooling2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

model = Sequential()
# trim image to only see section with road
model.add(Cropping2D(cropping=((60,25), (0,0)),dim_ordering='tf',  input_shape=(160,320,3)))
#normalization
model.add(Lambda(lambda x: x /255.0 - 0.5))
#pooling layer to reduce dimensionality
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
model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=5)

model.save('model.h5')
