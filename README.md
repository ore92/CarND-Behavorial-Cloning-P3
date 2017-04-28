# P3: Behavioural Cloning
 This project uses deep neural networks and convolutional neural networks to clone driving behavior. Using a set of images and predicted steering angles, a CNN is generates with Keras to predict steering angles from images using an Udaciy simulator. 

##  Included Files
* `model.py`: file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.
    * [Video of the model driving](https://www.youtube.com/watch?v=BuNYg8pbcI8) 
* `model.h5`: containing the trained convolution neural network
* `drive.py`: for driving the car in autonomous mode

## 2. Dataset Characteristics

### Data Generation: Udacity's Car-Driving Simulator
I used the data provided instead of generating my own since I couldn't easily figure out how to drive with the mouse to generate good data. As there are no right turns, I experimented with several augumentation techniques. The one's that improved my model were:
* Using the left and right captured images and predicting their steering by multiplying the center steering by 0.25
* I augumented images with steering angles > 0.3 and <-0.3 by increasing their occurrence in the dataset by 10x i.e each sample that met the above characteristic now occured 10 times
* For samples that with steering angles > 0.3 and <-0.3 ; I flipped the images and the sign of the steering angles for further augumentation
* I also added a little noise(overfitting purposed) to the saturation values of the images; i modified this randomly between -5 to +5.

#### Sample Data
Here is an examples of an image in the data set.

![Center Image](examples/center_2016_12_01_13_39_28_024.jpg)

<table>
<th>Steering Angle</th>
<tr><td>-0.9426954</td>
</table>

### How the model was trained
The model was trained using the images provided and the augumentation techniques described above. In total, 21196 images were used for training and 5336 for validation.

## 3. Solution Design
As the goal was to predict steering angles on a road; I cropped out unuseful portions of the images to reduce what the network had to learn. I cropped out the top 60 and bottom 25 pixels. After augumenting the data set; I added some noise to images by flickering(+/- 5) the saturation values of the image in an attempt to combat overfitting. I then used Convolutional networks,Pooling layers and Fully connected Layers for my model; I used mean squared error has my loss function since this is a regression problem,adam optimizer and RELUs for activation; I used Dropout as a regularization Technique. The model is described in full below

## 4. Model architecture

From previous projects; I guaged the power of CNNs we created before and felt this problem could be solved with simpler networks since there were not too many features to discriminate (atleast with a human eye),also it was a simulator were the patterns were repeated exactly. In short, this was a much easier road to drive on than the real world with less noise. So I wanted to build a simple fast model that could not on my CPU in minutes.

My model is similar to the Nvidia model with less neurons and an Average pooling layer initially to reduce the size of inputs. I ran the model for 5 epochs about 20 minutes on my CPU.


The model code:
```
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

```


<table>
	<th>Layer</th><th>Details</th>
	<tr>
		<td>Average Pooling Layer 1</td>
		<td>
			<ul>
				<li>Pool Size: 2 x 2</li>
			</ul>
		</td>
	</tr>
	<tr>
		<td>Convolution Layer 1</td>
		<td>
			<ul>
				<li>Filters: 24</li>
				<li>Kernel: 2 x 2</li>
				<li>Stride: 2 x 2</li>
				<li>Padding: SAME</li>
				<li>Activation: relu</li>
			</ul>
		</td>
	</tr>
	<tr>
		<td>Convolution Layer 2</td>
		<td>
			<ul>
				<li>Filters: 36</li>
				<li>Kernel: 2 x 2</li>
				<li>Stride: 2 x 2</li>
				<li>Padding: SAME</li>
				<li>Activation: relu</li>
				<li>DropOut: 0.7</li>
			</ul>
		</td>
	</tr>
	<tr>
		<td>Convolution Layer 3</td>
		<td>
			<ul>
				<li>Filters: 48</li>
				<li>Kernel: 2 x 2</li>
				<li>Stride: 2 x 2</li>
				<li>Padding: SAME</li>
				<li>Activation: relu</li>
			</ul>
		</td>
	</tr>
	<tr>
		<td>Convolution Layer 4</td>
		<td>
			<ul>
				<li>Filters: 64</li>
				<li>Kernel: 2 x 2</li>
				<li>Stride: 2 x 2</li>
				<li>Padding: SAME</li>
				<li>Activation: relu</li>
			</ul>
		</td>
	</tr>
	<tr>
		<td>Max Pooling Layer 1</td>
		<td>
			<ul>
				<li>Pool Size: 2 x 2</li>
			</ul>
		</td>
	</tr>
	<tr>
		<td>Flatten layer</td>
		<td>
			<ul>
			</ul>
		</td>
	</tr>
	<tr>
		<td>Fully Connected Layer 1</td>
		<td>
			<ul>
				<li>Neurons: 48</li>
				<li>Activation: relu</li>
			</ul>
		</td>
	</tr>
   	<tr>
		<td>Fully Connected Layer 2</td>
		<td>
			<ul>
				<li>Neurons: 24</li>
				<li>DropOut: 0.7</li>
				<li>Activation: relu</li>
			</ul>
		</td>
	</tr>
	<tr>
		<td>Fully Connected Layer 3</td>
		<td>
			<ul>
				<li>Neurons: 12</li>
				<li>Activation: relu</li>
			</ul>
		</td>
	</tr>
	<tr>
		<td>Fully Connected Layer 4</td>
		<td>
			<ul>
				<li>Neurons: 6</li>
				<li>Activation: relu</li>
			</ul>
		</td>
	</tr>
	<tr>
		<td>Fully Connected Layer 5</td>
		<td>
			<ul>
				<li>Neurons: 1</li>
			</ul>
		</td>
	</tr>

</table>

## 5. Discussion

Adding the dropout layers was key in getting the model to work; while the validation losses were slightly lower without them. On the actual tracks, the car skidded of the road until I added droput to the middle layers of the convolutional and fully connected sections. Augumenting parts of data with steering angles > 0.3/< -0.3 also helped with the car skidding of roads at turns. As a challenge , I wanted to see how much the model using only images from lane 1 could be generalized to lane2 while using only training images from lane1 wasn't completely successful for lane 2; Flickering the HSV vaules within a threshold showed the most progress.
