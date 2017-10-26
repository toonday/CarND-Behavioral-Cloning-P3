#**Behavioral Cloning** 

##Tunde Oladimeji - Writeup

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

---

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 72 (model.py lines 168-176) 

The model includes RELU layers after each convolution and fully connected layer to introduce nonlinearity (model.py lines 168-183), and the data is normalized in the model using a Keras lambda layer (model.py line 167). 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py line 178). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (model.py line 27). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 193).

####4. Appropriate training data

I used the training data provided by udacity which I believe used a combination of center lane driving, recovering from the left and right sides of the road to help the model learn how to stay on track and also recover back on track if the car gets lost. 

---

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with a simple model and keep increasing the complexity of the architecture until I found an appropriate solution.

My first step was to use a simple fully connected layer just to ensure my pipeline could work as expected.
Next, I used a convolution neural network model similar to the LeNet architecture. I thought this model might be appropriate because it was quick to implement and had performed well on pattern recognition tasks (e.g handwritten character recognition).

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that I added a dropout layer after the final pooling layer.

Then I used more data for trainging to reduce overfitting by using the left and right camera with a correction factor. I also augmented all the images by flipping them horizontally and reversing the steering angle. I needed to resize the images to have enough memory to work with the augmented data set.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track (for example where there was a gap in the road, the car teneded to go into the gap and go off track). To improve the driving behavior in these cases, I had to use a more complex neural network model. I also ensured that the color of the predicted images matched the color that the images where trained on (The OpenCV was used for training and it reads images as BGR while the PIL library was used during simulation and it reads images as RGB. I had to convert the images in the simulation from RGB to BGR).

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 166-184) consisted of a convolution neural network with the following layers and layer sizes ...
My final model was a model similar to the NVIDIA convolution neural network model consisting of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Lambda         		| Normalization Layer   						| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x24 	|
| RELU					|												|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 24x24x36 	|
| RELU					|												|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 20x20x48 	|
| RELU					|												|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 18x18x60 	|
| RELU					|												|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 16x16x72 	|
| RELU					|												|
| Dropout				|												|
| Fully connected		|												|
| RELU					|												|
| Fully connected		|												|
| RELU					|												|
| Fully connected		|												|
| RELU					|												|
| Fully connected		|												|
| RELU					|												|
| Fully connected		|												|
|						|												|

####3. Creation of the Training Set & Training Process

I did not capture any more data since I went with the assumption that the data I have was good enough.
If I found that my model was overfitting the training data set, I would have revisited this assumption.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by the loss error on the validation set increasing while the loss error on the training set kept reducing after about 5 epochs (signs of overfitting). I used an adam optimizer so that manually training the learning rate wasn't necessary.
