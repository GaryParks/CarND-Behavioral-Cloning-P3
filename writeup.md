#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/arch.png "Model Visualization"
[image2]: ./images/center_2017_08_16_14_08_28_883.jpg "Center lane driving"
[image3]: ./images/center_2017_08_17_17_07_08_441.jpg "Dirt edge"
[image4]: ./images/center_2017_08_16_14_09_14_552.jpg "Bridge"
[image5]: ./images/Original-Image.png "Normal Image"
[image6]: ./images/Flipped-Image.png "Flipped Image"
[image7]: ./images/data.png "Gathered data statistics"
[image8]: ./images/MSE_Loss_Graph.png "Loss Graph"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md summarizing the results
* run1.mp4 containing 1 lap of track one in autonomus mode

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network based on Nvidia's E2E CNN.  The model consisted of 3 convolutional neural network layers with 5x5 kernel size and 2x2 stride with RELU, followed by 2 more convolution layers with 3x3 kernel size with RELU. This was then followed by 3 fully connected layers and 1 output layer. I used a dropout of 0.4 for the fully connected layers to avoid overfitting of the network, along with additional RELU.(model.py lines 188-214) 

The model includes RELU layers to introduce nonlinearity (code lines 188,190,192,194,196,202,206,210), and the data is normalized in the model using a Keras lambda layer (code line 186). The top and bottom of the images were also cropped to remove potentially distracting parts of the images (code line 184).

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 201, 205, 209). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 133-139). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 222).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, along with extra data of the more unique areas (bridge and dirt sides).

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

My first step was to use a convolution neural network model similar to the Nvidia model used in the course. I thought this model might be appropriate because of the large performance increase compared to the more basic model used in the beginning of the lectures.

In order to gauge how well the model was working, I split 20% of my image and steering angle data into a validation set, leaving the rest for training. I found that my first model had a low mean squared error but could not drive through the course completely, so I increased the amount of data and modified the correction factor that I was using for the left and right images.

To combat the overfitting, I modified the model so that with additional dropout layeers and RELUs to add nonlinearity.

I also used a generator from Keras with a batch size of 32 to generate the flipped data at run time. I hoped that this would help in performance.  I have a GPU on this PC, but not a very high performance one.  Each epoch ended up taking about 150s.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 179-214) consisted of a convolution neural network based on Nvidias E2E CNN.

Here is a visualization of the architecture:

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I used the mouse for steering input and first recorded two laps on track one going clockwise and counter-clockwise using center lane driving. I saw that there was a provided dataset from Udacity, but I was not sure about the quality so I decided not to use it.  Here is an example image of center lane driving:

![alt text][image2]

I then recorded extra data the vehicle in the unique areas (bridge and dirt sides) along with an additional lap of data (clockwise and counter-clockwise) due to noticing some strange behavior in the first test.

Here are examples of the bridge and dirt edge driving:

![alt text][image4]
![alt text][image3]
After the collection process, I had 7705 number of data points.


I then preprocessed this data by cropping the top of the images by 60 pixels and the bottom by 25 to remove the scenery and vehicle.

To augment the data sat, I also flipped images and angles thinking that this would help keep the car in the center. For example, here is an image that has then been flipped:

![alt text][image5]
![alt text][image6]


This gave me a total of 15410 data points to work with.
![alt text][image7]


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10 as evidenced by both the training and validation losses continuing to decrease until the final epoch. 

![alt text][image8]

I used an adam optimizer so that manually training the learning rate wasn't necessary.
