# Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


## Drawing boxes

The `Box` class stores a box represented by 2 `Point`. The `draw` method allows to draw a box on a given image

![Draw a box][draw_box]

## Color spaces

The `ExtractFeature` class stores the features of an image. The `_color_space` method extracts the features of a given image with respect to the color space. 

|image|histogram of HSV color space|
|-----|----------------------------|
|![cropped_black_car]|![color_space]|

The `_hog` method extracts the histogram of oriented gradients from an image

|image|HOG|
|-----|----------------------------|
|![cropped_black_car]|![hog]|

## Car classifier

The `CarClassifier` class stores the classifier to tell cars and non cars appart:

 - `_process_data` : This method loads a set of images and extract the features from each of them.
 - `train`: This method trains an SVM classifier based on a set of cars and non cars images

The `_process_data` method uses the `ExtractFeatures` class which is responsible of extracting the features from an image. For each image in the dataset, the class will extract :

- The HOG of an image
- The spatial feature and histogram features from the color space

When all features have been extracted for both cars and non cars images, the `train` method uses those to train an SVM classifier. The data is first scaled using the `StandardScaler` class from SKLearn. The dataset is split into two dataset, one for training, one for testing purposes.

After the training phase, my model had a 98.789% accuracy

## Detect cars on an image

The `Window` class implements the `slide` method. This method will iterate on small portion of the image to classify them as cars or not. For each window on the image, the features are going to be extracted and those features will be handed to the classifier to predict whether it contains a car.

Here is an example of a positive and negative classification:

![classification]

[//]: # (Image References)
[draw_box]: ./output_images/draw_box.jpg "Drawing a box"
[cropped_black_car]: ./output_images/cropped_black_car.jpg "Black car"
[color_space]: ./output_images/color_space.jpg "Color space features"
[hog]: ./output_images/hog.jpg "HOG"
[classification]: ./output_images/classification.png "Classification"