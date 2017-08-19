# Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


## Drawing boxes

The `Box` class stores a box represented by 2 `Point`. The `draw` method allows to draw a box on a given image

![Draw a box][draw_box]

## Color spaces

The `ImageFeature` class stores the features of an image. The `color_space` method extracts the features of a given image with respect to the color space. 

|image|histogram of HSV color space|
|-----|----------------------------|
|![cropped_black_car]|![color_space]|

The `hog` method extracts the histogram of oriented gradients from an image

|image|HOG|
|-----|----------------------------|
|![cropped_black_car]|![hog]|


[//]: # (Image References)
[draw_box]: ./output_images/draw_box.jpg "Drawing a box"
[cropped_black_car]: ./output_images/cropped_black_car.jpg "Black car"
[color_space]: ./output_images/color_space.jpg "Color space features"
[hog]: ./output_images/hog.jpg "HOG"