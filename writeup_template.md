## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_notcar.png
[image2]: ./output_images/car_features.png
[image3]: ./output_images/notcar_features.png
[image4]: ./output_images/search_area.jpg
[image5]: ./output_images/window1.jpg
[image6]: ./output_images/window3.jpg
[image7]: ./output_images/seven_frames.png
[image8]: ./output_images/labels.jpg
[image9]: ./output_images/out_img.jpg
[video1]: ./output_project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in from the first to the third code cells of the IPython notebook `detect_vehicles.ipynb`.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `HSV` color space and HOG parameters of `orientations=32`, `pixels_per_cell=8` and `cells_per_block=2` for both a car image and a noncar image:

![alt text][image2]
![alt text][image3]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and came to the conclusion that `orientations=32`, `pixels_per_cell=8`, and `cells_per_block=2` are good values for extractinng HOG features, because it was by far the most important feature in classifying cars in an image. Spatial and color histogram features didn't seem very useful for the model, but I still wanted to maintain some information about the color of the cars, so I used the values `spatial_size=(4, 4)` and `hist_bins = 4` for those parameters.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I extracted features from the car and noncar images, fit a `StandardScaler` from the `sklearn.preprocessing` library to those features, and split the normalized data to a training and a testing set, all in the seventh cell of the IPython notebook. After this, I trained a linear SVM in the eighth cell. The `C` parameter didn't seem to affect the performance of the model all that much if at all.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

After some experimentation, I decided that I wanted my sliding window search to detect/draw as few as possible, and as correctly as possible. This lead to three window scales, from search window size 130 (`scale = 2.03`) to size 64 (`scale = 1`), and to `cells_per_step = 2`. The sliding window implementation was done in the eleventh cell of the notebook. I also modified the search area so that each window only searched from an area of interest. Here is a visualization of the window search areas:

![alt text][image4]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on three scales using HSV 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result. Here are some example images:

![alt text][image5]
![alt text][image6]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_images/output_project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected. (cells 10, 13, and 14 in the IPython notebook `detect_vehicles.ipynb`)

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are seven frames and their corresponding heatmaps:

![alt text][image7]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all seven frames:
![alt text][image8]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image9]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The pipeline seems to have some problems detecting the white car in the test images and in the video stream. That is, it detects the black car more reliably than the white one. Better optimization of parameters could help improve their detection. Also, as in mahcine learning in general, more training data could help the pipeline's performance quite a bit. The pipeline also has some trouble with positive detections in the edges of the image. This could be improved by increasing the number of search windows there.
