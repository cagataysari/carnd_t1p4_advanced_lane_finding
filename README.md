##CarND - Project 4 Advanced Lane Finding
###The objective is to recognize a lane and compute the curvature on a road. The road video is recorded from a car, so the departure from lane center can also be obtained. 

---

**Advanced Lane Finding Project**

The project is broken down into the following steps:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./doc/undistortion_calibration_img.jpg "Undistorted"
[image2]: ./doc/undistortion_road_img.jpg "Road Transformed"
[image3]: ./doc/binary_combo_example.png "Binary Example"
[image3_filters]: ./doc/binary_filters.png "Binary Filters"
[image4]: ./doc/persp_birds_eye_view.jpg "Warp Example"
[image5_orig]: ./doc/transformed_processed_test4.jpg "Original Image for fit"
[image5]: ./doc/computed_transformed_processed_test4.jpg "Fit Visual"
[image6]: ./doc/example_output_1036.jpg "Output"
[video1]: ./Processsed_project_video.mp4 "Video"


---

###Camera Calibration

####1. Compute the camera matrix and distortion coefficients. 

The calibration is based on the raw photos of chessboard pattern. The photos are distorted due to inherent camera and lens properties. The first step is to generate the meshgrid of logical coordination of corners. Then using OpenCV function of drawChessboardCorners(), we are able to obtain the actual pixel position of corners in the picture. By correlating the logical coordinations and the physical pixel locations, OpenCV function calibrateCamera() is able to calculate the camera matrix and distortion coefficients.

All the incoming frame picture will have distortion from this camera. Therefore, we save the matrix and coefficients, and apply them to correct distortion in every future frame.  

The calibration logic is encapsulated at source code `util_camera.py`.

An example image after correction is as following.

![alt text][image1]

###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like the below one. It is less obvious to see the distortion if it is not chess-patterned. But look closely at the horizontal line on the windshield, it is warped before correction and straight afterwards.

![alt text][image2]

####2. Filtering for thresholded binary image.  

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at the function processImg() in `util_vision.py`). 

The image are passed through the intermediate filters. The most difficult issue is the noise from shadow. To tackle the problem, it is found that u-channel of YUV color space is perfect in identifying the yellow lane mark, while l-channel of HLS has excellent property in isolating the white lane mark. The down-side of the color thresholding is color noise from random objects or marks on the road.

On the other hand, magnitude Sobel gradient filter is not sensitive to the color noise, but is prone to shadow noise. Combining the good properties of the two, the yellow lane mark can be cleanly filtered by logically AND the u-channel thresholding and magnitude Sobel filter.

The L-channel thresholding filter can robustly identify the white lane mark with noise from other objects. It is logically OR with the u-channel result. To compensate the color noise in white lane marks, the directional Sobel filter as the last layer filters out the noise that does not resemble the lane lines.

The intermediate filter outputs on a challenging shadowy image is as following

![alt text][image3_filters]

Here's an example of the final filtering output for this step. 

![alt text][image3]


####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])

```
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:


Image for fitting:         |  Fit result
:-------------------------:|:-------------------------:
![alt text][image5_orig]   |![alt text][image5] 


####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in function findLinePixels() in `util_findLane.py`

####6. An example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./Processsed_project_video.mp4). Or it can be viewed on youtube by clicking the following image:

[![Lane-finding Video](http://img.youtube.com/vi/R_oIGwcXz1Y/0.jpg)](https://www.youtube.com/embed/R_oIGwcXz1Y "Lane-finding Video")

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

