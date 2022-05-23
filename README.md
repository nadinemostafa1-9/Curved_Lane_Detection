# Curved_Lane_Detection
This project support detecting the lane in videos and images taken by a camera fixed at the center of front of a car.

It has 3 modes of operation:
- debug mode:     detects the lane in a video, calculates the curvature radius and car offset from the lane center, and displays the intermediate pipeline steps outputs, 
                  and store the video displayed in the path secified by the user.
                  
- no-debug mode:  detects the lane in a video, calculates the curvature radius and car offset from the lane center, and store the video displayed in the path 
                  secified by the user
                  
- image mode:     detects the lane in an image and calculates the curvature radius and car offset from the lane center

To run the project, you can use the laneDetection.bat shell using command:

  laneDetection.bat [src video or image] [dst video] [mode]
  
or equivalently you can use:

  python main.py [src video or image] [dst video] [mode]
  
Please notice that all the arguments are mandatory.

[mode] can only be, to specifiy one of the modes mentioned ealier:
- debug
- no-debug
- image
  
  Car detection steps :
 -Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
 -Apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
for those first two steps normalize your features and randomize a selection for training and testing.
-Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
Run the pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
Estimate a bounding box for vehicles detected.
