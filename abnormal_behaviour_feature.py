#!/usr/bin/env python
# coding: utf-8

# ### Optical Flow Feature for Abnormal Crowd Behavior Detection

# #### Based on the difference angle between the optical flows of the current and previous frame


import numpy as np
import cv2 as cv
import argparse


# deviation angle

# The optical flow of each frame is represented in a 2d array of shape(n,2) where n is the number of pixels being tracked
# opt_flw s used for the current's frame optical flow
# opt_flw_old s used for the previous' frame optical flow


def angle_difference_feature(opt_flw,opt_flw_old):
    rms=np.sqrt(np.sum(opt_flw**2,axis=1))
    angle_difference = np.sum(np.multiply(opt_flw,opt_flw_old),axis=1)/(rms * np.sqrt(np.sum(opt_flw_old**2,axis=1)))
    return rms * angle_difference



#This code calculates the optical flow for video frames and the difference angle between the optical flows
#   - the video frames will be passed through a mask filter first


path_to_video = 'test.mp4'
cap = cv.VideoCapture(path_to_video)

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0,255,(100,3))

# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

# create a list to store the differnce angles of each frame
angle_diff_vid=[]

# variable used to skip calculating the difference angle for the first frame 
frame_num = 0

while(1):
    ret,frame = cap.read()
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # calculate optical flow
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points
    if p1 is not None:
        good_new = p1[st==1]
        good_old = p0[st==1]

    
    ### Calculate the optical flow
    opt_flw= good_new - good_old
    st = st.flatten()
    
    
    ### Calculate the difference angle
    if frame_num!=0 :
        opt_flw_old = opt_flw_old[st==1]
        angle_diff_frame = angle_difference_feature(opt_flw,opt_flw_old)
        angle_diff_frame[::-1].sort()
        angle_diff_frame = angle_diff_frame[0:101]
        angle_diff_vid.append(angle_diff_frame)
        
        
    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)
    
    ### update the previous optical flow
    opt_flw_old= opt_flw
    
    ### skip first frame difference angle
    if frame_num == 0:
        frame_num= frame_num + 1
    