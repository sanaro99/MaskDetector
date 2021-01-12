#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import required libraries
import numpy as np
import cv2
import sys


# In[2]:


# Load the classifiers
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
eyes_with_eyeglasses_cascade = cv2.CascadeClassifier('haarcascade_eyeglasses.xml')
nose_cascade = cv2.CascadeClassifier('haarcascade_nose.xml')
mouth_cascade = cv2.CascadeClassifier('haarcascade_mouth.xml')


# In[3]:


# Start video capture
video_capture = cv2.VideoCapture(0)
video_capture.isOpened()

# Specify text label parameters
font = cv2.FONT_HERSHEY_SIMPLEX
textPosition = (100,50)
fontScale = 1
fontColor = (255,255,255)
lineType = 2

while True:
    
    # Specify label text
    mask_text = "Good!"
    
    # Capture video frame by frame
    ret, frame = video_capture.read()
    
    # Convert frame to grayscale for better working of the classifiers
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply eye classifier
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    # For each eye found, draw a rectangle around it of BGR color (255,0,0) and thickness 2 px
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(frame, (ex,ey), (ex+ew,ey+eh), (255,0,0), 2)
    
    eyes_with_eyeglasses = eyes_with_eyeglasses_cascade.detectMultiScale(gray, scaleFactor=1.4, minNeighbors=6)
    for (ex,ey,ew,eh) in eyes_with_eyeglasses:
        cv2.rectangle(frame,(ex,ey),(ex+ew,ey+eh),(255,0,0),2)
        
    # If no eyes detected, no face detected
    if(len(eyes)==0 and len(eyes_with_eyeglasses)==0):
        mask_text = "No face detected"
        
    # Apply nose classifier
    nose = nose_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=8)
    # If nose detected => mask not worn properly 
    for (ex,ey,ew,eh) in nose:
        mask_text = "Wear mask properly!"
        cv2.rectangle(frame,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    # Apply mouth classifier
    mouth = mouth_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=7)
    # If mouth detected => mask not worn
    for (ex,ey,ew,eh) in mouth:
        mask_text = "Mask not detected!"
        cv2.rectangle(frame,(ex,ey),(ex+ew,ey+eh),(80,80,80),2)

    # Put label on the frame
    cv2.putText(frame, mask_text, textPosition, font, fontScale, fontColor, lineType)
    
    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Press 'q' to exit the video capture
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()

