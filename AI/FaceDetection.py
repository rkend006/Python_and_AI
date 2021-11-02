#!/usr/bin/env python
# coding: utf-8

# # First we will need to import OpenCV and Sys
# Sys is going to be so we can use our webcam

# In[1]:


import cv2
import sys


# # Now we will start the cascading XML file

# In[2]:


# Set the path for the cascade and load it into the memory to be used by opencv
cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

# Opens a video file or a capturing device or an IP video stream for video capturing with API Preference.
video_capture = cv2.VideoCapture(0)


# # This is where the camera turns on and the program waits for user termination

# In[3]:


while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Used to convert an image from one color space to another
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        #  Parameter specifying how much the image size is reduced at each image scale.
        scaleFactor=1.1,
        
        #  Parameter specifying how many neighbors each candidate rectangle should have to retain it.
        minNeighbors=5,
        
        # Minimum possible object size. Objects smaller than that are ignored.
        minSize=(30, 30),
        
        
        # flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around the faces.
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame.
    cv2.imshow('Video', frame)

    # Stop the program with ctr 'q'.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()

print('This will never print since the program is waiting to terminate, camera still on.')


# In[ ]:




