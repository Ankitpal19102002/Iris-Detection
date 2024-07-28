#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install opencv-python')


# In[2]:


import urllib.request

# URLs for Haar cascade XML files
face_cascade_url = 'https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_alt.xml?raw=true'
eye_cascade_url = 'https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_eye.xml?raw=true'

# Paths where the files will be saved
face_cascade_path = 'haarcascade_frontalface_alt.xml'
eye_cascade_path = 'haarcascade_eye.xml'

# Download and save the files
urllib.request.urlretrieve(face_cascade_url, face_cascade_path)
urllib.request.urlretrieve(eye_cascade_url, eye_cascade_path)

print("Files downloaded and saved.")


# In[ ]:


import cv2
import numpy as np

# Load Haar cascade classifiers
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Check if classifiers are loaded properly
if face_cascade.empty():
    print("Error: Could not load face cascade classifier.")
    exit()
if eye_cascade.empty():
    print("Error: Could not load eye cascade classifier.")
    exit()

# Initialize camera
cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame. Check camera connection.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Region of interest (ROI) for eyes
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        # Detect eyes within the face ROI
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
        
        for (ex, ey, ew, eh) in eyes:
            # Draw rectangle around the eyes
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Camera Feed', frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()


# In[ ]:




