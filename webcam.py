#!/usr/bin/env python3

import sys
import numpy as np
import cv2
# import dlib
import face_recognition
import time

cvXMLPath = '/usr/share/opencv4/haarcascades/'

# Load the cascade # set the correct path
face_cascade = cv2.CascadeClassifier(cvXMLPath + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

# based on face_recognition  examples

sample_image_path = "images/mikhail_krasikau.png"
# sample_image_path = "images/bruce_willis.jpg"
mkras_image = face_recognition.load_image_file(sample_image_path)
mkras_encoding = face_recognition.face_encodings(mkras_image)[0]

known_face_encodings = [
    mkras_encoding,
]
known_face_names = [
    "Mikhail Krasikau",
]

print (len(known_face_encodings))

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
        
    # get it into the correct format
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]    
    rgb_frame = frame[:, :, ::-1]    

    # Detect the faces
    faces = face_cascade.detectMultiScale(rgb_frame)
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # faces = face_cascade.detectMultiScale(gray)
    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        cropped = frame[y: y+h, x: x+w]
        rgb_cropped = cropped[:, :, ::-1]
        face_image = rgb_cropped

        # cv2.imshow("face image", face_image)

        face_sample = cv2.imread(sample_image_path)
        cv2.imshow("face sample", face_sample)

        face_encoding_list = face_recognition.face_encodings(face_image)

        bbreak = False
        if len(face_encoding_list) < 1:
            #print ("Can not encode face")
            bbreak = True
        else:
            print ("Face encoded")
        
        if bbreak: break

        # continue
        face_encoding = face_encoding_list[0]

        # See if the face is a match for the known face(s)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        print ("face distances: {}".format(face_distances))
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        if matches == True:
          print ("!!!!!Match")
        else:
          print ("Not Match")        

    # Display
    cv2.imshow('face detector', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    time.sleep(0.25)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()