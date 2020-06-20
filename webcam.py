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

# predictor_path = sys.argv[1]
# face_rec_model_path = sys.argv[2]

# based on example: http://dlib.net/face_recognition.py.html

predictor_path = './data/shape_predictor_5_face_landmarks.dat'
face_rec_model_path = './data/dlib_face_recognition_resnet_model_v1.dat'


# # Load all the models we need: a detector to find the faces, a shape predictor
# # to find face landmarks so we can precisely localize the face, and finally the
# # face recognition model.
# detector = dlib.get_frontal_face_detector()
# sp = dlib.shape_predictor(predictor_path)
# facerec = dlib.face_recognition_model_v1(face_rec_model_path)

mkras_image = face_recognition.load_image_file("images/mikhail_krasikau.png")
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
    # face_landmarks_list = face_recognition.face_landmarks(rgb_small_frame)

    # print ("ret '{}' ftame '{}'".format(ret, frame))
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

    # Detect the faces
    #faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    faces = face_cascade.detectMultiScale(gray)
    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        
        cropped = frame[y: y+h, x: x+w]

        small_cropped = cv2.resize(cropped, (0, 0), fx=0.25, fy=0.25)
        rgb_small_cropped = small_cropped[:, :, ::-1]
        # face_landmarks_list = face_recognition.face_landmarks(rgb_small_cropped)

        # cv2.imshow("Show Boxes", cropped)
        # rgb_cropped = cropped[:, :, ::-1]
        gray_cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        gray_cropped = cv2.resize(gray_cropped,(0,0),fx=0.5,fy=0.5)

        # rgb_frame = frame[:, :, ::-1]
        # dirty hack - save image to disk and load it
        # cv2.imwrite('images/face_webcam.jpg', rgb_cropped)
        # cv2.imwrite('images/face_webcam.jpg', rgb_frame)
        # face_image = face_recognition.load_image_file("images/face_webcam.jpg")
        # face_image = face_recognition.load_image_file("images/mikhail_krasikau.png")
        # face_image = face_recognition.load_image_file("images/mkras_tall.jpg")
        face_image = rgb_small_frame #rgb_small_cropped #gray_cropped

        cv2.imshow("face image", face_image)

        face_encoding_list = face_recognition.face_encodings(face_image)

        bbreak = False
        if len(face_encoding_list) < 1:
            #print ("Can not encode face")
            bbreak = True
        else:
            print ("Face encoded")
        
        if bbreak: break

        # comtinue
        face_encoding = face_encoding_list[0]

        # face_encoding = face_recognition.face_encodings(rgb_cropped)[0]
        # print ("face_encoding {}".format(face_encoding))
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        if matches == True:
          print ("Match")
        else:
          print ("Not Match")
        
        

    # Display
    cv2.imshow('face detector', frame)

    # # Display the resulting frame
    # cv2.imshow('frame',gray)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    time.sleep(0.25)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()