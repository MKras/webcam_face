#!/usr/bin/env python3

import sys
import numpy as np
import cv2
# import dlib
import face_recognition
import time
import os
 

cvXMLPath = '/usr/share/opencv4/haarcascades/'

# Load the cascade # set the correct path
face_cascade = cv2.CascadeClassifier(cvXMLPath + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

def prepare_data():
    known_face_encodings_map = {}
    files = os.listdir('images')
    for f in files:
        image  = face_recognition.load_image_file("images/{}".format(f))
        image_encoding = face_recognition.face_encodings(image)[0]
        f = f[ : f.find('.')]
        f = f.replace("_", " ")
        known_face_encodings_map.update({f: image_encoding})
    return known_face_encodings_map

def main():
    
    known_face_encodings_map = prepare_data()

    # print (len(known_face_encodings))
    face_name = ""
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # get it into the correct format
        rgb_frame = frame[:, :, ::-1]    

        # Detect the faces
        min_face_distances = 1
        faces = face_cascade.detectMultiScale(rgb_frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            # print ("Rectangle coords: {} {} {} {}".format(x, y, x+w, y+h))
            
            cropped = frame[y: y+h, x: x+w]
            rgb_cropped = cropped[:, :, ::-1]
            face_image = rgb_cropped

            # cv2.imshow("face image", face_image)
            face_encoding_list = face_recognition.face_encodings(face_image)

            if len(face_encoding_list) < 1: break 

            # continue
            face_encoding = face_encoding_list[0]

            min_face_distances = 1
            
            for name, face in known_face_encodings_map.items():
                face_distances = face_recognition.face_distance([face], face_encoding)
                if min_face_distances > face_distances:
                    min_face_distances = face_distances
                    face_name = name            
            # print ("{} face distance {}".format(face_name, min_face_distances))

            # Display
            font = cv2.FONT_HERSHEY_SIMPLEX #cv2.FONT_HERSHEY_DUPLEX
            if face_name != "" :
                cv2.putText(frame, "{} - {}".format(face_name, min_face_distances), (x + 6, y - 6), font, 0.5, (255, 255, 255), 1)
            cv2.imshow('face detector', frame)

        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(0.25)

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()