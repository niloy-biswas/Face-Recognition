import numpy as np
import cv2
import os
import pickle

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('cascade/data/haarcascade_smile.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")

#load name from pickel dict
labels = {"Person_name": 1}
with open("labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for (x,y,w,h) in faces:
        #print(x,y,w,h)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        id_, conf =recognizer.predict(roi_gray) #recognizer
        if conf>=45 and conf<=85:
            #print(id_)
            print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255, 255, 255)
            stroke = 2
            cv2.putText(frame, name.capitalize(), (x,y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

        img_item = "my-image.png"
        cv2.imwrite(img_item, roi_gray)

        color = (255, 0, 0) #BGR - Blue, green red 0-255
        stroke = 2
        end_cord_x = x+w #width
        end_cord_y = y+h #height
        cv2.rectangle(frame, (x,y), (end_cord_x, end_cord_y), color, stroke)

        '''eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

        smiles = smile_cascade.detectMultiScale(roi_gray)
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (255, 0, 0), 2)'''


    cv2.imshow('frame', frame) #imgshow
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()