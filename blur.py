import cv2
import numpy as np

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('/Users/vlyubin/code/lauzhack/lauzhack/haarcascade_frontalface_default.xml')

while( cap.isOpened() ) :
    ret,img = cap.read()
    img = cv2.resize(img, None, None, 0.5, 0.5)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
      #cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
      face = img[y:y+h, x:x+w, :]
      face = cv2.GaussianBlur(face,(45,45),0)
      img[y:y+h, x:x+w] = face
    
    cv2.imshow('output',img)
    #cv2.imshow('input',img)

    k = cv2.waitKey(10)
    if k == 27:
        break
