import cv2
import numpy as np

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('/Users/vlyubin/code/lauzhack/lauzhack/haarcascade_frontalface_default.xml')

iron_man = cv2.imread('/Users/vlyubin/code/lauzhack/lauzhack/ironman3.png')
print(iron_man[0,0])

while( cap.isOpened() ) :
    ret,img = cap.read()
    img = cv2.resize(img, None, None, 0.5, 0.5)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
      # Modify x,y,w,h a bit
      x_new = max(x - int(0.2 * w), 0)
      y_new = max(y - int(0.2 * h), 0)
      w = int(1.2 * w)
      h = int(1.2 * h)

      # TODO: this can break!
      x = x_new
      y = y_new

      cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
      iron_man_resized = cv2.resize(iron_man, (w, h)) 
      img[y:y+h, x:x+w] = iron_man_resized
    
    cv2.imshow('output',img)
    #cv2.imshow('input',img)

    k = cv2.waitKey(10)
    if k == 27:
        break
