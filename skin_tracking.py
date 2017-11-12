#gesture tracking using skin colour detection

import cv2
import numpy as np
from numpy import sqrt,arccos,rad2deg

camera = cv2.VideoCapture(0)

cv2.namedWindow("Finger tracking",cv2.WINDOW_NORMAL)

while True:
	#run()
	ret, image = camera.read()
	image = cv2.flip(image,1)

	OriginalImg = image.copy()
	NoFilterImg = image.copy()

	image = cv2.blur(image,(5,5)) # *** try blurring with different kernel sizes to check effectiveness ***

	"""OpenCV uses different ranges for HSV as compared to other applications. h:0-180 s:0-255 v:0-255
	hsv color range for use in OpenCV [0,30,60 - 20,150,255] OR [0,40,60-20,150,255] OR [0,10,60-20,150,255]
	NOTE: since skin color can have a wide range, can use markers on finger tips to target a smaller and easy to use color range """
	MIN = np.array([0,30,60],np.uint8)
	MAX = np.array([20,150,179],np.uint8) #HSV: V-79%
	HSVImg = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

	filterImg = cv2.inRange(HSVImg,MIN,MAX) #filtering by skin color
	filterImg = cv2.erode(filterImg,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))) #eroding the image
	filterImg = cv2.dilate(filterImg,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))) #dilating the image

	#getting all the contours
	_, contours, heirarchy = cv2.findContours(filterImg,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)

	#try using only the contour with max area

	if len(contours)==0:
		continue

	Index = []
	index_val = 0

	for cnt in contours:
		tempImage = image.copy()
		tempImage = cv2.subtract(tempImage,image)

		#finding convex hull
		hull = cv2.convexHull(cnt)
		last = None
		for h in hull:
			if last is None:
				cv2.circle(tempImage,tuple(h[0]),5,(0,255,255),2) #0,255,255 - bgr value for yellow - marks convex hull pts i.e. fingertips
			else:
				x = abs(last[0]-tuple(h[0])[0])
				y = abs(last[1]-tuple(h[0])[1])
				distance = sqrt(x**2+y**2)
				if distance>10:
					cv2.circle(tempImage,tuple(h[0]),5,(0,255,255),2) #yellow - marks convex hull pts i.e. fingertips
			last = tuple(h[0])

		m = cv2.moments(cnt)
		if(m["m00"]==0):
			continue
		c_x = int(m["m10"]/m["m00"])
		c_y = int(m["m01"]/m["m00"])
		cv2.circle(tempImage,(c_x,c_y),10,(255,255,0),2) #255,255,0 - bgr value for cyan

		hull = cv2.convexHull(cnt,returnPoints=False)
		defects = cv2.convexityDefects(cnt,hull)
		if defects is None:
			continue

		for i in range(defects.shape[0]):
			s,e,f,d = defects[i,0]
			if d>1000:
				start = tuple(cnt[s][0])
				end = tuple(cnt[e][0])
				far = tuple(cnt[f][0])

				cv2.circle(tempImage,far,5,(255,255,255),-2) #255,255,255 - bgr for white - marks convex defect points
				cv2.line(tempImage,start,far,(0,255,0),5) #lime green color - lines from one convex defect to another - gives outline to obj
				cv2.line(tempImage,far,end,(0,255,0),5) #lime green color - lines from one convex defect to another - gives outline to obj

		OriginalImg = cv2.add(OriginalImg,tempImage)
		index_val += 1

	cv2.drawContours(OriginalImg,contours,-1,(255,0,0),-2) #blue color - marks hands

	cv2.imshow("Finger tracking",OriginalImg)
	if cv2.waitKey(10)==27:
		break
