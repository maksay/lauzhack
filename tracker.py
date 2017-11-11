import cv2
import numpy as np

def detect_object(img):
    blurValue = 41  # GaussianBlur parameter
    minH = 0
    maxH = 255
    minS = 150
    maxS = 255
    minV = 150
    maxV = 255

    out = cv2.GaussianBlur(img, (blurValue, blurValue), 0)
    out = cv2.cvtColor(out, cv2.COLOR_BGR2HSV);
    mask = cv2.inRange(out, np.asarray((minH, minS, minV)),
                            np.asarray((maxH, maxS, maxV)))
    connectivity = 4
    output = cv2.connectedComponentsWithStats(mask, connectivity, cv2.CV_32S)
    num_labels = output[0]

    if num_labels == 0: return None

    labels = output[1]
    stats = output[2]
    centroids = output[3]

    # Remove largest - background
    stats[np.argmax(stats[:, cv2.CC_STAT_AREA]), cv2.CC_STAT_AREA] = 0

    max_component = np.argmax(stats[:, cv2.CC_STAT_AREA])

    if stats[max_component, cv2.CC_STAT_AREA] < 50: return None

    return centroids[max_component]

cap = cv2.VideoCapture(1)
cnt = 0
while( cap.isOpened() ) :
    ret,img = cap.read()
    img = cv2.resize(img, None, None, 0.2, 0.2)
    pos = detect_object(img)
    if pos is not None:
        img = cv2.circle(img, (int(pos[0]), int(pos[1])), 10, (255, 0, 0))

    cv2.imshow('orig',img)

    k = cv2.waitKey(10)
    if k == 27:
        break
    continue

    #hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV);
    #mid = img[img.shape[0] // 2, img.shape[1] // 2, :]

    #diff = np.sqrt(np.sum((img - mid)**2, axis = 2, keepdims = True))
    #diff = np.tile(diff, (1, 1, 3))
    #diff[diff > 10] = 0
    #diff[diff > 0] = 1
    #img *= np.uint8(diff)
    cv2.imshow('input',hsv)

    k = cv2.waitKey(10)
    if k == 27:
        break
    continue

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)

    ret,thresh1 = cv2.threshold(blur,70,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    _, contours, hierarchy = cv2.findContours(thresh1,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE)
    drawing = np.zeros(img.shape,np.uint8)

    max_area=0

    for i in range(len(contours)):
            cnt=contours[i]
            area = cv2.contourArea(cnt)
            if(area>max_area):
                max_area=area
                ci=i
    cnt=contours[ci]
    hull = cv2.convexHull(cnt)
    moments = cv2.moments(cnt)
    if moments['m00']!=0:
                cx = int(moments['m10']/moments['m00']) # cx = M10/M00
                cy = int(moments['m01']/moments['m00']) # cy = M01/M00

    centr=(cx,cy)
    cv2.circle(img,centr,5,[0,0,255],2)
    cv2.drawContours(drawing,[cnt],0,(0,255,0),2)
    cv2.drawContours(drawing,[hull],0,(0,0,255),2)

    cnt = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
    hull = cv2.convexHull(cnt,returnPoints = False)

    if(1):
               defects = cv2.convexityDefects(cnt,hull)
               mind=0
               maxd=0
               for i in range(defects.shape[0]):
                    s,e,f,d = defects[i,0]
                    start = tuple(cnt[s][0])
                    end = tuple(cnt[e][0])
                    far = tuple(cnt[f][0])
                    dist = cv2.pointPolygonTest(cnt,centr,True)
                    cv2.line(img,start,end,[0,255,0],2)

                    cv2.circle(img,far,5,[0,0,255],-1)
               print(i)
               i=0
    cv2.imshow('output',drawing)
    cv2.imshow('input',img)

    k = cv2.waitKey(10)
    if k == 27:
        break
