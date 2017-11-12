import cv2
import numpy as np
from skimage import filters

barwin = np.zeros((1,512,3), np.uint8)
cv2.namedWindow('BarWindow')

#face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#face2_cascade = cv2.CascadeClassifier('haarcascade_profileface.xml')
face_cascades = [
cv2.CascadeClassifier(path)
#for path in ['hands.xml']
for path in ['haarcascade_frontalface_default.xml',
             'haarcascade_profileface.xml']
]

fist_cascades = [
cv2.CascadeClassifier(path)
#for path in ['hands.xml']
for path in ['palm.xml']
]

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    # compute the area of intersection rectangle
    interArea = (xB - xA + 1) * (yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def detect_face(gray, cascades):

    for cascade in cascades:

        faces = cascade.detectMultiScale(gray, 1.3, 5)
        face_size = [w * h for (x, y, w, h) in faces]

        if len(face_size) > 0:
            max_face = np.argmax(face_size)
            face = faces[max_face]
            return face

    return (None, None, None, None)

def track_face(img, old_face, tracker, cascades):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    new_face = detect_face(gray, cascades)
    if old_face[0] is not None:
        face = tracker.update(img)
    else:
        face = (None, None, None, None)

    if face[0] is not None:
        track, face = tracker.update(img)
        if not track:
            face = (None, None, None, None)

    if face[0] is None and new_face[0] is None:
        face = old_face
    elif face[0] is None and new_face[0] is not None:
        face = new_face
    elif face[0] is not None and new_face[0] is None:
        face = face
    else:
        metric =  bb_intersection_over_union(face, new_face)
        if metric < 0.7:
            face = new_face

    return face

def detect_hands(thresholded, face):

    (x, y, w, h) = face
    left_box = (0, 0, x - w // 2, y + h)
    right_box = (x + w + w // 2, 0, img.shape[1], y + h)
    top_box = (left_box[2], 0, right_box[0], max(0, y - h // 2))

    kernel = np.ones((5,5),np.uint8)
    eroded = cv2.erode(thresholded, kernel, iterations = 1)
    #eroded = np.copy(thresholded)

    MIN_PIX = 100

    if np.sum(eroded[top_box[1] : top_box[3], top_box[0] : top_box[2]] > 0) > MIN_PIX:
        return True, None, None, left_box, right_box, top_box


    py, px = np.where(eroded[left_box[1] : left_box[3], left_box[0] : left_box[2]] > 0)
    if len(px) > MIN_PIX:
        idx = np.argmin(py * img.shape[1] + px)
        lft = (px[idx] + left_box[0], left_box[1] + py[idx])
    else:
        lft = None


    py, px = np.where(eroded[right_box[1] : right_box[3], right_box[0] : right_box[2]] > 0)
    if len(px) > MIN_PIX:
        idx = np.argmin(py * img.shape[1] + px)
        rgt = (px[idx] + right_box[0], py[idx] + right_box[1])
    else:
        rgt = None

    return False, lft, rgt, left_box, right_box, top_box

face = (None, None, None, None)
face_tracker = cv2.Tracker_create("MIL")

fist = (None, None, None, None)
fist_tracker = cv2.Tracker_create("MIL")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,720);
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,450);
cnt = 0

left_history = []
right_history = []

while( cap.isOpened() ) :
    ret,img = cap.read()
    img = cv2.resize(img, None, None, 0.5, 0.5)
    img = cv2.flip(img, 1)
    img2 = np.copy(img)

    # Face detection
    face = track_face(img, face, face_tracker, face_cascades)
    if face[0] is not None:
        (x, y, w, h) = face
        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)
        cv2.rectangle(img2,(x,y),(x+w,y+h),(255,0,0),2)
        fface = img2[y:y+h, x:x+w, :]
        fface = cv2.GaussianBlur(fface,(45,45),0)
        img2[y:y+h, x:x+w] = fface
        #print(face)
    else:
        continue

    # BG SUB
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    if cnt < 30:
        if cnt == 0:
            bg = gray.copy().astype("float")
            cnt = 1
        else:
            cv2.accumulateWeighted(gray, bg, 0.5)
            cnt += 1


    diff = cv2.absdiff(bg.astype("uint8"), gray)
    thresholded = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]

    # Hand detection
    on_top, pos_lft, pos_rgt, left_box, right_box, top_box = detect_hands(thresholded, face)

    img2 = cv2.rectangle(img2, (left_box[0], left_box[1]), (left_box[2], left_box[3]), (255, 0, 0))
    img2 = cv2.rectangle(img2, (right_box[0], right_box[1]), (right_box[2], right_box[3]), (255, 0, 0))
    if not on_top:
        img2 = cv2.rectangle(img2, (top_box[0], top_box[1]), (top_box[2], top_box[3]), (255, 0, 0))
    else:
        img2 = cv2.rectangle(img2, (top_box[0], top_box[1]), (top_box[2], top_box[3]), (0, 0, 255))

    if pos_lft is not None:
        img2 = cv2.circle(img2, (pos_lft[0], pos_lft[1]), 10, (0, 0, 255))
    if pos_rgt is not None:
        img2 = cv2.circle(img2, (pos_rgt[0], pos_rgt[1]), 10, (0, 0, 255))



    if pos_rgt is not None:
        right_history.append(np.copy(pos_rgt))
        print("Adding %d" % len(right_history))
    else:
        if len(right_history) <= 15 and len(right_history) >= 7:
            print(right_box)
            print(right_history)
            increasing = 0
            non_increasing = 0
            in_margin = 0
            if right_history[0][0] < right_box[0] + (right_box[2] - right_box[0]) * 0.2:
                if right_history[-1][0] > right_box[2] - (right_box[2] - right_box[0]) * 0.2:
                    print("EXIT")
                    exit(0)

        right_history = []




        ##midx = x + w // 2
        #thresholded[:, max(0, x - w // 2) : min(x + w + w // 2, img.shape[1])] = 0
        #thresholded[y + h : img.shape[0], :] = 0

    #img2 = cv2.resize(img2, None, None, 3, 3)
    #print(img2.shape)
    cv2.imshow('orig',img2)


    cv2.imshow('bgsub',thresholded)

        #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #mask = gray < filters.threshold_otsu(gray)
        #mask = np.uint8(mask)
        #mask[mask > 0] = 255
        ##mask = fgbg.apply(cv2.GaussianBlur(img, (41, 41), 0))

        ##mask |= cv2.inRange(cv2.GaussianBlur(img, (41, 41), 0)[:, :, :2],
        ##                   np.array([0, 90], np.uint8),
        ##                   np.array([120, 250], np.uint8))
        #cv2.imshow('mask',mask)



    #if cnt == 1:
    #    track, roi = tracker.update(img)
    #    if track:
    #        (x, y, w, h) = roi
    #        print(x, y, w, h)
    #        x = int(x)
    #        y = int(y)
    #        w = int(w)
    #        h = int(h)
    #        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    #        fface = img[y:y+h, x:x+w, :]
    #        fface = cv2.GaussianBlur(fface,(45,45),0)
    #        img[y:y+h, x:x+w] = fface
    #    else:
    #        print("lost tracker")
    #cv2.imshow('orig',img)

    #print(img.shape)
    #img = cv2.resize(img, None, None, 0.5, 0.5)
    #img = cv2.flip(img, 1)

    #pos = detect_object(img, fgbg, face, tracker)
    #if pos is not None:
    #    img = cv2.circle(img, (int(pos[0]), int(pos[1])), 10, (255, 0, 0))



    k = cv2.waitKey(10)
    if k == 27:
        break
    continue
