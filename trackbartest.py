import cv2
import numpy as np
import math
from control import *
from button import Button

DRAW_SLIDERS = True
APPLY_SLIDER_ACTIONS = True
FINAL_SCALE_FACTOR = 3

# Mutable flags
iron_man_on = False
blur_on = False

DRAW_GESTURES = False

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

def iron_man_toogle():
    global iron_man_on
    global blur_on

    if blur_on:
        blur_on = False

    iron_man_on = not iron_man_on

def blur_toogle():
    global iron_man_on
    global blur_on

    if iron_man_on:
        iron_man_on = False

    blur_on = not blur_on


sliders = [[(30//FINAL_SCALE_FACTOR, 30//FINAL_SCALE_FACTOR, 90//FINAL_SCALE_FACTOR, 390//FINAL_SCALE_FACTOR), 0.5, 'music'],
           [(150//FINAL_SCALE_FACTOR, 30//FINAL_SCALE_FACTOR, 90//FINAL_SCALE_FACTOR, 390//FINAL_SCALE_FACTOR), 0.5, 'brightness']] # each slider is (x,y,w,h) and slider_level
buttons = [Button(1.0, iron_man_toogle, (900//FINAL_SCALE_FACTOR, 90//FINAL_SCALE_FACTOR), 66//FINAL_SCALE_FACTOR),
           Button(1.0, blur_toogle, (750//FINAL_SCALE_FACTOR, 90//FINAL_SCALE_FACTOR), 66//FINAL_SCALE_FACTOR)]

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

def apply_sliders(img, pos1, pos2):
    img = np.array(img, dtype=np.float)
    col_idx = 0

    for slider in sliders:
        [(x,y,w,h), slider_level, tp] = slider
        # Set the new slider_level value
        old_slider_level = slider_level
        if pos1 is not None and y <= pos1[1] and pos1[1] <= y + h and x <= pos1[0] and pos1[0] <= x + w:
            slider_level = (y + h - pos1[1]) / 1.0 / h

        if pos2 is not None and y <= pos2[1] and pos2[1] <= y + h and x <= pos2[0] and pos2[0] <= x + w:
            slider_level = (y + h - pos2[1]) / 1.0 / h

        if old_slider_level != slider_level:
            sliders[col_idx][1] = slider_level
            if APPLY_SLIDER_ACTIONS:
                set_slider_value(sliders[col_idx][1], sliders[col_idx][2])

        col_idx += 1

def draw_sliders(img):
    img = np.array(img, dtype=np.float)
    col_idx = 0

    for slider in sliders:
        [(x,y,w,h), slider_level, tp] = slider
        x *= FINAL_SCALE_FACTOR
        y *= FINAL_SCALE_FACTOR
        w *= FINAL_SCALE_FACTOR
        h *= FINAL_SCALE_FACTOR
        cv2.rectangle(img,(x, y),(x + w, y + h),(255,0,0),2)
        # Set filled slider level
        img[y + h - int(h * slider_level) : y + h, x : x + w,:] += np.array([255, 0, 0], dtype=np.float)
        img[y + h - int(h * slider_level) : y + h, x : x + w,:] /= 2

        col_idx += 1


    img = np.clip(img, 0, 255)
    img = np.array(img, dtype=np.uint8)

    return img

face = (None, None, None, None)
try:
    face_tracker = cv2.Tracker_create("MIL")
except:
    face_tracker = cv2.TrackerMIL_create()
cap = cv2.VideoCapture(0)


cap.set(cv2.CAP_PROP_FRAME_WIDTH,720)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
cnt = 0

left_history = []
right_history = []
message_queue = []


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
        if blur_on:
            fface = img2[y:y+h, x:x+w, :]
            fface = cv2.GaussianBlur(fface,(45,45),0)
            img2[y:y+h, x:x+w] = fface
        #print(face)
    else:
        continue

    cnt += 1
    # BG SUB
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    if cnt <= 60:
        if cnt == 1:
            bg = gray.copy().astype("float")
        else:
            cv2.accumulateWeighted(gray, bg, 0.5)

        img2 = cv2.putText(img2, "INITIALIZATION", (0, img.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow('orig',img2)


    diff = cv2.absdiff(bg.astype("uint8"), gray)
    thresholded = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]

    if np.sum(thresholded > 0) > 0.5 * thresholded.shape[0] * thresholded.shape[1]:
        cnt = 1
        message_queue = []
        continue

    # Hand detection
    on_top, pos_lft, pos_rgt, left_box, right_box, top_box = detect_hands(thresholded, face)

    img2 = cv2.rectangle(img2, (left_box[0], left_box[1]), (left_box[2], left_box[3]), (255, 0, 0))
    img2 = cv2.rectangle(img2, (right_box[0], right_box[1]), (right_box[2], right_box[3]), (255, 0, 0))
    if not on_top:
        img2 = cv2.rectangle(img2, (top_box[0], top_box[1]), (top_box[2], top_box[3]), (255, 0, 0))
    else:
        img2 = cv2.rectangle(img2, (top_box[0], top_box[1]), (top_box[2], top_box[3]), (0, 0, 255))

    fingerradius = 100
    if pos_lft is not None:
        if isfinger(thresholded, left_box):
            img2 = cv2.circle(img2, (pos_lft[0], pos_lft[1]), 5, (0, 255, 0))
        else:
            img2 = cv2.circle(img2, (pos_lft[0], pos_lft[1]), 5, (0, 0, 255))

    if pos_rgt is not None:
        img2 = cv2.circle(img2, (pos_rgt[0], pos_rgt[1]), 20, (0, 0, 255))


    # L->R gesture, R->L gesture for right hand
    if pos_rgt is not None:
        right_history.append(np.copy(pos_rgt))
        #print("Adding %d" % len(right_history))
    else:
        if pos_lft is None:
            if len(right_history) <= 15 and len(right_history) >= 7:
                if right_history[0][0] < right_box[0] + (right_box[2] - right_box[0]) * 0.2:
                    if right_history[-1][0] > right_box[2] - (right_box[2] - right_box[0]) * 0.2:
                        message_queue.append((cnt, "RH: L->R"))
            if len(right_history) <= 15 and len(right_history) >= 7:
                if right_history[0][0] > right_box[2] - (right_box[2] - right_box[0]) * 0.2:
                    if right_history[-1][0] < right_box[0] + (right_box[2] - right_box[0]) * 0.2:
                        message_queue.append((cnt, "RH: R->L"))
        right_history = []

    # L->R gesture, R->L gesture for left hand
    if pos_lft is not None:
        left_history.append(np.copy(pos_lft))
        #print("Adding %d" % len(right_history))
    else:
        if pos_rgt is None:
            if len(left_history) <= 15 and len(left_history) >= 7:
                if left_history[0][0] < left_box[0] + (left_box[2] - left_box[0]) * 0.1:
                    if left_history[-1][0] > left_box[2] - (left_box[2] - left_box[0]) * 0.3:
                        message_queue.append((cnt, "LH: L->R"))
            if len(left_history) <= 15 and len(left_history) >= 7:
                if left_history[0][0] > left_box[2] - (left_box[2] - left_box[0]) * 0.3:
                    if left_history[-1][0] < left_box[0] + (left_box[2] - left_box[0]) * 0.1:
                        message_queue.append((cnt, "LH: R->L"))
        left_history = []

    # Simultaneous L-L, R-R
    if pos_lft is not None and pos_rgt is not None:
        if len(left_history) > 10 and len(right_history) > 10:
            dec_left = 0
            inc_left = 0
            for i in range(len(left_history) - 10, len(left_history)):
                if left_history[i - 1][0] > left_history[i][0]:
                    dec_left += 1
                if left_history[i - 1][0] < left_history[i][0]:
                    inc_left += 1
            inc_right = 0
            dec_right = 0
            for i in range(len(right_history) - 10, len(right_history)):
                if right_history[i - 1][0] < right_history[i][0]:
                    inc_right += 1
                if right_history[i - 1][0] > right_history[i][0]:
                    dec_right += 1
            if dec_left == 10 and inc_right == 10:
                left_history = []
                right_history = []
                message_queue.append((cnt, "Zoom: IN"))
            if inc_left == 10 and dec_right == 10:
                left_history = []
                right_history = []
                message_queue.append((cnt, "Zoom: OUT"))


    # Draw sliders
    if APPLY_SLIDER_ACTIONS:
        apply_sliders(img2, pos_lft, pos_rgt)
        for button in buttons:
            button.checkPressed(img2, pos_lft, pos_rgt)

    img2 = cv2.resize(img2, None, None, FINAL_SCALE_FACTOR, FINAL_SCALE_FACTOR)

    # Draw gestures
    if DRAW_GESTURES:
        while len(message_queue) > 0 and message_queue[0][0] < cnt - 30:
            message_queue = message_queue[1:]
        if len(message_queue) > 0 and message_queue[0][0] >= cnt - 30 and message_queue[0][0] <= cnt:
            img2 = cv2.putText(img2, message_queue[0][1], (0, img.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


    if DRAW_SLIDERS:
        img2 = draw_sliders(img2)
        for button in buttons:
            img2 = button.draw(img2)

    cv2.imshow('orig',img2)
    #cv2.imshow('bgsub',thresholded)

    k = cv2.waitKey(10)
    if k == 27:
        break
    continue
