import cv2
import numpy as np

slider_level = 0.5

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

def draw_slider(img):
    h,w,_ = img.shape
    cv2.rectangle(img,(w // 10, h // 10),(2 * w // 10, 9 * (h // 10)),(255,0,0),2)
    # Set filled slider level
    img[9 * (h // 10) - int(8 * (h // 10) * slider_level) : 9 * (h // 10), w // 10:2 * (w // 10),:] += np.array([255 // 5, 0, 0], dtype=np.float)
    img[9 * (h // 10) - int(8 * (h // 10) * slider_level) : 9 * (h // 10), w // 10:2 * (w // 10),:] /= 1.2
    return img

cap = cv2.VideoCapture(0)
cnt = 0

while cap.isOpened():
    ret,img = cap.read()
    img = cv2.resize(img, None, None, 0.5, 0.5)
    h,w,_ = img.shape

    slider_region = img[h // 10:9 * (h // 10), w // 10:2 * (w // 10),:]
    pos = detect_object(slider_region)
    pos_global = detect_object(img)

    if pos is not None:
        img = cv2.circle(img, (int(w // 10 + pos[0]), int(h // 10 + pos[1])), 10, (255, 0, 0))
        slider_level = 1.0 - (pos[1] / slider_region.shape[0])

    if pos_global is not None:
        img = cv2.circle(img, (int(pos_global[0]), int(pos_global[1])), 10, (0, 255, 0))

    # Recast img as float for easier manipulation
    img_float = np.array(img, dtype=np.float)

    img = draw_slider(img_float)

    # Recast img back into uint8 for easier manipulation
    img = np.clip(img, 0, 255)
    img = np.array(img, dtype=np.uint8)

    cv2.imshow('orig',img)

    k = cv2.waitKey(10)
    if k == 27:
        break
