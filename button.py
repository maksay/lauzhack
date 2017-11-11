import cv2
import numpy as np
import time 

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

class Button:
  def __init__(self, timeout, f):
    self.lastTime = 0
    self.timeout = timeout
    self.eps = 0
    
    self.function = f
    
  def setpos(self,pos,radius):
    self.pos = pos
    self.radius = radius
  
  def draw(self, img):
    self.checkPressed(img)
    cv2.circle(img, self.pos, self.radius, [255,0,0])
    cv2.circle(img, self.pos, int(self.radius*self.eps), [255,0,0],-1)
  
  def fillOutside(self,img):
    tmp = img.copy()
    for i in range(0,2*self.radius):
      for j in range(0,2*self.radius):
        if ((i-self.radius)**2 + (j-self.radius)**2 >= self.radius**2):
          tmp[i][j] = [0,0,0]
    return tmp
    
  def checkPressed(self, img):
    buttonRegion = img[self.pos[1]-self.radius:self.pos[1]+self.radius, self.pos[0]-self.radius:self.pos[0]+self.radius]
    buttonRegion = self.fillOutside(buttonRegion)
    pos = detect_object(buttonRegion)
    if (pos is None):
      self.lastTime = 0
      self.eps=0
      return
#    img = cv2.circle(img, (int(pos[0]+self.pos[0]-self.radius),int(pos[1]+self.pos[1]-self.radius)) , 10, (0, 255, 0))
    if (np.linalg.norm(pos-(self.radius,self.radius))<self.radius):
      if (self.lastTime == 0): self.lastTime = time.time()
      self.eps = min((time.time() - self.lastTime)/self.timeout,1)
      if (time.time() - self.lastTime > self.timeout):
        self.function()
        self.lastTime = 0
        self.eps=0
    else:
      self.lastTime = 0
      self.eps=0

cap = cv2.VideoCapture(0)
cnt = 0

b = Button(1, lambda: print("Yes"))

while cap.isOpened():
    ret,img = cap.read()
    img = cv2.resize(img, None, None, 0.5, 0.5)
    h,w,_ = img.shape

    pos_global = detect_object(img)

    if pos_global is not None:
        img = cv2.circle(img, (int(pos_global[0]), int(pos_global[1])), 10, (0, 255, 0))
       
    b.setpos((int(h//5),int(w//4)), 30)
    b.draw(img)

    # Recast img back into uint8 for easier manipulation
    img = np.clip(img, 0, 255)
    img = np.array(img, dtype=np.uint8)

    cv2.imshow('orig',img)

    k = cv2.waitKey(10)
    if k == 27:
        break
