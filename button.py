import cv2
import numpy as np
import time 

FINAL_SCALE_FACTOR = 3

class Button:
  def __init__(self, timeout, f, pos, radius, message = None):
    self.lastTime = 0
    self.timeout = timeout
    self.eps = 0
    self.state = 0 # Off
    self.pos = pos
    self.radius = radius
    
    self.text = message
    
    self.function = f
    
  def draw(self, img):
    scaled_pos = (self.pos[0] * FINAL_SCALE_FACTOR, self.pos[1] * FINAL_SCALE_FACTOR)
    img = cv2.circle(img, scaled_pos, FINAL_SCALE_FACTOR * self.radius, [189,233,0], 4)
    img = cv2.circle(img, scaled_pos, int(FINAL_SCALE_FACTOR * self.radius*self.eps), [221+(221-189)*self.eps,126+(233-128)*self.eps,0], -1)
    if self.text is not None:
      cv2.putText(img, self.text, (scaled_pos[0]-FINAL_SCALE_FACTOR*self.radius//2, scaled_pos[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255))
    return img

  def checkPressed(self, img, pos1, pos2):
    if (pos1 is not None and np.linalg.norm(np.array(pos1) - np.array(self.pos)) < self.radius) or \
        (pos2 is not None and np.linalg.norm(np.array(pos2) - np.array(self.pos)) < self.radius):
      if self.lastTime == 0: self.lastTime = time.time()
      self.eps = min((time.time() - self.lastTime) / self.timeout, 1)
      if time.time() - self.lastTime > self.timeout:
        self.function()
        self.lastTime = 0
        self.eps = 0
        self.state = 1 - self.state
    else:
      self.lastTime = 0
      self.eps = 0
