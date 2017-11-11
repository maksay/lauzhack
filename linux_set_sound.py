import os
def changeVol(x):
    os.system("pactl set-sink-volume 0 "+str(x)+"%")
