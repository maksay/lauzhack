import os

VOLUME_FILE = './volume_val'

def set_slider_value(val, valtype):
  if valtype == 'music':
    os.system('osascript -e "set Volume ' + str(val * 7) + '"')
