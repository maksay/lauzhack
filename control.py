VOLUME_FILE = './volume_val'

def set_slider_value(val, valtype):
  if valtype == 'music':
    f = open(VOLUME_FILE, 'w')
    f.write(str(int(val * 100)))
    f.close()
