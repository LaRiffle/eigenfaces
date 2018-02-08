import imageio
import numpy as np
from os import listdir
from os.path import isfile, join

path = '/home/pi/Documents/test/'
files = [path+file for file in listdir(path) if isfile(join(path, file))]

for file in files:
    print(file)
    img = imageio.imread(file)
    print(img.shape)
    width = img.shape[1]
    img = img[:,int(0.25*width):int(0.75*width),:]
    # img = np.apply_along_axis(np.mean, 2, img)
    imageio.imwrite(path + 'test.jpg', img)
    
