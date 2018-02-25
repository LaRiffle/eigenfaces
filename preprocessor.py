import imageio
from os import listdir
from os.path import isfile, join
from PIL import Image


# select path depending of the machine used
#path = '/home/pi/Documents/eigenfaces/'
path = '/Users/ryffel/Documents/TPE/eigenfaces/'
path_source = path + 'images/'
path_target = path + 'data/'

files = [path_source+file for file in listdir(path_source) if isfile(join(path_source, file)) and file != '.DS_Store']


# convert is necessary img in B&W
if False:
    for file in files:
        print(file)
        image_file = Image.open(file)  # open colour image
        image_file = image_file.convert('L')  # convert image to black and white
        image_file.save(file)


for file in files:
    print(file)

    # crop image
    img = imageio.imread(file)
    #print(img.shape)
    height = img.shape[0]
    width = img.shape[1]
    filename = file.split('/')[-1]
    label = filename.split('_')[0]
    if label == 'max':
        img = img[int(0.10*height):int(0.724*height), int(0.38*width):int(0.71*width)]
    elif label == 'victor':
        img = img[int(0.12*height):int(0.88*height), int(0.30 * width):int(0.70 * width)]
    imageio.imwrite(path_target + filename, img)

    # reduce quality
    basewidth = 37
    img = Image.open(path_target + filename)
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, hsize), Image.ANTIALIAS)
    img.save(path_target + filename)


