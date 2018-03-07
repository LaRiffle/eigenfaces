"""
===================================================
   Faces recognition experiment using eigenfaces
===================================================
"""

from pynput import keyboard
import picamera
from PIL import Image
import numpy as np
import imageio

from sklearn.externals import joblib

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


path = '/home/pi/Documents/eigenfaces/'
#path = '/Users/ryffel/Documents/TPE/eigenfaces/'

print('Prise de photo...')

camera = picamera.PiCamera()
camera.color_effects = (128,128)
camera.start_preview()


def on_press(key):
    filepath = path + 'test.jpg'
    camera.capture(filepath)
    camera.stop_preview()
    print('Détection...')

    # ensure img is in B&W
    image_file = Image.open(filepath)  # open colour image
    image_file = image_file.convert('L')  # convert image to black and white
    image_file.save(filepath)

    # crop
    img = imageio.imread(filepath)
    width = img.shape[1]
    height = img.shape[0]
    w = height  * 37 / 52
    l = (width - w)/2
    img = img[int(0*height):int(1*height), int(l):int(width - l)]
    imageio.imwrite(filepath, img)

    # reduce quality
    basewidth = 37
    img = Image.open(filepath)
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent))) + 1
    img = img.resize((basewidth, hsize), Image.ANTIALIAS)
    img.save(filepath)

    # load algorithm and make prediction
    target_names = ['max', 'victor']
    img = imageio.imread(filepath)
    X_test = np.array([img.flatten()])
    pca = joblib.load('pca.pkl')
    X_test_pca = pca.transform(X_test)
    clf = joblib.load('model.pkl')
    y_pred = clf.predict(X_test_pca)
    print('Le visage détecté est probablement celui de ' + target_names[y_pred[0]] + '.')
    return False

# Collect events until released
with keyboard.Listener(on_press=on_press) as listener:
    listener.join()

