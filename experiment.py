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


path = '/home/pi/Documents/eigenfaces/'
#path = '/Users/ryffel/Documents/TPE/eigenfaces/'

camera = picamera.PiCamera()
camera.color_effects = (128,128)
camera.start_preview()


def on_press(key):
    filepath = path + 'test.jpg'
    camera.capture(filepath)

    # crop
    img = imageio.imread(filepath)
    width = img.shape[1]
    height = img.shape[0]
    img = img[int(0*height):int(1*height), int(0.2669 * width):int(0.7330 * width)]
    imageio.imwrite(filepath, img)

    # reduce quality
    basewidth = 37
    img = Image.open(filepath)
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
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
with keyboard.Listener(
        on_press=on_press) as listener:
    listener.join()

